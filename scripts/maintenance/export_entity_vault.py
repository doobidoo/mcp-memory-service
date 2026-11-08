#!/usr/bin/env python3
"""
export_entity_vault.py - Read-only export of entity graph into a Markdown Vault.

Usage:
    python scripts/maintenance/export_entity_vault.py --out <dir> [--store <name>] [--limit N] [--db-path <path>]
"""

import argparse
import asyncio
import datetime
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from mcp_memory_service.config import SQLITE_VEC_PATH
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.server.handlers.graph import (
    extract_extractive_summary,
    get_entity_detail,
)
from mcp_memory_service.storage.graph import GraphStorage, normalize_entity_id


class ReadOnlyVaultStorage:
    """Lightweight read-only storage adapter for vault export.

    Opens SQLite in read-only URI mode without triggering migrations, embedding model
    loads, dimension checks, or write locks on the source database.
    """
    get_by_hash_is_store_scoped = True

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    async def initialize(self):
        resolved = Path(self.db_path).resolve()
        if not resolved.exists():
            # If the database file does not exist yet, treat as empty in-memory db
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            return
        # Open in read-only URI mode to guarantee zero writes to the source db.
        # Use as_uri() to properly percent-encode special characters like '#' and '?'.
        uri = f"{resolved.as_uri()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    async def get_by_hash(self, content_hash: str, store: Optional[str] = None) -> Optional[Memory]:
        if not self._conn:
            return None
        sql = """
            SELECT content_hash, content, tags, memory_type, metadata,
                   created_at, updated_at, created_at_iso, updated_at_iso
            FROM memories
            WHERE content_hash = ? AND deleted_at IS NULL
        """
        params = [content_hash]
        if store is not None and store != "all":
            sql += " AND store = ?"
            params.append(store)
        try:
            cursor = self._conn.execute(sql, tuple(params))
            row = cursor.fetchone()
        except sqlite3.Error:
            return None
        if not row:
            return None
        tags_str = row["tags"] or ""
        tags_list = [t.strip() for t in tags_str.split(",") if t.strip()]
        meta_str = row["metadata"]
        try:
            metadata = json.loads(meta_str) if meta_str else {}
        except Exception:
            metadata = {}
        return Memory(
            content=row["content"],
            content_hash=row["content_hash"],
            tags=tags_list,
            memory_type=row["memory_type"] or "general",
            metadata=metadata,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            created_at_iso=row["created_at_iso"],
            updated_at_iso=row["updated_at_iso"],
        )

    async def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


MANIFEST_FILENAME = ".vault_manifest.json"


def is_safe_vault_child(out_dir: Path, rel_path: str) -> Optional[Path]:
    """Validate that rel_path is a safe relative file path strictly inside out_dir.

    Rejects absolute paths, parent directory traversals ('..'), and any path
    whose resolved location does not reside strictly within out_dir.
    Returns the entry path (under out_dir) rather than the dereferenced target
    so unlinking removes the entry itself (e.g. symlink) instead of deleting
    the note the symlink points to.
    """
    if not rel_path or not isinstance(rel_path, str):
        return None
    # Reject paths with null bytes or path traversal markers
    if "\x00" in rel_path or ".." in rel_path.replace("\\", "/").split("/"):
        return None
    p = Path(rel_path)
    if p.is_absolute():
        return None
    try:
        resolved_out = out_dir.resolve()
        entry_path = out_dir / p
        resolved_target = entry_path.resolve()
        # Ensure target is strictly inside resolved_out
        if resolved_target == resolved_out:
            return None
        # On Python 3.9+, is_relative_to is standard
        if not resolved_target.is_relative_to(resolved_out):
            return None
        return entry_path
    except (ValueError, RuntimeError):
        return None


def safe_write_vault_file(out_dir: Path, filename: str, content: str) -> None:
    """Safely write content to filename inside out_dir.

    Replaces any existing symlink entry so writes never follow a symlink
    to a destination outside out_dir.
    """
    safe_path = is_safe_vault_child(out_dir, filename)
    if not safe_path:
        raise ValueError(f"Unsafe destination or path traversal detected: {filename}")
    if safe_path.is_symlink():
        safe_path.unlink()
    safe_path.write_text(content, encoding="utf-8")


def safe_filename(name: str) -> str:
    """Sanitize entity_id or name for safe filesystem and wikilink usage across Windows and POSIX."""
    # Replace characters not allowed in Windows/Linux filenames: < > : " / \ | ? * and control chars
    sanitized = re.sub(r'[\x00-\x1f<>:"/\\|?*]', '_', name.strip())
    # Strip trailing periods/spaces which Windows disallows
    sanitized = sanitized.rstrip('. ')
    return sanitized or "unnamed"


WINDOWS_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def entity_filename_stem(eid: str) -> str:
    """Return the safe filename stem and wikilink target for an entity_id.

    Reserves 'index' for index.md and prefixes Windows reserved device names
    (CON, PRN, AUX, NUL, COM1-9, LPT1-9) with 'entity-' to prevent OS write failures.
    """
    stem = safe_filename(eid)
    lower = stem.lower()
    if lower == "index" or lower in WINDOWS_RESERVED_NAMES:
        return f"entity-{stem}"
    return stem


def format_chunk_quote(chunk: Dict[str, Any]) -> str:
    """Format a memory chunk into markdown blockquote with hash and created_at."""
    content = (chunk.get("content") or "").strip()
    h = chunk.get("hash") or ""
    created_at = chunk.get("created_at")
    if created_at and isinstance(created_at, (int, float)):
        created_str = datetime.datetime.fromtimestamp(created_at, datetime.timezone.utc).isoformat()
    elif created_at:
        created_str = str(created_at)
    else:
        created_str = "unknown"

    quoted_lines = [f"> {line}" for line in content.splitlines()]
    quote_text = "\n".join(quoted_lines) if quoted_lines else "> [empty]"
    return f"{quote_text}\n> \n> — Hash: `{h}` | Created: {created_str}"


def build_entity_markdown(
    detail: Dict[str, Any],
    profile: Dict[str, Any],
    exported_entity_ids: Set[str],
    exported_at: str,
    name_to_eid: Optional[Dict[str, str]] = None,
) -> str:
    """Build the complete markdown page with YAML frontmatter for an entity."""
    entity_id = detail["entity_id"]
    name = detail["name"]
    entity_type = (profile.get("entity_types") or [""])[0] if profile else ""
    chunks = detail.get("chunks", [])

    # Filter out empty or missing chunks
    valid_chunks = [c for c in chunks if c.get("content")]

    # Source hashes are the content hashes of valid memories
    source_hashes = [c["hash"] for c in valid_chunks if c.get("hash")]

    # Sort chunks deterministically by relevance descending, then hash
    valid_chunks.sort(
        key=lambda c: (c.get("relevance") if c.get("relevance") is not None else -1.0, c.get("hash", "")),
        reverse=True,
    )

    # Frontmatter
    frontmatter_dict = {
        "entity_id": entity_id,
        "name": name,
        "entity_type": entity_type,
        "relation_count": profile.get("memory_count", len(valid_chunks)),
        "exported_at": exported_at,
        "source_hashes": source_hashes,
    }

    frontmatter_yaml = yaml.safe_dump(
        frontmatter_dict,
        sort_keys=False,
        allow_unicode=True,
    ).strip()

    # Extractive summary
    summary = extract_extractive_summary(valid_chunks)

    # Body parts
    body_sections = []
    if summary:
        body_sections.append(f"{summary}\n")

    if valid_chunks:
        chunk_quotes = [format_chunk_quote(c) for c in valid_chunks]
        body_sections.append("\n\n".join(chunk_quotes))

    # Related section with wikilinks
    # Only link to entities that are actually exported in this vault run (respecting --limit)
    related = detail.get("related_entities", [])
    name_map = name_to_eid or {}
    related_links = []
    for rel in related:
        rel_name = rel.get("name", "")
        rel_id = name_map.get(rel_name) or rel.get("entity_id") or normalize_entity_id(rel_name)
        if rel_id in exported_entity_ids and rel_id != entity_id:
            related_links.append(f"- [[{entity_filename_stem(rel_id)}]]")

    # Sort links deterministically
    related_links.sort()

    if related_links:
        body_sections.append(f"\n## Related\n\n" + "\n".join(related_links))

    body = "\n\n".join(s for s in body_sections if s)
    return f"---\n{frontmatter_yaml}\n---\n\n{body}\n"


def build_index_markdown(
    entities_by_type: Dict[str, List[Dict[str, Any]]],
    exported_at: str,
) -> str:
    """Build the index.md listing all exported entities grouped by type."""
    frontmatter_dict = {
        "title": "Entity Vault Index",
        "exported_at": exported_at,
        "total_entities": sum(len(el) for el in entities_by_type.values()),
    }
    frontmatter_yaml = yaml.safe_dump(frontmatter_dict, sort_keys=False).strip()

    sections = [f"---\n{frontmatter_yaml}\n---\n\n# Entity Vault Index\n"]

    # Sort entity types deterministically
    for etype in sorted(entities_by_type.keys()):
        heading = etype.capitalize() if etype else "Untyped"
        sections.append(f"## {heading}\n")
        # Sort entities deterministically by name / entity_id
        sorted_entities = sorted(entities_by_type[etype], key=lambda e: e["entity_id"])
        links = []
        for ent in sorted_entities:
            eid = ent["entity_id"]
            name = ent["name"]
            links.append(f"- [[{entity_filename_stem(eid)}]] — {name} ({ent.get('relation_count', 0)} links)")
        sections.append("\n".join(links) + "\n")

    return "\n".join(sections)


async def export_vault(
    out_dir: Path,
    store: Optional[str] = None,
    limit: Optional[int] = None,
    db_path: Optional[str] = None,
    force: bool = False,
) -> int:
    """Execute the vault export."""
    sqlite_path = db_path or SQLITE_VEC_PATH

    storage = ReadOnlyVaultStorage(db_path=sqlite_path)
    await storage.initialize()

    try:
        graph = GraphStorage(sqlite_path)

        # 1. Fetch entities without artificial cap
        fetch_limit = -1 if limit is None else max(limit * 10, 500)
        all_raw_entities = await graph.list_entities(limit=fetch_limit)
        if not all_raw_entities:
            print(
                "Error: Entity graph is empty.\n"
                'The entity graph needs to be populated via memory_quality {"action": "maintain", "dry_run": false}',
                file=sys.stderr,
            )
            return 1

        # Step A: Collect distinct raw entity names that pass store filtering
        raw_names_set: Set[str] = set()
        for raw in all_raw_entities:
            raw_name = raw.get("entity_name", "")
            if not raw_name or raw_name in raw_names_set:
                continue

            # Store scoping filter: check that entity has active memories in the requested store
            if store is not None and store != "all":
                ent_hashes = await graph.find_memories_by_entity(raw_name, limit=200)
                has_active_chunk = False
                for eh in ent_hashes:
                    mem = await storage.get_by_hash(eh, store=store)
                    if mem and not getattr(mem, "deleted_at", None):
                        has_active_chunk = True
                        break
                if not has_active_chunk:
                    continue

            raw_names_set.add(raw_name)

        if not raw_names_set:
            print(
                f"Error: No entities found matching store '{store}'.\n"
                'Ensure the entity graph is populated with memories in this store.',
                file=sys.stderr,
            )
            return 1

        # Step B: Group by base slug and sort names alphabetically to ensure 100% stable IDs
        # across exports, regardless of memory count changes or query order.
        slug_to_names: Dict[str, List[str]] = {}
        for name in sorted(raw_names_set):
            base_slug = normalize_entity_id(name)
            slug_to_names.setdefault(base_slug, []).append(name)

        name_to_eid: Dict[str, str] = {}
        candidate_entities = []
        for base_slug in sorted(slug_to_names.keys()):
            names = slug_to_names[base_slug]
            if len(names) == 1:
                name_to_eid[names[0]] = base_slug
                candidate_entities.append({"entity_id": base_slug, "name": names[0]})
            else:
                for idx, name in enumerate(names):
                    if idx == 0:
                        eid = base_slug
                    else:
                        h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
                        eid = f"{base_slug}-{h}"
                    name_to_eid[name] = eid
                    candidate_entities.append({"entity_id": eid, "name": name})

        # Sort entities deterministically
        candidate_entities.sort(key=lambda e: e["entity_id"])

        # Apply limit to candidate entities after store filtering
        selected_entities = candidate_entities[:limit] if limit is not None else candidate_entities
        exported_entity_ids = {e["entity_id"] for e in selected_entities}

        # Timestamp for exported_at (deterministic for this run)
        exported_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Prepare output dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Read previous manifest for safe cleanup
        manifest_path = out_dir / MANIFEST_FILENAME
        previous_files: Set[str] = set()
        if manifest_path.exists():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                previous_files = set(manifest_data.get("files", []))
            except Exception:
                previous_files = set()

        current_files: Set[str] = set()
        entities_by_type: Dict[str, List[Dict[str, Any]]] = {}

        # 2. Write one page per entity
        for ent_info in selected_entities:
            eid = ent_info["entity_id"]
            name = ent_info["name"]
            try:
                detail = await get_entity_detail(
                    storage,
                    graph,
                    entity_id=eid,
                    limit=50,
                    include_related=True,
                    store=store,
                    entity_name=name,
                )
            except ValueError as e:
                # Entity could not be resolved or scoped
                continue

            profile = await graph.get_entity_profile(detail["name"])
            entity_type = (profile.get("entity_types") or [""])[0] if profile else ""

            # Hydrate created_at for chunks from storage
            for chunk in detail.get("chunks", []):
                h = chunk.get("hash")
                if h:
                    mem = await storage.get_by_hash(h, store=store)
                    if mem and not getattr(mem, "deleted_at", None):
                        chunk["created_at"] = getattr(mem, "created_at", None)
                    else:
                        # Omit content for deleted memories so they aren't rendered
                        chunk["content"] = None

            md_content = build_entity_markdown(
                detail=detail,
                profile=profile,
                exported_entity_ids=exported_entity_ids,
                exported_at=exported_at,
                name_to_eid=name_to_eid,
            )

            filename = f"{entity_filename_stem(eid)}.md"
            file_path = out_dir / filename

            # Protect pre-existing user notes: if file exists and was NOT in previous_files, refuse unless --force
            if (file_path.exists() or file_path.is_symlink()) and filename not in previous_files and not force:
                print(
                    f"Refusing to overwrite existing unmanaged file: {file_path}.\n"
                    "Pass --force to allow overwriting pre-existing files in the vault directory.",
                    file=sys.stderr,
                )
                return 1

            try:
                safe_write_vault_file(out_dir, filename, md_content)
            except ValueError as ve:
                print(f"Error writing entity file {filename}: {ve}", file=sys.stderr)
                return 1

            current_files.add(filename)

            entities_by_type.setdefault(entity_type, []).append({
                "entity_id": eid,
                "name": detail["name"],
                "relation_count": profile.get("memory_count", len(detail.get("chunks", []))),
            })

        # 3. Write index.md
        index_content = build_index_markdown(entities_by_type, exported_at)
        index_path = out_dir / "index.md"
        if (index_path.exists() or index_path.is_symlink()) and "index.md" not in previous_files and not force:
            print(
                f"Refusing to overwrite existing unmanaged index file: {index_path}.\n"
                "Pass --force to allow overwriting pre-existing files in the vault directory.",
                file=sys.stderr,
            )
            return 1

        try:
            safe_write_vault_file(out_dir, "index.md", index_content)
        except ValueError as ve:
            print(f"Error writing index.md: {ve}", file=sys.stderr)
            return 1

        current_files.add("index.md")

        # 4. Cleanup stale files previously created by this script
        stale_files = previous_files - current_files
        for stale in stale_files:
            safe_target = is_safe_vault_child(out_dir, stale)
            if safe_target and (safe_target.is_file() or safe_target.is_symlink()):
                try:
                    safe_target.unlink()
                except OSError:
                    pass

        # 5. Save updated manifest
        manifest_payload = {
            "exported_at": exported_at,
            "files": sorted(list(current_files)),
        }
        try:
            safe_write_vault_file(out_dir, MANIFEST_FILENAME, json.dumps(manifest_payload, indent=2))
        except ValueError as ve:
            print(f"Error writing manifest: {ve}", file=sys.stderr)
            return 1

        print(f"Successfully exported {len(current_files) - 1} entities to {out_dir}")
        return 0

    finally:
        if 'graph' in locals() and graph and graph._connection:
            graph._connection.close()
        await storage.close()


def main():
    parser = argparse.ArgumentParser(description="Export entity graph into Markdown Vault.")
    parser.add_argument("--out", required=True, help="Output directory for the vault")
    parser.add_argument("--store", default=None, help="Store namespace to scope export to")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of entities to export")
    parser.add_argument("--db-path", default=None, help="Path to SQLite database file")
    parser.add_argument("--force", "-f", action="store_true", help="Allow overwriting pre-existing unmanaged files")

    args = parser.parse_args()
    exit_code = asyncio.run(
        export_vault(
            out_dir=Path(args.out),
            store=args.store,
            limit=args.limit,
            db_path=args.db_path,
            force=args.force,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

