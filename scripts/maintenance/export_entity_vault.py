#!/usr/bin/env python3
"""
export_entity_vault.py - Read-only export of entity graph into a Markdown Vault.

Usage:
    python scripts/maintenance/export_entity_vault.py --out <dir> [--store <name>] [--limit N] [--db-path <path>]
"""

import argparse
import asyncio
import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from mcp_memory_service.config import SQLITE_VEC_PATH
from mcp_memory_service.server.handlers.graph import (
    extract_extractive_summary,
    get_entity_detail,
)
from mcp_memory_service.storage.graph import GraphStorage, normalize_entity_id
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


MANIFEST_FILENAME = ".vault_manifest.json"


def safe_filename(name: str) -> str:
    """Sanitize entity_id or name for safe filesystem and wikilink usage across Windows and POSIX."""
    # Replace characters not allowed in Windows/Linux filenames: < > : " / \ | ? * and control chars
    sanitized = re.sub(r'[\x00-\x1f<>:"/\\|?*]', '_', name.strip())
    # Strip trailing periods/spaces which Windows disallows
    sanitized = sanitized.rstrip('. ')
    return sanitized or "unnamed"


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
    related_links = []
    for rel in related:
        rel_id = rel.get("entity_id") or normalize_entity_id(rel.get("name", ""))
        if rel_id in exported_entity_ids and rel_id != entity_id:
            related_links.append(f"- [[{safe_filename(rel_id)}]]")

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
            links.append(f"- [[{safe_filename(eid)}]] — {name} ({ent.get('relation_count', 0)} links)")
        sections.append("\n".join(links) + "\n")

    return "\n".join(sections)


async def export_vault(
    out_dir: Path,
    store: Optional[str] = None,
    limit: Optional[int] = None,
    db_path: Optional[str] = None,
) -> int:
    """Execute the vault export."""
    sqlite_path = db_path or SQLITE_VEC_PATH

    storage = SqliteVecMemoryStorage(db_path=sqlite_path)
    await storage.initialize()

    try:
        graph = GraphStorage(sqlite_path)

        # 1. Fetch entities
        all_raw_entities = await graph.list_entities(limit=limit or 500)
        if not all_raw_entities:
            print(
                "Error: Entity graph is empty.\n"
                'The entity graph needs to be populated via memory_quality {"action": "maintain", "dry_run": false}',
                file=sys.stderr,
            )
            return 1

        # Apply limit if specified
        selected_raw = all_raw_entities[:limit] if limit else all_raw_entities

        # Normalize and deduplicate entity IDs
        selected_entities = []
        exported_entity_ids = set()
        for raw in selected_raw:
            raw_name = raw.get("entity_name", "")
            if not raw_name:
                continue
            eid = normalize_entity_id(raw_name)
            if eid not in exported_entity_ids:
                exported_entity_ids.add(eid)
                selected_entities.append({"entity_id": eid, "name": raw_name})

        # Sort entities deterministically
        selected_entities.sort(key=lambda e: e["entity_id"])

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
            try:
                detail = await get_entity_detail(
                    storage,
                    graph,
                    entity_id=eid,
                    limit=50,
                    include_related=True,
                    store=store,
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
            )

            filename = f"{safe_filename(eid)}.md"
            file_path = out_dir / filename
            file_path.write_text(md_content, encoding="utf-8")
            current_files.add(filename)

            entities_by_type.setdefault(entity_type, []).append({
                "entity_id": eid,
                "name": detail["name"],
                "relation_count": profile.get("memory_count", len(detail.get("chunks", []))),
            })

        # 3. Write index.md
        index_content = build_index_markdown(entities_by_type, exported_at)
        index_path = out_dir / "index.md"
        index_path.write_text(index_content, encoding="utf-8")
        current_files.add("index.md")

        # 4. Cleanup stale files previously created by this script
        stale_files = previous_files - current_files
        for stale in stale_files:
            stale_path = out_dir / stale
            if stale_path.exists():
                try:
                    stale_path.unlink()
                except OSError:
                    pass

        # 5. Save updated manifest
        manifest_payload = {
            "exported_at": exported_at,
            "files": sorted(list(current_files)),
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

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

    args = parser.parse_args()
    exit_code = asyncio.run(
        export_vault(
            out_dir=Path(args.out),
            store=args.store,
            limit=args.limit,
            db_path=args.db_path,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

