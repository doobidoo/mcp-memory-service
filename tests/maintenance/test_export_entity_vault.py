import os
import sys
import sqlite3
import subprocess
from pathlib import Path
import pytest
import yaml

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.storage.graph import GraphStorage
from mcp_memory_service.utils import generate_content_hash


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "maintenance" / "export_entity_vault.py"


def _make_mem(content: str, tags: list = None) -> Memory:
    return Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=tags or [],
    )


def _ensure_memory_graph_table(db_path: str) -> None:
    """Create the memory_graph table if it doesn't exist.

    GraphStorage assumes the table already exists (created by migrations
    for production databases). For test databases that bypass migrations,
    we create it manually — matching the schema from tests/storage/conftest.py.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_graph (
            source_hash TEXT NOT NULL,
            target_hash TEXT NOT NULL,
            similarity REAL NOT NULL,
            connection_types TEXT NOT NULL,
            metadata TEXT,
            created_at REAL NOT NULL,
            relationship_type TEXT DEFAULT 'related',
            PRIMARY KEY (source_hash, target_hash)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_graph_source
        ON memory_graph(source_hash)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_graph_target
        ON memory_graph(target_hash)
    """)
    conn.commit()
    conn.close()


@pytest.fixture
async def seeded_storage(temp_db_path):
    """Seed real sqlite-vec storage and graph with test memories and entities.

    GraphStorage stores entities as memory_graph rows with
    relationship_type='has_entity' and entity_type in the metadata JSON.
    Use store_entity_link(memory_hash, entity_name, entity_type) — the only
    public API that creates these rows.
    """
    db_file = os.path.join(temp_db_path, "test_vault.db")
    storage = SqliteVecMemoryStorage(db_path=db_file)
    await storage.initialize()

    # Ensure memory_graph table exists (GraphStorage expects it pre-created)
    _ensure_memory_graph_table(db_file)

    # Seed 3 memories
    mem1 = _make_mem(
        "Alpha service handles user authentication and session tokens.",
        tags=["auth", "security"],
    )
    await storage.store(mem1)

    mem2 = _make_mem(
        "Beta gateway routes traffic to Alpha service and validates JWT.",
        tags=["gateway", "networking"],
    )
    await storage.store(mem2)

    mem3 = _make_mem(
        "Gamma database stores persistent records for users and sessions.",
        tags=["database", "storage"],
    )
    await storage.store(mem3)

    # 4 memories to form a common-neighbor triangle:
    # mem1 -> shared_mid
    # mem3 -> shared_mid
    # This makes mem1 and mem3 share 'shared_mid', but they are not directly connected!
    mem4 = _make_mem(
        "Shared auth token session cache used by Alpha and Beta.",
        tags=["cache", "shared"],
    )
    await storage.store(mem4)

    # Seed graph relationships using the real GraphStorage public API.
    # store_entity_link(memory_hash, entity_name, entity_type) writes a row in
    # memory_graph with relationship_type='has_entity' and entity_type in metadata.
    graph = GraphStorage(db_file)

    # Alpha Service — linked to mem1
    await graph.store_entity_link(mem1.content_hash, "Alpha Service", "service")

    # Beta Gateway — linked to mem3
    await graph.store_entity_link(mem3.content_hash, "Beta Gateway", "service")

    # Gamma Database — linked to mem2
    await graph.store_entity_link(mem2.content_hash, "Gamma Database", "database")

    # Add edges: mem1 <-> mem4, mem3 <-> mem4.
    # Common neighbors for mem1: candidate is mem3 because both connect to mem4!
    # And candidate mem3 is linked to entity 'Beta Gateway'.
    await graph.store_association(
        mem1.content_hash, mem4.content_hash, 0.8, ["semantic"], relationship_type="related"
    )
    await graph.store_association(
        mem3.content_hash, mem4.content_hash, 0.8, ["semantic"], relationship_type="related"
    )

    yield db_file, storage, graph, [mem1, mem2, mem3, mem4]

    await graph.close()
    await storage.close()


def run_export_script(*args, env=None):
    """Helper to run export_entity_vault.py in a subprocess."""
    cmd_env = os.environ.copy()
    if env:
        cmd_env.update(env)
    # Ensure src directory is in PYTHONPATH
    src_dir = str(REPO_ROOT / "src")
    existing_pythonpath = cmd_env.get("PYTHONPATH", "")
    cmd_env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_dir

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)] + list(args),
        capture_output=True,
        text=True,
        env=cmd_env,
    )
    return result


def test_empty_graph_exits_nonzero(temp_db_path):
    """If list_entities is empty, the script prints an explanation and exits non-zero without writing."""
    empty_db = os.path.join(temp_db_path, "empty.db")
    vault_dir = Path(temp_db_path) / "vault"

    result = run_export_script("--out", str(vault_dir), "--db-path", empty_db)

    assert result.returncode != 0
    assert 'memory_quality {"action": "maintain", "dry_run": false}' in result.stdout or \
           'memory_quality {"action": "maintain", "dry_run": false}' in result.stderr
    assert not vault_dir.exists() or len(list(vault_dir.iterdir())) == 0


@pytest.mark.asyncio
async def test_vault_export_creates_expected_structure(seeded_storage, temp_db_path):
    """Verify that export writes index.md, one file per entity, valid YAML frontmatter, and correct body."""
    db_file, storage, graph, memories = seeded_storage
    vault_dir = Path(temp_db_path) / "vault"

    result = run_export_script("--out", str(vault_dir), "--db-path", db_file)
    assert result.returncode == 0, f"Script failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

    # 1. index.md exists and groups by type
    index_file = vault_dir / "index.md"
    assert index_file.exists()
    index_content = index_file.read_text(encoding="utf-8")
    assert "[[alpha-service]]" in index_content
    assert "[[beta-gateway]]" in index_content
    assert "[[gamma-database]]" in index_content
    assert "service" in index_content.lower()

    # 2. Entity files exist
    alpha_file = vault_dir / "alpha-service.md"
    assert alpha_file.exists()
    content = alpha_file.read_text(encoding="utf-8")

    # 3. YAML frontmatter validation
    assert content.startswith("---")
    parts = content.split("---", 2)
    assert len(parts) >= 3
    frontmatter = yaml.safe_load(parts[1])

    assert frontmatter["entity_id"] == "alpha-service"
    assert frontmatter["name"] == "Alpha Service"
    assert frontmatter["entity_type"] == "service"
    assert "relation_count" in frontmatter
    assert "exported_at" in frontmatter
    assert isinstance(frontmatter["source_hashes"], list)
    assert memories[0].content_hash in frontmatter["source_hashes"]

    # 4. Body checks (summary + blockquotes with hashes + created_at)
    body = parts[2]
    assert "> " in body
    assert memories[0].content_hash in body
    assert "## Related" in body
    # Wikilink to related beta-gateway
    assert "[[beta-gateway]]" in body


@pytest.mark.asyncio
async def test_wikilinks_respect_limit(seeded_storage, temp_db_path):
    """Wikilinks must only point to entities whose pages are actually written (respecting --limit)."""
    db_file, storage, graph, memories = seeded_storage
    vault_dir = Path(temp_db_path) / "vault_limit"

    # Export with --limit 1 (only alpha_service will be written)
    result = run_export_script("--out", str(vault_dir), "--db-path", db_file, "--limit", "1")
    assert result.returncode == 0, f"Script failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

    # List written markdown files excluding index.md
    written_entities = [p.stem for p in vault_dir.glob("*.md") if p.name != "index.md"]
    assert len(written_entities) == 1

    # Check written file's wikilinks
    entity_file = vault_dir / f"{written_entities[0]}.md"
    content = entity_file.read_text(encoding="utf-8")

    # Wikilinks must only point to pages that exist
    for word in content.split("[["):
        if "]]" in word:
            target = word.split("]]")[0].strip()
            target_path = vault_dir / f"{target}.md"
            assert target_path.exists(), f"Wikilink [[{target}]] points to unwritten entity"


@pytest.mark.asyncio
async def test_deleted_memory_omitted(seeded_storage, temp_db_path):
    """Soft-deleted memories (deleted_at IS NOT NULL) must not appear in the exported vault."""
    db_file, storage, graph, memories = seeded_storage
    vault_dir = Path(temp_db_path) / "vault_deleted"

    # Delete mem1
    await storage.delete(memories[0].content_hash)

    result = run_export_script("--out", str(vault_dir), "--db-path", db_file)
    assert result.returncode == 0, f"Script failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

    alpha_file = vault_dir / "alpha-service.md"
    if alpha_file.exists():
        content = alpha_file.read_text(encoding="utf-8")
        assert memories[0].content_hash not in content


@pytest.mark.asyncio
async def test_rerun_removes_stale_pages_and_preserves_unrelated_files(seeded_storage, temp_db_path):
    """Re-running overwrites only what the script wrote, cleans up stale entities, and preserves unrelated files."""
    db_file, storage, graph, memories = seeded_storage
    vault_dir = Path(temp_db_path) / "vault_rerun"
    vault_dir.mkdir(parents=True, exist_ok=True)

    # 1. User adds an unrelated file
    unrelated_file = vault_dir / "my_custom_notes.txt"
    unrelated_file.write_text("User personal notes - must NOT be deleted", encoding="utf-8")

    # 2. Run export
    result = run_export_script("--out", str(vault_dir), "--db-path", db_file)
    assert result.returncode == 0

    assert (vault_dir / "gamma-database.md").exists()
    assert unrelated_file.exists()

    # 3. Simulate deletion of Gamma Database from graph.
    #    Entities live in memory_graph with relationship_type='has_entity'.
    #    _get_connection() is an async method returning a plain sync sqlite3.Connection.
    conn = await graph._get_connection()
    conn.execute(
        "DELETE FROM memory_graph WHERE target_hash = ? AND relationship_type = 'has_entity'",
        ("Gamma Database",),
    )
    conn.commit()

    # 4. Re-run export
    result2 = run_export_script("--out", str(vault_dir), "--db-path", db_file)
    assert result2.returncode == 0

    # 5. Verify gamma-database.md is cleaned up, and unrelated file is preserved
    assert not (vault_dir / "gamma-database.md").exists(), "Stale entity page was not removed"
    assert unrelated_file.exists(), "Unrelated file was erroneously deleted"
    assert unrelated_file.read_text(encoding="utf-8") == "User personal notes - must NOT be deleted"


@pytest.mark.asyncio
async def test_manifest_path_traversal_does_not_delete_outside_files(seeded_storage, temp_db_path):
    """Malicious entries in .vault_manifest.json (e.g., ../outside.txt or absolute paths) must be rejected."""
    import json
    db_file, storage, graph, memories = seeded_storage
    base_dir = Path(temp_db_path)
    vault_dir = base_dir / "vault_sec"
    vault_dir.mkdir(parents=True, exist_ok=True)

    # Create an important file outside the vault
    outside_file = base_dir / "critical_system_file.txt"
    outside_file.write_text("DO NOT DELETE", encoding="utf-8")

    # Write a malicious manifest referencing the outside file and an absolute path
    manifest_path = vault_dir / ".vault_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "exported_at": "2026-01-01T00:00:00Z",
            "files": ["../critical_system_file.txt", str(outside_file.resolve()), "stale_entity.md"],
        }),
        encoding="utf-8",
    )

    # Run export
    result = run_export_script("--out", str(vault_dir), "--db-path", db_file)
    assert result.returncode == 0

    # The outside file MUST still exist!
    assert outside_file.exists(), "Path traversal allowed deleting outside file!"
    assert outside_file.read_text(encoding="utf-8") == "DO NOT DELETE"


@pytest.mark.asyncio
async def test_collision_with_existing_unmanaged_file_refused_without_force(seeded_storage, temp_db_path):
    """Writing over pre-existing notes not in manifest is refused unless --force is given."""
    db_file, storage, graph, memories = seeded_storage
    vault_dir = Path(temp_db_path) / "vault_collision"
    vault_dir.mkdir(parents=True, exist_ok=True)

    # Pre-existing user note that clashes with an entity name
    pre_existing_note = vault_dir / "alpha-service.md"
    pre_existing_note.write_text("User manually created alpha service note", encoding="utf-8")

    # Attempt export without --force -> should fail and preserve note
    result = run_export_script("--out", str(vault_dir), "--db-path", db_file)
    assert result.returncode != 0
    assert "Refusing to overwrite existing unmanaged file" in result.stderr
    assert pre_existing_note.read_text(encoding="utf-8") == "User manually created alpha service note"

    # With --force -> should succeed and overwrite
    result_forced = run_export_script("--out", str(vault_dir), "--db-path", db_file, "--force")
    assert result_forced.returncode == 0
    assert "---" in pre_existing_note.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_entity_named_index_does_not_clash_with_directory_index(seeded_storage, temp_db_path):
    """An entity named 'Index' must be written to entity-index.md and not be overwritten by index.md."""
    db_file, storage, graph, memories = seeded_storage
    vault_dir = Path(temp_db_path) / "vault_index_clash"

    # Add an entity named "Index"
    await graph.store_entity_link(memories[0].content_hash, "Index", "concept")

    result = run_export_script("--out", str(vault_dir), "--db-path", db_file)
    assert result.returncode == 0

    # 1. index.md exists and is the vault directory index
    index_md = vault_dir / "index.md"
    assert index_md.exists()
    index_content = index_md.read_text(encoding="utf-8")
    assert "Entity Vault Index" in index_content
    # Wikilink to the entity named Index must point to [[entity-index]]
    assert "[[entity-index]]" in index_content

    # 2. entity-index.md exists and contains the entity's frontmatter and quotes
    entity_index_md = vault_dir / "entity-index.md"
    assert entity_index_md.exists()
    entity_content = entity_index_md.read_text(encoding="utf-8")
    assert "name: Index" in entity_content
    assert memories[0].content_hash in entity_content


@pytest.mark.asyncio
async def test_store_scoping_selects_only_relevant_entities(seeded_storage, temp_db_path):
    """--store must select only entities that have active memories in that specific store."""
    db_file, storage, graph, memories = seeded_storage
    vault_dir = Path(temp_db_path) / "vault_scoped"

    # Assign mem1 to 'work' store directly in memories table
    conn = sqlite3.connect(db_file)
    conn.execute("UPDATE memories SET store = 'work' WHERE content_hash = ?", (memories[0].content_hash,))
    conn.commit()
    conn.close()

    # Export with --store work
    result = run_export_script("--out", str(vault_dir), "--db-path", db_file, "--store", "work")
    assert result.returncode == 0

    # Only Alpha Service has memories in 'work'
    assert (vault_dir / "alpha-service.md").exists()
    assert not (vault_dir / "beta-gateway.md").exists()
    assert not (vault_dir / "gamma-database.md").exists()


@pytest.mark.asyncio
async def test_stale_symlink_does_not_delete_target_note(seeded_storage, temp_db_path):
    """If a previously exported file becomes an internal symlink, cleanup unlinks the symlink without deleting target."""
    import json
    db_file, storage, graph, memories = seeded_storage
    vault_dir = Path(temp_db_path) / "vault_symlink"
    vault_dir.mkdir(parents=True, exist_ok=True)

    # 1. User's personal note inside the vault
    target_note = vault_dir / "my_precious_notes.md"
    target_note.write_text("User custom research notes", encoding="utf-8")

    # 2. Create a symlink stale_page.md -> my_precious_notes.md
    symlink_page = vault_dir / "stale_page.md"
    try:
        symlink_page.symlink_to(target_note.name)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported in this environment")

    # 3. Simulate previous manifest listing stale_page.md
    manifest_path = vault_dir / ".vault_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "exported_at": "2026-01-01T00:00:00Z",
            "files": ["stale_page.md"],
        }),
        encoding="utf-8",
    )

    # 4. Run export
    result = run_export_script("--out", str(vault_dir), "--db-path", db_file)
    assert result.returncode == 0

    # 5. Verify the stale symlink was unlinked, but target_note is STILL INTACT!
    assert not symlink_page.is_symlink() and not symlink_page.exists(), "Stale symlink was not removed"
    assert target_note.exists(), "Target note was erroneously deleted!"
    assert target_note.read_text(encoding="utf-8") == "User custom research notes"


@pytest.mark.asyncio
async def test_writes_do_not_escape_vault_via_external_symlinks(seeded_storage, temp_db_path):
    """If a destination file is a symlink pointing outside out_dir, writing must not overwrite the outside file."""
    import json
    db_file, storage, graph, memories = seeded_storage
    base_dir = Path(temp_db_path)
    vault_dir = base_dir / "vault_external_sym"
    vault_dir.mkdir(parents=True, exist_ok=True)

    # 1. Critical outside file that must NEVER be overwritten
    outside_file = base_dir / "important_host_config.json"
    outside_file.write_text('{"host_state": "intact"}', encoding="utf-8")

    # 2. Symlinks inside the vault pointing to the outside file
    # (e.g. index.md -> outside_file, and managed page -> outside_file)
    index_link = vault_dir / "index.md"
    entity_link = vault_dir / "alpha-service.md"
    manifest_link = vault_dir / ".vault_manifest.json"

    try:
        index_link.symlink_to(outside_file)
        entity_link.symlink_to(outside_file)
        manifest_link.symlink_to(outside_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported in this environment")

    # Mark them in manifest so they are treated as managed / previous files
    manifest_link.unlink(missing_ok=True)
    manifest_link.write_text(
        json.dumps({
            "exported_at": "2026-01-01T00:00:00Z",
            "files": ["index.md", "alpha-service.md"],
        }),
        encoding="utf-8",
    )
    # Re-link manifest to outside file
    manifest_link.unlink(missing_ok=True)
    manifest_link.symlink_to(outside_file)

    # Run export with --force to test that writing to an escaping symlink is safely refused
    result = run_export_script("--out", str(vault_dir), "--db-path", db_file, "--force")
    assert result.returncode != 0
    assert "Unsafe destination or path traversal detected" in result.stderr or "Refusing" in result.stderr

    # The external file must be COMPLETELY UNTOUCHED!
    assert outside_file.read_text(encoding="utf-8") == '{"host_state": "intact"}'


@pytest.mark.asyncio
async def test_database_path_with_special_characters(seeded_storage, temp_db_path):
    """Database paths containing special characters like '#' and '?' must be opened correctly using as_uri()."""
    db_file, storage, graph, memories = seeded_storage

    # Use SQLite's online backup to safely copy the entire database (including WAL) to a special path
    special_db = Path(temp_db_path) / "special#project?db.sqlite"
    if sys.platform == "win32":
        special_db = Path(temp_db_path) / "special#project_db.sqlite"

    # Backup from source to special_db
    src_conn = sqlite3.connect(db_file)
    dst_conn = sqlite3.connect(str(special_db))
    src_conn.backup(dst_conn)
    dst_conn.close()
    src_conn.close()

    vault_dir = Path(temp_db_path) / "vault_special_path"
    result = run_export_script("--out", str(vault_dir), "--db-path", str(special_db))
    assert result.returncode == 0
    assert (vault_dir / "index.md").exists()
    assert (vault_dir / "alpha-service.md").exists()


@pytest.mark.asyncio
async def test_colliding_slugs_disambiguated_without_dropping_entities(seeded_storage, temp_db_path):
    """Distinct raw entity names with identical slugs (e.g. foo.py and foo/py) must both be exported with disambiguated IDs."""
    db_file, storage, graph, memories = seeded_storage
    vault_dir = Path(temp_db_path) / "vault_slug_collision"

    # Add two entities that slugify identically to 'foo-py'
    await graph.store_entity_link(memories[0].content_hash, "foo.py", "file")
    await graph.store_entity_link(memories[1].content_hash, "foo/py", "file")

    result = run_export_script("--out", str(vault_dir), "--db-path", db_file)
    assert result.returncode == 0

    # Both entities must have pages in the vault
    all_pages = [p.name for p in vault_dir.glob("*.md")]
    foo_pages = sorted([p for p in all_pages if p.startswith("foo-py")])
    assert len(foo_pages) == 2, f"Expected 2 disambiguated pages for foo.py and foo/py, found: {foo_pages}"

    # Verify both entities appear in index.md
    index_content = (vault_dir / "index.md").read_text(encoding="utf-8")
    assert "foo.py" in index_content
    assert "foo/py" in index_content

    # Now add more memories to foo/py so its memory count exceeds foo.py
    # Re-run export to assert that the assigned filenames remain 100% STABLE!
    await graph.store_entity_link(memories[2].content_hash, "foo/py", "file")
    await graph.store_entity_link(memories[3].content_hash, "foo/py", "file")

    result2 = run_export_script("--out", str(vault_dir), "--db-path", db_file)
    assert result2.returncode == 0

    all_pages_after = sorted([p.name for p in vault_dir.glob("*.md") if p.name.startswith("foo-py")])
    assert foo_pages == all_pages_after, "Entity pages changed filenames after popularity/memory count changed!"


@pytest.mark.asyncio
async def test_windows_reserved_device_names_exported_safely(seeded_storage, temp_db_path):
    """Entities matching Windows reserved device names (AUX, CON, NUL, COM1) must use safe stems and not crash."""
    db_file, storage, graph, memories = seeded_storage
    vault_dir = Path(temp_db_path) / "vault_windows_reserved"

    # Add entities with reserved Windows device names
    await graph.store_entity_link(memories[0].content_hash, "aux", "keyword")
    await graph.store_entity_link(memories[1].content_hash, "CON", "keyword")

    result = run_export_script("--out", str(vault_dir), "--db-path", db_file)
    assert result.returncode == 0

    # Pages must be created with the safe 'entity-' prefix
    assert (vault_dir / "entity-aux.md").exists()
    assert (vault_dir / "entity-CON.md").exists() or (vault_dir / "entity-con.md").exists()

    # index.md must link to the safe stems
    index_content = (vault_dir / "index.md").read_text(encoding="utf-8")
    assert "[[entity-aux]]" in index_content
