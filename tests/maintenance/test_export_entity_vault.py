import os
import sys
import subprocess
from pathlib import Path
import pytest
import yaml

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.storage.graph import GraphStorage


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "maintenance" / "export_entity_vault.py"


@pytest.fixture
async def seeded_storage(temp_db_path):
    """Seed real sqlite-vec storage and graph with test memories and entities."""
    db_file = os.path.join(temp_db_path, "test_vault.db")
    storage = SqliteVecMemoryStorage(db_path=db_file)
    await storage.initialize()

    # Seed 3 memories
    mem1 = await storage.store(
        content="Alpha service handles user authentication and session tokens.",
        tags=["auth", "security"],
    )
    mem2 = await storage.store(
        content="Beta gateway routes traffic to Alpha service and validates JWT.",
        tags=["gateway", "networking"],
    )
    mem3 = await storage.store(
        content="Gamma database stores persistent records for users and sessions.",
        tags=["database", "storage"],
    )

    # Seed graph relationships
    graph = GraphStorage(db_file)
    await graph.add_entity("Alpha Service", entity_type="service")
    await graph.add_entity("Beta Gateway", entity_type="service")
    await graph.add_entity("Gamma Database", entity_type="database")

    # Link memories to entities
    await graph.link_memory_to_entity(mem1.content_hash, "Alpha Service")
    await graph.link_memory_to_entity(mem2.content_hash, "Alpha Service")
    await graph.link_memory_to_entity(mem2.content_hash, "Beta Gateway")
    await graph.link_memory_to_entity(mem3.content_hash, "Gamma Database")

    # Link entities (memories shared)
    await graph.create_relationship(mem1.content_hash, mem2.content_hash, "causes")

    yield db_file, storage, graph, [mem1, mem2, mem3]

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
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # 1. index.md exists and groups by type
    index_file = vault_dir / "index.md"
    assert index_file.exists()
    index_content = index_file.read_text(encoding="utf-8")
    assert "[[alpha_service]]" in index_content
    assert "[[beta_gateway]]" in index_content
    assert "[[gamma_database]]" in index_content
    assert "service" in index_content.lower()

    # 2. Entity files exist
    alpha_file = vault_dir / "alpha_service.md"
    assert alpha_file.exists()
    content = alpha_file.read_text(encoding="utf-8")

    # 3. YAML frontmatter validation
    assert content.startswith("---")
    parts = content.split("---", 2)
    assert len(parts) >= 3
    frontmatter = yaml.safe_load(parts[1])

    assert frontmatter["entity_id"] == "alpha_service"
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
    # Wikilink to related beta_gateway
    assert "[[beta_gateway]]" in body


@pytest.mark.asyncio
async def test_wikilinks_respect_limit(seeded_storage, temp_db_path):
    """Wikilinks must only point to entities whose pages are actually written (respecting --limit)."""
    db_file, storage, graph, memories = seeded_storage
    vault_dir = Path(temp_db_path) / "vault_limit"

    # Export with --limit 1 (only alpha_service will be written)
    result = run_export_script("--out", str(vault_dir), "--db-path", db_file, "--limit", "1")
    assert result.returncode == 0, f"Script failed: {result.stderr}"

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
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    alpha_file = vault_dir / "alpha_service.md"
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

    assert (vault_dir / "gamma_database.md").exists()
    assert unrelated_file.exists()

    # 3. Simulate deletion of Gamma Database from graph
    async with graph._get_connection() as conn:
        await conn.execute("DELETE FROM entities WHERE entity_name = 'Gamma Database'")
        await conn.execute("DELETE FROM memory_entities WHERE entity_name = 'Gamma Database'")
        await conn.commit()

    # 4. Re-run export
    result2 = run_export_script("--out", str(vault_dir), "--db-path", db_file)
    assert result2.returncode == 0

    # 5. Verify gamma_database.md is cleaned up, and unrelated file is preserved
    assert not (vault_dir / "gamma_database.md").exists(), "Stale entity page was not removed"
    assert unrelated_file.exists(), "Unrelated file was erroneously deleted"
    assert unrelated_file.read_text(encoding="utf-8") == "User personal notes - must NOT be deleted"

