import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOUDFLARE_SCRIPT = REPO_ROOT / "scripts" / "testing" / "test_cloudflare_backend.py"
SETUP_CLOUDFLARE_SCRIPT = (
    REPO_ROOT / "scripts" / "installation" / "setup_cloudflare_resources.py"
)

REPAIR_EMBEDDINGS_SCRIPT = (
    REPO_ROOT / "scripts" / "maintenance" / "repair_missing_embeddings_onnx.py"
)

CLOUDFLARE_ENV = {
    "CLOUDFLARE_API_TOKEN": "test-token",
    "CLOUDFLARE_ACCOUNT_ID": "test-account",
    "CLOUDFLARE_VECTORIZE_INDEX": "test-index",
    "CLOUDFLARE_D1_DATABASE_ID": "test-database",
}


def isolated_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "MCP_MEMORY_BASE_DIR": str(tmp_path / "base"),
            "MCP_MEMORY_SQLITE_PATH": str(tmp_path / "memory.db"),
            **overrides,
        }
    )
    return env


def test_cloudflare_help_does_not_run_backend() -> None:
    result = subprocess.run(
        [sys.executable, str(CLOUDFLARE_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "Run Cloudflare backend integration tests." in result.stdout
    assert "Initializing Cloudflare storage" not in output
    assert "Testing memory storage" not in output


def test_cloudflare_decline_prints_target_without_running_backend(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, str(CLOUDFLARE_SCRIPT)],
        input="n\n",
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=isolated_env(tmp_path, **CLOUDFLARE_ENV),
        check=False,
    )

    assert result.returncode == 0
    assert "Backend: Cloudflare" in result.stdout
    assert "Account ID: test-account" in result.stdout
    assert "Vectorize index: test-index" in result.stdout
    assert "D1 database ID: test-database" in result.stdout
    assert "Tests cancelled by user." in result.stdout
    assert "Initializing Cloudflare storage" not in result.stderr
    assert "Testing memory storage" not in result.stderr


def test_repair_embeddings_help_does_not_run_repair() -> None:
    result = subprocess.run(
        [sys.executable, str(REPAIR_EMBEDDINGS_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "Generate missing embeddings" in result.stdout
    assert "Missing embeddings:" not in output
    assert "Progress:" not in output


def test_repair_embeddings_decline_prints_target_without_running_repair(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "memory.db"
    result = subprocess.run(
        [sys.executable, str(REPAIR_EMBEDDINGS_SCRIPT)],
        input="n\n",
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=isolated_env(
            tmp_path,
            MCP_MEMORY_STORAGE_BACKEND="sqlite_vec",
            MCP_MEMORY_SQLITE_PATH=str(database_path),
        ),
        check=False,
    )

    assert result.returncode == 0
    assert "Backend: sqlite_vec" in result.stdout
    assert f"Database: {database_path}" in result.stdout
    assert "Operation cancelled by user." in result.stdout
    assert "Missing embeddings:" not in result.stderr
    assert "Progress:" not in result.stderr
    assert not database_path.exists()


def test_repair_embeddings_refuses_cloudflare_even_with_yes(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(REPAIR_EMBEDDINGS_SCRIPT), "--yes"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=isolated_env(
            tmp_path,
            MCP_MEMORY_STORAGE_BACKEND="cloudflare",
            **CLOUDFLARE_ENV,
        ),
        check=False,
    )

    assert result.returncode == 2
    assert "Backend: cloudflare" in result.stdout
    assert "Database: None" in result.stdout
    assert "has no local SQLite database to repair" in result.stdout
    assert "Missing embeddings:" not in result.stderr
    assert "Progress:" not in result.stderr


def test_cloudflare_setup_help_does_not_create_resources() -> None:
    result = subprocess.run(
        [sys.executable, str(SETUP_CLOUDFLARE_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "Create Cloudflare resources" in result.stdout
    assert "Creating Vectorize index" not in output
    assert "Creating D1 database" not in output


def test_cloudflare_setup_decline_prints_target_without_creating_resources(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, str(SETUP_CLOUDFLARE_SCRIPT), "--no-r2"],
        input="n\n",
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=isolated_env(tmp_path, **CLOUDFLARE_ENV),
        check=False,
    )

    assert result.returncode == 0
    assert "Backend: Cloudflare" in result.stdout
    assert "Account ID: test-account" in result.stdout
    assert "Vectorize index: mcp-memory-index" in result.stdout
    assert "D1 database: mcp-memory-db" in result.stdout
    assert "Setup cancelled by user." in result.stdout
    assert "Creating Vectorize index" not in result.stderr
    assert "Creating D1 database" not in result.stderr
