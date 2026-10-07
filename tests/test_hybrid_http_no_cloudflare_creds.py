"""Regression test for #1304: hybrid + HTTP secondary must not require Cloudflare creds.

Bug: config/storage.py validates CLOUDFLARE_* and calls sys.exit(1) whenever
STORAGE_BACKEND == 'hybrid', regardless of MCP_HYBRID_SECONDARY_BACKEND. When the
secondary is 'http' (RemoteHTTPStorage), Cloudflare is never used, so demanding
CLOUDFLARE_API_TOKEN/ACCOUNT_ID/VECTORIZE_INDEX/D1_DATABASE_ID is wrong and makes
hybrid-over-HTTP impossible without dummy Cloudflare values.

The validation module runs at import time and may call sys.exit(1); we therefore
import it in a clean subprocess with a controlled environment.
"""
import os
import subprocess
import sys

import pytest


def _import_storage_config(env: dict) -> subprocess.CompletedProcess:
    """Import mcp_memory_service.config.storage in a subprocess with the given env.

    Returns the CompletedProcess (returncode 0 = import succeeded without sys.exit).
    """
    full_env = os.environ.copy()
    # Force-empty the Cloudflare creds so a local .env cannot reintroduce them.
    # config.base loads .env with load_dotenv(override=False), so setting these to
    # "" (rather than popping) guarantees dotenv will NOT restore real values, and
    # the validator treats empty strings as missing (`if not CLOUDFLARE_...`).
    for k in (
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_VECTORIZE_INDEX",
        "CLOUDFLARE_D1_DATABASE_ID",
    ):
        full_env[k] = ""
    full_env.update(env)
    code = (
        "import importlib; "
        "import mcp_memory_service.config.storage as s; "
        "import importlib as _i; _i.reload(s); "
        "print('BACKEND=' + str(s.STORAGE_BACKEND)); "
        "print('SECONDARY=' + str(s.MCP_HYBRID_SECONDARY_BACKEND))"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_hybrid_http_secondary_does_not_require_cloudflare_creds():
    """hybrid + secondary=http WITHOUT any Cloudflare creds must import cleanly."""
    result = _import_storage_config(
        {
            "MCP_MEMORY_STORAGE_BACKEND": "hybrid",
            "MCP_HYBRID_SECONDARY_BACKEND": "http",
            "MCP_HYBRID_SECONDARY_URL": "https://example.test/memory",
        }
    )
    assert result.returncode == 0, (
        "hybrid+http should not sys.exit on missing Cloudflare creds.\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "BACKEND=hybrid" in result.stdout
    assert "SECONDARY=http" in result.stdout


def test_hybrid_cloudflare_secondary_still_requires_creds():
    """Guard against over-correction: hybrid + secondary=cloudflare WITHOUT creds must still fail."""
    result = _import_storage_config(
        {
            "MCP_MEMORY_STORAGE_BACKEND": "hybrid",
            "MCP_HYBRID_SECONDARY_BACKEND": "cloudflare",
        }
    )
    assert result.returncode != 0, (
        "hybrid+cloudflare without creds must still sys.exit(1).\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )


def test_cloudflare_backend_still_requires_creds():
    """Pure cloudflare backend without creds must still fail (unchanged behavior)."""
    result = _import_storage_config(
        {"MCP_MEMORY_STORAGE_BACKEND": "cloudflare"}
    )
    assert result.returncode != 0, (
        "cloudflare backend without creds must still sys.exit(1).\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
