"""Tests for the production-settings guard of the xerj agent-memory adapter."""
import importlib.util
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "benchmarks",
                     "xerj_agent_memory", "run_mms.py")
_spec = importlib.util.spec_from_file_location("xerj_run_mms", _PATH)
run_mms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_mms)


@pytest.fixture
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(run_mms.LEAK_PREFIXES) or key in run_mms.LEAK_KEYS:
            monkeypatch.delenv(key)
    return monkeypatch


def test_clean_environment_passes(clean_env):
    run_mms.check_leaks()


@pytest.mark.parametrize("key", ["MCP_API_KEY", "CLOUDFLARE_API_TOKEN", "MCP_SYNC_HUB_URL",
                                 "MCP_SEMANTIC_DEDUP_THRESHOLD"])
def test_production_setting_exits(clean_env, key):
    clean_env.setenv(key, "x")
    with pytest.raises(SystemExit, match=key):
        run_mms.check_leaks()


def test_nodedup_flag_is_allowed_only_when_named(clean_env):
    clean_env.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    run_mms.check_leaks(allowed=("MCP_SEMANTIC_DEDUP_ENABLED",))
    with pytest.raises(SystemExit):
        run_mms.check_leaks()


def test_isolate_points_storage_at_workdir(clean_env, tmp_path):
    clean_env.chdir(tmp_path)
    for key in ("HOME", "MCP_MEMORY_BASE_DIR", "MCP_MEMORY_STORAGE_BACKEND", "MCP_MEMORY_SQLITE_PATH"):
        clean_env.setenv(key, "unchanged")
    try:
        run_mms.isolate(str(tmp_path), "nodedup")
        assert os.environ["MCP_MEMORY_SQLITE_PATH"] == str(tmp_path / "mem.db")
        assert os.environ["HOME"] == str(tmp_path)
        assert os.environ["MCP_SEMANTIC_DEDUP_ENABLED"] == "false"
    finally:
        os.environ.pop("MCP_SEMANTIC_DEDUP_ENABLED", None)  # set by isolate(), not monkeypatch
