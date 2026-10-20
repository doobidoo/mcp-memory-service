"""Tests for scripts/hermes/install.py.

The installer registers the MCP server with Hermes and proves a write/read round
trip. The load-bearing cases are the ones the maintainer's review called out:

* a failed store must not look like success (`memory_search` always returns one
  text block, including "No memories found" and "Error searching memories");
* a constant probe is rejected as a duplicate on reinstall, so the search would
  find the previous run's probe and never exercise the new write;
* a registration counts only when it points at this port and is enabled.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "hermes"))


def _mod():
    """Load the script by path: `import install` collides with unrelated modules."""
    import importlib.util

    path = REPO_ROOT / "scripts" / "hermes" / "install.py"
    spec = importlib.util.spec_from_file_location("hermes_install", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # @dataclass resolves its own module through sys.modules while executing.
    sys.modules["hermes_install"] = module
    spec.loader.exec_module(module)
    return module


# --- location ----------------------------------------------------------------


def test_explicit_home_is_used_literally(tmp_path):
    """`--hermes-home <dir>` means <dir>, not <dir>/.hermes (maintainer review)."""
    instalar = _mod()
    profile = tmp_path / "profiles" / "work"
    profile.mkdir(parents=True)

    assert instalar.resolve_hermes_home({"HERMES_HOME": str(profile)}, Path.home()) == profile


def test_default_home_appends_dot_hermes(tmp_path):
    instalar = _mod()
    (tmp_path / ".hermes").mkdir()

    assert instalar.resolve_hermes_home({}, tmp_path) == tmp_path / ".hermes"


def test_missing_profile_is_reported_not_created(tmp_path):
    instalar = _mod()

    with pytest.raises(RuntimeError):
        instalar.resolve_hermes_home({"HERMES_HOME": str(tmp_path / "nope")}, tmp_path)


def test_env_file_supplies_the_port(tmp_path):
    """`--env-file` must feed the port, not just the environment."""
    instalar = _mod()
    env_file = tmp_path / "service.env"
    env_file.write_text("# comment\nMCP_HTTP_PORT=3296\n")

    assert instalar.env_file_port(env_file) == 3296


# --- registration ------------------------------------------------------------


def test_registration_requires_the_right_port_and_enabled_state():
    instalar = _mod()

    good = lambda _argv: (0, "  mms   http://localhost:8000/mcp   all   enabled\n")
    other_port = lambda _argv: (0, "  mms   http://localhost:9999/mcp   all   enabled\n")
    disabled = lambda _argv: (0, "  mms   http://localhost:8000/mcp   all   disabled\n")
    unrelated = lambda _argv: (0, "  codebase-memory   /usr/bin/cbm   all   enabled\n")

    assert instalar.mcp_is_registered(8000, good) is True
    assert instalar.mcp_is_registered(8000, other_port) is False
    assert instalar.mcp_is_registered(8000, disabled) is False
    assert instalar.mcp_is_registered(8000, unrelated) is False


def test_plan_is_empty_when_registered():
    instalar = _mod()

    assert instalar.plan(8000, installed=True) == []
    actions = instalar.plan(8000, installed=False)
    assert len(actions) == 1
    assert actions[0].argv[:3] == ["hermes", "mcp", "add"]


def test_plan_has_no_service_or_skill_actions():
    """Service backend and skill/hooks were removed from this PR by review."""
    instalar = _mod()
    rendered = " ".join(a.render() for a in instalar.plan(8000, installed=False))

    for forbidden in ("systemctl", "schtasks", "skills.mms", "hooks.mms"):
        assert forbidden not in rendered


# --- round trip (the guarantee the PR is built on) ---------------------------


def _call(store=None, search=None):
    def call(_port, tool, _args):
        return store if tool == "memory_store" else search

    return call


def test_round_trip_returns_the_probe_hash():
    instalar = _mod()
    token = "abc123"
    store = {"result": {"content": [{"type": "text", "text": "Memory stored successfully"}]}}
    search = {
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": f"Found 1 memory for '{token}'\n\n1. [note] install probe {token}\n"
                    f"Hash: {'a' * 64}",
                }
            ]
        }
    }

    assert instalar.verify_round_trip(8000, token, call=_call(store, search)) == "a" * 64


def test_failed_store_is_not_success():
    """`memory_search` always returns a text block — the store must be inspected."""
    instalar = _mod()
    store = {"error": {"code": -32603, "message": "boom"}}
    search = {"result": {"content": [{"type": "text", "text": "No memories found"}]}}

    with pytest.raises(RuntimeError):
        instalar.verify_round_trip(8000, "tok", call=_call(store, search))


def test_store_rejected_as_duplicate_is_not_success():
    instalar = _mod()
    store = {"result": {"content": [{"type": "text", "text": "Error: duplicate content"}]}}

    with pytest.raises(RuntimeError):
        instalar.verify_round_trip(8000, "tok", call=_call(store, {}))


def test_empty_search_is_not_success():
    instalar = _mod()
    store = {"result": {"content": [{"type": "text", "text": "stored"}]}}
    search = {"result": {"content": [{"type": "text", "text": "No memories found for 'tok'"}]}}

    with pytest.raises(RuntimeError):
        instalar.verify_round_trip(8000, "tok", call=_call(store, search))


def test_search_error_is_not_success():
    instalar = _mod()
    store = {"result": {"content": [{"type": "text", "text": "stored"}]}}
    search = {"result": {"content": [{"type": "text", "text": "Error searching memories: x"}]}}

    with pytest.raises(RuntimeError):
        instalar.verify_round_trip(8000, "tok", call=_call(store, search))


def test_search_must_return_the_probe_itself():
    """A previous run's probe must not satisfy this run's write."""
    instalar = _mod()
    store = {"result": {"content": [{"type": "text", "text": "stored"}]}}
    search = {"result": {"content": [{"type": "text", "text": "install probe OUTRA-COISA"}]}}

    with pytest.raises(RuntimeError):
        instalar.verify_round_trip(8000, "tok-desta-run", call=_call(store, search))


def test_probe_is_deleted_after_the_round_trip():
    instalar = _mod()
    calls = []

    instalar.delete_probe(8000, "b" * 64, call=lambda p, t, a: calls.append((t, a)) or {})

    assert calls and calls[0][0] == "memory_delete"
    assert calls[0][1]["content_hash"] == "b" * 64


def test_delete_probe_without_hash_is_a_noop():
    instalar = _mod()
    calls = []

    instalar.delete_probe(8000, "", call=lambda p, t, a: calls.append(t) or {})

    assert calls == []


# --- CLI ---------------------------------------------------------------------


def test_dry_run_writes_nothing(tmp_path):
    instalar = _mod()
    profile = tmp_path / "profile"
    profile.mkdir()

    rc = instalar.main(["--hermes-home", str(profile), "--dry-run"])

    assert rc == 0
    assert list(profile.rglob("*")) == []


def test_uninstall_failure_returns_nonzero(tmp_path, monkeypatch):
    instalar = _mod()
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setattr(instalar, "default_runner", lambda argv: (1, "boom"))

    rc = instalar.main(["--hermes-home", str(profile), "--uninstall"])

    assert rc == 1, "falha ao desfazer precisa sair com codigo != 0"


def test_manifest_round_trips_the_probe_hash(tmp_path):
    instalar = _mod()
    path = tmp_path / "mms-install.json"

    instalar.write_manifest(path, "c" * 64)

    assert instalar.read_manifest(path)["probe_hash"] == "c" * 64


def test_corrupted_manifest_degrades(tmp_path):
    instalar = _mod()
    path = tmp_path / "mms-install.json"
    path.write_text("{ not json")

    assert instalar.read_manifest(path) == {}


# --- regressions found in review round 2 -------------------------------------


def test_uninstall_dry_run_changes_nothing(tmp_path, monkeypatch):
    """`--uninstall --dry-run` previewed by removing the registration."""
    instalar = _mod()
    profile = tmp_path / "profile"
    profile.mkdir()
    manifest = profile / "mms-install.json"
    instalar.write_manifest(manifest, "d" * 64)
    calls = []
    monkeypatch.setattr(instalar, "default_runner", lambda argv: (calls.append(list(argv)), (0, ""))[1])

    rc = instalar.main(["--hermes-home", str(profile), "--uninstall", "--dry-run"])

    assert rc == 0
    assert calls == [], "dry-run nao pode executar `hermes mcp remove`"
    assert manifest.is_file(), "dry-run nao pode apagar o manifesto"


def test_registration_aborts_when_health_fails(tmp_path, monkeypatch):
    """The abort-before-writing promise: no Hermes change on a down server."""
    instalar = _mod()
    profile = tmp_path / "profile"
    profile.mkdir()
    calls = []
    monkeypatch.setattr(instalar, "mcp_is_registered", lambda port, runner=None: False)
    monkeypatch.setattr(instalar, "health_ok", lambda port: (False, "service not answering"))
    monkeypatch.setattr(instalar, "default_runner", lambda argv: (calls.append(list(argv)), (0, ""))[1])

    rc = instalar.main(["--hermes-home", str(profile), "--port", "8000"])

    assert rc == 1
    assert calls == [], "nada pode ser executado antes de o servico responder"


def test_api_key_is_written_to_the_profile_env(tmp_path, monkeypatch):
    """Hermes reads MCP_MMS_API_KEY from the profile .env; without it the
    registration exists but cannot authenticate."""
    instalar = _mod()
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / ".env").write_text("OUTRA=coisa\n")
    monkeypatch.setenv("MCP_API_KEY", "s3cret")
    monkeypatch.setattr(instalar, "mcp_is_registered", lambda port, runner=None: False)
    monkeypatch.setattr(instalar, "health_ok", lambda port: (True, ""))
    monkeypatch.setattr(instalar, "default_runner", lambda argv: (0, ""))
    monkeypatch.setattr(instalar, "verify_round_trip", lambda port, token, call=None: "e" * 64)
    monkeypatch.setattr(instalar, "delete_probe", lambda port, h, call=None: None)

    instalar.main(["--hermes-home", str(profile), "--port", "8000"])

    env_text = (profile / ".env").read_text()
    assert "MCP_MMS_API_KEY=s3cret" in env_text
    assert "OUTRA=coisa" in env_text, "o .env existente nao pode ser reescrito"


# --- regressions found in review round 3 -------------------------------------


def test_health_falls_back_to_api_health(monkeypatch):
    """`memory launch` serves /api/health; `memory server` serves /health.

    Checking only one aborted installation against a healthy server.
    """
    import urllib.error

    instalar = _mod()
    seen = []

    def fake_urlopen(request, timeout=None):
        route = request.full_url.split("8000", 1)[1]
        seen.append(route)
        if route == "/health":
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)
        return object()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    ok, reason = instalar.health_ok(8000)

    assert ok is True, reason
    assert seen == ["/health", "/api/health"]


def test_remove_env_key_removes_only_our_key(tmp_path):
    instalar = _mod()
    env = tmp_path / ".env"
    env.write_text("OUTRA=coisa\nMCP_MMS_API_KEY=s3cret\n")

    instalar.remove_env_key(tmp_path)

    text = env.read_text()
    assert "MCP_MMS_API_KEY" not in text
    assert "OUTRA=coisa" in text


def test_uninstall_keeps_a_key_it_did_not_add(tmp_path, monkeypatch):
    """A key the user already had must survive uninstall."""
    instalar = _mod()
    profile = tmp_path / "profile"
    profile.mkdir()
    env = profile / ".env"
    env.write_text("MCP_MMS_API_KEY=do-usuario\n")
    instalar.write_manifest(profile / "mms-install.json", "", added_env_key=False)
    monkeypatch.setattr(instalar, "default_runner", lambda argv: (0, ""))

    instalar.main(["--hermes-home", str(profile), "--uninstall"])

    assert "MCP_MMS_API_KEY=do-usuario" in env.read_text()


def test_install_records_the_probe_hash_in_the_manifest(tmp_path, monkeypatch):
    """Without the hash the uninstall cannot delete the probe it created."""
    instalar = _mod()
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.delenv("MCP_API_KEY", raising=False)
    # False before registration (plan), True after (effect check).
    calls = {"n": 0}

    def registered(port, runner=None):
        calls["n"] += 1
        return calls["n"] > 1

    monkeypatch.setattr(instalar, "mcp_is_registered", registered)
    monkeypatch.setattr(instalar, "health_ok", lambda port: (True, ""))
    monkeypatch.setattr(instalar, "default_runner", lambda argv: (0, ""))
    monkeypatch.setattr(instalar, "verify_round_trip", lambda port, token, call=None: "f" * 64)
    monkeypatch.setattr(instalar, "delete_probe", lambda port, h, call=None: None)

    assert instalar.main(["--hermes-home", str(profile), "--port", "8000"]) == 0

    assert instalar.read_manifest(profile / "mms-install.json")["probe_hash"] == "f" * 64


# --- review round 4: a repeat install must still verify -----------------------


def test_repeat_install_still_verifies_the_round_trip(tmp_path, monkeypatch):
    """Already registered must not mean "nothing to check"."""
    instalar = _mod()
    profile = tmp_path / "profile"
    profile.mkdir()
    calls = []
    monkeypatch.setattr(instalar, "mcp_is_registered", lambda port, runner=None: True)
    monkeypatch.setattr(instalar, "health_ok", lambda port: (True, ""))
    monkeypatch.setattr(
        instalar, "verify_round_trip", lambda port, token, call=None: calls.append(token) or "a" * 64
    )
    monkeypatch.setattr(instalar, "delete_probe", lambda port, h, call=None: None)

    rc = instalar.main(["--hermes-home", str(profile), "--port", "8000"])

    assert rc == 0
    assert calls, "a repeat install precisa rodar a sonda de novo"


def test_repeat_install_fails_when_the_server_is_down(tmp_path, monkeypatch):
    instalar = _mod()
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setattr(instalar, "mcp_is_registered", lambda port, runner=None: True)
    monkeypatch.setattr(instalar, "health_ok", lambda port: (False, "service not answering"))

    rc = instalar.main(["--hermes-home", str(profile), "--port", "8000"])

    assert rc == 1, "registro existente com servidor fora do ar nao pode sair com sucesso"
