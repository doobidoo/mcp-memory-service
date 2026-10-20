"""RED tests for scripts/hermes/instalar.py (D7 installer — not yet implemented).

The module under test does not exist. Every test imports it lazily so the
failure is an explicit ModuleNotFoundError today, and the same test starts
exercising the real contract once G4 creates the module.

Contract under test (docs/rfc/planned/spec-d7-instalador.md, R1–R14):
install mode detection, platform backend, idle plan when already installed,
post-write verification, surgical uninstall, abort before writing when the
service is down, and commands instead of YAML edits.

No real `hermes`, `systemctl` or `schtasks` calls: tmp_path plus injection only.
"""

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_HERMES_DIR = REPO_ROOT / "scripts" / "hermes"
if str(SCRIPTS_HERMES_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_HERMES_DIR))


def _mod():
    """Import the installer lazily so RED is a clear ModuleNotFoundError."""
    return importlib.import_module("instalar")


# --- R1: HERMES_HOME resolution -------------------------------------------


def test_resolve_hermes_home_prefers_env(tmp_path):
    instalar = _mod()
    env_home = tmp_path / "custom_hermes"
    env_home.mkdir()

    resolved = instalar.resolve_hermes_home(
        {"HERMES_HOME": str(env_home)}, home=tmp_path
    )

    assert resolved == env_home


def test_resolve_hermes_home_falls_back_to_dot_hermes(tmp_path):
    instalar = _mod()
    (tmp_path / ".hermes").mkdir()

    assert instalar.resolve_hermes_home({}, home=tmp_path) == tmp_path / ".hermes"


def test_resolve_hermes_home_rejects_unusable_path(tmp_path):
    instalar = _mod()
    bogus = tmp_path / "missing" / "nested"

    with pytest.raises(Exception) as exc:
        instalar.resolve_hermes_home({"HERMES_HOME": str(bogus)}, home=tmp_path)

    assert "missing" in str(exc.value) or "HERMES_HOME" in str(exc.value)


# --- R12/R13: platform backend --------------------------------------------


def test_detect_backend_maps_supported_platforms():
    instalar = _mod()

    assert instalar.detect_backend("Linux") == "systemd"
    assert instalar.detect_backend("Windows") == "schtasks"


def test_detect_backend_rejects_unsupported_platform():
    instalar = _mod()

    with pytest.raises(Exception) as exc:
        instalar.detect_backend("Darwin")

    assert "Darwin" in str(exc.value)


# --- R12/CA8: Windows path must not use POSIX tools -----------------------


def test_build_schtasks_command_is_onlogon_and_posix_free(tmp_path):
    instalar = _mod()

    cmd = instalar.build_schtasks_command(
        python_exe=Path("C:/Python311/python.exe"),
        script=Path("C:/mms/serve.py"),
        task_name="mms-memory-service",
    )
    joined = " ".join(str(part) for part in cmd)

    assert "/sc" in cmd and "onlogon" in cmd
    assert "mms-memory-service" in joined
    for forbidden in ("bash", "systemctl", "chmod"):
        assert forbidden not in joined


# --- R6: idempotency ------------------------------------------------------


def test_plan_is_empty_when_everything_is_installed(tmp_path):
    instalar = _mod()
    installed = {"mcp": True, "skill": True, "service": True, "hooks": True}

    assert instalar.plan(tmp_path, 3202, "systemd", installed) == []


def test_plan_has_actions_on_a_fresh_machine(tmp_path):
    instalar = _mod()

    actions = instalar.plan(tmp_path, 3202, "systemd", {})

    assert len(actions) > 0


# --- R5/R8: post-write verification --------------------------------------


def test_verify_search_returns_memories_when_found():
    instalar = _mod()
    memories = [{"content": "probe"}]

    assert instalar.verify_search(lambda: memories) == memories


def test_verify_search_fails_with_clear_message_on_empty_store():
    instalar = _mod()

    with pytest.raises(Exception) as exc:
        instalar.verify_search(lambda: [])

    assert "nenhuma memoria" in str(exc.value).lower()


# --- R7/R9/CA3: surgical uninstall ---------------------------------------


def test_uninstall_removes_only_manifest_entries(tmp_path):
    instalar = _mod()
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    ours = hermes_home / "mms-skill.md"
    ours.write_text("created by the installer")
    user_file = hermes_home / "config-do-usuario.txt"
    user_file.write_text("pre-existing")

    manifest = tmp_path / "manifest.json"
    instalar.write_manifest(manifest, [str(ours)])

    removed = instalar.uninstall(manifest, hermes_home)

    assert str(ours) in " ".join(str(item) for item in removed)
    assert not ours.exists()
    assert user_file.exists(), "uninstall must never delete a file it did not create"


# --- R2/CA4: abort before writing when the service is down ----------------


def test_apply_writes_nothing_when_health_is_down(tmp_path):
    instalar = _mod()

    def down():
        raise ConnectionRefusedError("connection refused")

    def boom():
        raise AssertionError("apply must not run when health is down")

    with pytest.raises(Exception):
        instalar.apply_or_abort(
            hermes_home=tmp_path,
            port=3202,
            health_get=down,
            executor=boom,
        )

    assert list(tmp_path.iterdir()) == [], "no artefact may be written on abort"


# --- R3/CA5: configuration via commands, never YAML edits ----------------


def test_plan_configures_mcp_through_hermes_cli(tmp_path):
    instalar = _mod()

    actions = instalar.plan(tmp_path, 3202, "systemd", {})
    mcp_actions = [
        a for a in actions if "mcp" in " ".join(str(p) for p in getattr(a, "argv", a))
    ]

    assert mcp_actions, "an action must register the MCP server"
    joined = " ".join(str(p) for a in mcp_actions for p in getattr(a, "argv", a))
    assert "hermes" in joined and "add" in joined


def test_mcp_action_points_at_the_mcp_endpoint(tmp_path):
    """The MCP server lives at /mcp, not at the host root.

    Regression found while inspecting the GREEN output: the plan registered
    `http://localhost:3202` (no path), which Hermes would try to speak MCP to
    and fail. The original test only checked `hermes mcp add`, so it stayed green.
    """
    instalar = _mod()

    actions = instalar.plan(tmp_path, 3202, "systemd", {})
    joined = " ".join(str(p) for a in actions for p in getattr(a, "argv", []))

    assert "hermes mcp add" in joined
    assert "http://localhost:3202/mcp" in joined


class TestCliIsIdempotent:
    """The bug the G5 found: `main()` planned everything on every run.

    `plan()` was idempotent, but the CLI handed it `installed={}` unconditionally,
    so a second run reinstalled from scratch. These tests drive `main()`.
    """

    def _run_cli(self, tmp_path, monkeypatch, monkeypatch_installed):
        instalar = _mod()
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        executed = []

        # The developer's shell may export HERMES_HOME (it did), which would send
        # every artefact outside tmp_path and make this test lie.
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(
            instalar, "detect_installed", lambda *a, **k: monkeypatch_installed
        )

        def fake_executor(actions):
            for action in actions:
                executed.append(list(action.argv))
            return []

        monkeypatch.setattr(instalar, "default_executor", fake_executor)
        monkeypatch.setattr(instalar, "default_health_get", lambda port: {"status": 200})
        # Never let a CLI test reach a real service: verify_installation writes a
        # probe, and the port in the fixture is not the test's.
        monkeypatch.setattr(instalar, "verify_installation", lambda port: None)

        rc = instalar.main(["--hermes-home", str(tmp_path), "--port", "3202"])
        return rc, executed, hermes_home

    def test_second_run_does_nothing_when_already_installed(self, tmp_path, monkeypatch):
        rc, executed, _ = self._run_cli(
            tmp_path,
            monkeypatch,
            {"mcp": True, "skill": True, "service": True, "hooks": True},
        )

        assert rc == 0
        assert executed == [], "an installed machine must not run any command again"

    def test_first_run_executes_actions_and_writes_undo_commands(self, tmp_path, monkeypatch):
        rc, executed, hermes_home = self._run_cli(tmp_path, monkeypatch, {})

        assert rc == 0
        assert any(cmd[:3] == ["hermes", "mcp", "add"] for cmd in executed)
        manifest = hermes_home / "mms-install-manifest.json"
        assert manifest.is_file()
        instalar = _mod()
        undo = instalar.read_manifest_undo(manifest)
        assert ["hermes", "mcp", "remove", "mms"] in undo


class TestUninstallRobustness:
    def test_corrupted_manifest_does_not_raise(self, tmp_path):
        instalar = _mod()
        manifest = tmp_path / "mms-install-manifest.json"
        manifest.write_text("{ isto nao e json valido")

        assert instalar.uninstall(manifest, tmp_path) == []

    def test_partial_manifest_skips_missing_files(self, tmp_path):
        instalar = _mod()
        gone = tmp_path / "ja-nao-existe.md"
        manifest = tmp_path / "mms-install-manifest.json"
        instalar.write_manifest(manifest, [str(gone)])

        assert instalar.uninstall(manifest, tmp_path) == []

    def test_uninstall_runs_undo_commands(self, tmp_path):
        instalar = _mod()
        manifest = tmp_path / "mms-install-manifest.json"
        instalar.write_manifest(manifest, [], [["hermes", "mcp", "remove", "mms"]])
        ran = []

        removed = instalar.uninstall(
            manifest, tmp_path, runner=lambda cmd: (ran.append(list(cmd)), (0, ""))[1]
        )

        assert ran == [["hermes", "mcp", "remove", "mms"]]
        assert "hermes mcp remove mms" in removed


def test_each_action_runs_exactly_once(tmp_path, monkeypatch):
    """Regression: `apply_or_abort` replanned with `{}` and executed, then
    `main` executed the same plan again — every command ran twice."""
    instalar = _mod()
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    executed = []

    monkeypatch.setattr(instalar, "detect_installed", lambda *a, **k: {})
    monkeypatch.setattr(instalar, "default_health_get", lambda port: {"status": 200})

    def counting_executor(actions):
        executed.extend(list(a.argv) for a in actions)
        return []

    monkeypatch.setattr(instalar, "default_executor", counting_executor)
    monkeypatch.setattr(instalar, "verify_installation", lambda port: None)

    instalar.main(["--hermes-home", str(tmp_path), "--port", "3202"])

    assert len(executed) == len(set(map(tuple, executed))), (
        f"comando repetido na mesma execucao: {executed}"
    )
