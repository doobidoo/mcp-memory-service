import importlib.util
import sys
from pathlib import Path


def _load_install_py_module(monkeypatch, base_dir: Path):
    repo_root = Path(__file__).resolve().parents[2]
    # Avoid touching user-global app support directories during import.
    monkeypatch.setenv("MCP_MEMORY_BASE_DIR", str(base_dir))
    install_py = repo_root / "install.py"
    spec = importlib.util.spec_from_file_location("mcp_memory_service_install_py", install_py)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_install_py_uses_pip_when_available(monkeypatch, tmp_path):
    install_module = _load_install_py_module(monkeypatch, tmp_path)

    recorded = {}

    def fake_check_call(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(install_module, "_pip_available", lambda: True)
    monkeypatch.setattr(install_module, "subprocess", install_module.subprocess)
    monkeypatch.setattr(install_module.subprocess, "check_call", fake_check_call)

    install_module._install_python_packages(["sqlite-vec"], extra_args=["--no-deps"], silent=True)

    assert recorded["cmd"][:4] == [sys.executable, "-m", "pip", "install"]
    assert "--no-deps" in recorded["cmd"]
    assert "sqlite-vec" in recorded["cmd"]
    assert "stdout" in recorded["kwargs"]
    assert "stderr" in recorded["kwargs"]


def test_install_py_uses_uv_when_pip_missing(monkeypatch, tmp_path):
    install_module = _load_install_py_module(monkeypatch, tmp_path)

    recorded = {}

    def fake_check_call(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(install_module, "_pip_available", lambda: False)
    monkeypatch.setattr(install_module, "_uv_executable", lambda: "/opt/homebrew/bin/uv")
    monkeypatch.setattr(install_module.subprocess, "check_call", fake_check_call)

    install_module._install_python_packages(["sqlite-vec"])

    assert recorded["cmd"][:3] == ["/opt/homebrew/bin/uv", "pip", "install"]
    assert "--python" in recorded["cmd"]
    assert sys.executable in recorded["cmd"]
    assert "sqlite-vec" in recorded["cmd"]


def test_scripts_installer_uses_uv_when_pip_module_missing(monkeypatch):
    from scripts.installation import install as scripts_install

    recorded = {}

    def fake_run_command_safe(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return True, None

    monkeypatch.setattr(scripts_install.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(scripts_install.shutil, "which", lambda name: "/opt/homebrew/bin/uv")
    monkeypatch.setattr(scripts_install, "run_command_safe", fake_run_command_safe)

    ok = scripts_install.install_package_safe("sqlite-vec", fallback_in_venv=False)
    assert ok is True
    assert recorded["cmd"][:3] == ["/opt/homebrew/bin/uv", "pip", "install"]
    assert "--python" in recorded["cmd"]
    assert sys.executable in recorded["cmd"]
    assert "sqlite-vec" in recorded["cmd"]


def test_scripts_installer_returns_false_when_no_pip_and_no_uv(monkeypatch):
    from scripts.installation import install as scripts_install

    monkeypatch.setattr(scripts_install.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(scripts_install.shutil, "which", lambda name: None)

    ok = scripts_install.install_package_safe("sqlite-vec", fallback_in_venv=False)
    assert ok is False
