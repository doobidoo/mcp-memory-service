"""Keep the package manager bootstrap above the compromised CA bundle versions."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _installer():
    path = Path(__file__).resolve().parents[2] / "scripts/installation/install_uv.py"
    spec = importlib.util.spec_from_file_location("install_uv_security", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_upgrades_pip_before_installing_fixed_uv(monkeypatch, tmp_path):
    installer = _installer()
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(installer.subprocess, "check_call", calls.append)

    installer.main()

    assert calls == [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip>=26.2.2"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "uv>=0.13.4"],
    ]


def test_failed_pip_upgrade_stops_uv_installation(monkeypatch, tmp_path):
    installer = _installer()
    calls = []
    monkeypatch.chdir(tmp_path)

    def fail(command):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(installer.subprocess, "check_call", fail)
    with pytest.raises(SystemExit, match="1"):
        installer.main()
    assert calls == [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip>=26.2.2"]
    ]
