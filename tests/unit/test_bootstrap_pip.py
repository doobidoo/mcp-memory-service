"""Bootstrap pip through a checked CA bundle across runner pip versions."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
SCRIPT = Path(__file__).resolve().parents[2] / "scripts/ci/bootstrap_pip.py"


@pytest.fixture
def bootstrap(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("bootstrap_pip", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("version,legacy", [
    ("24.0", False),
    ("24.2", True),
    ("26.2.1", True),
    ("26.2.2", False),
])
def test_checked_bundle_and_compatible_flags(bootstrap, monkeypatch, tmp_path, version, legacy):
    bundle = tmp_path / "checked.pem"
    events = []

    def inspect(path):
        assert Path(path) == bundle
        events.append("checked")
        return 1, set()

    def run(command, **kwargs):
        assert events == ["checked"]
        events.append("installed")
        expected = [sys.executable, "-m", "pip", "install", "--upgrade"]
        if legacy:
            expected.append("--use-deprecated=legacy-certs")
        expected.extend(["--cert", str(bundle), "pip>=26.2.2"])
        assert command == expected
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bootstrap, "inspect_bundle", inspect)
    monkeypatch.setattr(bootstrap.metadata, "version", lambda name: version)
    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    assert bootstrap.main(["--cert", str(bundle)]) == 0
    assert events == ["checked", "installed"]


def test_blocked_roots_stop_before_download(bootstrap, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(bootstrap, "inspect_bundle", lambda path: (1, {"AAA Certificate Services"}))
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **kw: pytest.fail("pip ran"))
    assert bootstrap.main(["--cert", str(tmp_path / "blocked.pem")]) == 1
    assert "AAA Certificate Services" in capsys.readouterr().err


def test_invalid_bundle_stops_before_download(bootstrap, monkeypatch, tmp_path):
    bundle = tmp_path / "invalid.pem"
    bundle.write_text("not a CA bundle")
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **kw: pytest.fail("pip ran"))
    assert bootstrap.main(["--cert", str(bundle)]) == 1


def test_cli_runs_the_bootstrap_entrypoint(tmp_path):
    bundle = tmp_path / "invalid.pem"
    bundle.write_text("not a CA bundle")
    result = subprocess.run([sys.executable, str(SCRIPT), "--cert", str(bundle)],
                            check=False, capture_output=True, text=True)
    assert result.returncode == 1
    assert "FAIL" in result.stderr


def test_pip_failure_is_propagated(bootstrap, monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "inspect_bundle", lambda path: (1, set()))
    monkeypatch.setattr(bootstrap.metadata, "version", lambda name: "24.0")
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=2))
    assert bootstrap.main(["--cert", str(tmp_path / "safe.pem")]) == 2
