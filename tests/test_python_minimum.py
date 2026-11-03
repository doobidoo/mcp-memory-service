"""Keep installation checks and locked dependencies safe at the Python floor."""

import importlib.util
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path):
    spec = importlib.util.spec_from_file_location("minimum_check", ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("manifest", ["pyproject.toml", "pyproject-lite.toml"])
def test_distribution_requires_supported_python(manifest):
    project = tomllib.loads((ROOT / manifest).read_text())["project"]
    supported = SpecifierSet(project["requires-python"])
    assert "3.10" not in supported
    assert "3.11" in supported
    assert "3.12" in supported
    assert "Programming Language :: Python :: 3.10" not in project["classifiers"]


def test_lock_has_no_vulnerable_filelock():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    filelocks = [p for p in lock["package"] if p["name"] == "filelock"]
    assert filelocks
    assert all(Version(p["version"]) >= Version("4.2.1") for p in filelocks)


@pytest.mark.parametrize("minor", [10, 11, 12])
def test_service_installer_python_floor(monkeypatch, minor):
    installer = load_script("scripts/installation/install_service.py")
    monkeypatch.setattr(
        installer,
        "sys",
        SimpleNamespace(version_info=(3, minor), version=f"3.{minor}", exit=sys.exit),
    )
    if minor < 11:
        with pytest.raises(SystemExit) as error:
            installer.check_python_version()
        assert error.value.code == 1
    else:
        installer.check_python_version()


@pytest.mark.parametrize("minor", [10, 11, 12])
def test_installation_verifier_python_floor(monkeypatch, minor):
    verifier = load_script("scripts/testing/test_installation.py")
    monkeypatch.setattr(
        verifier.platform, "python_version_tuple", lambda: ("3", str(minor), "0")
    )
    assert verifier.check_python_version() is (minor >= 11)


@pytest.mark.parametrize("minor", [10, 11, 12])
def test_environment_verifier_accepts_newer_python(monkeypatch, minor):
    verifier = load_script("scripts/validation/verify_environment.py")
    monkeypatch.setattr(
        verifier,
        "sys",
        SimpleNamespace(version_info=(3, minor), version=f"3.{minor}.0"),
    )
    instance = verifier.EnvironmentVerifier.__new__(verifier.EnvironmentVerifier)
    instance.critical_failures = []
    instance.verification_results = []
    instance.verify_python_version()
    assert bool(instance.critical_failures) is (minor < 11)
    assert bool(instance.verification_results) is (minor >= 11)


def test_main_installer_rejects_python310_before_installing(monkeypatch):
    installer = load_script("scripts/installation/install.py")
    monkeypatch.setattr(
        installer, "sys", SimpleNamespace(version_info=(3, 10), exit=sys.exit)
    )
    with pytest.raises(SystemExit) as error:
        installer.main()
    assert error.value.code == 1


def test_service_dependencies_reject_python310(monkeypatch):
    service = load_script("scripts/service/service_utils.py")
    monkeypatch.setattr(
        service, "sys", SimpleNamespace(version_info=(3, 10), version="3.10")
    )
    ok, message = service.check_dependencies()
    assert not ok
    assert "3.11" in message
