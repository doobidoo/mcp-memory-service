"""
Log-injection tests for server/environment.py (#1146).

The module logs facts about the host it runs on: site-packages paths, the
installed package version and the detected platform. None of those are values
this code produced, so a newline in one must not reach the log as a line break.
"""

import logging
import os
import site
import sys
from types import SimpleNamespace

import pytest

from mcp_memory_service.server import environment

FORGED = "FORGED admin authenticated"


def _messages(caplog):
    return [record.getMessage() for record in caplog.records]


@pytest.fixture
def restore_environ():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def restore_sys_path():
    saved = list(sys.path)
    yield
    sys.path[:] = saved


@pytest.mark.unit
def test_site_packages_paths_do_not_carry_newlines(monkeypatch, caplog, restore_sys_path):
    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    monkeypatch.setattr(site, "getusersitepackages", lambda: f"/home/u/user\n{FORGED}")
    monkeypatch.setattr(site, "getsitepackages", lambda: [f"/usr/lib/global\n{FORGED}"])

    with caplog.at_level(logging.DEBUG):
        environment.setup_python_paths()

    messages = _messages(caplog)
    assert any("Added user site-packages: /home/u/user\\nFORGED" in m for m in messages)
    assert any("Added global site-packages: /usr/lib/global\\nFORGED" in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)


@pytest.mark.unit
def test_site_packages_error_does_not_carry_newlines(monkeypatch, caplog, restore_sys_path):
    def boom():
        raise RuntimeError(f"no site\n{FORGED}")

    monkeypatch.setattr(site, "getsitepackages", boom)

    with caplog.at_level(logging.DEBUG):
        environment.setup_python_paths()

    messages = _messages(caplog)
    assert any("Could not access site-packages: no site\\nFORGED" in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)


@pytest.mark.unit
def test_version_mismatch_does_not_carry_newlines(monkeypatch, caplog):
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: f"1.2.3\n{FORGED}")
    monkeypatch.setitem(sys.modules, "pkg_resources", None)

    with caplog.at_level(logging.DEBUG):
        environment.check_version_consistency()

    messages = _messages(caplog)
    assert any("Installed:   v1.2.3\\nFORGED" in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)


@pytest.mark.unit
def test_detected_system_does_not_carry_newlines(monkeypatch, caplog, restore_environ):
    info = SimpleNamespace(
        os_name=f"linux\n{FORGED}",
        architecture=f"x86_64\n{FORGED}",
        memory_gb=16.0,
        accelerator=f"cuda\n{FORGED}",
    )
    monkeypatch.setattr(environment, "get_system_info", lambda: info)

    with caplog.at_level(logging.INFO):
        environment.configure_environment()

    messages = _messages(caplog)
    assert any("Detected system: linux\\nFORGED admin authenticated x86_64\\nFORGED admin authenticated" in m for m in messages)
    assert any("Accelerator: cuda\\nFORGED admin authenticated" in m for m in messages)
    assert any("Memory: 16.00 GB" in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)
