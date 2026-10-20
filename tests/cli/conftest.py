"""Fixtures for the doctor CLI tests.

Two boundaries must not leak from the test process to the host:

* the readiness window — the poll is bounded by ``ready_timeout`` (45 s default,
  overridable through ``MCP_DOCTOR_READY_TIMEOUT``); tests that drive ``main()``
  cannot inject a no-op sleep, so a health check that never recovers would spend
  39 s of backoff. Zeroing it keeps the suite honest and fast.
* the unit state — since the G5 fix (B1.1) the restart decision always consults
  the real state, which in production is the honest behaviour but in tests would
  let this machine's ``active`` unit suppress the restart under test. Declaring
  ``failed`` here makes the intent explicit; tests that need other states pass
  ``unit_state=`` themselves.

The ``MCP_DOCTOR_READY_TIMEOUT`` reading itself is covered by
``test_resolve_ready_timeout_*``, which clears the env var first.
"""

import pytest

from mcp_memory_service.cli import doctor


@pytest.fixture(autouse=True)
def _controlled_environment(monkeypatch):
    monkeypatch.setenv("MCP_DOCTOR_READY_TIMEOUT", "0")
    monkeypatch.setattr(doctor, "_default_unit_state", lambda: "failed")
