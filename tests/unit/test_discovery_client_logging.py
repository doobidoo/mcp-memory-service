"""
Log-injection tests for discovery/client.py (#1146).

The discovery client logs the name and url of services another host announced
over mDNS, the error text of a failed health check, and the exception text of
a failed capabilities fetch. A newline in any of them must not reach the log as
a line break.
"""

import asyncio
import logging

import pytest

from mcp_memory_service.discovery import client as client_module
from mcp_memory_service.discovery.client import DiscoveryClient, HealthStatus
from mcp_memory_service.discovery.mdns_service import ServiceDetails

FORGED = "FORGED service verified"
LOGGER = "mcp_memory_service.discovery.client"


def _service(name: str) -> ServiceDetails:
    return ServiceDetails(
        name=name,
        host="192.0.2.10",
        port=8000,
        https=False,
        api_version="2.0.0",
        requires_auth=False,
        service_info=None,
    )


class _Discovery:
    def __init__(self, services):
        self._services = services

    async def discover_services(self):
        return self._services


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> DiscoveryClient:
    """A client whose network discovery returns one service with a forged name."""
    discovery_client = DiscoveryClient.__new__(DiscoveryClient)
    discovery_client.discovery_timeout = 1
    discovery_client._discovery = _Discovery([_service(f"memory\n{FORGED}")])
    return discovery_client


def _assert_clean(caplog, expected):
    messages = [record.getMessage() for record in caplog.records]
    assert any(expected in m for m in messages), messages
    assert not any(f"\n{FORGED}" in m for m in messages), messages


def test_discovered_service_log_does_not_carry_newlines(client, caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        services = asyncio.run(client.discover_services())

    assert len(services) == 1
    _assert_clean(caplog, "Found 1 MCP Memory Services")
    _assert_clean(caplog, "  - memory")


def test_failed_health_check_log_does_not_carry_newlines(client, monkeypatch, caplog):
    async def check_service_health(_service):
        return HealthStatus(
            healthy=False,
            status="unhealthy",
            backend="unknown",
            statistics={},
            response_time_ms=0.0,
            error=f"HTTP 503\n{FORGED}",
        )

    monkeypatch.setattr(client, "check_service_health", check_service_health)

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        best = asyncio.run(client.find_best_service(validate_health=True))

    assert best is None
    _assert_clean(caplog, "failed health check: HTTP 503")


def test_selected_service_logs_do_not_carry_newlines(client, monkeypatch, caplog):
    async def check_service_health(_service):
        return HealthStatus(
            healthy=True,
            status="healthy",
            backend="sqlite_vec",
            statistics={},
            response_time_ms=1.0,
        )

    monkeypatch.setattr(client, "check_service_health", check_service_health)

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        healthy = asyncio.run(client.find_best_service(validate_health=True))
        first = asyncio.run(client.find_best_service(validate_health=False))

    assert healthy is not None and first is not None
    _assert_clean(caplog, "Selected healthy service: memory")
    _assert_clean(caplog, "Selected service: memory")


def test_capabilities_error_log_does_not_carry_newlines(client, monkeypatch, caplog):
    class _Session:
        def __init__(self, **_kwargs):
            raise RuntimeError(f"connection refused\n{FORGED}")

    monkeypatch.setattr(client_module.aiohttp, "ClientSession", _Session)

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        capabilities = asyncio.run(client.get_service_capabilities(_service("memory")))

    assert capabilities is None
    _assert_clean(caplog, "Failed to get service capabilities: connection refused")
