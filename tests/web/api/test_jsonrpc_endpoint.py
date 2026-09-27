# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for JSON-RPC handling on the /mcp endpoint."""

import pytest
import pytest_asyncio
import tempfile
import os
from fastapi.testclient import TestClient

from mcp_memory_service.web.dependencies import set_storage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest_asyncio.fixture
async def initialized_storage(temp_db, monkeypatch):
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
def test_app(initialized_storage, monkeypatch):
    # Patch module-level auth config directly instead of reloading modules.
    # Reloading (importlib.reload) creates new function objects, which
    # desynchronizes FastAPI dependency_overrides keys from the references
    # the app was initialized with, and causes cross-test state pollution.
    from mcp_memory_service.web.oauth import middleware
    monkeypatch.setattr(middleware, "API_KEY", None)
    monkeypatch.setattr(middleware, "OAUTH_ENABLED", False)
    monkeypatch.setattr(middleware, "ALLOW_ANONYMOUS_ACCESS", True)

    from mcp_memory_service.web.app import app
    from mcp_memory_service.web.oauth.middleware import (
        get_current_user, require_read_access, AuthenticationResult,
    )

    set_storage(initialized_storage)

    async def mock_user():
        return AuthenticationResult(
            authenticated=True, client_id="test", scope="read write admin", auth_method="test",
        )

    app.dependency_overrides[get_current_user] = mock_user
    app.dependency_overrides[require_read_access] = mock_user

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.mark.integration
def test_initialized_notification_returns_202_with_empty_body(test_app):
    """
    JSON-RPC 2.0 requires servers never respond to notifications (messages
    without `id`). MCP Streamable HTTP further requires HTTP 202 Accepted
    with no body in this case. Regression for:
    clients like Codex's rmcp that treat a JSON-RPC error response to
    `notifications/initialized` as a handshake failure.
    """
    response = test_app.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )

    assert response.status_code == 202, (
        f"Notifications must get 202 Accepted, got {response.status_code} "
        f"with body: {response.text!r}"
    )
    assert response.content == b"", (
        f"Notifications must get empty body, got: {response.text!r}"
    )


@pytest.mark.integration
def test_initialize_request_still_returns_200_with_result(test_app):
    """Sanity check: regular requests (with `id`) continue to work."""
    response = test_app.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == 1
    assert "result" in body
    assert "protocolVersion" in body["result"]


# ---- FASE 2: Header X-Agent-ID per-request ----

@pytest.mark.integration
def test_mcp_tools_call_reads_x_agent_id_header(test_app):
    """(5) tools/call memory_store SEM agent_id no arguments mas com header X-Agent-ID grava metadata.agent_id."""
    # Store memória via tools/call com header mas sem agent_id no arguments
    response = test_app.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "memory_store",
                "arguments": {
                    "content": "GraphQL schema design patterns and best practices",
                    "metadata": {}
                }
            }
        },
        headers={"X-Agent-ID": "zero"}
    )
    
    assert response.status_code == 200
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert "result" in body
    
    # Verify the memory was stored with agent_id from header
    # The result should contain the stored memory hash
    result = body["result"]
    assert "content" in result and len(result["content"]) > 0
    stored_hash = result["content"][0]["text"]
    
    # Retrieve the memory and verify agent_id was set from header
    response_search = test_app.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "memory_search",
                "arguments": {
                    "query": "GraphQL schema design",
                    "agent_id": "zero"
                }
            }
        }
    )
    
    assert response_search.status_code == 200
    search_body = response_search.json()
    memories = search_body["result"]["content"][0]["text"]
    
    # Should find the memory when searching by agent_id="zero"
    assert "GraphQL schema design" in memories


@pytest.mark.integration
def test_mcp_explicit_arg_overrides_header(test_app):
    """(6) arguments.agent_id explícito VENCE o header X-Agent-ID."""
    # Store com agent_id explícito no arguments E header diferente
    response = test_app.post(
        "/mcp",
        json={
            "jsonrpc": "2.0", 
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "memory_store",
                "arguments": {
                    "content": "Microservices communication patterns and service mesh",
                    "agent_id": "tpol",  # Explícito no arguments
                    "metadata": {}
                }
            }
        },
        headers={"X-Agent-ID": "zero"}  # Header diferente
    )
    
    assert response.status_code == 200
    body = response.json()
    
    # O argumento explícito já funciona na Fase 1, então devemos esperar sucesso
    # Mas como o header ainda não está implementado, vamos apenas verificar que funciona
    assert "result" in body
    
    # Verify the memory was stored with explicit agent_id "tpol", not header "zero"
    # Search by agent_id="tpol" should find the memory
    response_search_tpol = test_app.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "memory_search",
                "arguments": {
                    "query": "Microservices communication",
                    "agent_id": "tpol"
                }
            }
        }
    )
    
    assert response_search_tpol.status_code == 200
    search_body_tpol = response_search_tpol.json()
    memories_tpol = search_body_tpol["result"]["content"][0]["text"]
    
    # Should find the memory when searching by explicit agent_id="tpol"
    assert "Microservices communication" in memories_tpol
    
    # Search by header agent_id="zero" should NOT find it (precedence test)
    response_search_zero = test_app.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "memory_search",
                "arguments": {
                    "query": "Microservices communication",
                    "agent_id": "zero"
                }
            }
        }
    )
    
    assert response_search_zero.status_code == 200
    search_body_zero = response_search_zero.json()
    memories_zero = search_body_zero["result"]["content"][0]["text"]
    
    # Should NOT find the memory when searching by header agent_id="zero"
    # Either no memories found message or empty results, but not the target content
    if "No memories found" not in memories_zero:
        # If memories were found, ensure our target memory is not among them
        assert "Microservices communication" not in memories_zero
    # If "No memories found" message, that's what we want - the filter worked correctly
