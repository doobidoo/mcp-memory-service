"""Autoria em ``memory_store_session`` — o caminho mais usado por clientes que colhem sessão.

Defeito: ``handle_store_session`` chamava ``store_memory`` sem ``agent_id``, então o
``agent_id`` do payload era ignorado e a precedência caía no header ou no
``MCP_AGENT_ID`` do serviço. Medido em produção: 388 memórias colhidas ficaram
todas com o autor do env (``zero``).

Precedência correta (utils/agent_context.py):
    argumento explícito > header X-Agent-ID > MCP_AGENT_ID > metadata
"""

import uuid

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from tests.integration.test_agent_id_streamable_header import ENV_AGENT, StreamableService

SESSION_TOOL = "memory_store_session"


async def _store_session(mcp_url: str, headers: dict[str, str], turn: str, session_id: str, **extra) -> None:
    payload = {"turns": [{"role": "user", "content": turn}], "session_id": session_id, **extra}
    async with streamablehttp_client(mcp_url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool(SESSION_TOOL, payload)


@pytest.fixture
async def service(tmp_path):
    svc = StreamableService(str(tmp_path / "store_session_author.db"))
    await svc.start()
    try:
        yield svc
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_argument_wins_over_header_and_env(service):
    """O ``agent_id`` do payload vence header e env — é o autor que o cliente declarou."""
    turn = f"store session argument author {uuid.uuid4()}"
    await _store_session(
        service.mcp_url, {"X-Agent-ID": "beta"}, turn, "sess-arg", agent_id="alfa"
    )

    stored = await service.read_stored_agent_ids()
    assert stored.get(f"[user] {turn}") == "alfa", (
        f"autor gravado={stored.get(f'[user] {turn}')!r}, esperado 'alfa' "
        f"(argumento vence header 'beta' e MCP_AGENT_ID={ENV_AGENT!r})"
    )


@pytest.mark.asyncio
async def test_header_still_used_without_argument(service):
    """Guarda: sem argumento, o header continua sendo o autor (não regride para o env)."""
    turn = f"store session header author {uuid.uuid4()}"
    await _store_session(service.mcp_url, {"X-Agent-ID": "beta"}, turn, "sess-hdr")

    stored = await service.read_stored_agent_ids()
    assert stored.get(f"[user] {turn}") == "beta", (
        f"autor gravado={stored.get(f'[user] {turn}')!r}, esperado 'beta' "
        f"(sem argumento, o header vence MCP_AGENT_ID={ENV_AGENT!r})"
    )
