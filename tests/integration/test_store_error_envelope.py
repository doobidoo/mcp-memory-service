"""Envelope de erro do MCP: gravação recusada tem de sair com ``isError=True``.

Defeito (#1601): ``handle_store_memory`` devolvia ``TextContent`` com o texto
"Error storing memory: ..." dentro de um resultado bem-sucedido. O ``isError``
do envelope ficava ``False``, então automação (colheita, importação, sondas de
teste) contava como gravado algo que nunca foi para o banco.

O padrão do repo já está documentado no dispatch de ``server_impl``: levantar
exceção faz o SDK responder ``isError=True``.
"""

import uuid

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from tests.integration.test_agent_id_streamable_header import StreamableService


@pytest.fixture
async def service(tmp_path):
    svc = StreamableService(str(tmp_path / "store_error.db"))
    await svc.start()
    try:
        yield svc
    finally:
        await svc.stop()


async def _store(mcp_url: str, content: str):
    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool("memory_store", {"content": content})


@pytest.mark.asyncio
async def test_refused_write_reports_iserror(service):
    """Segunda gravação do mesmo conteúdo é recusada e o envelope diz isError."""
    content = f"envelope probe {uuid.uuid4()}"

    first = await _store(service.mcp_url, content)
    assert not first.isError, "a primeira gravação deveria ser aceita"

    second = await _store(service.mcp_url, content)
    assert second.isError is True, (
        "gravação recusada deve sair com isError=True; "
        f"recebido isError={second.isError!r}"
    )
    text = " ".join(getattr(c, "text", "") for c in second.content)
    assert "Error" in text or "Duplicate" in text, "a mensagem do erro deve chegar ao cliente"


@pytest.mark.asyncio
async def test_accepted_write_keeps_iserror_false(service):
    """Guarda: gravação aceita continua com isError=False."""
    result = await _store(service.mcp_url, f"envelope ok {uuid.uuid4()}")
    assert not result.isError
