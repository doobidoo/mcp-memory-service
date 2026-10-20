"""G3 RED — autoria por header ``X-Agent-ID`` no transporte streamable-http.

Contrato: ``docs/rfc/planned/spec-agent-id-streamable.md`` (R1-R5, CA1-CA5).

Defeito sob teste (spec §1): o produto lê ``X-Agent-ID`` **apenas** no caminho
da Web API (``src/mcp_memory_service/web/api/mcp.py``). O serviço roda em
streamable-http (``MCP_STREAMABLE_HTTP_MODE=1`` →
``ServerRunManager.run_streamable_http`` →
``StreamableHTTPSessionManager(app=server_instance, event_store=None,
stateless=True)``), e esse caminho **não inspeciona headers**. Logo, uma
chamada ``memory_store`` sem o argumento ``agent_id`` grava a autoria a partir
do env ``MCP_AGENT_ID`` (``zero``), ignorando o header do cliente.

Estes testes sobem o **serviço real** (subprocess, transporte streamable-http,
porta efêmera, backend sqlite_vec) e conectam **clientes MCP reais**
(``streamablehttp_client``) com headers ``X-Agent-ID`` distintos. NÃO há mock do
transporte — o defeito vive exatamente na camada que um mock substituiria
(spec §5: "Teste unitário com mock não serve").

Autoria resolvida: não é autenticação; o header apenas ATRIBUI o autor
(spec §3.4). A leitura de volta é feita abrindo o mesmo banco sqlite_vec com a
classe de storage real (``get_all_memories``) — a mesma API que a suíte verde
``tests/test_agent_id.py`` usa —, evitando qualquer dependência de ranqueamento
de busca para que a mensagem de falha nomeie o defeito (``agent_id`` gravado
como ``zero``), não um erro de preparação.

Estado esperado em ``main`` (antes da correção): RED.
- CA1 / CA4: ``agent_id`` gravado = ``zero`` (env), esperado o valor do header.
- CA2: concorrência — todos os registros carimbados ``zero``, 0 com ``alfa``/``beta``.
- CA3: guarda de regressão — sem header, ``MCP_AGENT_ID`` continua valendo (GREEN
  hoje e depois da correção; falha só se a correção quebrar o fallback).
"""

import asyncio
import os
import socket
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio

# Transporte MCP real: cliente streamable-http + sessão. Se o SDK não estiver
# disponível, o módulo falha na coleta (RED válido por ambiente ausente, não um
# except que mascara a ausência). O serviço depende deste SDK em produção.
from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamablehttp_client  # noqa: E402

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage  # noqa: E402

REPO_SRC = str(Path(__file__).resolve().parents[2] / "src")
ENV_AGENT = "zero"  # MCP_AGENT_ID do serviço (espelha a prova medida na spec §1)


def _free_port() -> int:
    """Reserva uma porta efêmera livre em loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_health(url: str, timeout: float = 30.0) -> None:
    """Aguarda o endpoint /health do serviço responder."""
    import httpx

    deadline = time.time() + timeout
    last_err: Exception | None = None
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            try:
                r = await client.get(url, timeout=2.0)
                if r.status_code == 200:
                    return
            except Exception as e:  # noqa: BLE001 — probe de readiness, não é o caminho sob teste
                last_err = e
            await asyncio.sleep(0.25)
    raise RuntimeError(f"serviço streamable-http não respondeu em {url}: {last_err!r}")


class StreamableService:
    """Serviço mcp-memory real rodando em streamable-http (subprocess)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.port = _free_port()
        self.proc: asyncio.subprocess.Process | None = None

    @property
    def mcp_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    @property
    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/health"

    async def start(self) -> None:
        env = dict(os.environ)
        env.update(
            {
                "PYTHONPATH": REPO_SRC,
                "MCP_STREAMABLE_HTTP_MODE": "1",  # transporte real sob teste
                "MCP_SSE_HOST": "127.0.0.1",      # loopback → não dispara o guard de bind autenticado
                "MCP_SSE_PORT": str(self.port),
                "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
                "MCP_MEMORY_SQLITE_PATH": self.db_path,
                "MCP_AGENT_ID": ENV_AGENT,         # env que o defeito faz vencer o header
                "MCP_MEMORY_USE_ONNX": "0",
                "MCP_MEMORY_EMBEDDING_MODEL": "",  # força hash embeddings (sem download de modelo)
                "MCP_OAUTH_ENABLED": "false",
                "MCP_ALLOW_ANONYMOUS_ACCESS": "true",
            },
        )
        env.pop("MCP_API_KEY", None)  # sem API key → /mcp anônimo em loopback
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "mcp_memory_service.server_impl",
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            await _wait_health(self.health_url)
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()

    async def read_stored_agent_ids(self) -> dict[str, str | None]:
        """Lê de volta {content: agent_id} do mesmo banco via storage real."""
        storage = SqliteVecMemoryStorage(self.db_path)
        await storage.initialize()
        try:
            memories = await storage.get_all_memories()
            return {m.content: m.agent_id for m in memories}
        finally:
            await storage.close()


async def _store_via_header(mcp_url: str, agent_header: str, content: str) -> None:
    """Chama memory_store via MCP streamable real, com X-Agent-ID e SEM argumento agent_id."""
    headers = {"X-Agent-ID": agent_header}
    async with streamablehttp_client(mcp_url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool(
                "memory_store",
                {"content": content, "metadata": {"tags": "rok,agent-id-red"}},
            )


@pytest.fixture
async def service(tmp_path):
    db_path = str(tmp_path / "streamable_agent_id.db")
    svc = StreamableService(db_path)
    await svc.start()
    try:
        yield svc
    finally:
        await svc.stop()


# --------------------------------------------------------------------------- #
# CA1 — header sem argumento → autor correto em 3/3 tentativas
# --------------------------------------------------------------------------- #

async def test_ca1_header_sets_author_three_of_three(service):
    """CA1: 3 stores com X-Agent-ID=alfa, sem argumento agent_id → 3/3 com agent_id='alfa'.

    RED em main: o transporte streamable ignora o header; os 3 registros ficam
    com agent_id='zero' (MCP_AGENT_ID). A asserção pede o alvo (header vence o
    env) e portanto falha hoje e passa após a correção.
    """
    contents = [f"ca1 header author attempt {i} :: unique-{i}" for i in range(3)]
    for c in contents:
        await _store_via_header(service.mcp_url, "alfa", c)

    stored = await service.read_stored_agent_ids()
    got = [stored.get(c) for c in contents]
    assert got == ["alfa", "alfa", "alfa"], (
        f"X-Agent-ID ignorado no transporte streamable: agent_id gravado={got}, "
        f"esperado ['alfa','alfa','alfa'] (defeito: header perde para MCP_AGENT_ID='{ENV_AGENT}')"
    )


# --------------------------------------------------------------------------- #
# CA2 — 2 clientes concorrentes, 20 chamadas intercaladas → 0 cruzamento
# --------------------------------------------------------------------------- #

async def test_ca2_concurrent_clients_no_crossing(service):
    """CA2: alfa e beta gravam intercalados (20 chamadas); cada registro carrega
    o autor do SEU cliente — 0 cruzamento e 0 gravado como env.

    RED em main: sem leitura de header, TODOS os 20 ficam agent_id='zero'
    (nenhum alfa/beta). A asserção exige 10 alfa + 10 beta.
    """
    alfa_contents = [f"ca2 alfa msg {i} :: uniq-a-{i}" for i in range(10)]
    beta_contents = [f"ca2 beta msg {i} :: uniq-b-{i}" for i in range(10)]

    # Intercala as 20 chamadas (alfa, beta, alfa, beta, ...) concorrentemente.
    tasks = []
    for i in range(10):
        tasks.append(_store_via_header(service.mcp_url, "alfa", alfa_contents[i]))
        tasks.append(_store_via_header(service.mcp_url, "beta", beta_contents[i]))
    await asyncio.gather(*tasks)

    stored = await service.read_stored_agent_ids()
    alfa_got = [stored.get(c) for c in alfa_contents]
    beta_got = [stored.get(c) for c in beta_contents]

    crossed = [c for c in alfa_contents if stored.get(c) != "alfa"] + [
        c for c in beta_contents if stored.get(c) != "beta"
    ]
    assert not crossed, (
        "cruzamento/perda de autoria no transporte streamable concorrente: "
        f"alfa={alfa_got} beta={beta_got}; "
        f"esperado 10×'alfa' + 10×'beta' (defeito: todos caem para MCP_AGENT_ID='{ENV_AGENT}')"
    )


# --------------------------------------------------------------------------- #
# CA4 — precedência: argumento explícito > header
# --------------------------------------------------------------------------- #

async def test_ca4_explicit_argument_beats_header(service):
    """CA4: com header X-Agent-ID=beta E argumento agent_id='alfa', vence o argumento.

    Após a correção, o argumento explícito tem precedência sobre o header.
    Em main o argumento JÁ é respeitado (caminho da resolução em
    memory_service), então esta asserção serve de guarda para a correção NÃO
    inverter a precedência. (Pode já estar GREEN em main — é o par de CA1/CA2.)
    """
    content = "ca4 explicit arg beats header :: uniq-ca4"
    headers = {"X-Agent-ID": "beta"}
    async with streamablehttp_client(service.mcp_url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool(
                "memory_store",
                {"content": content, "agent_id": "alfa", "metadata": {"tags": "rok"}},
            )

    stored = await service.read_stored_agent_ids()
    assert stored.get(content) == "alfa", (
        f"precedência violada: agent_id gravado={stored.get(content)!r}, "
        "esperado 'alfa' (argumento explícito > header)"
    )


# --------------------------------------------------------------------------- #
# CA3 — sem header → MCP_AGENT_ID continua valendo (guarda de regressão)
# --------------------------------------------------------------------------- #

async def test_ca3_no_header_falls_back_to_env(service):
    """CA3: SEM header e sem argumento → agent_id = MCP_AGENT_ID ('zero').

    GREEN hoje (comportamento atual) e deve continuar GREEN após a correção:
    a correção não pode quebrar o fallback para o env. R4/CA3 da spec.
    """
    content = "ca3 no header falls back to env :: uniq-ca3"
    async with streamablehttp_client(service.mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool(
                "memory_store",
                {"content": content, "metadata": {"tags": "rok"}},
            )

    stored = await service.read_stored_agent_ids()
    assert stored.get(content) == ENV_AGENT, (
        f"regressão no fallback de env: agent_id gravado={stored.get(content)!r}, "
        f"esperado '{ENV_AGENT}' (MCP_AGENT_ID sem header)"
    )


# --------------------------------------------------------------------------- #
# CA4 estendido — evolve honra o header (R2: precedência nos 3 caminhos)
# --------------------------------------------------------------------------- #

async def test_ca4_evolve_header_sets_author(service):
    """CA4 estendido / R2: evoluir uma memória com X-Agent-ID e SEM argumento
    agent_id → a NOVA versão carrega o autor do header.

    O caminho de escrita ``evolve_memory`` (harvest) ignorava o header em main
    (usava ``env or metadata``). A correção roteia os TRÊS caminhos de escrita
    pelo mesmo ``resolve_agent_id``; aqui provamos o caminho de evolve.

    Transporte NÃO mockado: o serviço streamable real está no ar (prova viva de
    CA1/CA2); a semente é gravada pelo transporte real com o header. A evolução
    usa o MESMO mecanismo que o middleware ASGI usa para propagar o header — o
    ContextVar real (``set_current_agent_id``), que é exatamente a função que
    ``startup_orchestrator`` chama por requisição. Lê-se de volta pelo storage
    real (``get_all_memories``), como o resto desta suíte.
    """
    from mcp_memory_service.services.memory_service import MemoryService
    from mcp_memory_service.utils.agent_context import (
        reset_current_agent_id,
        set_current_agent_id,
    )

    seed = "ca4 evolve seed :: uniq-evolve"
    # Semente gravada pelo transporte real, carimbada pelo header 'alfa'.
    await _store_via_header(service.mcp_url, "alfa", seed)

    storage = SqliteVecMemoryStorage(service.db_path)
    await storage.initialize()
    try:
        before = await storage.get_all_memories()
        seed_hash = next(m.content_hash for m in before if m.content == seed)

        svc = MemoryService(storage)
        # O middleware liga o header num ContextVar por requisição; reproduzimos
        # EXATAMENTE isso (sem mock) para provar que evolve_memory o consulta.
        token = set_current_agent_id("gamma")
        try:
            ok, _msg, new_hash = await svc.evolve_memory(
                seed_hash,
                "ca4 evolve seed :: uniq-evolve (evolved v2)",
                reason="test evolve header author",
            )
        finally:
            reset_current_agent_id(token)

        assert ok and new_hash and new_hash != seed_hash
        evolved = await storage.get_by_hash(new_hash)
        assert evolved is not None
        assert evolved.agent_id == "gamma", (
            f"evolve ignorou o header: agent_id da nova versão={evolved.agent_id!r}, "
            "esperado 'gamma' (header > env > metadata na precedência única)"
        )
    finally:
        await storage.close()


# --------------------------------------------------------------------------- #
# R5 — log DEBUG com a ORIGEM da resolução (sem o valor do header em produção)
# --------------------------------------------------------------------------- #

async def test_r5_debug_log_reports_origin(caplog):
    """R5: uma linha DEBUG por chamada nomeando a origem da resolução.

    origem='header' quando veio do ContextVar; origem='env' quando não veio e
    o MCP_AGENT_ID está setado. O valor é mascarado (nunca o header cru).
    Não depende do transporte: exercita o ponto único ``resolve_agent_id`` +
    ``log_agent_id_resolution`` que os três caminhos de escrita chamam.
    """
    import logging

    from mcp_memory_service.utils import agent_context
    from mcp_memory_service.utils.agent_context import (
        log_agent_id_resolution,
        reset_current_agent_id,
        resolve_agent_id,
        set_current_agent_id,
    )

    logger_name = agent_context.logger.name

    # Caso 1: header presente → origin=header, valor mascarado (não 'tpolsecret').
    token = set_current_agent_id("tpolsecret")
    try:
        with caplog.at_level(logging.DEBUG, logger=logger_name):
            res = resolve_agent_id(explicit=None, metadata=None, default=None)
            log_agent_id_resolution(res, context="store_memory")
    finally:
        reset_current_agent_id(token)

    assert res.origin == "header"
    header_lines = [r.getMessage() for r in caplog.records if r.name == logger_name]
    assert any("origin=header" in m for m in header_lines), header_lines
    # Produção-safe: o valor cru nunca aparece no log.
    assert all("tpolsecret" not in m for m in header_lines), header_lines

    caplog.clear()

    # Caso 2: sem header, MCP_AGENT_ID setado → origin=env.
    import os

    prev = os.environ.get("MCP_AGENT_ID")
    os.environ["MCP_AGENT_ID"] = "envbot"
    try:
        with caplog.at_level(logging.DEBUG, logger=logger_name):
            res_env = resolve_agent_id(explicit=None, metadata=None, default=None)
            log_agent_id_resolution(res_env, context="evolve_memory")
    finally:
        if prev is None:
            os.environ.pop("MCP_AGENT_ID", None)
        else:
            os.environ["MCP_AGENT_ID"] = prev

    assert res_env.origin == "env"
    env_lines = [r.getMessage() for r in caplog.records if r.name == logger_name]
    assert any("origin=env" in m for m in env_lines), env_lines
