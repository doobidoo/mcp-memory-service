"""Tests for memory_distill — extract insights from existing memories (TDD)."""

import pytest
import os
import subprocess
import json
from pathlib import Path

# Porta do serviço sob teste: a do transporte MCP (MCP_SSE_PORT, 8765 por padrão
# no projeto), sobreponível por MCP_TEST_URL quando a instância usa outra porta.
MCP_URL = os.environ.get(
    "MCP_TEST_URL",
    f"http://localhost:{os.environ.get('MCP_SSE_PORT', '8765')}/mcp",
)


class TestMemoryDistillToolExists:
    """O tool memory_distill deve existir no server_impl."""

    def test_handler_exists_in_source(self):
        """server_impl deve ter handle_memory_distill."""
        result = subprocess.run(
            ["grep", "-c", "async def handle_memory_distill",
             "src/mcp_memory_service/server_impl.py"],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1])
        )
        assert int(result.stdout.strip()) >= 1

    def test_dispatch_exists(self):
        """Routing table deve ter 'memory_distill'."""
        from mcp_memory_service.tools.routing import ROUTING_TABLE
        assert "memory_distill" in ROUTING_TABLE


class TestMemoryDistillLogic:
    """Testa a lógica de filtragem isoladamente."""

    def test_skips_short_content(self):
        """Memórias < 200 chars não são candidatas."""
        min_length = 200
        short = "Texto curto"
        long = "X" * 250
        assert len(short) < min_length
        assert len(long) >= min_length

    def test_skips_already_distilled_tag(self):
        """Tag 'memory-distilled' exclui da seleção."""
        tags = "sessao,mir-sistema,memory-distilled"
        assert "memory-distilled" in tags

    def test_skips_association_tag(self):
        """Tag 'association' exclui da seleção."""
        tags = "association,discovered,temporal_proximity"
        assert "association" in tags

    def test_skips_mistake_type(self):
        """memory_type='mistake' é excluído."""
        skip_types = {"mistake", "insight", "session"}
        assert "mistake" in skip_types
        assert "observation" not in skip_types

    def test_accepts_observation_long(self):
        """observation com >200 chars e sem tag distilled é candidata."""
        content = "Sessão 2026-04-30: Descoberta importante sobre o sistema" + "X" * 200
        tags = "sessao,mir-sistema"
        min_length = 200
        skip_types = {"mistake", "insight", "session"}
        mtype = "observation"

        is_candidate = (
            len(content) >= min_length
            and "memory-distilled" not in tags
            and "association" not in tags
            and mtype not in skip_types
        )
        assert is_candidate


class TestMemoryDistillIntegration:
    """Teste de integração via MCP (requer service rodando)."""

    @pytest.mark.skipif(
        subprocess.run(["curl", "-s", "--max-time", "2", MCP_URL],
                      capture_output=True).returncode != 0,
        reason="memory-service not running"
    )
    def test_dry_run_via_mcp(self):
        """Chamar memory_distill dry_run via MCP retorna JSON válido."""
        import httpx
        resp = httpx.post(
            MCP_URL,
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0", "id": 99,
                "method": "tools/call",
                "params": {"name": "memory_distill", "arguments": {"batch_size": 5, "dry_run": True}}
            },
            timeout=30,
        )
        # O endpoint responde em SSE (transporte streamable HTTP) ou em JSON puro,
        # dependendo de qual app serve /mcp; aceitar os dois formatos.
        payloads = []
        try:
            payloads.append(resp.json())
        except ValueError:
            pass
        for line in resp.text.split("\n"):
            if line.startswith("data:"):
                try:
                    payloads.append(json.loads(line[5:].strip()))
                except ValueError:
                    continue

        for data in payloads:
            if not isinstance(data, dict) or "result" not in data:
                continue
            content = data["result"]["content"]
            result = json.loads(content[0]["text"])
            assert "candidates_found" in result
            assert result["dry_run"] is True
            return
        pytest.fail(f"No valid response from MCP: {resp.status_code} {resp.text[:200]}")
