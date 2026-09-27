"""Tests for harvest pipeline v2 improvements (TDD — written before implementation)."""

import inspect
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

try:
    from mcp_memory_service.storage.mixins.embeddings import SENTENCE_TRANSFORMERS_AVAILABLE
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@pytest.fixture(autouse=True)
def set_harvest_locale(monkeypatch):
    """Tests expect pt_BR keywords in meta/temporal filter."""
    monkeypatch.setenv("HARVEST_LOCALE", "pt_BR")


class TestMetaFilterExpanded:
    """Passo 1: keywords expandidas no filtro pós-LLM."""

    def _get_harvester(self):
        from mcp_memory_service.harvest.harvester import SessionHarvester
        return SessionHarvester(Path("/tmp/fake"))

    def test_rejects_prompt_discussion(self):
        h = self._get_harvester()
        assert h._is_meta_or_temporal("O prompt mais rigoroso vai filtrar nas próximas colheitas")

    def test_rejects_threshold_jaccard(self):
        h = self._get_harvester()
        assert h._is_meta_or_temporal("É necessário ajustar o threshold de similaridade Jaccard")

    def test_rejects_harvest_mention(self):
        h = self._get_harvester()
        assert h._is_meta_or_temporal("Rodar harvest em sessões de trabalho real para obter insights")

    def test_rejects_bootstrap_mention(self):
        h = self._get_harvester()
        assert h._is_meta_or_temporal("O bootstrap profile vai melhorar organicamente")

    def test_rejects_generic_importance(self):
        h = self._get_harvester()
        assert h._is_meta_or_temporal("É importante documentar tudo para evitar perda de conhecimento")

    def test_accepts_real_convention(self):
        h = self._get_harvester()
        assert not h._is_meta_or_temporal("Nunca usar strReplace em MEMORY.md — usar insert append-only")

    def test_accepts_real_bug(self):
        h = self._get_harvester()
        assert not h._is_meta_or_temporal("useLocation ausente faz o código acessar window.location.state que é undefined")

    def test_accepts_real_decision(self):
        h = self._get_harvester()
        assert not h._is_meta_or_temporal("Utilizar EnvironmentFile= com .env sincronizado para cada .service local")


class TestContextAccumulated:
    """Passo 2: rewriter recebe contexto de candidatos já aceitos."""

    def _get_harvester(self):
        from mcp_memory_service.harvest.harvester import SessionHarvester
        return SessionHarvester(Path("/tmp/fake"))

    def test_harvest_passes_context_to_rewriter(self):
        """O rewriter deve receber lista de insights já aceitos."""
        h = self._get_harvester()
        # O método rewrite_sync deve aceitar parâmetro `already_extracted`
        from mcp_memory_service.harvest.rewriter import HarvestRewriter
        rewriter = HarvestRewriter()
        # Verificar que o método aceita o parâmetro
        sig = inspect.signature(rewriter.rewrite)
        assert "already_extracted" in sig.parameters, \
            "rewrite() deve aceitar parâmetro 'already_extracted'"

    def test_rewrite_sync_accepts_already_extracted(self):
        """rewrite_sync também deve aceitar already_extracted."""
        from mcp_memory_service.harvest.rewriter import HarvestRewriter
        rewriter = HarvestRewriter()
        sig = inspect.signature(rewriter.rewrite_sync)
        assert "already_extracted" in sig.parameters, \
            "rewrite_sync() deve aceitar parâmetro 'already_extracted'"


@pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="Requires real embedding model for semantic consolidation"
)
class TestBugConsolidation:
    """Passo 3: bugs similares na mesma sessão são consolidados."""

    def _get_harvester(self):
        from mcp_memory_service.harvest.harvester import SessionHarvester
        return SessionHarvester(Path("/tmp/fake"))

    def test_consolidate_similar_exists(self):
        """Método _consolidate_similar deve existir."""
        h = self._get_harvester()
        assert hasattr(h, "_consolidate_similar"), \
            "SessionHarvester deve ter método _consolidate_similar"

    def test_three_similar_bugs_become_one(self):
        """3 bugs sobre o mesmo tema → consolidados (menos que 3)."""
        from mcp_memory_service.harvest.models import HarvestCandidate
        h = self._get_harvester()
        candidates = [
            HarvestCandidate(content="O curl com SSE não grava em arquivo corretamente", memory_type="bug", confidence=0.85, tags=["harvest:bug"]),
            HarvestCandidate(content="O curl em background com SSE pode causar problemas de timeout", memory_type="bug", confidence=0.85, tags=["harvest:bug"]),
            HarvestCandidate(content="O curl com SSE não grava em arquivo corretamente, mas o problema pode ser contornado chamando o serviço diretamente via Python", memory_type="bug", confidence=0.85, tags=["harvest:bug"]),
        ]
        result = h._consolidate_similar(candidates)
        assert len(result) < len(candidates), f"Expected consolidation, got {len(result)}"
        # Mantém o mais completo (maior content)
        assert any("Python" in c.content for c in result)

    def test_different_bugs_kept_separate(self):
        """Bugs sobre temas diferentes são mantidos separados."""
        from mcp_memory_service.harvest.models import HarvestCandidate
        h = self._get_harvester()
        candidates = [
            HarvestCandidate(content="useLocation ausente faz o código acessar window.location.state", memory_type="bug", confidence=0.85, tags=["harvest:bug"]),
            HarvestCandidate(content=".gitignore pode excluir .env.production se não configurado", memory_type="bug", confidence=0.85, tags=["harvest:bug"]),
        ]
        result = h._consolidate_similar(candidates)
        assert len(result) == 2


class TestMultiProvider:
    """Passo 5: Multi-provider LLM com fallback."""

    def test_rewriter_has_providers_list(self):
        """Rewriter deve ter lista de providers configurável."""
        os.environ["HARVEST_LLM_PROVIDERS"] = "groq,ollama"
        os.environ["HARVEST_LLM_OLLAMA_BASE_URL"] = "http://localhost:11434/v1"
        os.environ["HARVEST_LLM_OLLAMA_MODEL"] = "qwen2.5:14b"
        from mcp_memory_service.harvest.rewriter import HarvestRewriter
        rewriter = HarvestRewriter()
        assert hasattr(rewriter, "_providers"), "Rewriter deve ter _providers"
        assert len(rewriter._providers) >= 1, "Deve ter pelo menos 1 provider"

    def test_call_openai_compatible_method_exists(self):
        """Método _call_openai_compatible deve existir."""
        from mcp_memory_service.harvest.rewriter import HarvestRewriter
        rewriter = HarvestRewriter()
        assert hasattr(rewriter, "_call_openai_compatible"), \
            "Rewriter deve ter _call_openai_compatible"

    @pytest.mark.asyncio
    async def test_fallback_on_rate_limit(self):
        """Se provider 1 retorna 429, tenta provider 2."""
        from mcp_memory_service.harvest.rewriter import HarvestRewriter, LLMProvider
        os.environ["HARVEST_LLM_PROVIDERS"] = "groq,ollama"
        os.environ["HARVEST_LLM_GROQ_BASE_URL"] = "https://api.groq.com/openai/v1"
        os.environ["HARVEST_LLM_GROQ_MODEL"] = "llama-3.3-70b-versatile"
        os.environ["HARVEST_LLM_GROQ_API_KEY"] = "fake"
        os.environ["HARVEST_LLM_OLLAMA_BASE_URL"] = "http://localhost:11434/v1"
        os.environ["HARVEST_LLM_OLLAMA_MODEL"] = "qwen2.5:14b"
        rewriter = HarvestRewriter()
        # Mock: first provider raises rate limit, second succeeds
        call_log = []

        async def mock_call(base_url, model, api_key, prompt, timeout):
            call_log.append(base_url)
            if "groq" in base_url:
                raise Exception("Rate limit exceeded 429")
            return "convention: Test insight"

        rewriter._call_openai_compatible = mock_call
        result = await rewriter._call_llm("test prompt", timeout=HarvestRewriter._CALL_TIMEOUT_SINGLE)
        assert len(call_log) == 2, f"Should try 2 providers, tried {len(call_log)}"
        assert "11434" in call_log[1]
