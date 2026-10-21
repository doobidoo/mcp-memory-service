"""Orçamento de tempo no NLI on-store (issue #1630).

Medido em 21/out: com `MCP_NLI_ON_STORE=true` uma gravação levava ~27 s porque o
loop comparava até 20 crenças, uma chamada de LLM por crença, sem teto de tempo,
sem teto configurável e sem uma linha de log.

Estes testes falham no código anterior e passam com o orçamento:

- o tempo total do check respeita `MCP_NLI_ON_STORE_BUDGET_S` (default 1 s);
- o número de crenças comparadas respeita `MCP_QUARANTINE_MAX_BELIEFS`;
- estourar o orçamento não levanta exceção (fail-open: a gravação continua);
- uma linha INFO registra duração, crenças comparadas e motivo da parada;
- uma contradição encontrada dentro do orçamento continua quarentenando.
"""

import asyncio
import time

import pytest

from mcp_memory_service.consolidation import quarantine as quarantine_mod
from mcp_memory_service.consolidation.quarantine import check_beliefs_on_store
from mcp_memory_service.reasoning.nli import NLIResult


class _FakeBeliefs:
    def __init__(self, beliefs):
        self._b = beliefs

    async def get_beliefs(self, status="active", min_confidence=0.35):
        return [b for b in self._b if b.get("status", "active") == status]

    async def challenge_belief(self, belief_hash):
        return {"belief_hash": belief_hash, "new_status": "disputed"}


class _FakeStorage:
    def __init__(self):
        self.updates = []

    async def update_memory_metadata(self, content_hash, updates, preserve_timestamps=True):
        self.updates.append((content_hash, updates))
        return (True, "ok")

    async def search_by_tag(self, tags, time_start=None):
        return []


class _CountingClassifier:
    """Classificador falso que conta chamadas e simula a latência do LLM."""

    backend = "cascade"

    def __init__(self, delay=0.2, label="neutral", confidence=0.3):
        self.delay = delay
        self.label = label
        self.confidence = confidence
        self.calls = 0

    async def classify(self, premise, hypothesis):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return NLIResult(label=self.label, confidence=self.confidence)

    def max_achievable_confidence(self):
        return 0.95


def _beliefs(n):
    return [
        {
            "belief_hash": f"b{i}",
            "content": f"crença número {i}",
            "confidence": 0.9,
            "status": "active",
        }
        for i in range(n)
    ]


def _patch_classifier(classifier):
    from unittest.mock import patch

    return patch("mcp_memory_service.reasoning.nli.NLIClassifier", return_value=classifier)


class TestBudget:
    @pytest.mark.asyncio
    async def test_total_time_respects_default_budget(self, monkeypatch):
        monkeypatch.delenv("MCP_NLI_ON_STORE_BUDGET_S", raising=False)
        monkeypatch.delenv("MCP_QUARANTINE_MAX_BELIEFS", raising=False)
        clf = _CountingClassifier(delay=0.2)

        started = time.monotonic()
        with _patch_classifier(clf):
            result = await check_beliefs_on_store(
                _FakeStorage(), _FakeBeliefs(_beliefs(20)), "texto novo", "h1"
            )
        elapsed = time.monotonic() - started

        assert result is None
        assert elapsed < 2.8, f"check levou {elapsed:.2f}s; o default é 2 s de orçamento"
        assert clf.calls < 20, f"comparou {clf.calls} crenças; deveria parar pelo orçamento"
        assert clf.calls >= 1, "com 2 s de orçamento ao menos uma comparação precisa acontecer"

    @pytest.mark.asyncio
    async def test_budget_zero_compares_nothing(self, monkeypatch):
        monkeypatch.setenv("MCP_NLI_ON_STORE_BUDGET_S", "0")
        clf = _CountingClassifier(delay=0.2)

        started = time.monotonic()
        with _patch_classifier(clf):
            result = await check_beliefs_on_store(
                _FakeStorage(), _FakeBeliefs(_beliefs(5)), "texto novo", "h1"
            )
        elapsed = time.monotonic() - started

        assert result is None
        assert clf.calls == 0
        assert elapsed < 0.3

    @pytest.mark.asyncio
    async def test_max_beliefs_is_configurable(self, monkeypatch):
        monkeypatch.setenv("MCP_NLI_ON_STORE_BUDGET_S", "30")
        monkeypatch.setenv("MCP_QUARANTINE_MAX_BELIEFS", "3")
        clf = _CountingClassifier(delay=0.01)

        with _patch_classifier(clf):
            await check_beliefs_on_store(
                _FakeStorage(), _FakeBeliefs(_beliefs(20)), "texto novo", "h1"
            )

        assert clf.calls == 3

    @pytest.mark.asyncio
    async def test_timeout_does_not_raise(self, monkeypatch):
        """Classificador mais lento que o orçamento: a gravação não pode quebrar."""
        monkeypatch.setenv("MCP_NLI_ON_STORE_BUDGET_S", "0.2")
        clf = _CountingClassifier(delay=5.0)

        with _patch_classifier(clf):
            result = await check_beliefs_on_store(
                _FakeStorage(), _FakeBeliefs(_beliefs(3)), "texto novo", "h1"
            )

        assert result is None


class TestObservability:
    @pytest.mark.asyncio
    async def test_one_log_line_with_numbers(self, monkeypatch, caplog):
        monkeypatch.setenv("MCP_NLI_ON_STORE_BUDGET_S", "0.5")
        clf = _CountingClassifier(delay=0.01)

        with caplog.at_level("INFO"):
            with _patch_classifier(clf):
                await check_beliefs_on_store(
                    _FakeStorage(), _FakeBeliefs(_beliefs(10)), "texto novo", "h1"
                )

        linhas = [r for r in caplog.records if "on-store check" in r.getMessage().lower()]
        assert len(linhas) == 1, f"esperava uma linha de log, veio {len(linhas)}"
        texto = linhas[0].getMessage().lower()
        assert "compared" in texto or "comparadas" in texto
        assert any(ch.isdigit() for ch in texto)


class TestBehaviourPreserved:
    @pytest.mark.asyncio
    async def test_contradiction_within_budget_still_quarantines(self, monkeypatch):
        monkeypatch.setenv("MCP_NLI_ON_STORE_BUDGET_S", "5")
        monkeypatch.setenv("MCP_QUARANTINE_NLI_THRESHOLD", "0.5")
        storage = _FakeStorage()
        clf = _CountingClassifier(delay=0.01, label="contradiction", confidence=0.9)

        with _patch_classifier(clf):
            result = await check_beliefs_on_store(
                storage, _FakeBeliefs(_beliefs(5)), "texto contraditório", "h1"
            )

        assert result is not None and result["status"] == "quarantined"
        assert clf.calls == 1
        assert storage.updates, "a memória deveria ter recebido metadata de quarentena"

    @pytest.mark.asyncio
    async def test_no_beliefs_returns_none_without_calling_the_model(self, monkeypatch):
        monkeypatch.delenv("MCP_NLI_ON_STORE_BUDGET_S", raising=False)
        clf = _CountingClassifier(delay=0.01)

        with _patch_classifier(clf):
            result = await check_beliefs_on_store(
                _FakeStorage(), _FakeBeliefs([]), "texto novo", "h1"
            )

        assert result is None
        assert clf.calls == 0


class TestTruncationIsLoud:
    @pytest.mark.asyncio
    async def test_truncated_check_warns_even_with_info_suppressed(self, monkeypatch, caplog):
        """No modo Claude Desktop o INFO é suprimido; o truncamento precisa sair como WARNING."""
        monkeypatch.setenv("MCP_NLI_ON_STORE_BUDGET_S", "0.2")
        clf = _CountingClassifier(delay=0.05)

        with caplog.at_level("WARNING"):
            with _patch_classifier(clf):
                await check_beliefs_on_store(
                    _FakeStorage(), _FakeBeliefs(_beliefs(20)), "texto novo", "h1"
                )

        avisos = [r for r in caplog.records if "truncated" in r.getMessage().lower()]
        assert len(avisos) == 1, "um check truncado tem de aparecer como WARNING"
        assert "MCP_NLI_ON_STORE_BUDGET_S" in avisos[0].getMessage()

    @pytest.mark.asyncio
    async def test_clean_check_does_not_warn(self, monkeypatch, caplog):
        monkeypatch.setenv("MCP_NLI_ON_STORE_BUDGET_S", "30")
        monkeypatch.setenv("MCP_QUARANTINE_MAX_BELIEFS", "2")
        clf = _CountingClassifier(delay=0.01)

        with caplog.at_level("WARNING"):
            with _patch_classifier(clf):
                await check_beliefs_on_store(
                    _FakeStorage(), _FakeBeliefs(_beliefs(2)), "texto novo", "h1"
                )

        assert not [r for r in caplog.records if "truncated" in r.getMessage().lower()]


class TestAsyncMode:
    @pytest.mark.asyncio
    async def test_background_runs_both_checks(self, monkeypatch):
        """MCP_NLI_ON_STORE_ASYNC: as duas verificações rodam fora do caminho da escrita."""
        chamadas = []

        async def fake_detect(storage, memory_hash=None, dry_run=True):
            chamadas.append("detect")
            return {"pairs_detected": 0}

        async def fake_beliefs(*args, **kwargs):
            chamadas.append("beliefs")
            return None

        monkeypatch.setattr("mcp_memory_service.reasoning.nli.detect_contradictions_nli", fake_detect)
        monkeypatch.setattr(quarantine_mod, "check_beliefs_on_store", fake_beliefs)

        quarantine_mod.schedule_on_store_checks(_FakeStorage(), "texto", "hash-async")
        for _ in range(50):
            if len(chamadas) >= 2:
                break
            await asyncio.sleep(0.01)

        assert chamadas == ["detect", "beliefs"]

    @pytest.mark.asyncio
    async def test_background_failure_does_not_escape(self, monkeypatch, caplog):
        """Falha em background vira WARNING; nunca sobe para o caller."""
        chamadas = []

        async def explode(*args, **kwargs):
            raise RuntimeError("provider fora do ar")

        async def fake_beliefs(*args, **kwargs):
            chamadas.append("beliefs")
            return None

        monkeypatch.setattr("mcp_memory_service.reasoning.nli.detect_contradictions_nli", explode)
        monkeypatch.setattr(quarantine_mod, "check_beliefs_on_store", fake_beliefs)

        with caplog.at_level("WARNING"):
            quarantine_mod.schedule_on_store_checks(_FakeStorage(), "texto", "hash-fail")
            for _ in range(50):
                if chamadas:
                    break
                await asyncio.sleep(0.01)

        assert chamadas == ["beliefs"], "a segunda verificação deve rodar mesmo com a primeira falhando"
        assert any("background contradiction check failed" in r.getMessage() for r in caplog.records)
