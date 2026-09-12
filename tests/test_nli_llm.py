"""Tests for NLIClassifier._llm_classify method."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from mcp_memory_service.reasoning.nli import NLIClassifier, NLIResult


@pytest.fixture
def classifier():
    return NLIClassifier(backend="llm")


@pytest.mark.asyncio
async def test_returns_contradiction_with_high_confidence(classifier):
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="contradiction")
        result = await classifier._llm_classify("A is true", "A is false")
    assert result.label == "contradiction"
    assert result.confidence == 0.9


@pytest.mark.asyncio
async def test_returns_entailment_with_high_confidence(classifier):
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="entailment")
        result = await classifier._llm_classify("sky is blue", "sky is blue")
    assert result.label == "entailment"
    assert result.confidence == 0.9


@pytest.mark.asyncio
async def test_returns_neutral_with_low_confidence(classifier):
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="neutral")
        result = await classifier._llm_classify("cats are nice", "weather is warm")
    assert result.label == "neutral"
    assert result.confidence == 0.3


@pytest.mark.asyncio
async def test_falls_back_to_heuristic_on_empty_response(classifier):
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="")
        result = await classifier._llm_classify("redis enabled", "redis disabled")
    # Heuristic detects enabled/disabled antonym pair
    assert result.label == "contradiction"
    assert result.confidence <= 0.6


@pytest.mark.asyncio
async def test_falls_back_to_heuristic_on_exception(classifier):
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        result = await classifier._llm_classify("feature enabled", "feature disabled")
    assert result.label == "contradiction"
    assert result.confidence <= 0.6


@pytest.mark.asyncio
async def test_garbled_output_falls_back_to_heuristic(classifier):
    """Unknown/garbled label must fall back to the heuristic classifier,
    not be returned as a (wrong) answer. Regression for the review on #1215."""
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="xyzzy blorp 42 random garbage")
        # heuristic on unrelated texts → neutral; the point is it went through
        # _heuristic_classify, not that the label happens to be neutral.
        with patch.object(classifier, "_heuristic_classify",
                          return_value=NLIResult(label="entailment", confidence=0.42)) as heur:
            result = await classifier._llm_classify("some premise", "some hypothesis")
    heur.assert_called_once()
    assert result.label == "entailment"
    assert result.confidence == 0.42


@pytest.mark.asyncio
async def test_label_with_trailing_punctuation_is_parsed(classifier):
    """'Contradiction.' (with a period) must parse as contradiction, not neutral.
    Regression for the review on #1215 (one-word parser dropped punctuation)."""
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="Contradiction.")
        result = await classifier._llm_classify("feature enabled", "feature disabled")
    assert result.label == "contradiction"
    assert result.confidence == 0.9


@pytest.mark.asyncio
async def test_hedged_multilabel_falls_back_to_heuristic(classifier):
    """A hedged answer mentioning two distinct labels is unparseable and must
    fall back to the heuristic — NOT be acted on at 0.9 confidence.
    Regression for the second-round review on #1215."""
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="contradiction, but really neutral")
        with patch.object(classifier, "_heuristic_classify",
                          return_value=NLIResult(label="neutral", confidence=0.3)) as heur:
            result = await classifier._llm_classify("some premise", "some hypothesis")
    heur.assert_called_once()
    assert result.confidence <= 0.6


@pytest.mark.asyncio
async def test_negated_contradiction_falls_back_to_heuristic(classifier):
    """'there is no contradiction' must NOT parse as contradiction — the
    first-token anchor sends it to the heuristic. Regression for #1215
    (a word-boundary match would wrongly read it as contradiction@0.9)."""
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="there is no contradiction")
        with patch.object(classifier, "_heuristic_classify",
                          return_value=NLIResult(label="neutral", confidence=0.3)) as heur:
            result = await classifier._llm_classify("some premise", "some hypothesis")
    heur.assert_called_once()
    assert result.confidence <= 0.6


@pytest.mark.asyncio
async def test_classification_prefix_is_parsed(classifier):
    """'Classification: contradiction' must parse as contradiction — the
    optional leading 'classification:' prefix is stripped before anchoring.
    Regression for the second-round review on #1215."""
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="Classification: contradiction")
        result = await classifier._llm_classify("feature enabled", "feature disabled")
    assert result.label == "contradiction"
    assert result.confidence == 0.9


@pytest.mark.asyncio
async def test_env_backend_reaches_pipeline(monkeypatch):
    """INTEGRATION: MCP_NLI_BACKEND=cascade must actually reach the pipeline.
    Regression for the review on #1215 — detect_contradictions_nli hardcoded
    NLIClassifier(backend='heuristic'), so the env switch changed nothing.

    This drives the REAL detect_contradictions_nli (not a locally-built
    classifier), mocking storage+graph up to Stage 3, and asserts the LLM
    backend selected by the env is the one that runs. Without the fix
    (backend='auto' at the call-site) the LLM is never called → red on main."""
    from mcp_memory_service.reasoning import nli as nli_mod

    monkeypatch.setenv("MCP_NLI_ENABLED", "true")
    monkeypatch.setenv("MCP_NLI_BACKEND", "cascade")
    called = {"llm": 0}

    async def fake_call_llm(prompt, timeout=30):
        called["llm"] += 1
        return "contradiction"

    ha, hb = "hashA", "hashB"
    mem_a = MagicMock()
    mem_a.content = "the feature is enabled"
    mem_a.metadata = {"entities": ["feature"]}

    storage = MagicMock()
    storage.get_by_hash = AsyncMock(return_value=mem_a)
    storage.search_memories = AsyncMock(return_value={
        "memories": [{"content_hash": hb, "content": "the feature is disabled",
                      "similarity_score": 0.6}]
    })
    graph = MagicMock()
    graph.find_memories_by_entity = AsyncMock(return_value=[ha, hb])
    graph.store_association = AsyncMock()

    import sys, types  # inline import
    # Stub the graph handler module so the in-function import resolves without
    # triggering the lazy-loaded server package.
    graph_mod = types.ModuleType("mcp_memory_service.server.handlers.graph")
    graph_mod.get_graph_storage = AsyncMock(return_value=graph)
    monkeypatch.setitem(sys.modules, "mcp_memory_service.server.handlers.graph", graph_mod)

    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        MockRewriter.return_value._call_llm = AsyncMock(side_effect=fake_call_llm)
        res = await nli_mod.detect_contradictions_nli(storage, memory_hash=ha, dry_run=True)

    assert res["nli_calls"] >= 1, "pipeline never reached NLI classification"
    assert called["llm"] >= 1, "env backend did not reach the pipeline (LLM never called)"


@pytest.mark.asyncio
async def test_env_backend_reaches_quarantine_call_site(monkeypatch):
    """INTEGRATION: MCP_NLI_BACKEND=cascade must also reach the SECOND call site,
    quarantine.check_beliefs_on_store. Regression for the second-round review on
    #1215 — that call site builds its own NLIClassifier, so it needs its own
    proof that the env switch is honored there too (the nli.py integration test
    does not cover it). Red on main if quarantine.py:62 were hardcoded to
    'heuristic' (LLM never called)."""
    from mcp_memory_service.consolidation import quarantine as q_mod

    monkeypatch.setenv("MCP_NLI_BACKEND", "cascade")
    called = {"llm": 0}

    async def fake_call_llm(prompt, timeout=30):
        called["llm"] += 1
        return "contradiction"

    belief_service = MagicMock()
    belief_service.get_beliefs = AsyncMock(return_value=[
        {"content": "the feature is enabled", "belief_hash": "beliefA"}
    ])
    belief_service.challenge_belief = AsyncMock()

    storage = MagicMock()
    storage.update_memory_metadata = AsyncMock()
    storage.search_by_tag = AsyncMock(return_value=[])

    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        MockRewriter.return_value._call_llm = AsyncMock(side_effect=fake_call_llm)
        res = await q_mod.check_beliefs_on_store(
            storage, belief_service, "the feature is disabled", "hashNew"
        )

    assert called["llm"] >= 1, "env backend did not reach the quarantine call site (LLM never called)"
    assert res is not None and res.get("status") == "quarantined", \
        "contradiction@0.9 from cascade should have quarantined the memory"


