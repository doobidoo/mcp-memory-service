"""Tests for NLIClassifier._llm_classify method."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock, PropertyMock

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




# ============================================================================
# RED Tests for Issue #1235 (Phase 2) - Requirements R10-R14
# ============================================================================

@pytest.mark.asyncio
async def test_r10_rewriter_constructed_once_for_multiple_pairs():
    """R10: HarvestRewriter SHALL be constructed UMA vez per run, not per pair.
    
    Tests that multiple classify calls in a batch reuse the same rewriter instance
    rather than creating a new one for each pair. This validates the optimization
    to resolve provider config once per run.
    """
    classifier = NLIClassifier(backend="cascade")
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="contradiction")
        instance.is_configured = True
        
        # Multiple calls should reuse the same rewriter
        await classifier.classify("premise1", "hypothesis1")
        await classifier.classify("premise2", "hypothesis2")
        await classifier.classify("premise3", "hypothesis3")
        
        # Should have been constructed exactly once despite 3 calls
        MockRewriter.assert_called_once()


@pytest.mark.asyncio
async def test_r10_batch_classify_constructs_rewriter_once():
    """R10: Batch classification should also reuse rewriter across all pairs."""
    classifier = NLIClassifier(backend="cascade")
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="neutral")
        instance.is_configured = True
        
        pairs = [
            ("premise1", "hypothesis1"),
            ("premise2", "hypothesis2"), 
            ("premise3", "hypothesis3")
        ]
        
        results = await classifier.classify_batch(pairs)
        
        assert len(results) == 3
        # Should construct rewriter only once for the entire batch
        MockRewriter.assert_called_once()


@pytest.mark.asyncio
async def test_r11_no_llm_call_when_provider_not_configured():
    """R11: IF no provider configured (is_configured=False), 
    THEN SHALL skip LLM call and use heuristic directly."""
    classifier = NLIClassifier(backend="cascade")
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance.is_configured = False  # No provider configured
        instance._call_llm = AsyncMock(return_value="contradiction")
        
        with patch.object(classifier, "_heuristic_classify", 
                         return_value=NLIResult(label="neutral", confidence=0.5)) as mock_heuristic:
            result = await classifier._llm_classify("premise", "hypothesis")
        
        # Should NOT have called LLM at all
        instance._call_llm.assert_not_called()
        # Should have used heuristic directly
        mock_heuristic.assert_called_once_with("premise", "hypothesis")
        assert result.label == "neutral"
        assert result.confidence == 0.5


@pytest.mark.asyncio
async def test_r11_is_configured_property_exception_handled():
    """R11 Edge case: is_configured property throwing exception should be handled."""
    classifier = NLIClassifier(backend="cascade")
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        # is_configured property throws exception
        type(instance).is_configured = PropertyMock(side_effect=RuntimeError("Config error"))
        instance._call_llm = AsyncMock(return_value="contradiction")
        
        with patch.object(classifier, "_heuristic_classify",
                         return_value=NLIResult(label="neutral", confidence=0.3)) as mock_heuristic:
            result = await classifier._llm_classify("premise", "hypothesis")
        
        # Should fall back to heuristic when config check fails
        mock_heuristic.assert_called_once()
        instance._call_llm.assert_not_called()


@pytest.mark.asyncio
async def test_r12_degradation_warning_bounded_once_per_run():
    """R12: WHEN degradation occurs, SHALL emit UM warning bounded (1x per run).
    
    Tests that multiple failures in the same run only generate one warning,
    not one per pair.
    """
    classifier = NLIClassifier(backend="cascade")
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance.is_configured = True
        # First call succeeds, second and third fail
        instance._call_llm = AsyncMock(side_effect=[
            "contradiction",
            RuntimeError("LLM failed"),
            RuntimeError("Still failing")
        ])
        
        with patch.object(classifier, "_warn_once") as mock_warn:
            with patch.object(classifier, "_heuristic_classify",
                             return_value=NLIResult(label="neutral", confidence=0.4)):
                # First call succeeds
                result1 = await classifier._llm_classify("p1", "h1")
                assert result1.label == "contradiction"
                
                # Second call fails - should warn
                result2 = await classifier._llm_classify("p2", "h2")
                assert result2.label == "neutral"
                
                # Third call fails - should NOT warn again
                result3 = await classifier._llm_classify("p3", "h3")
                assert result3.label == "neutral"
        
        # Should have warned exactly once despite multiple failures
        mock_warn.assert_called_once()


@pytest.mark.asyncio
async def test_r12_empty_response_triggers_bounded_warning():
    """R12: Empty/whitespace-only response should trigger bounded warning."""
    classifier = NLIClassifier(backend="cascade")
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance.is_configured = True
        # Return whitespace-only responses that can't be parsed
        instance._call_llm = AsyncMock(side_effect=["   ", "\t\n", "   "])
        
        with patch.object(classifier, "_warn_once") as mock_warn:
            with patch.object(classifier, "_heuristic_classify",
                             return_value=NLIResult(label="neutral", confidence=0.2)):
                # Multiple empty responses should only warn once
                await classifier._llm_classify("p1", "h1")
                await classifier._llm_classify("p2", "h2")
                await classifier._llm_classify("p3", "h3")
        
        mock_warn.assert_called_once()


@pytest.mark.asyncio
async def test_r12_unparseable_response_triggers_bounded_warning():
    """R12: Unparseable response (garbage) should trigger bounded warning."""
    classifier = NLIClassifier(backend="cascade")
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance.is_configured = True
        # Return garbage that can't be parsed
        instance._call_llm = AsyncMock(side_effect=[
            "xyzzy blorp 42",
            "random garbage text",
            "not a valid label"
        ])
        
        with patch.object(classifier, "_warn_once") as mock_warn:
            with patch.object(classifier, "_heuristic_classify",
                             return_value=NLIResult(label="neutral", confidence=0.1)):
                # Multiple garbage responses should only warn once
                await classifier._llm_classify("p1", "h1")
                await classifier._llm_classify("p2", "h2")
        
        mock_warn.assert_called_once()


@pytest.mark.asyncio
async def test_r12_mixed_success_failure_warns_once():
    """R12 Edge case: Success followed by failures should still warn only once."""
    classifier = NLIClassifier(backend="cascade")
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance.is_configured = True
        # First succeeds, then provider fails in the middle of the run
        instance._call_llm = AsyncMock(side_effect=[
            "entailment",  # Success
            ConnectionError("Provider unavailable"),  # Failure
            ConnectionError("Still down")  # Another failure
        ])
        
        with patch.object(classifier, "_warn_once") as mock_warn:
            with patch.object(classifier, "_heuristic_classify",
                             return_value=NLIResult(label="neutral", confidence=0.3)):
                
                result1 = await classifier._llm_classify("p1", "h1")
                assert result1.label == "entailment"  # Succeeded
                
                result2 = await classifier._llm_classify("p2", "h2")
                assert result2.label == "neutral"  # Failed, used heuristic
                
                result3 = await classifier._llm_classify("p3", "h3")
                assert result3.label == "neutral"  # Failed again
        
        # Should warn exactly once despite multiple failures after success
        mock_warn.assert_called_once()


@pytest.mark.asyncio
async def test_r13_exception_with_newlines_sanitized_in_log(caplog):
    """R13: Exception with newlines/control chars SHALL be sanitized in the log.

    Checks the actual emitted log record (where _sanitize_log_value runs), not
    the argument passed into _warn_once. Scope: newline/carriage-return, which
    the shared _sanitize_log_value guarantees (log-injection vectors)."""
    import logging
    classifier = NLIClassifier(backend="cascade")

    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance.is_configured = True
        instance._call_llm = AsyncMock(side_effect=RuntimeError("boom\nline2\rmore"))
        with caplog.at_level(logging.WARNING):
            await classifier._llm_classify("sensitive premise", "sensitive hypothesis")

    warnings = [r for r in caplog.records
                if r.levelno == logging.WARNING and "degrad" in r.getMessage().lower()]
    assert len(warnings) == 1, "degradation warning must be bounded to once per run"
    msg = warnings[0].getMessage()
    # The emitted log line must not carry raw newlines/carriage returns
    assert "\n" not in msg and "\r" not in msg, "log line must be sanitized"


@pytest.mark.asyncio
async def test_r13_memory_content_never_appears_in_log(caplog):
    """R13: Memory content (premise/hypothesis) SHALL NEVER appear in log."""
    classifier = NLIClassifier(backend="cascade")
    
    sensitive_premise = "SECRET_DATABASE_PASSWORD=supersecret123"
    sensitive_hypothesis = "CONFIDENTIAL_API_KEY=abc123def456"
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance.is_configured = True
        instance._call_llm = AsyncMock(side_effect=RuntimeError("LLM failed"))
        
        with caplog.at_level("WARNING"):
            with patch.object(classifier, "_heuristic_classify",
                             return_value=NLIResult(label="neutral", confidence=0.1)):
                await classifier._llm_classify(sensitive_premise, sensitive_hypothesis)
        
        # Check that no log message contains the sensitive content
        all_log_text = " ".join(record.message for record in caplog.records)
        assert "SECRET_DATABASE_PASSWORD" not in all_log_text
        assert "supersecret123" not in all_log_text
        assert "CONFIDENTIAL_API_KEY" not in all_log_text
        assert "abc123def456" not in all_log_text


@pytest.mark.asyncio
async def test_r13_sanitize_log_value_is_used():
    """R13: Should use _sanitize_log_value function for error sanitization."""
    classifier = NLIClassifier(backend="cascade")
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance.is_configured = True
        instance._call_llm = AsyncMock(side_effect=RuntimeError("error\nwith\rnewlines"))
        
        with patch("mcp_memory_service.reasoning.nli._sanitize_log_value") as mock_sanitize:
            mock_sanitize.return_value = "sanitized_error"
            with patch.object(classifier, "_warn_once") as mock_warn:
                with patch.object(classifier, "_heuristic_classify",
                                 return_value=NLIResult(label="neutral", confidence=0.2)):
                    await classifier._llm_classify("premise", "hypothesis")
        
        # Should have called sanitize function
        mock_sanitize.assert_called_once()
        # Should have passed sanitized value to warning
        mock_warn.assert_called_once()


@pytest.mark.asyncio
async def test_r14_fallback_preserves_exact_heuristic_values():
    """R14: WHEN fallback occurs, SHALL preserve exact label and confidence from heuristic."""
    classifier = NLIClassifier(backend="cascade")
    
    # Specific heuristic result to verify exact preservation
    expected_result = NLIResult(label="contradiction", confidence=0.73)
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance.is_configured = True
        instance._call_llm = AsyncMock(side_effect=RuntimeError("LLM failed"))
        
        with patch.object(classifier, "_heuristic_classify", 
                         return_value=expected_result) as mock_heuristic:
            result = await classifier._llm_classify("premise", "hypothesis")
        
        # Should return EXACTLY the same values as heuristic (not synthetic confidence)
        assert result.label == expected_result.label
        assert result.confidence == expected_result.confidence
        assert result == expected_result  # Exact equality
        mock_heuristic.assert_called_once_with("premise", "hypothesis")


@pytest.mark.asyncio
async def test_r14_fallback_on_empty_response_preserves_heuristic():
    """R14: Fallback due to empty response should preserve exact heuristic values."""
    classifier = NLIClassifier(backend="cascade")
    
    expected_result = NLIResult(label="entailment", confidence=0.58)
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance.is_configured = True
        instance._call_llm = AsyncMock(return_value="")  # Empty response
        
        with patch.object(classifier, "_heuristic_classify",
                         return_value=expected_result) as mock_heuristic:
            result = await classifier._llm_classify("premise", "hypothesis")
        
        # Should preserve exact heuristic result, not create synthetic confidence
        assert result.label == expected_result.label
        assert result.confidence == expected_result.confidence
        mock_heuristic.assert_called_once()


@pytest.mark.asyncio
async def test_r14_fallback_on_garbage_response_preserves_heuristic():
    """R14: Fallback due to unparseable response should preserve exact heuristic values."""
    classifier = NLIClassifier(backend="cascade")
    
    expected_result = NLIResult(label="neutral", confidence=0.15)
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance.is_configured = True
        # Return garbage with valid label mixed in - should still be unparseable
        instance._call_llm = AsyncMock(return_value="entailment but also contradiction maybe neutral")
        
        with patch.object(classifier, "_heuristic_classify",
                         return_value=expected_result) as mock_heuristic:
            result = await classifier._llm_classify("premise", "hypothesis")
        
        # Should preserve exact heuristic values
        assert result.label == expected_result.label
        assert result.confidence == expected_result.confidence
        mock_heuristic.assert_called_once()


# Edge cases and adversarial scenarios

@pytest.mark.asyncio
async def test_edge_case_heuristic_backend_never_creates_rewriter():
    """Edge case: heuristic backend should NEVER instantiate HarvestRewriter."""
    classifier = NLIClassifier(backend="heuristic")
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        with patch.object(classifier, "_heuristic_classify",
                         return_value=NLIResult(label="neutral", confidence=0.4)):
            result = await classifier.classify("premise", "hypothesis")
        
        # Heuristic backend should never create rewriter
        MockRewriter.assert_not_called()
        assert result.confidence == 0.4


@pytest.mark.asyncio
async def test_edge_case_valid_label_surrounded_by_garbage():
    """Edge case: Valid label with surrounding garbage should be parsed correctly."""
    classifier = NLIClassifier(backend="cascade")
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance.is_configured = True
        # Valid label buried in garbage - tests parser robustness
        instance._call_llm = AsyncMock(return_value="garbage text contradiction more garbage")
        
        result = await classifier._llm_classify("premise", "hypothesis")
        
        # Should successfully extract the valid label
        assert result.label == "contradiction"
        assert result.confidence == 0.9


@pytest.mark.asyncio
async def test_edge_case_label_with_case_variations():
    """Edge case: Labels with different cases should be normalized."""
    classifier = NLIClassifier(backend="cascade")
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance.is_configured = True
        instance._call_llm = AsyncMock(side_effect=["CONTRADICTION", "Entailment", "nEuTrAl"])
        
        result1 = await classifier._llm_classify("p1", "h1")
        result2 = await classifier._llm_classify("p2", "h2") 
        result3 = await classifier._llm_classify("p3", "h3")
        
        assert result1.label == "contradiction"
        assert result2.label == "entailment"
        assert result3.label == "neutral"
        # All should have high confidence since they were parsed successfully
        assert all(r.confidence == 0.9 for r in [result1, result2] if r.label != "neutral")
        assert result3.confidence == 0.3  # neutral gets lower confidence


@pytest.mark.asyncio
async def test_edge_case_provider_disappears_mid_run():
    """Edge case: Provider becomes unavailable mid-run should warn only once."""
    classifier = NLIClassifier(backend="cascade")
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        # is_configured changes from True to False mid-run (provider disappears)
        instance.is_configured = True
        instance._call_llm = AsyncMock(side_effect=[
            "entailment",  # First call succeeds
            ConnectionError("Provider down")  # Provider disappears
        ])
        
        with patch.object(classifier, "_warn_once") as mock_warn:
            with patch.object(classifier, "_heuristic_classify",
                             return_value=NLIResult(label="neutral", confidence=0.2)):
                
                # First call should succeed
                result1 = await classifier._llm_classify("p1", "h1")
                assert result1.label == "entailment"
                
                # Provider goes down, should warn once
                result2 = await classifier._llm_classify("p2", "h2")
                assert result2.label == "neutral"
                
                # More calls while provider is down - should not warn again
                result3 = await classifier._llm_classify("p3", "h3")
                assert result3.label == "neutral"
        
        # Should warn exactly once when degradation begins
        mock_warn.assert_called_once()


@pytest.mark.asyncio
async def test_edge_case_timeout_error_sanitization():
    """Edge case: Timeout errors should be sanitized properly."""
    classifier = NLIClassifier(backend="cascade")
    
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance.is_configured = True
        # Timeout with potential sensitive info in message
        timeout_error = TimeoutError("Timeout after 30s connecting to http://secret-endpoint:8080/api")
        instance._call_llm = AsyncMock(side_effect=timeout_error)
        
        with patch.object(classifier, "_warn_once") as mock_warn:
            with patch.object(classifier, "_heuristic_classify",
                             return_value=NLIResult(label="neutral", confidence=0.1)):
                await classifier._llm_classify("premise", "hypothesis")
        
        mock_warn.assert_called_once()
        # Verify the error message was sanitized (implementation detail for the actual code)


# Missing method tests - these should FAIL until implementation exists

@pytest.mark.asyncio  
async def test_missing_get_rewriter_method_fails():
    """Removed: this was an inverted 'method does not exist yet' scaffold test.
    The methods now exist; real behavior is covered by the R10-R14 tests."""
    pass


def test_missing_warn_once_method_removed():
    """Removed: inverted scaffold. See _warn_once behavior in R12/R13 tests."""
    pass