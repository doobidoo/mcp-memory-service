"""Tests for LLM-based harvest classifier (Phase 2)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from mcp_memory_service.harvest.classifier import (
    HarvestClassifier,
    ClassificationResult,
)
from mcp_memory_service.harvest.models import HarvestCandidate


@pytest.fixture
def classifier():
    """Create classifier with mocked Groq bridge."""
    c = HarvestClassifier(groq_api_key="test-key")
    return c


@pytest.fixture
def sample_candidates():
    """Sample candidates for testing."""
    return [
        HarvestCandidate(
            content="Decided to use SQLite-Vec over FAISS because WAL mode supports concurrent access",
            memory_type="decision",
            tags=["harvest:decision"],
            confidence=0.75,
            source_line="I decided to use SQLite-Vec over FAISS...",
        ),
        HarvestCandidate(
            content="The root cause was FAISS not supporting concurrent access",
            memory_type="bug",
            tags=["harvest:bug"],
            confidence=0.75,
            source_line="Root cause: FAISS...",
        ),
        HarvestCandidate(
            content="Let me check the status of the tests",
            memory_type="context",
            tags=["harvest:context"],
            confidence=0.6,
            source_line="Let me check...",
        ),
    ]


def _mock_groq_response(keep, reason="test", refined=None, memory_type=None, confidence=0.8):
    """Helper to create a mock Groq response."""
    data = {
        "keep": keep,
        "reason": reason,
        "refined_content": refined,
        "memory_type": memory_type,
        "confidence": confidence,
    }
    return {"status": "success", "response": json.dumps(data), "model": "test", "tokens_used": 50}


class TestClassificationResult:
    def test_defaults(self):
        r = ClassificationResult(keep=True, reason="good")
        assert r.keep is True
        assert r.confidence == 0.0
        assert r.refined_content is None

    def test_with_all_fields(self):
        r = ClassificationResult(
            keep=True, reason="good", refined_content="refined", memory_type="decision", confidence=0.9
        )
        assert r.refined_content == "refined"
        assert r.memory_type == "decision"


class TestHarvestClassifier:
    def test_empty_candidates_returns_empty(self, classifier):
        assert classifier.classify([]) == []

    def test_no_api_key_returns_unfiltered(self, sample_candidates):
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}, clear=False):
            c = HarvestClassifier(groq_api_key=None)
            # Force re-init by ensuring bridge is None and key is empty
            c._api_key = None
            result = c.classify(sample_candidates)
        assert len(result) == len(sample_candidates)

    def test_classify_filters_rejected(self, classifier, sample_candidates):
        """LLM rejects low-quality candidates."""
        responses = [
            _mock_groq_response(keep=True, confidence=0.85),   # decision: keep
            _mock_groq_response(keep=True, confidence=0.8),    # bug: keep
            _mock_groq_response(keep=False, reason="fragment"),  # context: reject
        ]
        mock_bridge = MagicMock()
        mock_bridge.call_model.side_effect = responses
        classifier._groq_bridge = mock_bridge

        # Mock dedup to return all
        dedup_response = {"status": "success", "response": "[0, 1]", "model": "test", "tokens_used": 10}
        mock_bridge.call_model.side_effect = responses + [dedup_response]

        result = classifier.classify(sample_candidates, context_messages=["some context"])
        assert len(result) == 2
        assert all("llm-verified" in c.tags for c in result)

    def test_classify_refines_content(self, classifier):
        """LLM refines candidate content."""
        candidate = HarvestCandidate(
            content="So basically what happened was we decided to go with SQLite-Vec because of WAL",
            memory_type="decision",
            tags=["harvest:decision"],
            confidence=0.65,
        )
        refined = "Use SQLite-Vec over FAISS: WAL mode enables concurrent access"
        mock_bridge = MagicMock()
        mock_bridge.call_model.return_value = _mock_groq_response(
            keep=True, refined=refined, confidence=0.9
        )
        classifier._groq_bridge = mock_bridge

        result = classifier.classify([candidate])
        assert len(result) == 1
        assert result[0].content == refined
        assert result[0].confidence == 0.9

    def test_classify_updates_memory_type(self, classifier):
        """LLM can correct memory type and preserve non-harvest tags."""
        candidate = HarvestCandidate(
            content="Learned that WAL mode is required for concurrent SQLite access",
            memory_type="bug",
            tags=["harvest:bug", "custom-tag"],
            confidence=0.7,
        )
        mock_bridge = MagicMock()
        mock_bridge.call_model.return_value = _mock_groq_response(
            keep=True, memory_type="learning", confidence=0.85
        )
        classifier._groq_bridge = mock_bridge

        result = classifier.classify([candidate])
        assert result[0].memory_type == "learning"
        assert "harvest:learning" in result[0].tags
        assert "custom-tag" in result[0].tags
        assert "harvest:bug" not in result[0].tags

    def test_parse_classification_json(self, classifier):
        """Parse clean JSON response."""
        response = '{"keep": true, "reason": "good", "refined_content": null, "memory_type": "decision", "confidence": 0.85}'
        result = classifier._parse_classification(response)
        assert result.keep is True
        assert result.confidence == 0.85

    def test_parse_classification_markdown_wrapped(self, classifier):
        """Parse JSON wrapped in markdown code blocks."""
        response = '```json\n{"keep": false, "reason": "fragment", "confidence": 0.3}\n```'
        result = classifier._parse_classification(response)
        assert result.keep is False
        assert result.reason == "fragment"

    def test_parse_classification_with_extra_text(self, classifier):
        """Parse JSON embedded in other text."""
        response = 'Here is my analysis:\n{"keep": true, "reason": "valid", "confidence": 0.7}\nThat is my assessment.'
        result = classifier._parse_classification(response)
        assert result.keep is True
        assert result.confidence == 0.7

    def test_parse_classification_invalid_returns_keep(self, classifier):
        """Unparseable response defaults to keep=True (fail-open)."""
        result = classifier._parse_classification("I can't parse this as JSON")
        assert result.keep is True
        assert result.confidence == 0.5

    def test_dedup_removes_duplicates(self, classifier):
        """Deduplication removes near-duplicate candidates."""
        candidates = [
            HarvestCandidate(content="Use WAL mode for concurrent SQLite access", memory_type="convention", confidence=0.8, tags=["llm-verified"]),
            HarvestCandidate(content="Always enable WAL mode for SQLite concurrent access", memory_type="convention", confidence=0.75, tags=["llm-verified"]),
            HarvestCandidate(content="Root cause: missing WAL pragma", memory_type="bug", confidence=0.85, tags=["llm-verified"]),
        ]
        mock_bridge = MagicMock()
        mock_bridge.call_model.return_value = {
            "status": "success", "response": "[0, 2]", "model": "test", "tokens_used": 20
        }
        classifier._groq_bridge = mock_bridge

        result = classifier._deduplicate(candidates)
        assert len(result) == 2
        assert result[0].memory_type == "convention"
        assert result[1].memory_type == "bug"

    def test_dedup_single_candidate_noop(self, classifier):
        """Single candidate passes through without LLM call."""
        candidate = HarvestCandidate(content="test", memory_type="decision", confidence=0.8, tags=[])
        result = classifier._deduplicate([candidate])
        assert result == [candidate]

    def test_rate_limit_fallback(self, classifier):
        """Rate limit on first model falls back to second."""
        mock_bridge = MagicMock()
        mock_bridge.call_model.side_effect = [
            {"status": "error", "error": "429 rate limit"},
            _mock_groq_response(keep=True, confidence=0.8),
        ]
        classifier._groq_bridge = mock_bridge

        candidate = HarvestCandidate(content="test decision", memory_type="decision", confidence=0.7, tags=["harvest:decision"])
        result = classifier._classify_single(candidate, "context")
        assert result.keep is True
        assert mock_bridge.call_model.call_count == 2

    def test_all_models_fail_keeps_candidate(self, classifier):
        """If all LLM models fail, candidate is kept (fail-open)."""
        mock_bridge = MagicMock()
        mock_bridge.call_model.side_effect = Exception("network error")
        classifier._groq_bridge = mock_bridge

        candidate = HarvestCandidate(content="test", memory_type="decision", confidence=0.7, tags=[])
        result = classifier._classify_single(candidate, "context")
        assert result.keep is True


class TestHarvesterLLMIntegration:
    """Test harvester integration with LLM classifier."""

    def test_use_llm_false_skips_classifier(self, sample_project_dir):
        """use_llm=False does not invoke classifier."""
        from mcp_memory_service.harvest.harvester import SessionHarvester
        from mcp_memory_service.harvest.models import HarvestConfig

        harvester = SessionHarvester(project_dir=sample_project_dir)
        config = HarvestConfig(sessions=1, use_llm=False)
        results = harvester.harvest(config)
        assert len(results) > 0
        # No llm-verified tag
        for r in results:
            for c in r.candidates:
                assert "llm-verified" not in c.tags

    def test_use_llm_true_invokes_classifier(self, sample_project_dir):
        """use_llm=True invokes classifier on candidates."""
        from mcp_memory_service.harvest.harvester import SessionHarvester
        from mcp_memory_service.harvest.models import HarvestConfig

        harvester = SessionHarvester(project_dir=sample_project_dir)
        config = HarvestConfig(sessions=1, use_llm=True)

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = [
            HarvestCandidate(
                content="refined content",
                memory_type="decision",
                tags=["harvest:decision", "llm-verified"],
                confidence=0.9,
            )
        ]
        harvester._classifier = mock_classifier

        results = harvester.harvest(config)
        assert len(results) == 1
        assert results[0].found == 1
        assert results[0].candidates[0].content == "refined content"
        mock_classifier.classify.assert_called_once()

    def test_use_llm_config_default_false(self):
        """HarvestConfig defaults use_llm to False."""
        from mcp_memory_service.harvest.models import HarvestConfig
        config = HarvestConfig()
        assert config.use_llm is False
