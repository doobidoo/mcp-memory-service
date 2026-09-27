import pytest
from mcp_memory_service.harvest.extractor import PatternExtractor
from mcp_memory_service.harvest.parser import ParsedMessage


@pytest.fixture
def extractor():
    return PatternExtractor()


class TestPatternExtractor:
    def test_extract_decision(self, extractor):
        msg = ParsedMessage(role="assistant", text="I decided to use Redis over Memcached for caching because of pub/sub support.")
        candidates = extractor.extract(msg)
        types = [c.memory_type for c in candidates]
        assert "decision" in types

    def test_extract_bug(self, extractor):
        msg = ParsedMessage(role="assistant", text="The root cause was a race condition in the connection pool that blocked all queries.")
        candidates = extractor.extract(msg)
        types = [c.memory_type for c in candidates]
        assert "bug" in types

    def test_extract_convention(self, extractor):
        msg = ParsedMessage(role="assistant", text="Convention: always use WAL mode for concurrent SQLite access in production.")
        candidates = extractor.extract(msg)
        types = [c.memory_type for c in candidates]
        assert "convention" in types

    def test_extract_learning(self, extractor):
        msg = ParsedMessage(role="user", text="I learned that ONNX models need warmup on first inference to avoid latency spikes.")
        candidates = extractor.extract(msg, min_confidence=0.6)
        types = [c.memory_type for c in candidates]
        assert "learning" in types

    def test_no_false_positives_on_short_text(self, extractor):
        msg = ParsedMessage(role="assistant", text="OK")
        candidates = extractor.extract(msg)
        assert len(candidates) == 0

    def test_no_false_positives_on_code(self, extractor):
        msg = ParsedMessage(role="assistant", text="```python\ndef decide():\n    return True\n```")
        candidates = extractor.extract(msg)
        assert len(candidates) == 0

    def test_confidence_scaling(self, extractor):
        msg = ParsedMessage(role="assistant", text="The critical root cause of the production outage was the missing WAL pragma. The fix was adding it to startup.")
        candidates = extractor.extract(msg)
        bug_candidates = [c for c in candidates if c.memory_type == "bug"]
        assert len(bug_candidates) > 0
        assert bug_candidates[0].confidence >= 0.6

    def test_extract_multiple_types(self, extractor):
        msg = ParsedMessage(role="assistant", text="I decided to fix the bug by always using WAL mode because the root cause was missing pragma configuration.")
        candidates = extractor.extract(msg)
        types = set(c.memory_type for c in candidates)
        assert len(types) >= 2

    def test_min_text_length(self, extractor):
        msg = ParsedMessage(role="user", text="decided")
        candidates = extractor.extract(msg)
        assert len(candidates) == 0

    def test_meta_discussion_without_convention_is_filtered(self, extractor):
        """The meta-discussion filter's escape hatch is convention-only
        (Codeberg PR #322):
        a bug/decision/learning candidate about the harvest system itself
        must still be discarded, even though it matches a real pattern."""
        msg = ParsedMessage(
            role="assistant",
            text="The harvest pipeline's root cause was a race condition in the rewriter that blocked all extraction.",
        )
        candidates = extractor.extract(msg)
        assert candidates == []

    def test_meta_discussion_with_convention_survives(self, extractor):
        """The narrower escape hatch still lets real conventions about the
        system itself through, which is the one case it exists for."""
        msg = ParsedMessage(
            role="assistant",
            text="Convention: always use WAL mode for the harvest pipeline's extractor cache.",
        )
        candidates = extractor.extract(msg)
        types = [c.memory_type for c in candidates]
        assert "convention" in types
