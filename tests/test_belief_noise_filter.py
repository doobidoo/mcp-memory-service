"""Tests for belief noise filter, entropy, and dedup logic."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from mcp_memory_service.consolidation.belief_service import (
    _is_noise,
    _content_entropy,
    BeliefService,
)


class TestNoiseFilter:
    """Tests for the _is_noise content filter."""

    def test_noise_filter_session_summary(self):
        """Session summary headers should be filtered as noise."""
        assert _is_noise("# Session Summary - Project X") is True
        assert _is_noise("## Session Summary\nSome content") is True
        assert _is_noise("### Session Summary") is True

    def test_noise_filter_transcript(self):
        """Content with User:/Assistant: lines (transcripts) should be noise."""
        content = "Some context\nUser: how do I fix this?\nAssistant: try X"
        assert _is_noise(content) is True

    def test_noise_filter_long_degenerate_content(self):
        """Long but degenerate content (one giant token, no real words) is still noise."""
        long_content = "x" * 601
        assert _is_noise(long_content) is True

    def test_noise_filter_long_low_entropy_dump(self):
        """Long AND repetitive (low entropy) content is a dump → noise. Uses bulletless
        prose so it is NOT caught by the pre-existing bullet-ratio rule — it must be
        rejected by the entropy branch specifically (per Greptile review on #1533)."""
        dump = "status ok item " * 60
        assert len(dump) > 500
        assert not any(l.lstrip().startswith("- ") for l in dump.split("\n"))
        assert _is_noise(dump) is True

    def test_noise_filter_long_substantive_insight_not_noise(self):
        """A long but SUBSTANTIVE actionable insight (high entropy, many words) must
        NOT be filtered just for exceeding 500 chars. Guards #f0e2c306: the old
        length>500 rule discarded 538 real decision/learning memories (99.8% of them
        substantive, entropy>=0.49) — length alone is a bad proxy for 'dump'."""
        insight = (
            "RER architectural analysis: drop the Git submodules and keep independent "
            "repositories. Core becomes the deploy orchestrator via docker-compose and "
            "install.sh, and also serves the project landing page for external visitors. "
            "This removes the submodule sync pain and lets each service version on its own. "
            "The gateway routes requests to backend, frontend and geoserver; authentication "
            "flows through Keycloak with a dedicated realm, and the frontend redirect was "
            "fixed so relative URLs no longer break the login handshake across containers. "
            "Each repository publishes its own image to the registry and the compose file "
            "pins explicit tags, so a broken build in one service never blocks the others."
        )
        assert len(insight) > 500
        assert _is_noise(insight) is False

    def test_noise_filter_bullet_dump(self):
        """Content that is mostly bullet points (>60%) should be noise."""
        content = "\n".join([f"- Item {i}" for i in range(10)])
        assert _is_noise(content) is True

    def test_noise_filter_valid_insight(self):
        """Short, actionable insights should NOT be filtered."""
        assert _is_noise("Always pin versions in pyproject.toml") is False

    def test_noise_filter_checkpoint_prefix(self):
        """[CHECKPOINT] prefix should be noise."""
        assert _is_noise("[CHECKPOINT] session saved") is True

    def test_noise_filter_mined_prefix(self):
        """[MINED: prefix should be noise."""
        assert _is_noise("[MINED: 2026-01-01] some content") is True

    def test_noise_filter_task_notification(self):
        """<task-notification> content should be noise."""
        assert _is_noise("Before text <task-notification> something") is True

    def test_noise_filter_ide_opened_file(self):
        """<ide_opened_file> content should be noise."""
        assert _is_noise("Some prefix <ide_opened_file> path/to/file") is True

    def test_noise_filter_topics_discussed(self):
        """Topics Discussed: pattern should be noise."""
        assert _is_noise("Some preamble\nTopics Discussed: A, B, C") is True


class TestContentEntropy:
    """Tests for the _content_entropy function."""

    def test_entropy_low(self):
        """Highly repetitive content should have low entropy."""
        result = _content_entropy("the the the the the")
        assert result < 0.3

    def test_entropy_normal(self):
        """Normal varied content should have reasonable entropy."""
        result = _content_entropy("Redis has no persistence configured in production")
        assert result > 0.5

    def test_entropy_empty(self):
        """Empty content should return 0."""
        assert _content_entropy("") == 0.0

    def test_entropy_single_word(self):
        """Single word should have entropy 1.0."""
        assert _content_entropy("hello") == 1.0

    def test_entropy_all_unique(self):
        """All unique words should have entropy 1.0."""
        assert _content_entropy("every word here is different") == 1.0


class TestDedupPreventsIdenticalClusters:
    """Test that content-hash dedup prevents near-identical clusters."""

    @pytest.mark.asyncio
    async def test_dedup_prevents_duplicate_clusters(self):
        """Near-identical observations should only produce 1 group."""
        # Create mock storage with retrieve that returns empty
        mock_storage = MagicMock()
        mock_storage.retrieve = AsyncMock(return_value=[])

        svc = BeliefService(mock_storage)

        # Create observations with identical content but different hashes
        obs1 = MagicMock()
        obs1.content = "Always use async for IO"
        obs1.content_hash = "hash_aaa"

        obs2 = MagicMock()
        obs2.content = "Always use async for IO"
        obs2.content_hash = "hash_bbb"

        obs3 = MagicMock()
        obs3.content = "Pin versions in requirements"
        obs3.content_hash = "hash_ccc"

        groups = await svc._group_observations([obs1, obs2, obs3])

        # obs1 and obs2 have same content → should be in same cluster (exact match)
        # obs3 is different → separate group
        # Total groups should be 2
        assert len(groups) == 2

    @pytest.mark.asyncio
    async def test_dedup_normalizes_whitespace(self):
        """Observations with different whitespace but same words → 1 group."""
        mock_storage = MagicMock()
        mock_storage.retrieve = AsyncMock(return_value=[])

        svc = BeliefService(mock_storage)

        obs1 = MagicMock()
        obs1.content = "always use async"
        obs1.content_hash = "hash_1"

        obs2 = MagicMock()
        obs2.content = "always  use  async"  # extra spaces
        obs2.content_hash = "hash_2"

        groups = await svc._group_observations([obs1, obs2])

        # After normalization, both have same content hash → only 1 group
        assert len(groups) == 1


class TestSessionLegacyNoiseFilter:
    """Regression tests for session-legacy noise filter (#121 follow-up)."""

    def test_noise_filter_session_legacy_uuid(self):
        """commit_session_legacy entries (Session: UUID) should be noise."""
        content = "Session: a1b2c3d4-e5f6-7890-abcd-ef1234567890 | Agent: kiro | Duration: 45min"
        assert _is_noise(content) is True

    def test_noise_filter_session_prefix_with_short_hash(self):
        """Session: followed by hex hash should be noise."""
        assert _is_noise("Session: deadbeef | Topics: coding") is True

    def test_noise_filter_session_mid_text_not_noise(self):
        """The word 'session' in mid-text should NOT be noise."""
        assert _is_noise("Redis loses session data on restart") is False

    def test_noise_filter_session_without_colon_not_noise(self):
        """'Session' without colon-space should NOT be noise."""
        assert _is_noise("Session management is important for auth") is False
