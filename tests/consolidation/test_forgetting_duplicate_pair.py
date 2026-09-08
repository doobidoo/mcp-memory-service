"""Deduplication must leave a survivor.

`_appears_to_be_duplicate` is a *symmetric* predicate: if A is a near-duplicate of B then
B is a near-duplicate of A. `_identify_forgetting_candidates` asks it once per memory
against the whole list, so both members of a near-duplicate pair are flagged
`potential_duplicate`, both get `can_be_deleted=True` (this reason bypasses the
time-horizon restriction, so it fires at every horizon), and `_apply_forgetting_results`
calls `storage.delete_memory()` on both.

The realistic shape is a note and its slightly edited follow-up: different content, so
genuinely different hashes, but >80% word overlap. Deduplicating them deletes *both*.
"""

import pytest
from datetime import datetime, timedelta, timezone

from mcp_memory_service.consolidation.forgetting import ControlledForgettingEngine
from mcp_memory_service.consolidation.decay import RelevanceScore
from mcp_memory_service.models.memory import Memory


ORIGINAL = (
    "The deployment pipeline retries a failed migration three times before it gives up "
    "and rolls the release back to the previous known-good revision of the service."
)
FOLLOW_UP = ORIGINAL + " Confirmed again on staging."
THIRD = ORIGINAL + " Confirmed again on staging and in production."


def _mem(content, h, days_old=1):
    ts = (datetime.now(timezone.utc) - timedelta(days=days_old)).timestamp()
    return Memory(
        content=content,
        content_hash=h,
        tags=["standard"],
        memory_type="observation",
        embedding=[0.1] * 8,
        created_at=ts,
        updated_at=ts,
    )


def _score(h, total=0.9):
    return RelevanceScore(
        memory_hash=h, total_score=total, base_importance=1.0, decay_factor=1.0,
        connection_boost=1.0, access_boost=1.0, metadata={},
    )


@pytest.mark.unit
class TestDuplicatePairSurvivor:

    @pytest.fixture
    def engine(self, consolidation_config):
        return ControlledForgettingEngine(consolidation_config)

    async def _deleted(self, engine, memories):
        scores = [_score(m.content_hash) for m in memories]
        results = await engine.process(memories, scores, access_patterns={},
                                       time_horizon="weekly")
        return [r.memory_hash for r in results if r.action_taken == "deleted"]

    @pytest.mark.asyncio
    async def test_near_duplicate_pair_keeps_one_survivor(self, engine):
        """A note and its edited follow-up: at most one may be deleted."""
        memories = [_mem(ORIGINAL, "hash_original"), _mem(FOLLOW_UP, "hash_follow_up")]
        deleted = await self._deleted(engine, memories)
        assert len(deleted) <= 1, (
            "deduplication deleted every copy (%s): nothing is left in storage and only a "
            "JSON backup under the archive path survives" % deleted
        )

    @pytest.mark.asyncio
    async def test_three_near_duplicates_keep_exactly_one(self, engine):
        """Three revisions of the same note: two may go, one must stay."""
        memories = [_mem(ORIGINAL, "hash_original"), _mem(FOLLOW_UP, "hash_follow_up"),
                    _mem(THIRD, "hash_third")]
        deleted = await self._deleted(engine, memories)
        assert len(deleted) == len(memories) - 1, (
            "expected exactly one survivor out of %d near-duplicates, %d were deleted (%s)"
            % (len(memories), len(deleted), deleted)
        )

    @pytest.mark.asyncio
    async def test_survivor_is_the_highest_scoring_copy(self, engine):
        """The copy the system rates highest is the one that stays."""
        memories = [_mem(ORIGINAL, "hash_low"), _mem(FOLLOW_UP, "hash_high")]
        scores = [_score("hash_low", 0.30), _score("hash_high", 0.95)]
        results = await engine.process(memories, scores, access_patterns={},
                                       time_horizon="weekly")
        deleted = [r.memory_hash for r in results if r.action_taken == "deleted"]
        assert "hash_high" not in deleted, (
            "the best-rated copy was deleted and a worse one kept: %s" % deleted
        )

    @pytest.mark.asyncio
    async def test_unique_memory_is_never_a_duplicate(self, engine):
        """Control: a memory with no twin is never deleted as a duplicate."""
        deleted = await self._deleted(engine, [_mem(ORIGINAL, "hash_only")])
        assert deleted == []
