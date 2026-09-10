import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from mcp_memory_service.harvest.harvester import SessionHarvester
from mcp_memory_service.harvest.models import HarvestConfig


class TestSessionHarvester:
    def test_harvest_dry_run(self, sample_project_dir):
        """Dry run should find candidates but not store them."""
        config = HarvestConfig(sessions=1, dry_run=True)
        harvester = SessionHarvester(project_dir=sample_project_dir)
        results = harvester.harvest(config)
        assert len(results) == 1
        result = results[0]
        assert result.found > 0
        assert result.stored == 0  # dry run

    def test_harvest_type_filter(self, sample_project_dir):
        """Should filter by requested types."""
        config = HarvestConfig(sessions=1, dry_run=True, types=["bug"])
        harvester = SessionHarvester(project_dir=sample_project_dir)
        results = harvester.harvest(config)
        result = results[0]
        for c in result.candidates:
            assert c.memory_type == "bug"

    def test_harvest_confidence_filter(self, sample_project_dir):
        """Should filter by min_confidence."""
        config = HarvestConfig(sessions=1, dry_run=True, min_confidence=0.99)
        harvester = SessionHarvester(project_dir=sample_project_dir)
        results = harvester.harvest(config)
        result = results[0]
        assert result.found == 0  # Nothing above 0.99

    def test_harvest_no_sessions(self, tmp_path):
        """Should return empty results if no sessions found."""
        config = HarvestConfig(sessions=1, dry_run=True)
        harvester = SessionHarvester(project_dir=tmp_path)
        results = harvester.harvest(config)
        assert results == []

    @pytest.mark.asyncio
    async def test_harvest_store(self, sample_project_dir):
        """Non-dry-run should call memory_service.store_memory."""
        mock_service = AsyncMock()
        mock_service.storage = AsyncMock()
        mock_service.storage.retrieve = AsyncMock(return_value=[])  # No similar → store new
        mock_result = MagicMock()
        mock_result.success = True
        mock_service.store_memory.return_value = mock_result
        config = HarvestConfig(sessions=1, dry_run=False)
        harvester = SessionHarvester(project_dir=sample_project_dir, memory_service=mock_service)
        results = await harvester.harvest_and_store(config)
        result = results[0]
        if result.found > 0:
            assert mock_service.store_memory.call_count == result.found
            assert result.stored == result.found

    @pytest.mark.asyncio
    async def test_harvest_store_partial_failure(self, sample_project_dir):
        """If some store_memory calls fail, stored < found and harvester continues."""
        call_count = 0

        async def flaky_store(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated storage failure")
            result = MagicMock()
            result.success = True
            return result

        mock_service = AsyncMock()
        mock_service.storage = AsyncMock()
        mock_service.storage.retrieve = AsyncMock(return_value=[])  # No similar → store new
        mock_service.store_memory = flaky_store
        config = HarvestConfig(sessions=1, dry_run=False)
        harvester = SessionHarvester(project_dir=sample_project_dir, memory_service=mock_service)
        results = await harvester.harvest_and_store(config)
        result = results[0]
        if result.found > 1:
            assert result.stored < result.found


class TestHarvestEvolutionConfig:
    def test_harvest_config_has_evolution_fields(self):
        """HarvestConfig should have similarity_threshold and min_confidence_to_evolve."""
        config = HarvestConfig()
        assert config.similarity_threshold == 0.85
        assert config.min_confidence_to_evolve == 0.3

    def test_harvest_config_from_env(self, monkeypatch):
        """Environment variables should override defaults."""
        from mcp_memory_service.harvest.models import harvest_config_from_env
        monkeypatch.setenv("MCP_HARVEST_SIMILARITY_THRESHOLD", "0.90")
        monkeypatch.setenv("MCP_HARVEST_MIN_CONFIDENCE_TO_EVOLVE", "0.5")
        config = harvest_config_from_env()
        assert config.similarity_threshold == 0.90
        assert config.min_confidence_to_evolve == 0.5

    def test_harvest_config_from_env_with_overrides(self, monkeypatch):
        """Explicit overrides should beat env vars."""
        from mcp_memory_service.harvest.models import harvest_config_from_env
        monkeypatch.setenv("MCP_HARVEST_SIMILARITY_THRESHOLD", "0.90")
        config = harvest_config_from_env(similarity_threshold=0.75)
        assert config.similarity_threshold == 0.75


class TestHarvestEvolution:
    """Tests for P4 evolve-instead-of-duplicate behavior."""

    @pytest.mark.asyncio
    async def test_evolve_similar_memory(self, sample_project_dir):
        """When a similar active memory exists (score > threshold), evolve it."""
        mock_service = AsyncMock()
        mock_query_result = MagicMock()
        mock_query_result.relevance_score = 0.92  # Above 0.85 threshold
        mock_query_result.memory = MagicMock()
        mock_query_result.memory.content_hash = "existing-hash-123"
        mock_service.storage = AsyncMock()
        mock_service.storage.retrieve = AsyncMock(return_value=[mock_query_result])
        mock_service.storage.update_memory_versioned = AsyncMock(
            return_value=(True, "Updated", "new-hash-456")
        )

        config = HarvestConfig(sessions=1, dry_run=False, similarity_threshold=0.85)
        harvester = SessionHarvester(
            project_dir=sample_project_dir, memory_service=mock_service
        )
        results = await harvester.harvest_and_store(config)
        result = results[0]
        assert result.found > 0, "Fixture must produce candidates"

        mock_service.storage.retrieve.assert_called()
        mock_service.storage.update_memory_versioned.assert_called()
        mock_service.store_memory.assert_not_called()
        assert result.stored == result.found

    @pytest.mark.asyncio
    async def test_store_novel_content(self, sample_project_dir):
        """When no similar memory exists, fall back to normal store_memory."""
        mock_service = AsyncMock()
        mock_service.storage = AsyncMock()
        mock_service.storage.retrieve = AsyncMock(return_value=[])  # No matches
        mock_result = MagicMock()
        mock_result.success = True
        mock_service.store_memory.return_value = mock_result

        config = HarvestConfig(sessions=1, dry_run=False)
        harvester = SessionHarvester(
            project_dir=sample_project_dir, memory_service=mock_service
        )
        results = await harvester.harvest_and_store(config)
        result = results[0]
        assert result.found > 0, "Fixture must produce candidates"

        mock_service.store_memory.assert_called()
        mock_service.storage.update_memory_versioned.assert_not_called()
        assert result.stored == result.found

    @pytest.mark.asyncio
    async def test_skip_evolve_stale_memory(self, sample_project_dir):
        """Stale memory below min_confidence_to_evolve should not be evolved."""
        mock_service = AsyncMock()
        mock_service.storage = AsyncMock()
        # Simulate: retrieve with high min_confidence returns nothing
        mock_service.storage.retrieve = AsyncMock(return_value=[])
        mock_result = MagicMock()
        mock_result.success = True
        mock_service.store_memory.return_value = mock_result

        config = HarvestConfig(
            sessions=1, dry_run=False, min_confidence_to_evolve=0.8
        )
        harvester = SessionHarvester(
            project_dir=sample_project_dir, memory_service=mock_service
        )
        results = await harvester.harvest_and_store(config)
        result = results[0]
        assert result.found > 0, "Fixture must produce candidates"

        # retrieve was called with high min_confidence, returned nothing
        call_args = mock_service.storage.retrieve.call_args
        assert call_args.kwargs.get("min_confidence") == 0.8
        # Fell through to store_memory
        mock_service.store_memory.assert_called()

    @pytest.mark.asyncio
    async def test_below_threshold_similarity_stores_new(self, sample_project_dir):
        """Match below similarity_threshold should create new memory, not evolve."""
        mock_service = AsyncMock()
        mock_query_result = MagicMock()
        mock_query_result.relevance_score = 0.60  # Below 0.85 threshold
        mock_query_result.memory = MagicMock()
        mock_query_result.memory.content_hash = "low-sim-hash"
        mock_service.storage = AsyncMock()
        mock_service.storage.retrieve = AsyncMock(return_value=[mock_query_result])
        mock_result = MagicMock()
        mock_result.success = True
        mock_service.store_memory.return_value = mock_result

        config = HarvestConfig(sessions=1, dry_run=False, similarity_threshold=0.85)
        harvester = SessionHarvester(
            project_dir=sample_project_dir, memory_service=mock_service
        )
        results = await harvester.harvest_and_store(config)
        result = results[0]
        assert result.found > 0, "Fixture must produce candidates"

        mock_service.store_memory.assert_called()
        mock_service.storage.update_memory_versioned.assert_not_called()

    @pytest.mark.asyncio
    async def test_superseded_memory_not_evolved(self, sample_project_dir):
        """Superseded memories should not be evolved — retrieve filters them out.

        The storage layer's retrieve() excludes superseded versions
        (WHERE superseded_by IS NULL). If the only similar memory is
        superseded, retrieve returns empty and we create a new memory.
        """
        mock_service = AsyncMock()
        mock_service.storage = AsyncMock()
        # retrieve returns empty because the only match is superseded (filtered by storage)
        mock_service.storage.retrieve = AsyncMock(return_value=[])
        mock_result = MagicMock()
        mock_result.success = True
        mock_service.store_memory.return_value = mock_result

        config = HarvestConfig(sessions=1, dry_run=False)
        harvester = SessionHarvester(
            project_dir=sample_project_dir, memory_service=mock_service
        )
        results = await harvester.harvest_and_store(config)
        result = results[0]
        assert result.found > 0, "Fixture must produce candidates"

        mock_service.store_memory.assert_called()
        mock_service.storage.update_memory_versioned.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_when_no_storage(self, sample_project_dir):
        """If memory_service has no storage attr, fall back to store_memory."""
        mock_service = AsyncMock()
        # No .storage attribute — simulates pre-P4 MemoryService
        del mock_service.storage
        mock_result = MagicMock()
        mock_result.success = True
        mock_service.store_memory.return_value = mock_result

        config = HarvestConfig(sessions=1, dry_run=False)
        harvester = SessionHarvester(
            project_dir=sample_project_dir, memory_service=mock_service
        )
        results = await harvester.harvest_and_store(config)
        result = results[0]
        assert result.found > 0, "Fixture must produce candidates"

        mock_service.store_memory.assert_called()


class TestHarvestBatchRewrite:
    """Tests for batch LLM rewrite path (issue #1108)."""

    @pytest.mark.asyncio
    async def test_batch_rewrite_used_instead_of_per_candidate(self, sample_project_dir):
        """When LLM rewrite is enabled, rewrite_batch_sync should be called
        once with all candidates, not rewrite_sync per candidate.

        This exercises the batch API introduced in issue #1108.
        Without the change, rewrite_sync would be called per candidate.
        """
        from mcp_memory_service.harvest.rewriter import RewriteResult

        mock_service = AsyncMock()
        mock_service.storage = AsyncMock()
        mock_service.storage.retrieve = AsyncMock(return_value=[])
        mock_result = MagicMock()
        mock_result.success = True
        mock_service.store_memory.return_value = mock_result

        # Create a mock rewriter
        mock_rewriter = MagicMock()
        mock_rewriter.is_configured = True
        # Return a RewriteResult for each item
        def fake_batch(items):
            return [
                RewriteResult(content=f"rewritten: {item['content'][:20]}", memory_type=item['memory_type'])
                for item in items
            ]
        mock_rewriter.rewrite_batch_sync.side_effect = fake_batch

        config = HarvestConfig(sessions=1, dry_run=False, use_llm=True)
        harvester = SessionHarvester(
            project_dir=sample_project_dir, memory_service=mock_service
        )
        # Patch _get_rewriter to return our mock
        harvester._get_rewriter = lambda: mock_rewriter

        results = await harvester.harvest_and_store(config)
        result = results[0]

        if result.found > 0:
            # rewrite_batch_sync should be called (not rewrite_sync)
            assert mock_rewriter.rewrite_batch_sync.call_count >= 1
            mock_rewriter.rewrite_sync.assert_not_called()
        assert result.stored == result.found
