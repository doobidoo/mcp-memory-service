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
        mock_service.store_memory = flaky_store
        config = HarvestConfig(sessions=1, dry_run=False)
        harvester = SessionHarvester(project_dir=sample_project_dir, memory_service=mock_service)
        results = await harvester.harvest_and_store(config)
        result = results[0]
        if result.found > 1:
            assert result.stored < result.found
            assert result.stored > 0
