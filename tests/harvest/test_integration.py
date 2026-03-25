import pytest
import json
from pathlib import Path
from mcp_memory_service.harvest.harvester import SessionHarvester
from mcp_memory_service.harvest.models import HarvestConfig


class TestHarvestIntegration:
    """Integration test with realistic multi-message transcripts."""

    @pytest.fixture
    def realistic_session(self, tmp_path):
        """A more realistic session with varied content."""
        lines = [
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "text", "text": "The tests are failing with a database lock error on the CI pipeline"}
            ]}, "timestamp": "2026-03-25T10:00:00Z", "sessionId": "int-001", "uuid": "u1"},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "I found the root cause — the SQLite database was opened without WAL mode, causing lock contention between the HTTP and MCP servers."}
            ]}, "timestamp": "2026-03-25T10:01:00Z", "sessionId": "int-001", "uuid": "a1"},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "I decided to use journal_mode=WAL with busy_timeout=15000 instead of switching to PostgreSQL. The fix was adding MCP_MEMORY_SQLITE_PRAGMAS to the environment."}
            ]}, "timestamp": "2026-03-25T10:02:00Z", "sessionId": "int-001", "uuid": "a2"},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "Convention: always set journal_mode=WAL when multiple processes access the same SQLite database."}
            ]}, "timestamp": "2026-03-25T10:03:00Z", "sessionId": "int-001", "uuid": "a3"},
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "text", "text": "Great, I learned that SQLite can actually handle concurrent access well with WAL mode enabled."}
            ]}, "timestamp": "2026-03-25T10:04:00Z", "sessionId": "int-001", "uuid": "u2"},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "```python\nos.environ['MCP_MEMORY_SQLITE_PRAGMAS'] = 'journal_mode=WAL,busy_timeout=15000'\n```"}
            ]}, "timestamp": "2026-03-25T10:05:00Z", "sessionId": "int-001", "uuid": "a4"},
        ]
        filepath = tmp_path / "int-001.jsonl"
        with open(filepath, 'w') as f:
            for line in lines:
                f.write(json.dumps(line) + '\n')
        return tmp_path

    def test_full_pipeline(self, realistic_session):
        config = HarvestConfig(sessions=1, dry_run=True, min_confidence=0.6)
        harvester = SessionHarvester(project_dir=realistic_session)
        results = harvester.harvest(config)

        assert len(results) == 1
        r = results[0]
        assert r.session_id == "int-001"
        assert r.found >= 3  # bug, decision, convention, learning
        assert "bug" in r.by_type
        assert "decision" in r.by_type
        assert "convention" in r.by_type

    def test_code_blocks_not_extracted(self, realistic_session):
        """Code-only messages should not produce candidates."""
        config = HarvestConfig(sessions=1, dry_run=True)
        harvester = SessionHarvester(project_dir=realistic_session)
        results = harvester.harvest(config)
        r = results[0]
        # The code block message (a4) should not produce false positives
        # All candidates should come from natural language messages
        for c in r.candidates:
            assert "```" not in c.content

    def test_json_serialization(self, realistic_session):
        """Results should be JSON-serializable for MCP response."""
        config = HarvestConfig(sessions=1, dry_run=True)
        harvester = SessionHarvester(project_dir=realistic_session)
        results = harvester.harvest(config)

        output = {
            "results": [
                {
                    "session_id": r.session_id,
                    "total_messages": r.total_messages,
                    "found": r.found,
                    "stored": r.stored,
                    "by_type": r.by_type,
                    "candidates": [
                        {
                            "type": c.memory_type,
                            "content": c.content,
                            "confidence": c.confidence,
                            "tags": c.tags,
                        }
                        for c in r.candidates
                    ]
                }
                for r in results
            ]
        }
        serialized = json.dumps(output)
        parsed = json.loads(serialized)
        assert "results" in parsed
        assert len(parsed["results"]) == 1
