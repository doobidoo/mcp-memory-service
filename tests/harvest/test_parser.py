import pytest
from mcp_memory_service.harvest.models import HarvestCandidate, HarvestResult, HarvestConfig
from mcp_memory_service.harvest.parser import TranscriptParser, ParsedMessage


class TestHarvestModels:
    def test_candidate_creation(self):
        c = HarvestCandidate(
            content="Chose SQLite-Vec over FAISS for local storage",
            memory_type="decision",
            tags=["architecture", "storage"],
            confidence=0.8,
            source_line="I decided to go with SQLite-Vec because..."
        )
        assert c.memory_type == "decision"
        assert c.confidence == 0.8
        assert "architecture" in c.tags

    def test_result_creation(self):
        r = HarvestResult(
            candidates=[],
            session_id="abc123",
            total_messages=50,
            found=0,
            by_type={}
        )
        assert r.found == 0
        assert r.session_id == "abc123"

    def test_config_defaults(self):
        cfg = HarvestConfig()
        assert cfg.min_confidence == 0.6
        assert cfg.dry_run is True
        assert "decision" in cfg.types


class TestTranscriptParser:
    def test_parse_jsonl_file(self, sample_jsonl):
        filepath, session_id = sample_jsonl
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        # Should skip 'progress' and 'thinking' blocks
        assert len(messages) == 4  # 2 user texts + 2 assistant texts

    def test_parsed_message_fields(self, sample_jsonl):
        filepath, _ = sample_jsonl
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        msg = messages[0]
        assert msg.role == "user"
        assert "SQLite-Vec" in msg.text
        assert msg.timestamp is not None

    def test_find_sessions(self, sample_project_dir):
        parser = TranscriptParser()
        sessions = parser.find_sessions(sample_project_dir, count=1)
        assert len(sessions) == 1

    def test_parse_empty_file(self, tmp_path):
        filepath = tmp_path / "empty.jsonl"
        filepath.write_text("")
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        assert messages == []

    def test_parse_whitespace_only_file(self, tmp_path):
        filepath = tmp_path / "whitespace.jsonl"
        filepath.write_text("  \n\n  \n")
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        assert messages == []

    def test_parse_corrupt_line_skipped(self, tmp_path):
        filepath = tmp_path / "corrupt.jsonl"
        filepath.write_text('not-json\n{"type":"progress"}\n')
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        assert messages == []  # progress is skipped, corrupt is skipped
