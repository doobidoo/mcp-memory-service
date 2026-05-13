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

    def test_system_reminders_filtered(self, tmp_path):
        """System reminder injections should be filtered out."""
        import json
        lines = [
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>You have tools available</system-reminder>"},
                {"type": "text", "text": "Fix the bug in the parser"}
            ]}, "timestamp": "2026-03-25T10:00:00Z", "sessionId": "s1", "uuid": "u1"},
        ]
        filepath = tmp_path / "sys.jsonl"
        with open(filepath, 'w') as f:
            for line in lines:
                f.write(json.dumps(line) + '\n')
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        assert len(messages) == 1
        assert "Fix the bug" in messages[0].text

    def test_skill_outputs_filtered(self, tmp_path):
        """Skill/command outputs should be filtered out."""
        import json
        lines = [
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "text", "text": "<command-name>/release</command-name>"},
            ]}, "timestamp": "2026-03-25T10:00:00Z", "sessionId": "s1", "uuid": "u1"},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "I decided to use WAL mode for the database because of concurrent access needs."},
            ]}, "timestamp": "2026-03-25T10:01:00Z", "sessionId": "s1", "uuid": "a1"},
        ]
        filepath = tmp_path / "skill.jsonl"
        with open(filepath, 'w') as f:
            for line in lines:
                f.write(json.dumps(line) + '\n')
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        assert len(messages) == 1
        assert "WAL mode" in messages[0].text

    def test_very_long_text_filtered(self, tmp_path):
        """Very long text blocks (>2000 chars) should be filtered as likely skill definitions."""
        import json
        long_text = "x" * 2001
        lines = [
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": long_text},
                {"type": "text", "text": "Short real message about a bug fix"},
            ]}, "timestamp": "2026-03-25T10:00:00Z", "sessionId": "s1", "uuid": "a1"},
        ]
        filepath = tmp_path / "long.jsonl"
        with open(filepath, 'w') as f:
            for line in lines:
                f.write(json.dumps(line) + '\n')
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        assert len(messages) == 1
        assert "bug fix" in messages[0].text


class TestKiroCliParser:
    """Tests for Kiro CLI JSONL format parsing."""

    def test_parse_kiro_file(self, sample_kiro_jsonl):
        filepath, session_id = sample_kiro_jsonl
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        # Should get: 2 user (Prompt) + 2 assistant (AssistantMessage with text)
        # ToolResults are skipped
        assert len(messages) == 4

    def test_kiro_roles_mapped(self, sample_kiro_jsonl):
        filepath, _ = sample_kiro_jsonl
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_kiro_text_extracted(self, sample_kiro_jsonl):
        filepath, _ = sample_kiro_jsonl
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        assert "SQLite-Vec" in messages[0].text

    def test_kiro_tool_use_excluded(self, sample_kiro_jsonl):
        filepath, _ = sample_kiro_jsonl
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        # Assistant message a1 has toolUse — only text should be extracted
        for msg in messages:
            assert "toolUse" not in msg.text
            assert "shell" not in msg.text

    def test_kiro_uuid_preserved(self, sample_kiro_jsonl):
        filepath, _ = sample_kiro_jsonl
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        assert messages[0].uuid == "u1"

    def test_kiro_timestamp_converted(self, sample_kiro_jsonl):
        filepath, _ = sample_kiro_jsonl
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        # Unix 1778593355 should be converted to ISO string
        assert messages[0].timestamp is not None
        assert "2026" in messages[0].timestamp

    def test_kiro_tool_results_skipped(self, tmp_path):
        """ToolResults lines should be completely skipped."""
        import json
        lines = [
            {"version": "v1", "kind": "ToolResults", "data": {
                "message_id": "tr1",
                "content": [{"kind": "toolResult", "data": {"toolUseId": "t1", "content": [], "status": "success"}}]
            }},
        ]
        filepath = tmp_path / "tools_only.jsonl"
        with open(filepath, 'w') as f:
            for line in lines:
                f.write(json.dumps(line) + '\n')
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        assert messages == []

    def test_mixed_format_file(self, tmp_path):
        """A file with both formats should parse both correctly."""
        import json
        lines = [
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "text", "text": "Claude Code message"}
            ]}, "timestamp": "2026-03-25T10:00:00Z", "sessionId": "s1", "uuid": "cc1"},
            {"version": "v1", "kind": "Prompt", "data": {
                "message_id": "k1",
                "content": [{"kind": "text", "data": "Kiro CLI message"}],
                "meta": {"timestamp": 1778593355}
            }},
        ]
        filepath = tmp_path / "mixed.jsonl"
        with open(filepath, 'w') as f:
            for line in lines:
                f.write(json.dumps(line) + '\n')
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        assert len(messages) == 2
        assert messages[0].text == "Claude Code message"
        assert messages[1].text == "Kiro CLI message"

    def test_kiro_empty_text_skipped(self, tmp_path):
        """Kiro lines with empty text blocks should be skipped."""
        import json
        lines = [
            {"version": "v1", "kind": "AssistantMessage", "data": {
                "message_id": "a1",
                "content": [
                    {"kind": "text", "data": ""},
                    {"kind": "toolUse", "data": {"toolUseId": "t1", "name": "shell", "input": {}}}
                ],
                "meta": {"timestamp": 1778593355}
            }},
        ]
        filepath = tmp_path / "empty_text.jsonl"
        with open(filepath, 'w') as f:
            for line in lines:
                f.write(json.dumps(line) + '\n')
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        assert messages == []
