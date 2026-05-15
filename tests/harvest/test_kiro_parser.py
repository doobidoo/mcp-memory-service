"""Tests for Kiro CLI session format support in TranscriptParser."""

import json
import tempfile
from pathlib import Path

import pytest

from mcp_memory_service.harvest.parser import TranscriptParser


# Real Kiro CLI JSONL fixture (anonymized)
KIRO_SESSION_LINES = [
    {"version": "v1", "kind": "Prompt", "data": {"content": [{"kind": "text", "data": "Implement the login page"}]}, "timestamp": "2026-05-15T10:00:00Z", "uuid": "p1"},
    {"version": "v1", "kind": "Response", "data": {"content": [{"kind": "text", "data": "I'll create the login component with email and password fields."}]}, "timestamp": "2026-05-15T10:00:05Z", "uuid": "r1"},
    {"version": "v1", "kind": "Prompt", "data": {"content": [{"kind": "text", "data": "Add validation"}]}, "timestamp": "2026-05-15T10:01:00Z", "uuid": "p2"},
    {"version": "v1", "kind": "Response", "data": {"content": [{"kind": "text", "data": "Added email format validation and required field checks."}]}, "timestamp": "2026-05-15T10:01:10Z", "uuid": "r2"},
]

# Kiro with plain string content (alternative format)
KIRO_STRING_CONTENT = [
    {"version": "v1", "kind": "Prompt", "data": {"content": "Fix the bug"}, "timestamp": "2026-05-15T11:00:00Z"},
    {"version": "v1", "kind": "Response", "data": {"content": "Done, the null check was missing."}, "timestamp": "2026-05-15T11:00:05Z"},
]

# Claude Code format for comparison
CLAUDE_SESSION_LINES = [
    {"type": "user", "message": {"content": [{"type": "text", "text": "Hello"}]}, "timestamp": "2026-05-15T09:00:00Z", "uuid": "c1"},
    {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hi there!"}]}, "timestamp": "2026-05-15T09:00:01Z", "uuid": "c2"},
]

UNKNOWN_FORMAT_LINES = [
    {"foo": "bar", "baz": 123},
]


def _write_jsonl(lines, tmpdir, filename="session.jsonl"):
    path = Path(tmpdir) / filename
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


class TestKiroParser:
    def setup_method(self):
        self.parser = TranscriptParser()

    def test_parse_kiro_session(self, tmp_path):
        path = _write_jsonl(KIRO_SESSION_LINES, tmp_path)
        messages = self.parser.parse_file(path)
        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[0].text == "Implement the login page"
        assert messages[1].role == "assistant"
        assert messages[1].text == "I'll create the login component with email and password fields."
        assert messages[2].role == "user"
        assert messages[3].role == "assistant"

    def test_parse_kiro_preserves_metadata(self, tmp_path):
        path = _write_jsonl(KIRO_SESSION_LINES, tmp_path)
        messages = self.parser.parse_file(path)
        assert messages[0].timestamp == "2026-05-15T10:00:00Z"
        assert messages[0].uuid == "p1"

    def test_parse_kiro_string_content(self, tmp_path):
        path = _write_jsonl(KIRO_STRING_CONTENT, tmp_path)
        messages = self.parser.parse_file(path)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].text == "Fix the bug"
        assert messages[1].role == "assistant"

    def test_parse_claude_still_works(self, tmp_path):
        path = _write_jsonl(CLAUDE_SESSION_LINES, tmp_path)
        messages = self.parser.parse_file(path)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].text == "Hello"
        assert messages[1].role == "assistant"

    def test_unknown_format_returns_empty(self, tmp_path):
        path = _write_jsonl(UNKNOWN_FORMAT_LINES, tmp_path)
        messages = self.parser.parse_file(path)
        assert messages == []

    def test_empty_file_returns_empty(self, tmp_path):
        path = Path(tmp_path) / "empty.jsonl"
        path.write_text("")
        messages = self.parser.parse_file(path)
        assert messages == []

    def test_kiro_skips_non_prompt_response(self, tmp_path):
        lines = [
            {"version": "v1", "kind": "ToolUse", "data": {"content": [{"kind": "text", "data": "internal"}]}},
            {"version": "v1", "kind": "Prompt", "data": {"content": [{"kind": "text", "data": "real message"}]}},
        ]
        path = _write_jsonl(lines, tmp_path)
        messages = self.parser.parse_file(path)
        assert len(messages) == 1
        assert messages[0].text == "real message"
