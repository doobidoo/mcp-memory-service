"""Tests for Kiro CLI v4 (payload-wrapped) parser support.

These tests ensure the TranscriptParser can handle the new kiro-cli-v4 format
where messages have the structure:
{"id": "...", "timestamp": "...", "payload": {"type": "...", "content": "...", ...}}

The new format differs from legacy formats:
- Legacy Claude: {"type": "...", "message": {...}} (type at top level)
- Legacy Kiro: {"kind": "...", "data": {...}} (kind at top level)  
- New Kiro CLI v4: {"payload": {"type": "..."}} (type inside payload)
"""

import json
import pytest
from pathlib import Path
from mcp_memory_service.harvest.parser import TranscriptParser, ParsedMessage


@pytest.fixture
def kiro_v4_messages_file(tmp_path):
    """Create a messages.jsonl file with kiro-cli-v4 payload-wrapped format."""
    messages = [
        {
            "id": "msg-001-user",
            "timestamp": "2026-01-01T10:00:00Z",
            "payload": {
                "type": "user",
                "content": "Hello, can you help me with Python?",
                "images": [],
                "documents": [],
                "_meta": {"source": "cli"}
            }
        },
        {
            "id": "msg-002-assistant", 
            "timestamp": "2026-01-01T10:00:05Z",
            "payload": {
                "type": "assistant",
                "content": "Of course! I'd be happy to help you with Python programming.",
                "operationType": "text_generation",
                "executionId": "exec-001",
                "_meta": {"model": "claude-3"}
            }
        },
        {
            "id": "msg-003-tool-call",
            "timestamp": "2026-01-01T10:00:10Z", 
            "payload": {
                "type": "tool_call",
                "toolName": "code_execution",
                "args": {"language": "python", "code": "print('hello')"},
                "actionType": "execute",
                "status": "pending",
                "toolCallId": "tc-001"
            }
        },
        {
            "id": "msg-004-tool-result",
            "timestamp": "2026-01-01T10:00:12Z",
            "payload": {
                "type": "tool_result", 
                "content": "hello\n",
                "success": True,
                "durationMs": 150,
                "toolCallId": "tc-001",
                "executionId": "exec-001"
            }
        },
        {
            "id": "msg-005-metadata",
            "timestamp": "2026-01-01T10:00:15Z",
            "payload": {
                "type": "session_metadata",
                "sessionId": "sess-001",
                "metadata": {"version": "4.0.0"}
            }
        },
        {
            "id": "msg-006-user-empty",
            "timestamp": "2026-01-01T10:00:20Z",
            "payload": {
                "type": "user",
                "content": "",  # Empty content should be filtered out
                "_meta": {}
            }
        },
        {
            "id": "msg-007-tool-result-rich",
            "timestamp": "2026-01-01T10:00:25Z",
            "payload": {
                "type": "tool_result",
                "content": '{"status": "success", "files_created": ["test.py"], "output": "File created successfully"}',
                "success": True,
                "durationMs": 500,
                "toolCallId": "tc-002"
            }
        }
    ]
    
    jsonl_file = tmp_path / "messages.jsonl"
    with open(jsonl_file, 'w') as f:
        for msg in messages:
            f.write(json.dumps(msg) + '\n')
    
    return jsonl_file


@pytest.fixture
def claude_format_file(tmp_path):
    """Create a messages.jsonl file with legacy Claude format (type at top level)."""
    messages = [
        {
            "type": "user",
            "timestamp": "2026-01-01T09:00:00Z",
            "uuid": "claude-user-001",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello Claude"}
                ]
            }
        },
        {
            "type": "assistant", 
            "timestamp": "2026-01-01T09:00:05Z",
            "uuid": "claude-asst-001",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello! How can I help you?"}
                ]
            }
        }
    ]
    
    jsonl_file = tmp_path / "claude_messages.jsonl"
    with open(jsonl_file, 'w') as f:
        for msg in messages:
            f.write(json.dumps(msg) + '\n')
    
    return jsonl_file


@pytest.fixture 
def kiro_legacy_file(tmp_path):
    """Create a messages.jsonl file with legacy Kiro format (kind at top level)."""
    messages = [
        {
            "kind": "Prompt",
            "timestamp": "2026-01-01T08:00:00Z",
            "uuid": "kiro-prompt-001",
            "data": {
                "content": "Hello legacy Kiro"
            }
        },
        {
            "kind": "Response",
            "timestamp": "2026-01-01T08:00:05Z", 
            "uuid": "kiro-response-001",
            "data": {
                "content": "Hello! This is legacy Kiro format."
            }
        }
    ]
    
    jsonl_file = tmp_path / "kiro_legacy_messages.jsonl"
    with open(jsonl_file, 'w') as f:
        for msg in messages:
            f.write(json.dumps(msg) + '\n')
    
    return jsonl_file


class TestKiroCliV4Parser:
    """Tests for the new kiro-cli-v4 payload-wrapped format parser."""

    def test_detects_kiro_cli_v4_format(self, kiro_v4_messages_file):
        """Test that parser detects and processes kiro-cli-v4 format successfully.
        
        Should return >0 ParsedMessage objects instead of empty list with 
        "Unknown session format, skipping" warning.
        """
        parser = TranscriptParser()
        messages = parser.parse_file(kiro_v4_messages_file)
        
        # This MUST fail (RED) because kiro-cli-v4 detection doesn't exist yet
        assert len(messages) > 0, "Should detect and parse kiro-cli-v4 format, not skip as unknown"

    def test_extracts_user_and_assistant_messages(self, kiro_v4_messages_file):
        """Test that user and assistant messages are extracted with correct roles and content."""
        parser = TranscriptParser()
        messages = parser.parse_file(kiro_v4_messages_file)
        
        # Find the user message
        user_messages = [m for m in messages if m.role == "user"]
        assert len(user_messages) == 1, "Should extract exactly one user message (empty content filtered out)"
        
        user_msg = user_messages[0]
        assert user_msg.text == "Hello, can you help me with Python?"
        assert user_msg.timestamp == "2026-01-01T10:00:00Z"
        assert user_msg.uuid == "msg-001-user"
        
        # Find the assistant message
        assistant_messages = [m for m in messages if m.role == "assistant"]
        assert len(assistant_messages) >= 1, "Should extract at least one assistant message"
        
        # Check the direct assistant message (not tool_result)
        direct_assistant = [m for m in assistant_messages if "Of course!" in m.text]
        assert len(direct_assistant) == 1
        
        asst_msg = direct_assistant[0]
        assert asst_msg.text == "Of course! I'd be happy to help you with Python programming."
        assert asst_msg.timestamp == "2026-01-01T10:00:05Z"
        assert asst_msg.uuid == "msg-002-assistant"

    def test_tool_result_captured_as_assistant(self, kiro_v4_messages_file):
        """Test that tool_result messages are captured as assistant messages with rich content."""
        parser = TranscriptParser()
        messages = parser.parse_file(kiro_v4_messages_file)
        
        # Find tool_result messages converted to assistant
        tool_result_messages = [
            m for m in messages 
            if m.role == "assistant" and ("hello\n" in m.text or "files_created" in m.text)
        ]
        
        assert len(tool_result_messages) == 2, "Should capture both tool_result messages"
        
        # Check simple tool result
        simple_result = [m for m in tool_result_messages if "hello\n" in m.text][0]
        assert simple_result.text == "hello\n"
        assert simple_result.timestamp == "2026-01-01T10:00:12Z"
        assert simple_result.uuid == "msg-004-tool-result"
        
        # Check rich JSON tool result
        rich_result = [m for m in tool_result_messages if "files_created" in m.text][0]
        expected_content = '{"status": "success", "files_created": ["test.py"], "output": "File created successfully"}'
        assert rich_result.text == expected_content
        assert rich_result.timestamp == "2026-01-01T10:00:25Z"
        assert rich_result.uuid == "msg-007-tool-result-rich"

    def test_tool_call_and_metadata_dropped_but_counted_in_coverage(self, kiro_v4_messages_file):
        """Test that tool_call and session_metadata are not extracted but are counted in coverage."""
        parser = TranscriptParser()
        messages = parser.parse_file(kiro_v4_messages_file)
        
        # Verify no tool_call or session_metadata messages were extracted
        extracted_types = set()
        for msg in messages:
            # All extracted messages should be user or assistant
            assert msg.role in ["user", "assistant"]
        
        # Check coverage report shows these types were seen but dropped
        coverage = parser.coverage_report()
        
        # tool_call should be seen but not extracted
        assert "tool_call" in coverage
        tool_call_stats = coverage["tool_call"]
        assert tool_call_stats["seen"] == 1
        assert tool_call_stats["extracted"] == 0
        assert tool_call_stats["dropped"] == 1
        
        # session_metadata should be seen but not extracted  
        assert "session_metadata" in coverage
        metadata_stats = coverage["session_metadata"]
        assert metadata_stats["seen"] == 1
        assert metadata_stats["extracted"] == 0
        assert metadata_stats["dropped"] == 1

    def test_coverage_report_populated_for_all_types(self, kiro_v4_messages_file):
        """Test that coverage_report() contains entries for all payload types encountered."""
        parser = TranscriptParser()
        messages = parser.parse_file(kiro_v4_messages_file)
        
        coverage = parser.coverage_report()
        
        # Should have coverage entries for all types in the file
        expected_types = ["user", "assistant", "tool_call", "tool_result", "session_metadata"]
        for msg_type in expected_types:
            assert msg_type in coverage, f"Coverage should track {msg_type}"
            stats = coverage[msg_type]
            assert "seen" in stats and stats["seen"] > 0
            assert "extracted" in stats 
            assert "dropped" in stats
            assert stats["seen"] == stats["extracted"] + stats["dropped"]
        
        # Verify correct extraction counts
        assert coverage["user"]["extracted"] == 1  # One non-empty user message
        assert coverage["assistant"]["extracted"] == 1  # One direct assistant message  
        assert coverage["tool_result"]["extracted"] == 2  # Two tool results
        assert coverage["tool_call"]["extracted"] == 0  # Tool calls not extracted
        assert coverage["session_metadata"]["extracted"] == 0  # Metadata not extracted

    def test_old_claude_format_still_works(self, claude_format_file):
        """Test that legacy Claude format (type at top level) continues to parse without regression."""
        parser = TranscriptParser()
        messages = parser.parse_file(claude_format_file)
        
        # Should still extract Claude format messages
        assert len(messages) == 2
        
        user_msg = [m for m in messages if m.role == "user"][0]
        assert user_msg.text == "Hello Claude"
        assert user_msg.uuid == "claude-user-001"
        
        asst_msg = [m for m in messages if m.role == "assistant"][0]
        assert asst_msg.text == "Hello! How can I help you?"
        assert asst_msg.uuid == "claude-asst-001"

    def test_old_kiro_format_still_works(self, kiro_legacy_file):
        """Test that legacy Kiro format (kind at top level) continues to parse without regression."""
        parser = TranscriptParser()
        messages = parser.parse_file(kiro_legacy_file)
        
        # Should still extract legacy Kiro format messages  
        assert len(messages) == 2
        
        user_msg = [m for m in messages if m.role == "user"][0] 
        assert user_msg.text == "Hello legacy Kiro"
        assert user_msg.uuid == "kiro-prompt-001"
        
        asst_msg = [m for m in messages if m.role == "assistant"][0]
        assert asst_msg.text == "Hello! This is legacy Kiro format."
        assert asst_msg.uuid == "kiro-response-001"

    def test_empty_and_system_content_filtering(self, tmp_path):
        """Test that empty content and system content are properly filtered out."""
        messages = [
            {
                "id": "msg-empty",
                "timestamp": "2026-01-01T10:00:00Z",
                "payload": {"type": "user", "content": ""}  # Empty - should be filtered
            },
            {
                "id": "msg-whitespace", 
                "timestamp": "2026-01-01T10:00:05Z",
                "payload": {"type": "user", "content": "   \n  "}  # Whitespace only - should be filtered
            },
            {
                "id": "msg-system",
                "timestamp": "2026-01-01T10:00:10Z", 
                "payload": {"type": "assistant", "content": "<system-reminder>System prompt here</system-reminder>"}
            },
            {
                "id": "msg-valid",
                "timestamp": "2026-01-01T10:00:15Z",
                "payload": {"type": "user", "content": "This should be kept"}
            }
        ]
        
        jsonl_file = tmp_path / "filter_test.jsonl"
        with open(jsonl_file, 'w') as f:
            for msg in messages:
                f.write(json.dumps(msg) + '\n')
        
        parser = TranscriptParser()
        result_messages = parser.parse_file(jsonl_file)
        
        # Should only extract the valid message
        assert len(result_messages) == 1
        assert result_messages[0].text == "This should be kept"


    def test_malformed_payload_does_not_abort_run(self, tmp_path):
        """Greptile P1: a record with null/non-dict payload or unhashable type must
        not raise and abort harvesting — the remaining records must still parse."""
        messages = [
            {"id": "m1", "timestamp": "2026-01-01T10:00:00Z",
             "payload": {"type": "user", "content": "first valid"}},
            {"id": "m2", "timestamp": "2026-01-01T10:00:01Z", "payload": None},
            {"id": "m3", "timestamp": "2026-01-01T10:00:02Z", "payload": "not-an-object"},
            {"id": "m4", "timestamp": "2026-01-01T10:00:03Z",
             "payload": {"type": ["array", "type"], "content": "weird"}},
            {"id": "m5", "timestamp": "2026-01-01T10:00:04Z",
             "payload": {"type": "assistant", "content": "last valid"}},
        ]
        jsonl_file = tmp_path / "malformed.jsonl"
        with open(jsonl_file, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        parser = TranscriptParser()
        # Must not raise, and must still get the two valid records.
        result = parser.parse_file(jsonl_file)
        texts = [m.text for m in result]
        assert "first valid" in texts
        assert "last valid" in texts

    def test_tool_result_applies_system_content_filter(self, tmp_path):
        """Greptile P1: a tool_result carrying a system-reminder must NOT become a
        harvested memory — the same _is_system_content filter as user/assistant."""
        messages = [
            {"id": "t1", "timestamp": "2026-01-01T10:00:00Z",
             "payload": {"type": "tool_result",
                         "content": "<system-reminder>do not harvest me</system-reminder>"}},
            {"id": "t2", "timestamp": "2026-01-01T10:00:01Z",
             "payload": {"type": "tool_result", "content": "SELECT count(*) -> 42 rows"}},
        ]
        jsonl_file = tmp_path / "toolresult_filter.jsonl"
        with open(jsonl_file, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        parser = TranscriptParser()
        result = parser.parse_file(jsonl_file)
        texts = [m.text for m in result]
        assert "SELECT count(*) -> 42 rows" in texts
        assert not any("system-reminder" in t for t in texts)
    def test_long_tool_result_is_kept_not_dropped(self, tmp_path):
        """Greptile P1 (round 2): a tool_result over 10k chars is rich data (query
        dump / report), not injected context — it must be harvested (truncated if
        huge), NOT dropped by the length cutoff that applies to user/assistant."""
        big = "row data " * 2000  # ~18k chars, well over the 10k user/assistant cutoff
        messages = [
            {"id": "big1", "timestamp": "2026-01-01T10:00:00Z",
             "payload": {"type": "tool_result", "content": big}},
            {"id": "inj1", "timestamp": "2026-01-01T10:00:01Z",
             "payload": {"type": "tool_result",
                         "content": "<system-reminder>injected</system-reminder>"}},
        ]
        jsonl_file = tmp_path / "long_toolresult.jsonl"
        with open(jsonl_file, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        parser = TranscriptParser()
        result = parser.parse_file(jsonl_file)
        # The big tool result is kept (harvested) verbatim, the injected one dropped.
        # No parser-side truncation: the extractor caps each candidate downstream, so
        # nothing after an arbitrary cutoff is silently lost (Greptile P1 r3).
        assert len(result) == 1
        assert result[0].text == big
