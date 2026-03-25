import pytest
import json


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a minimal JSONL transcript file."""
    session_id = "test-session-001"
    lines = [
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "text", "text": "I decided to use SQLite-Vec over FAISS for local storage because it supports WAL mode"}
        ]}, "timestamp": "2026-03-25T10:00:00Z", "sessionId": session_id, "uuid": "u1"},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Let me consider the options..."},
            {"type": "text", "text": "Good choice. The root cause of the previous crash was FAISS not supporting concurrent access. I learned that SQLite-Vec handles WAL mode natively."}
        ]}, "timestamp": "2026-03-25T10:01:00Z", "sessionId": session_id, "uuid": "a1"},
        {"type": "progress", "data": {}, "timestamp": "2026-03-25T10:01:01Z", "sessionId": session_id},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Convention: always use journal_mode=WAL for concurrent SQLite access."}
        ]}, "timestamp": "2026-03-25T10:02:00Z", "sessionId": session_id, "uuid": "a2"},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "text", "text": "The bug was that we forgot to close the connection pool, which caused database locks"}
        ]}, "timestamp": "2026-03-25T10:03:00Z", "sessionId": session_id, "uuid": "u2"},
    ]
    filepath = tmp_path / f"{session_id}.jsonl"
    with open(filepath, 'w') as f:
        for line in lines:
            f.write(json.dumps(line) + '\n')
    return filepath, session_id


@pytest.fixture
def sample_project_dir(sample_jsonl):
    """Return the directory containing the sample JSONL."""
    filepath, session_id = sample_jsonl
    return filepath.parent
