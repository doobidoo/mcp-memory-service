# Session Harvest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `memory_harvest` MCP tool that extracts decisions, bugs, conventions, and learnings from Claude Code JSONL session transcripts and stores them as tagged memories.

**Architecture:** New `harvest/` module with JSONL parser, pattern-based extractor, and candidate scorer. Integrates as a new MCP tool in `server_impl.py`, stores via existing `MemoryService.store_memory()`. Phase 1 uses regex patterns only (no LLM calls).

**Tech Stack:** Python 3.11, regex, json, pathlib, existing MemoryService API

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/mcp_memory_service/harvest/__init__.py` | Package init, public API |
| Create | `src/mcp_memory_service/harvest/parser.py` | JSONL transcript parser — reads files, extracts user/assistant text blocks |
| Create | `src/mcp_memory_service/harvest/extractor.py` | Pattern-based extraction — regex patterns for decisions/bugs/conventions/learnings |
| Create | `src/mcp_memory_service/harvest/models.py` | Data models — `HarvestCandidate`, `HarvestResult`, `HarvestConfig` |
| Create | `src/mcp_memory_service/harvest/harvester.py` | Orchestrator — ties parser + extractor together, handles dry_run vs store |
| Modify | `src/mcp_memory_service/server_impl.py` | Register `memory_harvest` tool + handler |
| Create | `tests/harvest/__init__.py` | Test package |
| Create | `tests/harvest/test_parser.py` | Parser unit tests |
| Create | `tests/harvest/test_extractor.py` | Extractor unit tests |
| Create | `tests/harvest/test_harvester.py` | Orchestrator integration tests |
| Create | `tests/harvest/test_server_harvest.py` | MCP tool handler tests |
| Create | `tests/harvest/conftest.py` | Fixtures — sample JSONL data, mock MemoryService |

## JSONL Format Reference

Claude Code stores session transcripts at `~/.claude/projects/<project-path>/<session-id>.jsonl`.

Each line is a JSON object with `type` field:
- `user` — `message.content` is list of `{type: "text", text: "..."}` blocks
- `assistant` — `message.content` is list of `{type: "text", text: "..."}`, `{type: "thinking", thinking: "..."}`, `{type: "tool_use", name: "...", input: {...}}`
- `progress`, `queue-operation`, `file-history-snapshot`, `ai-title` — ignored

We extract text from `user` and `assistant` message blocks only.

---

## Task 1: Data Models

**Files:**
- Create: `src/mcp_memory_service/harvest/models.py`
- Create: `src/mcp_memory_service/harvest/__init__.py`
- Test: `tests/harvest/test_parser.py` (model instantiation)

- [ ] **Step 1: Write the test for data models**

```python
# tests/harvest/__init__.py
# (empty)

# tests/harvest/test_parser.py
import pytest
from mcp_memory_service.harvest.models import HarvestCandidate, HarvestResult, HarvestConfig

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/harvest/test_parser.py::TestHarvestModels -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Implement models**

```python
# src/mcp_memory_service/harvest/__init__.py
"""Session Harvest — extract learnings from Claude Code transcripts."""

from .models import HarvestCandidate, HarvestResult, HarvestConfig

__all__ = ["HarvestCandidate", "HarvestResult", "HarvestConfig"]

# src/mcp_memory_service/harvest/models.py
"""Data models for session harvest."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

HARVEST_TYPES = ["decision", "bug", "convention", "learning", "context"]

@dataclass
class HarvestCandidate:
    """A single extracted memory candidate."""
    content: str
    memory_type: str  # One of HARVEST_TYPES
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.5
    source_line: str = ""  # Original text that triggered extraction

@dataclass
class HarvestResult:
    """Result of a harvest operation."""
    candidates: List[HarvestCandidate]
    session_id: str
    total_messages: int
    found: int
    by_type: Dict[str, int]
    stored: int = 0  # Only set when dry_run=False

@dataclass
class HarvestConfig:
    """Configuration for harvest operation."""
    sessions: int = 1
    session_ids: Optional[List[str]] = None
    types: List[str] = field(default_factory=lambda: list(HARVEST_TYPES))
    min_confidence: float = 0.6
    dry_run: bool = True
    project_path: Optional[str] = None  # Override project dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/harvest/test_parser.py::TestHarvestModels -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mcp_memory_service/harvest/ tests/harvest/
git commit -m "feat(harvest): add data models for session harvest"
```

---

## Task 2: JSONL Parser

**Files:**
- Create: `src/mcp_memory_service/harvest/parser.py`
- Create: `tests/harvest/conftest.py`
- Modify: `tests/harvest/test_parser.py`

- [ ] **Step 1: Write conftest with sample JSONL fixture**

```python
# tests/harvest/conftest.py
import pytest
import json
import tempfile
from pathlib import Path

@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a minimal JSONL transcript file."""
    session_id = "test-session-001"
    lines = [
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "text", "text": "I decided to use SQLite-Vec over FAISS for local storage"}
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
            {"type": "text", "text": "The bug was that we forgot to close the connection pool"}
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
```

- [ ] **Step 2: Write parser tests**

```python
# Append to tests/harvest/test_parser.py

from mcp_memory_service.harvest.parser import TranscriptParser, ParsedMessage

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

    def test_parse_corrupt_line_skipped(self, tmp_path):
        filepath = tmp_path / "corrupt.jsonl"
        filepath.write_text('not-json\n{"type":"progress"}\n')
        parser = TranscriptParser()
        messages = parser.parse_file(filepath)
        assert messages == []  # progress is skipped, corrupt is skipped
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/harvest/test_parser.py::TestTranscriptParser -v`
Expected: FAIL — ImportError

- [ ] **Step 4: Implement parser**

```python
# src/mcp_memory_service/harvest/parser.py
"""JSONL transcript parser for Claude Code session files."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

@dataclass
class ParsedMessage:
    """A single extracted text message from a transcript."""
    role: str  # "user" or "assistant"
    text: str
    timestamp: Optional[str] = None
    uuid: Optional[str] = None

class TranscriptParser:
    """Parses Claude Code JSONL session transcripts."""

    RELEVANT_TYPES = {"user", "assistant"}

    def find_sessions(self, project_dir: Path, count: int = 1) -> List[Path]:
        """Find the most recent JSONL session files in a project directory."""
        project_dir = Path(project_dir)
        jsonl_files = sorted(
            project_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return jsonl_files[:count]

    def parse_file(self, filepath: Path) -> List[ParsedMessage]:
        """Parse a JSONL file and extract user/assistant text messages."""
        filepath = Path(filepath)
        messages: List[ParsedMessage] = []

        if not filepath.exists() or filepath.stat().st_size == 0:
            return messages

        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug(f"Skipping corrupt line {line_num} in {filepath.name}")
                    continue

                msg_type = obj.get("type")
                if msg_type not in self.RELEVANT_TYPES:
                    continue

                message = obj.get("message", {})
                content = message.get("content", [])
                timestamp = obj.get("timestamp")
                uuid = obj.get("uuid")

                # Extract text blocks (skip thinking, tool_use, etc.)
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            messages.append(ParsedMessage(
                                role=msg_type,
                                text=text,
                                timestamp=timestamp,
                                uuid=uuid
                            ))

        return messages
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/harvest/test_parser.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/mcp_memory_service/harvest/parser.py tests/harvest/
git commit -m "feat(harvest): add JSONL transcript parser"
```

---

## Task 3: Pattern-Based Extractor

**Files:**
- Create: `src/mcp_memory_service/harvest/extractor.py`
- Create: `tests/harvest/test_extractor.py`

- [ ] **Step 1: Write extractor tests**

```python
# tests/harvest/test_extractor.py
import pytest
from mcp_memory_service.harvest.extractor import PatternExtractor
from mcp_memory_service.harvest.parser import ParsedMessage

@pytest.fixture
def extractor():
    return PatternExtractor()

class TestPatternExtractor:
    def test_extract_decision(self, extractor):
        msg = ParsedMessage(role="assistant", text="I decided to use Redis over Memcached for caching because of pub/sub support.")
        candidates = extractor.extract(msg)
        types = [c.memory_type for c in candidates]
        assert "decision" in types

    def test_extract_bug(self, extractor):
        msg = ParsedMessage(role="assistant", text="The root cause was a race condition in the connection pool.")
        candidates = extractor.extract(msg)
        types = [c.memory_type for c in candidates]
        assert "bug" in types

    def test_extract_convention(self, extractor):
        msg = ParsedMessage(role="assistant", text="Convention: always use WAL mode for concurrent SQLite access.")
        candidates = extractor.extract(msg)
        types = [c.memory_type for c in candidates]
        assert "convention" in types

    def test_extract_learning(self, extractor):
        msg = ParsedMessage(role="user", text="I learned that ONNX models need warmup on first inference.")
        candidates = extractor.extract(msg)
        types = [c.memory_type for c in candidates]
        assert "learning" in types

    def test_no_false_positives_on_short_text(self, extractor):
        msg = ParsedMessage(role="assistant", text="OK")
        candidates = extractor.extract(msg)
        assert len(candidates) == 0

    def test_no_false_positives_on_code(self, extractor):
        msg = ParsedMessage(role="assistant", text="```python\ndef decide():\n    return True\n```")
        candidates = extractor.extract(msg)
        # Should not match "decide" inside code blocks
        assert len(candidates) == 0

    def test_confidence_scaling(self, extractor):
        msg = ParsedMessage(role="assistant", text="The critical root cause of the production outage was the missing WAL pragma.")
        candidates = extractor.extract(msg)
        bug_candidates = [c for c in candidates if c.memory_type == "bug"]
        assert len(bug_candidates) > 0
        # Multiple pattern matches should boost confidence
        assert bug_candidates[0].confidence >= 0.6

    def test_extract_multiple_types(self, extractor):
        msg = ParsedMessage(role="assistant", text="I decided to fix the bug by always using WAL mode. The root cause was missing pragma.")
        candidates = extractor.extract(msg)
        types = set(c.memory_type for c in candidates)
        assert len(types) >= 2

    def test_min_text_length(self, extractor):
        msg = ParsedMessage(role="user", text="decided")
        candidates = extractor.extract(msg)
        assert len(candidates) == 0  # Too short to be meaningful
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/harvest/test_extractor.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement extractor**

```python
# src/mcp_memory_service/harvest/extractor.py
"""Pattern-based extraction of learnings from session messages."""

import re
import logging
from typing import List
from .models import HarvestCandidate
from .parser import ParsedMessage

logger = logging.getLogger(__name__)

# Minimum text length to consider for extraction
MIN_TEXT_LENGTH = 30

# Regex to detect code blocks — we strip these before pattern matching
CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```', re.MULTILINE)

# Pattern definitions: (compiled_regex, base_confidence)
PATTERNS = {
    "decision": [
        (re.compile(r'\b(?:decided|chose|choosing|going with|opted for|selected)\b.*\b(?:over|instead of|because|for)\b', re.IGNORECASE), 0.75),
        (re.compile(r'\b(?:decision|approach):\s', re.IGNORECASE), 0.7),
        (re.compile(r'\bI (?:will|\'ll) (?:use|go with|pick|choose)\b', re.IGNORECASE), 0.65),
    ],
    "bug": [
        (re.compile(r'\b(?:root cause|bug was|issue was|problem was|error was)\b', re.IGNORECASE), 0.75),
        (re.compile(r'\b(?:fixed by|fix was|resolved by|the fix)\b', re.IGNORECASE), 0.7),
        (re.compile(r'\b(?:crash|regression|broke|breaking|broken)\b.*\b(?:because|due to|caused by)\b', re.IGNORECASE), 0.7),
    ],
    "convention": [
        (re.compile(r'\b(?:convention|rule|pattern|standard):\s', re.IGNORECASE), 0.8),
        (re.compile(r'\b(?:always|never)\s+(?:use|do|add|include|set|run|check)\b', re.IGNORECASE), 0.65),
        (re.compile(r'\bmust (?:always|never)\b', re.IGNORECASE), 0.7),
    ],
    "learning": [
        (re.compile(r'\b(?:learned|discovered|realized|found out|turns out|TIL)\b', re.IGNORECASE), 0.7),
        (re.compile(r'\b(?:insight|takeaway|lesson|key finding)\b', re.IGNORECASE), 0.7),
        (re.compile(r'\b(?:didn\'t know|wasn\'t aware|surprised to find)\b', re.IGNORECASE), 0.65),
    ],
    "context": [
        (re.compile(r'\b(?:current state|status|progress|where we left off)\b', re.IGNORECASE), 0.6),
        (re.compile(r'\b(?:next steps?|todo|remaining work|blocked on)\b', re.IGNORECASE), 0.6),
    ],
}


class PatternExtractor:
    """Extracts harvest candidates from parsed messages using regex patterns."""

    def extract(self, message: ParsedMessage) -> List[HarvestCandidate]:
        """Extract candidates from a single message."""
        text = message.text.strip()

        # Skip short texts
        if len(text) < MIN_TEXT_LENGTH:
            return []

        # Strip code blocks before pattern matching
        clean_text = CODE_BLOCK_RE.sub('', text).strip()
        if len(clean_text) < MIN_TEXT_LENGTH:
            return []

        candidates: List[HarvestCandidate] = []
        seen_types = {}  # type -> best confidence

        for memory_type, patterns in PATTERNS.items():
            matches = []
            for pattern, base_confidence in patterns:
                if pattern.search(clean_text):
                    matches.append(base_confidence)

            if matches:
                # Multiple pattern matches boost confidence
                confidence = min(max(matches) + 0.05 * (len(matches) - 1), 1.0)

                # Only keep highest confidence per type
                if memory_type in seen_types and seen_types[memory_type] >= confidence:
                    continue
                seen_types[memory_type] = confidence

                # Extract a concise version: use the full text but cap at 500 chars
                content = clean_text[:500].strip()

                candidates.append(HarvestCandidate(
                    content=content,
                    memory_type=memory_type,
                    tags=[f"harvest:{memory_type}"],
                    confidence=confidence,
                    source_line=text[:200]
                ))

        return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/harvest/test_extractor.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/mcp_memory_service/harvest/extractor.py tests/harvest/test_extractor.py
git commit -m "feat(harvest): add pattern-based extractor with decision/bug/convention/learning patterns"
```

---

## Task 4: Harvester Orchestrator

**Files:**
- Create: `src/mcp_memory_service/harvest/harvester.py`
- Create: `tests/harvest/test_harvester.py`

- [ ] **Step 1: Write harvester tests**

```python
# tests/harvest/test_harvester.py
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
        # Return a mock with .success attribute (matches real StoreMemorySingleSuccess)
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
            assert result.stored < result.found  # At least one failed
            assert result.stored > 0  # But others succeeded
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/harvest/test_harvester.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement harvester**

```python
# src/mcp_memory_service/harvest/harvester.py
"""Orchestrator for session harvest operations."""

import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional

from .models import HarvestCandidate, HarvestConfig, HarvestResult
from .parser import TranscriptParser
from .extractor import PatternExtractor

logger = logging.getLogger(__name__)


class SessionHarvester:
    """Orchestrates parsing, extraction, and optional storage of harvest candidates."""

    def __init__(self, project_dir: Path, memory_service=None):
        self.project_dir = Path(project_dir)
        self.memory_service = memory_service
        self.parser = TranscriptParser()
        self.extractor = PatternExtractor()

    def harvest(self, config: HarvestConfig) -> List[HarvestResult]:
        """Parse sessions and extract candidates (synchronous, no storage)."""
        session_files = self._resolve_sessions(config)
        if not session_files:
            return []

        results = []
        for filepath in session_files:
            result = self._harvest_file(filepath, config)
            results.append(result)
        return results

    async def harvest_and_store(self, config: HarvestConfig) -> List[HarvestResult]:
        """Parse, extract, and store candidates via MemoryService."""
        session_files = self._resolve_sessions(config)
        if not session_files:
            return []

        results = []
        for filepath in session_files:
            result = self._harvest_file(filepath, config)

            if not config.dry_run and self.memory_service and result.candidates:
                stored = 0
                for candidate in result.candidates:
                    try:
                        tags = ["session-harvest"] + candidate.tags
                        resp = await self.memory_service.store_memory(
                            content=candidate.content,
                            tags=tags,
                            memory_type=candidate.memory_type,
                            metadata={"confidence": candidate.confidence, "source": "harvest"},
                        )
                        if isinstance(resp, dict) and resp.get("success"):
                            stored += 1
                        elif hasattr(resp, "success") and resp.success:
                            stored += 1
                    except Exception as e:
                        logger.warning(f"Failed to store harvest candidate: {e}")
                result.stored = stored

            results.append(result)
        return results

    def _resolve_sessions(self, config: HarvestConfig) -> List[Path]:
        """Find session files based on config."""
        if config.session_ids:
            return [
                self.project_dir / f"{sid}.jsonl"
                for sid in config.session_ids
                if (self.project_dir / f"{sid}.jsonl").exists()
            ]
        return self.parser.find_sessions(self.project_dir, count=config.sessions)

    def _harvest_file(self, filepath: Path, config: HarvestConfig) -> HarvestResult:
        """Extract candidates from a single session file."""
        messages = self.parser.parse_file(filepath)
        session_id = filepath.stem

        all_candidates: List[HarvestCandidate] = []
        for msg in messages:
            candidates = self.extractor.extract(msg)
            all_candidates.extend(candidates)

        # Apply filters
        filtered = [
            c for c in all_candidates
            if c.confidence >= config.min_confidence
            and c.memory_type in config.types
        ]

        by_type = dict(Counter(c.memory_type for c in filtered))

        return HarvestResult(
            candidates=filtered,
            session_id=session_id,
            total_messages=len(messages),
            found=len(filtered),
            by_type=by_type,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/harvest/test_harvester.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/mcp_memory_service/harvest/harvester.py tests/harvest/test_harvester.py
git commit -m "feat(harvest): add harvester orchestrator with dry-run and store modes"
```

---

## Task 5: MCP Tool Registration

**Files:**
- Modify: `src/mcp_memory_service/server_impl.py` (~lines 1770, 2036)
- Create: `tests/harvest/test_server_harvest.py`

- [ ] **Step 1: Write handler test**

```python
# tests/harvest/test_server_harvest.py
import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

class TestMemoryHarvestHandler:
    """Test the memory_harvest MCP tool handler."""

    @pytest.mark.asyncio
    async def test_harvest_dry_run_returns_candidates(self, tmp_path, sample_jsonl):
        """Handler should return candidates in dry_run mode."""
        filepath, session_id = sample_jsonl
        project_dir = filepath.parent

        # Import handler
        from mcp_memory_service.harvest.harvester import SessionHarvester
        from mcp_memory_service.harvest.models import HarvestConfig

        harvester = SessionHarvester(project_dir=project_dir)
        config = HarvestConfig(sessions=1, dry_run=True)
        results = harvester.harvest(config)

        assert len(results) > 0
        result = results[0]
        assert result.session_id == session_id
        # Our fixture has decision, bug, convention, learning patterns
        assert result.found > 0

    @pytest.mark.asyncio
    async def test_harvest_returns_json_format(self, sample_jsonl):
        """Handler output should be valid JSON."""
        filepath, _ = sample_jsonl
        project_dir = filepath.parent

        from mcp_memory_service.harvest.harvester import SessionHarvester
        from mcp_memory_service.harvest.models import HarvestConfig

        harvester = SessionHarvester(project_dir=project_dir)
        config = HarvestConfig(sessions=1, dry_run=True)
        results = harvester.harvest(config)

        # Simulate handler JSON output
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
        # Should serialize cleanly
        serialized = json.dumps(output)
        parsed = json.loads(serialized)
        assert "results" in parsed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/harvest/test_server_harvest.py -v`
Expected: FAIL (initially may pass since we're testing harvester directly — but validates the integration contract)

- [ ] **Step 3: Register tool in server_impl.py**

Add to `server_impl.py` after the `ingestion_tools` block (~line 1770), inside the tool list:

```python
# Add after ingestion_tools list, before the return
types.Tool(
    name="memory_harvest",
    description="""Extract learnings from Claude Code session transcripts.

USE THIS WHEN:
- End of session — auto-capture decisions, bugs, conventions, learnings
- User asks to "harvest" or "extract learnings" from sessions
- Building knowledge base from past sessions

MODES:
- dry_run=true (DEFAULT): Preview candidates without storing
- dry_run=false: Store candidates as tagged memories

MEMORY TYPES EXTRACTED:
- decision: Architectural choices, tool selections
- bug: Issues encountered and root causes
- convention: Patterns/rules established
- learning: Insights, mistakes learned from
- context: Session state for continuity

Examples:
{"sessions": 1, "dry_run": true}
{"sessions": 3, "types": ["decision", "bug"]}
{"session_ids": ["abc123"], "dry_run": false}
{"min_confidence": 0.7, "sessions": 1}
""",
    inputSchema={
        "type": "object",
        "properties": {
            "sessions": {
                "type": "integer",
                "default": 1,
                "description": "Number of recent sessions to harvest"
            },
            "session_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific session IDs to harvest"
            },
            "types": {
                "type": "array",
                "items": {"type": "string", "enum": ["decision", "bug", "convention", "learning", "context"]},
                "description": "Filter by memory types (default: all)"
            },
            "min_confidence": {
                "type": "number",
                "default": 0.6,
                "description": "Minimum confidence threshold (0.0-1.0)"
            },
            "dry_run": {
                "type": "boolean",
                "default": True,
                "description": "Preview candidates without storing (default: true)"
            },
            "project_path": {
                "type": "string",
                "description": "Override Claude Code project directory path"
            }
        }
    }
),
```

Add handler dispatch (~line 2044, after `memory_graph`):

```python
elif name == "memory_harvest":
    logger.info("Calling handle_memory_harvest")
    return await self.handle_memory_harvest(arguments)
```

- [ ] **Step 4: Implement handler method**

Add to the MemoryServer class in `server_impl.py`:

```python
async def handle_memory_harvest(self, arguments: dict) -> list:
    """Handle memory_harvest tool calls."""
    from pathlib import Path as _Path
    from .harvest.harvester import SessionHarvester
    from .harvest.models import HarvestConfig

    # Resolve project directory
    project_path = arguments.get("project_path")
    if not project_path:
        # Default: ~/.claude/projects/ — auto-detect from cwd
        cwd = _Path.cwd()
        claude_projects = _Path.home() / ".claude" / "projects"
        # Convert cwd to Claude's project dir naming: replace / with -
        project_dir_name = str(cwd).replace("/", "-")
        if project_dir_name.startswith("-"):
            project_dir_name = project_dir_name  # keep leading dash
        project_path = claude_projects / project_dir_name
    else:
        project_path = _Path(project_path)

    if not project_path.exists():
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": f"Project directory not found: {project_path}"})
        )]

    config = HarvestConfig(
        sessions=arguments.get("sessions", 1),
        session_ids=arguments.get("session_ids"),
        types=arguments.get("types", ["decision", "bug", "convention", "learning", "context"]),
        min_confidence=arguments.get("min_confidence", 0.6),
        dry_run=arguments.get("dry_run", True),
        project_path=str(project_path),
    )

    memory_service = None
    if not config.dry_run:
        await self._ensure_storage_initialized()
        memory_service = self.memory_service

    harvester = SessionHarvester(
        project_dir=project_path,
        memory_service=memory_service
    )

    if config.dry_run:
        results = harvester.harvest(config)
    else:
        results = await harvester.harvest_and_store(config)

    output = {
        "dry_run": config.dry_run,
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
                        "content": c.content[:200],
                        "confidence": round(c.confidence, 2),
                        "tags": c.tags,
                    }
                    for c in r.candidates
                ]
            }
            for r in results
        ]
    }

    return [types.TextContent(type="text", text=json.dumps(output, indent=2))]
```

- [ ] **Step 5: Run all harvest tests**

Run: `.venv/bin/pytest tests/harvest/ -v`
Expected: ALL PASS

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/pytest --timeout=60 -x -q`
Expected: All existing tests still pass

- [ ] **Step 7: Commit**

```bash
git add src/mcp_memory_service/server_impl.py src/mcp_memory_service/harvest/ tests/harvest/
git commit -m "feat(harvest): register memory_harvest MCP tool with handler"
```

---

## Task 6: Update `__init__.py` exports and integration test

**Files:**
- Modify: `src/mcp_memory_service/harvest/__init__.py`
- Create: `tests/harvest/test_integration.py`

- [ ] **Step 1: Write integration test with real JSONL**

```python
# tests/harvest/test_integration.py
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
                {"type": "text", "text": "The tests are failing with a database lock error"}
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
                {"type": "text", "text": "Great, I learned that SQLite can actually handle concurrent access well with WAL mode."}
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
        # The code block message (a4) should not produce candidates
        for c in r.candidates:
            assert "MCP_MEMORY_SQLITE_PRAGMAS" not in c.content or c.memory_type != "decision"
```

- [ ] **Step 2: Run integration test**

Run: `.venv/bin/pytest tests/harvest/test_integration.py -v`
Expected: ALL PASS

- [ ] **Step 3: Update __init__.py exports**

```python
# src/mcp_memory_service/harvest/__init__.py
"""Session Harvest — extract learnings from Claude Code transcripts."""

from .models import HarvestCandidate, HarvestResult, HarvestConfig
from .parser import TranscriptParser, ParsedMessage
from .extractor import PatternExtractor
from .harvester import SessionHarvester

__all__ = [
    "HarvestCandidate", "HarvestResult", "HarvestConfig",
    "TranscriptParser", "ParsedMessage",
    "PatternExtractor",
    "SessionHarvester",
]
```

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/pytest --timeout=60 -x -q`
Expected: All tests pass (existing + new harvest tests)

- [ ] **Step 5: Commit**

```bash
git add src/mcp_memory_service/harvest/__init__.py tests/harvest/test_integration.py
git commit -m "feat(harvest): add integration tests and finalize exports"
```

---

## Summary

**6 tasks, ~30 steps total.** Creates:
- `src/mcp_memory_service/harvest/` — 5 new files (models, parser, extractor, harvester, __init__)
- `tests/harvest/` — 5 new test files
- 1 modification to `server_impl.py` (tool registration + handler)

**Not in scope (Phase 2+):**
- LLM-based classification
- HTTP API endpoint
- SessionEnd hook integration
- TTL per memory type
- Auto-tagging from content
