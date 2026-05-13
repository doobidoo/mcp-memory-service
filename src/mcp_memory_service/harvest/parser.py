"""JSONL transcript parser for Claude Code and Kiro CLI session files."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
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
    """Parses JSONL session transcripts from Claude Code and Kiro CLI."""

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
        """Parse a JSONL file and extract user/assistant text messages.

        Auto-detects format per line:
        - Kiro CLI: {"version": "v1", "kind": "Prompt"|"AssistantMessage"|"ToolResults", ...}
        - Claude Code: {"type": "user"|"assistant", "message": {...}, ...}
        """
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

                if "version" in obj and "kind" in obj:
                    msg = self._parse_kiro_line(obj)
                elif "type" in obj:
                    msg = self._parse_claude_code_line(obj)
                else:
                    continue

                if msg:
                    messages.append(msg)

        return messages

    def _parse_kiro_line(self, obj: dict) -> Optional[ParsedMessage]:
        """Parse a Kiro CLI JSONL line.

        Format:
          {"version": "v1", "kind": "Prompt", "data": {"message_id": "...",
           "content": [{"kind": "text", "data": "..."}],
           "meta": {"timestamp": 1778593355}}}
        """
        kind = obj.get("kind")
        if kind not in ("Prompt", "AssistantMessage"):
            return None

        data = obj.get("data", {})
        message_id = data.get("message_id")
        content = data.get("content", [])
        meta = data.get("meta", {})

        role = "user" if kind == "Prompt" else "assistant"

        # Extract timestamp (Unix epoch → ISO string)
        timestamp = None
        raw_ts = meta.get("timestamp")
        if isinstance(raw_ts, (int, float)):
            timestamp = datetime.fromtimestamp(raw_ts, tz=timezone.utc).isoformat()

        # Extract text blocks only (skip toolUse)
        texts = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("kind") == "text":
                    text = block.get("data", "")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())

        combined = "\n".join(texts).strip()
        if combined and not self._is_system_content(combined):
            return ParsedMessage(
                role=role, text=combined, timestamp=timestamp, uuid=message_id
            )
        return None

    def _parse_claude_code_line(self, obj: dict) -> Optional[ParsedMessage]:
        """Parse a Claude Code JSONL line.

        Format:
          {"type": "user"|"assistant", "message": {"content": [{"type": "text", "text": "..."}]},
           "timestamp": "...", "uuid": "..."}
        """
        msg_type = obj.get("type")
        if msg_type not in self.RELEVANT_TYPES:
            return None

        message = obj.get("message", {})
        content = message.get("content", [])
        timestamp = obj.get("timestamp")
        uuid = obj.get("uuid")

        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").strip()
                if text and not self._is_system_content(text):
                    texts.append(text)

        if texts:
            return ParsedMessage(
                role=msg_type,
                text="\n".join(texts),
                timestamp=timestamp,
                uuid=uuid
            )
        return None

    @staticmethod
    def _is_system_content(text: str) -> bool:
        """Filter out system prompts, skill outputs, and injected content."""
        # System reminder tags injected by Claude Code
        if "<system-reminder>" in text or "</system-reminder>" in text:
            return True
        # Skill/command outputs (e.g. /release, /commit)
        if "<command-name>" in text or "<command-message>" in text:
            return True
        # IDE context injections
        if text.startswith("<ide_opened_file>"):
            return True
        # Very long blocks (>2000 chars) are typically skill definitions, not conversation
        if len(text) > 2000:
            return True
        return False
