"""JSONL transcript parser for Claude Code and Kiro CLI session files."""

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
    """Parses JSONL session transcripts (Claude Code and Kiro CLI).

    Format is detected from the first message in the file:
    - If "type" key exists → Claude Code format
    - If "kind" key exists → Kiro CLI format
    - Unknown → warning logged, returns empty
    """

    RELEVANT_TYPES = {"user", "assistant"}
    KIRO_KIND_MAP = {"Prompt": "user", "Response": "assistant"}

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

        Auto-detects format (Claude Code vs Kiro CLI) from first message.
        """
        filepath = Path(filepath)
        messages: List[ParsedMessage] = []

        if not filepath.exists() or filepath.stat().st_size == 0:
            return messages

        format_detected = None

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

                # Auto-detect format from first valid JSON line
                if format_detected is None:
                    if "type" in obj:
                        format_detected = "claude"
                    elif "kind" in obj:
                        format_detected = "kiro"
                    else:
                        logger.warning(f"Unknown session format in {filepath.name}, skipping")
                        return messages

                if format_detected == "claude":
                    msgs = self._parse_claude_line(obj)
                else:
                    msgs = self._parse_kiro_line(obj)

                if msgs:
                    messages.extend(msgs)

        return messages

    def _parse_claude_line(self, obj: dict) -> List[ParsedMessage]:
        """Parse a single Claude Code JSONL line."""
        msg_type = obj.get("type")
        if msg_type not in self.RELEVANT_TYPES:
            return []

        message = obj.get("message", {})
        content = message.get("content", [])
        timestamp = obj.get("timestamp")
        uuid = obj.get("uuid")

        results = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").strip()
                if text and not self._is_system_content(text):
                    results.append(ParsedMessage(role=msg_type, text=text, timestamp=timestamp, uuid=uuid))
        return results

    def _parse_kiro_line(self, obj: dict) -> List[ParsedMessage]:
        """Parse a single Kiro CLI JSONL line."""
        kind = obj.get("kind")
        role = self.KIRO_KIND_MAP.get(kind)
        if not role:
            return []

        data = obj.get("data", {})
        content = data.get("content", []) if isinstance(data.get("content"), list) else []
        timestamp = obj.get("timestamp")
        uuid = obj.get("uuid")

        # Handle plain string content
        if isinstance(data.get("content"), str):
            text = data["content"].strip()
            if text and not self._is_system_content(text):
                return [ParsedMessage(role=role, text=text, timestamp=timestamp, uuid=uuid)]
            return []

        results = []
        for block in content:
            if isinstance(block, dict) and block.get("kind") == "text":
                text = block.get("data", "").strip()
                if text and not self._is_system_content(text):
                    results.append(ParsedMessage(role=role, text=text, timestamp=timestamp, uuid=uuid))
        return results

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
