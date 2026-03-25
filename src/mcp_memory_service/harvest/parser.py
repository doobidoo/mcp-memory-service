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
