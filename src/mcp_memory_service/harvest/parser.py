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
    """Parses JSONL session transcripts (Claude Code, Kiro CLI, and OpenClaw).

    Format is detected from the first message in the file:
    - If "traceSchema" is "openclaw-trajectory" → OpenClaw gateway format
    - If "type" key exists → Claude Code format
    - If "kind" key exists → Kiro CLI format (legacy)
    - If "payload" with "type" exists → Kiro CLI v4 format (payload-wrapped)
    - Unknown → warning logged, returns empty
    """

    RELEVANT_TYPES = {"user", "assistant"}
    KIRO_KIND_MAP = {"Prompt": "user", "Response": "assistant", "AssistantMessage": "assistant"}
    OPENCLAW_MESSAGE_TYPES = {"prompt.submitted", "model.completed"}
    PAYLOAD_ROLE_MAP = {"user": "user", "assistant": "assistant"}

    # --- Phase 0 coverage instrument (#1287) -------------------------------
    # Counts, per block kind/type, how many were seen vs extracted vs dropped,
    # WITHOUT changing what is harvested. This makes the coverage gap measurable
    # ("N ToolResult blocks seen, 0 extracted") before any extractor changes, so a
    # later negative result is trustworthy. Lazily initialized to avoid an __init__.
    #
    # One call == one block seen, keyed by that block's kind. A harvestable message
    # records one entry per content block (so a text block kept alongside a
    # non-text block dropped both show up); a non-harvestable message (e.g. a Kiro
    # ToolResult) records one entry keyed by its message kind. This keeps
    # extracted <= seen per kind — the per-message len(results) counting broke that.
    def _record_coverage(self, kind, was_extracted: bool) -> None:
        cov = getattr(self, "_coverage", None)
        if cov is None:
            cov = {}
            self._coverage = cov
        entry = cov.setdefault(str(kind), {"seen": 0, "extracted": 0, "dropped": 0})
        entry["seen"] += 1
        if was_extracted:
            entry["extracted"] += 1
        else:
            entry["dropped"] += 1

    def coverage_report(self) -> dict:
        """Per-kind coverage since this parser instance was created.

        Returns {kind: {"seen": n, "extracted": n, "dropped": n}}. Empty until
        something is parsed. Read-only; does not affect harvesting.

        Accumulates across multiple parse_file() calls on the same instance (by
        design — a harvest run aggregates coverage over many sessions). Create a
        fresh parser to reset. Not thread-safe: the instrument assumes the
        sequential, single-parser use the harvest scheduler already has.

        Returns a deep copy: mutating the result never affects the internal
        counters or a later report.
        """
        return {kind: dict(counts) for kind, counts in (getattr(self, "_coverage", {}) or {}).items()}


    def find_sessions(self, project_dir: Path, count: int = 1) -> List[Path]:
        """Find the most recent JSONL session files in a project directory."""
        project_dir = Path(project_dir)
        # Support both .jsonl (Claude/Kiro) and .trajectory.jsonl (OpenClaw)
        all_jsonl = list(project_dir.glob("*.jsonl")) + list(project_dir.glob("*.trajectory.jsonl"))
        # Deduplicate (*.jsonl already matches *.trajectory.jsonl)
        seen = set()
        unique = []
        for p in all_jsonl:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        # Exclude checkpoints, resets, deleted, and trajectory-path metadata
        filtered = [
            p for p in unique
            if ".checkpoint." not in p.name
            and ".reset." not in p.name
            and ".deleted." not in p.name
            and not p.name.endswith("-path.json")
        ]
        # Prefer .trajectory.jsonl over plain .jsonl for same session ID
        trajectory_stems = {p.name.replace(".trajectory.jsonl", "") for p in filtered if ".trajectory.jsonl" in p.name}
        final = [
            p for p in filtered
            if not (p.name.endswith(".jsonl") and not ".trajectory." in p.name
                    and p.stem in trajectory_stems)
        ]
        jsonl_files = sorted(final, key=lambda p: p.stat().st_mtime, reverse=True)
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
                    if obj.get("traceSchema") == "openclaw-trajectory":
                        format_detected = "openclaw"
                    elif "type" in obj:
                        format_detected = "claude"
                    elif "kind" in obj:
                        format_detected = "kiro"
                    elif isinstance(obj.get("payload"), dict) and "type" in obj["payload"]:
                        format_detected = "kiro-cli-v4"
                    else:
                        logger.warning(f"Unknown session format in {filepath.name}, skipping")
                        return messages

                if format_detected == "claude":
                    msgs = self._parse_claude_line(obj)
                elif format_detected == "kiro":
                    msgs = self._parse_kiro_line(obj)
                elif format_detected == "kiro-cli-v4":
                    msgs = self._parse_kiro_v4_line(obj)
                elif format_detected == "openclaw":
                    msgs = self._parse_openclaw_line(obj)
                else:
                    msgs = None

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
            # Not a harvestable kind (e.g. ToolResult). Record it as seen+dropped
            # (keyed by message kind) so the coverage instrument shows what the
            # pipeline structurally skips at the message level.
            self._record_coverage(kind, was_extracted=False)
            return []

        data = obj.get("data", {})
        content = data.get("content", []) if isinstance(data.get("content"), list) else []
        timestamp = obj.get("timestamp")
        uuid = obj.get("uuid")

        # Handle plain string content (no blocks). Key it by the message kind.
        if isinstance(data.get("content"), str):
            text = data["content"].strip()
            if text and not self._is_system_content(text):
                self._record_coverage(kind, was_extracted=True)
                return [ParsedMessage(role=role, text=text, timestamp=timestamp, uuid=uuid)]
            self._record_coverage(kind, was_extracted=False)
            return []

        results = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_kind = block.get("kind")
            if block_kind == "text":
                text = block.get("data", "").strip()
                if text and not self._is_system_content(text):
                    results.append(ParsedMessage(role=role, text=text, timestamp=timestamp, uuid=uuid))
                    self._record_coverage(block_kind, was_extracted=True)
                else:
                    self._record_coverage(block_kind, was_extracted=False)
            else:
                # Non-text block (e.g. tool_use) — dropped, but now visible per kind.
                self._record_coverage(block_kind, was_extracted=False)
        return results

    def _parse_kiro_v4_line(self, obj: dict) -> List[ParsedMessage]:
        """Parse a single Kiro CLI v4 (payload-wrapped) JSONL line.
        
        Format: {"id": "...", "timestamp": "...", "payload": {"type": "...", "content": "...", ...}}
        """
        pl = obj.get("payload")
        # Guard against malformed records: a valid JSON line whose payload is null
        # or not an object, or whose type is unhashable (list/dict), must not abort
        # the whole run — record it as an unparseable block and move on.
        if not isinstance(pl, dict):
            self._record_coverage("kiro-cli-v4:invalid-payload", was_extracted=False)
            return []
        ptype = pl.get("type")
        if not isinstance(ptype, (str, type(None))):
            self._record_coverage("kiro-cli-v4:invalid-type", was_extracted=False)
            return []
        ts = obj.get("timestamp")
        uid = obj.get("id")
        
        # Handle user/assistant messages
        if ptype in self.PAYLOAD_ROLE_MAP:
            role = self.PAYLOAD_ROLE_MAP[ptype]
            content = pl.get("content")
            if isinstance(content, str) and content.strip() and not self._is_system_content(content):
                self._record_coverage(ptype, was_extracted=True)
                return [ParsedMessage(role=role, text=content.strip(), timestamp=ts, uuid=uid)]
            else:
                self._record_coverage(ptype, was_extracted=False)
                return []
        
        # Handle tool_result as assistant message with rich content.
        # Reject injected markers (system-reminder / command / ide) so an injected
        # payload inside a tool result cannot become a harvested memory — but do NOT
        # apply the >10k length cutoff that _is_system_content uses: a long tool
        # result is exactly the rich analytical data #1346 wants (query dumps,
        # diagnostic reports). The content is passed verbatim, like every other
        # parser; the extractor caps each candidate (MAX_CANDIDATE_CONTENT_LENGTH)
        # and scans the whole text, so nothing after an arbitrary parser-side cutoff
        # is silently lost.
        elif ptype == "tool_result":
            content = pl.get("content")
            if isinstance(content, str) and content.strip() and not self._is_injected_content(content):
                self._record_coverage("tool_result", was_extracted=True)
                return [ParsedMessage(role="assistant", text=content, timestamp=ts, uuid=uid)]
            else:
                self._record_coverage("tool_result", was_extracted=False)
                return []
        
        # Handle tool_call and metadata - not extracted but counted in coverage
        else:
            self._record_coverage(ptype, was_extracted=False)
            return []

    def _parse_openclaw_line(self, obj: dict) -> List[ParsedMessage]:
        """Parse a single OpenClaw gateway trajectory JSONL line.

        Format (openclaw-trajectory):
        - prompt.submitted  → user message from data.prompt
        - model.completed   → assistant from data.assistantTexts[]
        - session.started/ended, trace.metadata, context.compiled → skip
        """
        event_type = obj.get("type")
        if event_type not in self.OPENCLAW_MESSAGE_TYPES:
            return []

        data = obj.get("data", {})
        timestamp = obj.get("ts")

        if event_type == "prompt.submitted":
            text = data.get("prompt", "").strip()
            if text and not self._is_system_content(text):
                return [ParsedMessage(role="user", text=text, timestamp=timestamp)]

        elif event_type == "model.completed":
            assistant_texts = data.get("assistantTexts", [])
            if isinstance(assistant_texts, list):
                texts = [t.strip() for t in assistant_texts if isinstance(t, str) and t.strip()]
                if texts:
                    combined = "\n\n".join(texts)
                    if not self._is_system_content(combined):
                        return [ParsedMessage(role="assistant", text=combined, timestamp=timestamp)]

        return []

    @staticmethod
    def _is_injected_content(text: str) -> bool:
        """True if the text carries a harness-injected marker (reminder/command/ide).

        No length heuristic here: this is the shared 'is it injected?' check. Used
        directly for tool results, where large content is legitimate rich data.
        """
        if "<system-reminder>" in text or "</system-reminder>" in text:
            return True
        if "<command-name>" in text or "<command-message>" in text:
            return True
        if text.startswith("<ide_opened_file>"):
            return True
        return False

    @staticmethod
    def _is_system_content(text: str) -> bool:
        """Filter out system prompts, skill outputs, and injected content."""
        if TranscriptParser._is_injected_content(text):
            return True
        # Extremely long blocks (>10k chars) — likely injected context, not conversation.
        # This length cutoff is intentionally NOT applied to tool results (see
        # _parse_kiro_v4_line), where long output is the rich data we want.
        if len(text) > 10000:
            return True
        return False
