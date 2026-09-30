"""JSONL transcript parser for Claude Code and Kiro CLI session files."""

import copy
import json
import logging
import re
import sqlite3
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
    def _record_coverage(self, kind, was_extracted: bool, text: Optional[str] = None) -> None:
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
        # --- I0-lang: language dimension (R0.4) ----------------------------
        # Record the detected language ONLY for text-bearing blocks, split by
        # OUTCOME (extracted vs dropped). A block without text (tool_use,
        # invalid-payload) passes text=None and gets no language tally. Splitting
        # by outcome lets the report answer "how much of the DROPPED content is
        # pt-BR?" separately from the kept content — the coverage-gap metric,
        # which a single merged counter could not show. Measurement, not
        # inference: a cheap pt/en heuristic isolated in _detect_language.
        if text is not None:
            langs = entry.setdefault("languages", {"extracted": {}, "dropped": {}})
            bucket = langs["extracted" if was_extracted else "dropped"]
            lang = self._detect_language(text)
            bucket[lang] = bucket.get(lang, 0) + 1

    # Cheap, zero-dependency pt-BR vs. English detector for the Phase 0 language
    # dimension. Deliberately NOT a full langid model: the goal is to quantify
    # how much of the coverage gap is pt-BR, accurately enough to decide whether
    # the design extractor (I2/R3.1) must be multilingual — not to classify with
    # production precision. Marker words are frequent and near-exclusive to each
    # language; ties or no-signal return "unknown" rather than guessing. Kept in
    # one method so a later langid/fastText upgrade is a single-point change.
    _PT_MARKERS = frozenset({
        "não", "que", "para", "com", "está", "são", "foi", "uma", "por", "mais",
        "como", "mas", "isso", "ção", "então", "porque", "também", "já", "ser",
        "das", "dos", "análise", "decisão", "correto",
    })
    _EN_MARKERS = frozenset({
        "the", "and", "with", "this", "that", "was", "for", "not", "are", "were",
        "which", "because", "should", "would", "correct", "analysis", "decision",
        "keep", "before", "changing",
    })

    def _detect_language(self, text: Optional[str]) -> str:
        """Return "pt", "en", or "unknown" for a text block (cheap heuristic)."""
        if not text:
            return "unknown"
        # Text is already lower-cased; [a-zà-ÿ] covers ASCII letters plus the
        # Latin-1 accented range (á, ã, ç, é, ê, õ, ü, ...) without the duplicate
        # chars / overlapping ranges CodeQL flags.
        tokens = re.findall(r"[a-zà-ÿ]+", text.lower())
        if len(tokens) < 3:
            return "unknown"
        token_set = set(tokens)
        pt_hits = len(token_set & self._PT_MARKERS)
        en_hits = len(token_set & self._EN_MARKERS)
        # Portuguese-specific diacritics/cedilla are a strong pt signal on their own.
        if re.search(r"[ãõçâêô]|ção", text.lower()):
            pt_hits += 1
        if pt_hits == 0 and en_hits == 0:
            return "unknown"
        if pt_hits == en_hits:
            return "unknown"
        return "pt" if pt_hits > en_hits else "en"

    def coverage_report(self) -> dict:
        """Per-kind coverage since this parser instance was created.

        Returns {kind: {"seen": n, "extracted": n, "dropped": n,
        "languages": {"extracted": {lang: n}, "dropped": {lang: n}}}}.
        The "languages" sub-tally (I0-lang, R0.4) counts detected language per
        text-bearing block, split by outcome so the report can show how much of
        the DROPPED content (the coverage gap) is pt-BR vs. the kept content. It
        is absent for non-text kinds. Empty until something is parsed. Read-only;
        does not affect harvesting.

        Accumulates across multiple parse_file() calls on the same instance (by
        design — a harvest run aggregates coverage over many sessions). Create a
        fresh parser to reset. Not thread-safe: the instrument assumes the
        sequential, single-parser use the harvest scheduler already has.

        Returns a deep copy: mutating the result (including the nested
        "languages" dict) never affects the internal counters or a later report.
        """
        return copy.deepcopy(getattr(self, "_coverage", {}) or {})


    def find_sessions(self, project_dir: Path, count: int = 1) -> List[Path]:
        """Find the most recent session files under a directory.

        Discovers two layouts (RFC harvest-kiro-sessions v2.0, RA.1):
        - flat *.jsonl / *.trajectory.jsonl at the root (CLI mirror, Claude, OpenClaw);
        - nested {workspace_hash}/{session_uuid}/messages.jsonl (Kiro v4
          payload-wrapped workspace sessions, including migrated IDE sessions).
        The nested content is parsed by the existing v4 parser (#1366); only
        discovery was missing. Pointing directly at cli/ stays backward-compatible.
        """
        project_dir = Path(project_dir)
        # Support both .jsonl (Claude/Kiro) and .trajectory.jsonl (OpenClaw)
        all_jsonl = list(project_dir.glob("*.jsonl")) + list(project_dir.glob("*.trajectory.jsonl"))
        # The flat CLI mirror when the root is ~/.kiro/sessions (not .../cli):
        # scope to cli/ specifically rather than a wildcard */*.jsonl, so an
        # unrelated .jsonl in some other subdir is not pulled in.
        all_jsonl += list(project_dir.glob("cli/*.jsonl"))
        # Nested Kiro workspace sessions: {hash}/{uuid}/messages.jsonl
        all_jsonl += list(project_dir.glob("*/*/messages.jsonl"))
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

    def parse_sqlite(self, db_path: Path) -> List[ParsedMessage]:
        """Parse Kiro CLI v3 conversations from a SQLite database (read-only).

        The CLI v3 source of truth is `data.sqlite3`, table `conversations_v2`,
        where each row's `value` is a JSON blob with a structured `history[]`.
        The .jsonl mirror does not always cover these, so this reads the db
        directly (RFC harvest-kiro-sessions v2.0, RB.1-RB.3).

        Opened strictly read-only (`file:...?mode=ro`) — NEVER open the live
        Kiro db in write mode (RB.2). Feeds the same Phase 0 coverage instrument
        (with language) as the jsonl parsers, so SQLite content enters the
        coverage matrix. Missing table or malformed rows are skipped, not fatal.
        """
        db_path = Path(db_path)
        messages: List[ParsedMessage] = []
        if not db_path.exists():
            return messages

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.Error as e:
            logger.warning("parse_sqlite: cannot open %s read-only: %s", db_path.name, e)
            return messages

        try:
            cur = conn.cursor()
            try:
                cur.execute("SELECT value FROM conversations_v2")
            except sqlite3.Error:
                # No conversations_v2 table (or unreadable) — nothing to harvest.
                return messages
            # Iterate the cursor rather than fetchall(): each value can be ~2MB
            # and there are hundreds of rows, so streaming avoids loading the
            # whole table into memory at once.
            for (value,) in cur:
                if not isinstance(value, str):
                    continue
                try:
                    obj = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    logger.debug("parse_sqlite: skipping malformed conversation value")
                    continue
                # Valid JSON that is not an object (e.g. null, []) has no
                # .get — skip the row instead of letting AttributeError abort
                # the whole database read.
                if not isinstance(obj, dict):
                    continue
                history = obj.get("history")
                if isinstance(history, list):
                    messages.extend(self._extract_history_turns(history))
        finally:
            conn.close()

        return messages

    def _extract_history_turns(self, history: list) -> List[ParsedMessage]:
        """Extract user/assistant text from a conversations_v2 `history[]`.

        Each turn is {"user": {...}, "assistant": {...}}:
        - user text lives at content.Prompt.prompt (a ToolUseResults user turn
          carries no conversational text and is recorded as dropped);
        - assistant text lives at the `.content` of either a Response or a
          ToolUse variant (both carry a text preface).
        """
        results: List[ParsedMessage] = []
        for turn in history:
            if not isinstance(turn, dict):
                continue

            # --- user side ---
            user = turn.get("user")
            if isinstance(user, dict):
                content = user.get("content")
                if isinstance(content, dict):
                    prompt = content.get("Prompt")
                    if isinstance(prompt, dict):
                        text = (prompt.get("prompt") or "").strip()
                        if text and not self._is_system_content(text):
                            results.append(ParsedMessage(role="user", text=text))
                            self._record_coverage("sqlite:Prompt", was_extracted=True, text=text)
                        else:
                            self._record_coverage("sqlite:Prompt", was_extracted=False, text=text or None)
                    else:
                        # e.g. ToolUseResults — no conversational text.
                        self._record_coverage("sqlite:user-nontext", was_extracted=False)

            # --- assistant side ---
            assistant = turn.get("assistant")
            if isinstance(assistant, dict):
                # Text lives under a known variant's .content. Look for Response
                # or ToolUse explicitly rather than the first dict key — key order
                # is not a contract, and an unknown variant should not be mined
                # for arbitrary .content (avoids capturing non-conversational junk).
                for variant_key in ("Response", "ToolUse"):
                    variant = assistant.get(variant_key)
                    if not isinstance(variant, dict):
                        continue
                    text = (variant.get("content") or "").strip()
                    kind = f"sqlite:{variant_key}"
                    if text and not self._is_system_content(text):
                        results.append(ParsedMessage(role="assistant", text=text))
                        self._record_coverage(kind, was_extracted=True, text=text)
                    else:
                        self._record_coverage(kind, was_extracted=False, text=text or None)
                    break
                else:
                    # No known variant matched. Record the unsupported turn under
                    # its actual variant key so the coverage instrument still
                    # shows what the parser dropped (a future format gap must be
                    # measurable, not silently invisible).
                    unknown_key = next(iter(assistant), "unknown")
                    self._record_coverage(f"sqlite:{unknown_key}", was_extracted=False)
        return results

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
                self._record_coverage(kind, was_extracted=True, text=text)
                return [ParsedMessage(role=role, text=text, timestamp=timestamp, uuid=uuid)]
            self._record_coverage(kind, was_extracted=False, text=text or None)
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
                    self._record_coverage(block_kind, was_extracted=True, text=text)
                else:
                    self._record_coverage(block_kind, was_extracted=False, text=text or None)
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
                self._record_coverage(ptype, was_extracted=True, text=content.strip())
                return [ParsedMessage(role=role, text=content.strip(), timestamp=ts, uuid=uid)]
            else:
                self._record_coverage(ptype, was_extracted=False,
                                      text=content.strip() if isinstance(content, str) and content.strip() else None)
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
                self._record_coverage("tool_result", was_extracted=True, text=content)
                return [ParsedMessage(role="assistant", text=content, timestamp=ts, uuid=uid)]
            else:
                self._record_coverage("tool_result", was_extracted=False,
                                      text=content if isinstance(content, str) and content.strip() else None)
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
