"""Orchestrator for session harvest operations."""

import asyncio
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import HarvestCandidate, HarvestConfig, HarvestResult
from .parser import TranscriptParser
from .extractor import PatternExtractor
from .patterns import load_filters
from ..compat import _sanitize_log_value
from ..config.locale import get_active_locales
from ..utils.hashing import generate_content_hash
from ..utils.time_parser import memory_statement_time, statement_time

logger = logging.getLogger(__name__)

# Provenance: starts at 3 to align with RFC-harvest-provenance phases (provenance tagging,
# re-harvest safety, session digest). Increment when the harvest pipeline changes materially.
# 4: metadata.source_time and time-aware evolution (#1858).
HARVEST_PIPELINE_VERSION = 4


def _mtime_or_zero(path: Path) -> float:
    """Sort key for session files; a file that vanished sorts first and fails later, where it is handled."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


class SessionHarvester:
    """Orchestrates parsing, extraction, and optional storage of harvest candidates."""

    def __init__(self, project_dir: Path, memory_service=None):
        self.project_dir = Path(project_dir)
        self.memory_service = memory_service
        self._memory_service = memory_service
        self.parser = TranscriptParser()
        self.extractor = PatternExtractor()
        self._classifier = None

        # Load filters from locale YAMLs
        locale = ",".join(get_active_locales())
        filters = load_filters(locale)
        self._meta_keywords = filters["meta_keywords"]
        self._temporal_re = re.compile(
            "|".join(filters["temporal_filters"]), re.IGNORECASE
        ) if filters["temporal_filters"] else None
        self._generic_re = re.compile(
            "|".join(filters["generic_filters"]), re.IGNORECASE
        ) if filters["generic_filters"] else None

    def _session_id(self, filepath: Path, base_dir: Path = None) -> str:
        """Stable, unique session id for a session file.

        A Kiro workspace session is ``messages.jsonl`` nested exactly two
        directory levels under the sessions root
        (``{root}/{workspace_hash}/{session_dir}/messages.jsonl``), so its
        ``stem`` collides at ``"messages"`` and the session-dir name alone
        recurs across workspaces. Such a file is keyed by
        ``{workspace_hash}/{session_dir}`` — globally unique and reversible via
        ``_resolve_session_id``.

        A ``messages.jsonl`` that is not in that two-level nesting (e.g. flat at
        the sessions root) is not a workspace session and keeps its stem, so the
        id stays resolvable. Every other layout (CLI ``cli/foo.jsonl``, OpenClaw,
        Claude) also keeps ``filepath.stem`` — the id already in the harvest
        tracker for thousands of sessions. Semantic, not positional: the id does
        not change with the directory harvest is pointed at, keeping dedup intact.

        ``base_dir`` defaults to ``self.project_dir``; it anchors the nesting
        test so a flat ``messages.jsonl`` is not mistaken for a workspace one.
        """
        if filepath.name == "messages.jsonl":
            root = Path(base_dir) if base_dir is not None else self.project_dir
            try:
                rel_parts = filepath.resolve().relative_to(Path(root).resolve()).parts
            except (ValueError, OSError):
                rel_parts = filepath.parts
            # Workspace nesting is exactly {hash}/{session}/messages.jsonl:
            # two directory components above the file, relative to the root.
            if len(rel_parts) >= 3:
                return f"{rel_parts[-3]}/{rel_parts[-2]}"
        return filepath.stem

    def _resolve_session_id(self, sid: str, base_dir: Path) -> Optional[Path]:
        """Reverse a session id to its file, or None if it escapes base_dir or
        does not exist. Shared by _resolve_sessions and verify_session_coverage
        so both agree on how flat and workspace ids map back to paths.

        - flat id ``foo``            -> ``{base_dir}/foo.jsonl``
        - workspace id ``hash/sess`` -> ``{base_dir}/hash/sess/messages.jsonl``
        session_ids are caller-controlled, so both candidates are resolved and
        containment-checked (reject ``../`` traversal) before any I/O.
        """
        base = base_dir.resolve()
        flat = (base_dir / f"{sid}.jsonl").resolve()
        nested = (base_dir / sid / "messages.jsonl").resolve()
        if not (flat.is_relative_to(base) or nested.is_relative_to(base)):
            logger.warning(
                "Rejected out-of-directory session id %s",
                _sanitize_log_value(str(sid)),
            )
            return None
        if flat.is_relative_to(base) and flat.exists():
            return flat
        if nested.is_relative_to(base) and nested.exists():
            return nested
        return None

    def _get_classifier(self):
        """Lazy-init LLM classifier."""
        if self._classifier is None:
            from .classifier import HarvestClassifier
            self._classifier = HarvestClassifier()
        return self._classifier

    def _get_rewriter(self):
        """Lazy-init LLM rewriter. Returns None if no provider is configured."""
        if not hasattr(self, '_rewriter'):
            try:
                from .rewriter import HarvestRewriter
                rewriter = HarvestRewriter()
                # A provider chain from HARVEST_LLM_PROVIDERS counts as
                # configured. Checking GROQ_API_KEY alone silently disabled
                # rewriting for anyone pointing at an OpenAI-compatible
                # endpoint (issue #178).
                if not rewriter.is_configured:
                    logger.info(
                        "Harvest rewriter disabled: neither HARVEST_LLM_PROVIDERS "
                        "nor GROQ_API_KEY is configured"
                    )
                    self._rewriter = None
                else:
                    self._rewriter = rewriter
            except Exception:
                self._rewriter = None
        return self._rewriter

    _META_KEYWORDS = []  # loaded from YAML at init
    _TEMPORAL_RE = None
    _GENERIC_RE = None

    def _is_meta_or_temporal(self, text: str) -> bool:
        """Reject meta-discussion, temporal facts, and generic statements."""
        lower = text.lower()
        if any(kw in lower for kw in self._meta_keywords):
            return True
        if self._temporal_re and self._temporal_re.search(text):
            return True
        if self._generic_re and self._generic_re.search(text):
            return True
        return False

    def _consolidate_similar(self, candidates: List[HarvestCandidate], threshold: float = 0.35) -> List[HarvestCandidate]:
        """Consolidate similar candidates, keeping the most complete one per cluster."""
        if len(candidates) <= 1:
            return candidates

        def _jaccard(a: str, b: str) -> float:
            wa = set(a.lower().split())
            wb = set(b.lower().split())
            if not wa or not wb:
                return 0.0
            return len(wa & wb) / len(wa | wb)

        kept: List[HarvestCandidate] = []
        used = set()
        sorted_cands = sorted(enumerate(candidates), key=lambda x: -len(x[1].content))

        for i, cand in sorted_cands:
            if i in used:
                continue
            for j, other in sorted_cands:
                if j != i and j not in used and _jaccard(cand.content, other.content) > threshold:
                    used.add(j)
            kept.append(cand)
            used.add(i)

        return kept

    def harvest(self, config: HarvestConfig) -> List[HarvestResult]:
        """Parse sessions and extract candidates (synchronous, no storage)."""
        # Reset coverage counters to ensure per-run scope (P2-c fix)
        self.parser.reset_coverage()

        session_files = self._resolve_sessions(config)
        if not session_files:
            return []

        results = []
        for filepath in session_files:
            result = self._harvest_file(filepath, config)
            results.append(result)

        # Get coverage report aggregated across the run and add to all results
        coverage = self.parser.coverage_report()
        if coverage:
            logger.info("Harvest coverage: %s", self._format_coverage_summary(coverage))
        else:
            logger.info("Harvest coverage: none measured (parser recorded no counters for this run)")
        for result in results:
            result.coverage = coverage

        return results

    async def harvest_and_store(self, config: HarvestConfig) -> List[HarvestResult]:
        """Parse, extract, and store candidates via MemoryService.

        P4 Evolution: Before storing, checks for semantically similar active
        memories. If found above similarity_threshold, evolves via versioned
        update instead of creating a duplicate.
        """
        # Reset coverage counters to ensure per-run scope (P2-c fix)
        self.parser.reset_coverage()

        session_files = self._resolve_sessions(config)
        if not session_files:
            return []

        results = []
        # Oldest first within the batch, so a later session's statement evolves
        # an earlier one and never the other way round (#1858). Which sessions
        # make up the batch stays the callers' choice (newest pending first).
        for filepath in sorted(session_files, key=_mtime_or_zero):
            # _harvest_file does synchronous file I/O — offload so the event
            # loop stays responsive when harvest_and_store is called from HTTP.
            result = await asyncio.to_thread(self._harvest_file, filepath, config)

            if not config.dry_run and self.memory_service and result.candidates:
                outcomes = [
                    await self._store_candidate(candidate, config, result.session_id)
                    for candidate in result.candidates
                ]
                result.stored = outcomes.count("stored")
                result.skipped = outcomes.count("skipped")

            results.append(result)

        # Get coverage report aggregated across the run and add to all results
        coverage = self.parser.coverage_report()
        if coverage:
            logger.info("Harvest coverage: %s", self._format_coverage_summary(coverage))
        else:
            logger.info("Harvest coverage: none measured (parser recorded no counters for this run)")
        for result in results:
            result.coverage = coverage

        return results

    async def _store_candidate(self, candidate, config: HarvestConfig, session_id) -> str:
        """Evolve or store one candidate; returns ``stored``, ``skipped`` or ``failed``."""
        try:
            tags, metadata = self._provenance(candidate, session_id)
            evolved, _new_hash = await self._try_evolve(candidate, config, metadata)
            if evolved:
                return "stored"
            resp = await self.memory_service.store_memory(
                content=candidate.content,
                tags=tags,
                memory_type=candidate.memory_type,
                metadata=metadata,
                # An older statement than its closest match is kept next to it.
                # A conversation_id skips semantic dedup, which would otherwise
                # reject it against that same match (#1858).
                conversation_id=f"harvest:{session_id}" if candidate.older_than_match else None,
            )
        except Exception as e:
            logger.warning("Failed to store harvest candidate: %s", _sanitize_log_value(str(e)))
            return "failed"
        if isinstance(resp, dict):
            if resp.get("success"):
                return "stored"
            # #1560: deleted on the server — skipped, not stored, not an error.
            return "skipped" if resp.get("refused_recreate") else "failed"
        return "stored" if getattr(resp, "success", False) else "failed"

    def _format_coverage_summary(self, coverage: dict) -> str:
        """Format coverage report for readable logging."""
        if not coverage:
            return "empty"

        parts = []
        for kind, stats in coverage.items():
            seen = stats.get("seen", 0)
            extracted = stats.get("extracted", 0)
            dropped = stats.get("dropped", 0)

            if seen > 0:
                # kind is read raw from the session JSON (external) — sanitise it
                # before it reaches the log line (py/log-injection, same guard the
                # rest of this module applies to external values).
                safe_kind = _sanitize_log_value(str(kind))
                parts.append(f"{safe_kind}: {seen} seen, {extracted} extracted, {dropped} dropped")

        return "; ".join(parts) if parts else "empty"

    @staticmethod
    def _provenance(candidate, session_id=None):
        """Tags and metadata recording how a candidate was harvested.

        Provenance (RFC-harvest-provenance Phase 1). Derive method from the
        model signal: only the LLM path sets harvest_model, so its presence is
        the source of truth — a missing/defaulted harvest_method must not
        mislabel an LLM candidate.
        """
        model = getattr(candidate, "harvest_model", None)
        method = getattr(candidate, "harvest_method", None)
        if not method:
            method = "llm" if model else "heuristic"
        tags = ["session-harvest", f"harvest:method:{method}"] + candidate.tags
        metadata = {
            "confidence": candidate.confidence,
            "source": "harvest",
            "harvest_method": method,
            "harvest_model": model,
            "harvest_pipeline_version": HARVEST_PIPELINE_VERSION,
            "harvest_session_id": session_id,
        }
        source_time = getattr(candidate, "source_time", None)
        if source_time is not None:
            # The time belongs to this text only; memory_statement_time ignores it
            # once an edit has changed the content (#1858).
            metadata["source_time"] = datetime.fromtimestamp(source_time, tz=timezone.utc).isoformat()
            metadata["source_content_hash"] = generate_content_hash(candidate.content)
        return tags, metadata

    @staticmethod
    def _is_older_statement(candidate, existing) -> bool:
        """True when the candidate was said before the existing memory (#1858).

        Unknown on either side keeps the old behaviour (evolve).
        """
        candidate_time = getattr(candidate, "source_time", None)
        if candidate_time is None:
            return False
        existing_time = memory_statement_time(existing)
        return existing_time is not None and candidate_time < existing_time

    async def _try_evolve(self, candidate, config: "HarvestConfig", metadata=None):
        """Check for similar active memory; if found, evolve it.

        Returns ``(evolved, new_hash)``. ``evolved`` is True when an existing
        memory was updated in place, and the caller must not store the
        candidate again. ``new_hash`` is the versioned child's hash when the
        backend reports one.

        Evolution can succeed without a hash, so ``new_hash`` may be None while
        ``evolved`` is True. Branch on ``evolved``, never on ``new_hash``:
        treating "no hash" as "not evolved" stores the candidate a second time.
        """
        if not hasattr(self.memory_service, "storage") or not self.memory_service.storage:
            return False, None

        try:
            similar = await self.memory_service.storage.retrieve(
                candidate.content,
                n_results=1,
                min_confidence=config.min_confidence_to_evolve,
            )
        except Exception as e:
            logger.debug("Similarity check failed, falling back to store: %s", _sanitize_log_value(str(e)))
            return False, None

        if not similar or similar[0].relevance_score <= config.similarity_threshold:
            return False, None

        existing = similar[0].memory
        existing_hash = existing.content_hash
        if self._is_older_statement(candidate, existing):
            # An older statement must not become the current version of a newer
            # memory (#1858). The caller stores it as its own memory instead.
            logger.debug("Not evolving %s: candidate is the older statement", existing_hash[:8])
            candidate.older_than_match = True
            return False, None
        try:
            # Same provenance as the store path. Evolve through the service,
            # not storage.update_memory_versioned(), so the new version gets
            # quality scoring, entity linking and on_store plugins too.
            tags, default_metadata = self._provenance(candidate)
            ok, msg, new_hash = await self.memory_service.evolve_memory(
                existing_hash,
                candidate.content,
                tags=tags,
                memory_type=candidate.memory_type,
                metadata=metadata if metadata is not None else default_metadata,
                reason=f"Session harvest: {datetime.now(timezone.utc).isoformat()}",
            )
            if ok:
                logger.info("Evolved memory %s→%s", existing_hash[:8], new_hash[:8] if new_hash else "?")
                return True, new_hash
            else:
                logger.debug("Evolution failed (%s), falling back to store", _sanitize_log_value(str(msg)))
                return False, None
        except Exception as e:
            logger.debug("Evolution error, falling back to store: %s", _sanitize_log_value(str(e)))
            return False, None

    async def verify_session_coverage(self, session_id: str, threshold: float = 0.9,
                                      use_llm: bool = True) -> dict:
        """Check how well a session's insights are already in memory (R11/R12).

        Re-harvests the session in-memory (nothing is stored) and, for each
        candidate insight, looks for a semantically similar stored memory. Used
        to decide whether the source session is safe to delete: if some insight
        has no strong match, deleting the transcript would lose it for good.

        Returns:
            {
              "session_id": str,
              "coverage": float,            # fraction of insights with a strong match
              "total_insights": int,
              "missing_insights": [str],    # insights with no match >= threshold
              "low_quality_matches": [str], # insights whose best match is weak
              "session_found": bool,        # session file was located and processed
              "safe_to_delete": bool,       # session processed, coverage complete, no gaps
            }

        Raises:
            ValueError: if ``threshold`` is not a positive score in (0.0, 1.0].
        """
        from .models import HarvestConfig

        # Guard the public threshold: an absent match is scored 0.0, so a
        # threshold <= 0 would let every missing insight count as "covered"
        # and wrongly mark a session safe to delete.
        if not (0.0 < threshold <= 1.0):
            raise ValueError(
                f"threshold must be a score in (0.0, 1.0], got {threshold!r}"
            )

        # The session_id is caller-controlled and is turned into a filesystem
        # path. _resolve_session_id resolves both flat ({id}.jsonl) and workspace
        # ({hash}/{session}/messages.jsonl) ids, confirms containment under
        # project_dir (rejecting "../" traversal), and returns None otherwise —
        # the same mapping _resolve_sessions uses, so a workspace session can be
        # verified (and thus safely deleted), not just flat ones.
        base_dir = Path(self.project_dir).resolve()
        session_path = self._resolve_session_id(session_id, self.project_dir)
        session_found = session_path is not None

        if session_path is None and not (base_dir / f"{session_id}.jsonl").resolve().is_relative_to(base_dir):
            return {
                "session_id": session_id, "coverage": 0.0, "total_insights": 0,
                "missing_insights": [], "low_quality_matches": [],
                "session_found": False, "safe_to_delete": False,
            }

        cfg = HarvestConfig(sessions=1, session_ids=[session_id],
                            dry_run=True, use_llm=use_llm)
        # Offload synchronous harvesting (blocking file + LLM I/O) off the event
        # loop so this async check does not stall unrelated coroutines.
        results = await asyncio.to_thread(self.harvest, cfg)
        candidates = [c for r in results for c in r.candidates]

        if not session_found:
            # The transcript was never inspected — deleting it could lose data.
            return {
                "session_id": session_id, "coverage": 0.0, "total_insights": 0,
                "missing_insights": [], "low_quality_matches": [],
                "session_found": False, "safe_to_delete": False,
            }

        if not candidates:
            # Session was found and processed but yields nothing worth keeping →
            # deleting the transcript loses nothing.
            return {
                "session_id": session_id, "coverage": 1.0, "total_insights": 0,
                "missing_insights": [], "low_quality_matches": [],
                "session_found": True, "safe_to_delete": True,
            }

        missing, weak, covered = [], [], 0
        for cand in candidates:
            try:
                matches = await self.memory_service.storage.retrieve(cand.content, n_results=1)
            except Exception as e:
                logger.debug("coverage retrieve failed: %s", _sanitize_log_value(str(e)))
                matches = []
            best = matches[0].relevance_score if matches else 0.0
            if best >= threshold:
                covered += 1
            elif best > 0.0:
                weak.append(cand.content)
            else:
                missing.append(cand.content)

        coverage = covered / len(candidates)
        return {
            "session_id": session_id,
            "coverage": coverage,
            "total_insights": len(candidates),
            "missing_insights": missing,
            "low_quality_matches": weak,
            "session_found": True,
            "safe_to_delete": coverage >= 1.0 and not missing and not weak,
        }

    def _resolve_sessions(self, config: HarvestConfig) -> List[Path]:
        """Find session files based on config."""
        if config.session_ids:
            resolved: List[Path] = []
            for sid in config.session_ids:
                path = self._resolve_session_id(sid, self.project_dir)
                if path is not None:
                    resolved.append(path)
            return resolved
        return self.parser.find_sessions(self.project_dir, count=config.sessions)

    def _harvest_file(self, filepath: Path, config: HarvestConfig) -> HarvestResult:
        """Extract candidates from a single session file."""
        messages = self.parser.parse_file(filepath)
        session_id = self._session_id(filepath, self.project_dir)

        # OpenClaw trajectories: disable role_filter (context.compiled already
        # filtered by parser; both user and assistant messages have value)
        use_role_filter = ".trajectory." not in filepath.name

        all_candidates: List[HarvestCandidate] = []
        rewrite_dropped = 0
        for msg in messages:
            candidates = self.extractor.extract(msg, role_filter=use_role_filter)
            msg_time = statement_time(msg.timestamp)
            for candidate in candidates:
                candidate.source_time = msg_time
            all_candidates.extend(candidates)

        # Apply regex-level filters
        filtered = [
            c for c in all_candidates
            if c.confidence >= config.min_confidence
            and c.memory_type in config.types
        ]

        # Phase 2: LLM rewrite (preferred) or classification (legacy)
        if config.use_llm and filtered:
            rewriter = self._get_rewriter()
            if rewriter:
                # Use batch API when available — one LLM call instead of N
                batch_items = [
                    {"content": c.content, "memory_type": c.memory_type}
                    for c in filtered
                ]
                batch_results = rewriter.rewrite_batch_sync(batch_items)
                # A None here is a candidate the rewrite lost (failed call, line
                # omitted or unparseable); a deliberate SKIP comes back as the
                # falsy SKIPPED sentinel. Count only the losses, so the scheduler
                # does not mark the session finished without them (#1560).
                rewrite_dropped = sum(1 for result in batch_results if result is None)
                rewritten = []
                for candidate, result in zip(filtered, batch_results):
                    if result:
                        _model = (
                            f"{result.provider}/{result.model}"
                            if getattr(result, "provider", None) and getattr(result, "model", None)
                            else None
                        )
                        rewritten.append(HarvestCandidate(
                            content=result.content,
                            memory_type=result.memory_type,
                            tags=candidate.tags,
                            confidence=min(candidate.confidence + 0.1, 1.0),
                            source_line=candidate.source_line,
                            harvest_method="llm",
                            harvest_model=_model,
                            source_time=candidate.source_time,
                        ))
                logger.info(
                    "LLM rewrite: %d → %d candidates (%d skipped)",
                    len(filtered), len(rewritten), len(filtered) - len(rewritten),
                )
                # Post-LLM filter: reject meta-discussion and temporal facts
                filtered = [c for c in rewritten if not self._is_meta_or_temporal(c.content)]
                if len(filtered) < len(rewritten):
                    logger.info(
                        "Post-LLM filter: %d → %d (%d meta/temporal rejected)",
                        len(rewritten), len(filtered), len(rewritten) - len(filtered),
                    )
                # Passo 3: consolidate similar candidates
                before_consolidate = len(filtered)
                filtered = self._consolidate_similar(filtered)
                if len(filtered) < before_consolidate:
                    logger.info(
                        "Consolidation: %d → %d (%d duplicates merged)",
                        before_consolidate, len(filtered), before_consolidate - len(filtered),
                    )
            else:
                # Fallback to legacy classifier
                context_texts = [m.text for m in messages]
                classifier = self._get_classifier()
                before_count = len(filtered)
                filtered = classifier.classify(filtered, context_messages=context_texts)
                logger.info(
                    "LLM classification: %d → %d candidates (%d rejected)",
                    before_count, len(filtered), before_count - len(filtered),
                )

        by_type = dict(Counter(c.memory_type for c in filtered))

        return HarvestResult(
            candidates=filtered,
            session_id=session_id,
            total_messages=len(messages),
            found=len(filtered),
            by_type=by_type,
            rewrite_dropped=rewrite_dropped,
        )
