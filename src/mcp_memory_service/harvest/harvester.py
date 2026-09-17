"""Orchestrator for session harvest operations."""

import asyncio
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .models import HarvestCandidate, HarvestConfig, HarvestResult
from .parser import TranscriptParser
from .extractor import PatternExtractor
from .patterns import load_filters
from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)

# Provenance: starts at 3 to align with RFC-harvest-provenance phases (provenance tagging, 
# re-harvest safety, session digest). Increment when the harvest pipeline changes materially.
HARVEST_PIPELINE_VERSION = 3


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
        locale = os.environ.get("HARVEST_LOCALE", "en")
        filters = load_filters(locale)
        self._meta_keywords = filters["meta_keywords"]
        self._temporal_re = re.compile(
            "|".join(filters["temporal_filters"]), re.IGNORECASE
        ) if filters["temporal_filters"] else None
        self._generic_re = re.compile(
            "|".join(filters["generic_filters"]), re.IGNORECASE
        ) if filters["generic_filters"] else None

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
        session_files = self._resolve_sessions(config)
        if not session_files:
            return []

        results = []
        for filepath in session_files:
            result = self._harvest_file(filepath, config)
            results.append(result)
        return results

    async def harvest_and_store(self, config: HarvestConfig) -> List[HarvestResult]:
        """Parse, extract, and store candidates via MemoryService.

        P4 Evolution: Before storing, checks for semantically similar active
        memories. If found above similarity_threshold, evolves via versioned
        update instead of creating a duplicate.
        """
        session_files = self._resolve_sessions(config)
        if not session_files:
            return []

        results = []
        for filepath in session_files:
            # _harvest_file does synchronous file I/O — offload so the event
            # loop stays responsive when harvest_and_store is called from HTTP.
            result = await asyncio.to_thread(self._harvest_file, filepath, config)

            if not config.dry_run and self.memory_service and result.candidates:
                stored = 0
                for candidate in result.candidates:
                    try:
                        evolved = await self._try_evolve(candidate, config)
                        if evolved:
                            stored += 1
                        else:
                            # Provenance (RFC-harvest-provenance Phase 1).
                            # Derive method from the model signal: only the LLM
                            # path sets harvest_model, so its presence is the
                            # source of truth — a missing/defaulted
                            # harvest_method must not mislabel an LLM candidate.
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
                                "harvest_session_id": result.session_id,
                            }
                            resp = await self.memory_service.store_memory(
                                content=candidate.content,
                                tags=tags,
                                memory_type=candidate.memory_type,
                                metadata=metadata,
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

    async def _try_evolve(self, candidate, config: "HarvestConfig") -> bool:
        """Check for similar active memory; if found, evolve it.

        Returns True if an existing memory was evolved, False if caller
        should fall back to store_memory().
        """
        if not hasattr(self.memory_service, "storage") or not self.memory_service.storage:
            return False

        try:
            similar = await self.memory_service.storage.retrieve(
                candidate.content,
                n_results=1,
                min_confidence=config.min_confidence_to_evolve,
            )
        except Exception as e:
            logger.debug(f"Similarity check failed, falling back to store: {e}")
            return False

        if not similar or similar[0].relevance_score <= config.similarity_threshold:
            return False

        existing_hash = similar[0].memory.content_hash
        try:
            # Apply method provenance tagging (same as store path)
            method = getattr(candidate, "harvest_method", None)
            if not method:
                method = "llm" if getattr(candidate, "harvest_model", None) else "heuristic"
            tags = ["session-harvest", f"harvest:method:{method}"] + candidate.tags
            
            ok, msg, new_hash = await self.memory_service.storage.update_memory_versioned(
                existing_hash,
                candidate.content,
                new_tags=tags,
                new_memory_type=candidate.memory_type,
                reason=f"Session harvest: {datetime.now(timezone.utc).isoformat()}",
            )
            if ok:
                logger.info(f"Evolved memory {existing_hash[:8]}→{new_hash[:8] if new_hash else '?'}")
                return True
            else:
                logger.debug(f"Evolution failed ({msg}), falling back to store")
                return False
        except Exception as e:
            logger.debug(f"Evolution error, falling back to store: {e}")
            return False

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
        # path. Resolve it and confirm it stays under project_dir, rejecting
        # traversal (e.g. "../other/transcript") before any I/O — otherwise both
        # the existence check and _resolve_sessions would read a JSONL outside
        # the configured session directory (repo directive: validate user paths).
        base_dir = Path(self.project_dir).resolve()
        session_path = (base_dir / f"{session_id}.jsonl").resolve()
        contained = session_path.is_relative_to(base_dir)
        session_found = contained and session_path.exists()

        if not contained:
            logger.warning(
                "Rejected out-of-directory session id %s",
                _sanitize_log_value(session_id),
            )
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
                logger.debug(f"coverage retrieve failed: {e}")
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

        # OpenClaw trajectories: disable role_filter (context.compiled already
        # filtered by parser; both user and assistant messages have value)
        use_role_filter = ".trajectory." not in filepath.name

        all_candidates: List[HarvestCandidate] = []
        for msg in messages:
            candidates = self.extractor.extract(msg, role_filter=use_role_filter)
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
                        ))
                logger.info(
                    f"LLM rewrite: {len(filtered)} → {len(rewritten)} candidates "
                    f"({len(filtered) - len(rewritten)} skipped)"
                )
                # Post-LLM filter: reject meta-discussion and temporal facts
                filtered = [c for c in rewritten if not self._is_meta_or_temporal(c.content)]
                if len(filtered) < len(rewritten):
                    logger.info(
                        f"Post-LLM filter: {len(rewritten)} → {len(filtered)} "
                        f"({len(rewritten) - len(filtered)} meta/temporal rejected)"
                    )
                # Passo 3: consolidate similar candidates
                before_consolidate = len(filtered)
                filtered = self._consolidate_similar(filtered)
                if len(filtered) < before_consolidate:
                    logger.info(
                        f"Consolidation: {before_consolidate} → {len(filtered)} "
                        f"({before_consolidate - len(filtered)} duplicates merged)"
                    )
            else:
                # Fallback to legacy classifier
                context_texts = [m.text for m in messages]
                classifier = self._get_classifier()
                before_count = len(filtered)
                filtered = classifier.classify(filtered, context_messages=context_texts)
                logger.info(
                    f"LLM classification: {before_count} → {len(filtered)} candidates "
                    f"({before_count - len(filtered)} rejected)"
                )

        by_type = dict(Counter(c.memory_type for c in filtered))

        return HarvestResult(
            candidates=filtered,
            session_id=session_id,
            total_messages=len(messages),
            found=len(filtered),
            by_type=by_type,
        )
