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

logger = logging.getLogger(__name__)


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
                            tags = ["session-harvest"] + candidate.tags
                            resp = await self.memory_service.store_memory(
                                content=candidate.content,
                                tags=tags,
                                memory_type=candidate.memory_type,
                                metadata={
                                    "confidence": candidate.confidence,
                                    "source": "harvest",
                                },
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
            ok, msg, new_hash = await self.memory_service.storage.update_memory_versioned(
                existing_hash,
                candidate.content,
                new_tags=["session-harvest"] + candidate.tags,
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
                rewritten = []
                accepted_so_far = []  # Passo 2: contexto acumulado
                for candidate in filtered:
                    result = rewriter.rewrite_sync(
                        candidate.content,
                        suggested_type=candidate.memory_type,
                        already_extracted=accepted_so_far if accepted_so_far else None,
                    )
                    if result:
                        rewritten.append(HarvestCandidate(
                            content=result.content,
                            memory_type=result.memory_type,
                            tags=candidate.tags,
                            confidence=min(candidate.confidence + 0.1, 1.0),
                            source_line=candidate.source_line,
                        ))
                        accepted_so_far.append(result.content[:80])
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
