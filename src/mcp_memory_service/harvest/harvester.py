"""Orchestrator for session harvest operations."""

import asyncio
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List

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
        self._classifier = None

    def _get_classifier(self):
        """Lazy-init LLM classifier."""
        if self._classifier is None:
            from .classifier import HarvestClassifier
            self._classifier = HarvestClassifier()
        return self._classifier

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

        all_candidates: List[HarvestCandidate] = []
        for msg in messages:
            candidates = self.extractor.extract(msg)
            all_candidates.extend(candidates)

        # Apply regex-level filters
        filtered = [
            c for c in all_candidates
            if c.confidence >= config.min_confidence
            and c.memory_type in config.types
        ]

        # Phase 2: LLM classification
        if config.use_llm and filtered:
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
