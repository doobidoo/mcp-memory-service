"""Orchestrator for session harvest operations."""

import logging
from collections import Counter
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
