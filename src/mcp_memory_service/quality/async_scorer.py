"""
Async background scoring for quality evaluation.

Provides non-blocking quality scoring using a background task queue.
This allows memory retrieval to return immediately while quality scores
are calculated in the background.
"""

import asyncio
import logging
from typing import Optional
from queue import Queue
import threading

from ..models.memory import Memory
from .ai_evaluator import QualityEvaluator
from .scorer import QualityScorer

logger = logging.getLogger(__name__)


class AsyncQualityScorer:
    """
    Background task queue for quality scoring.

    Scores memories asynchronously without blocking the main retrieval path.
    Uses an async queue to manage scoring tasks and a background worker coroutine.
    """

    def __init__(self):
        """Initialize the async quality scorer."""
        self.evaluator = QualityEvaluator()
        self.scorer = QualityScorer()
        self.queue: asyncio.Queue = None
        self.running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Statistics for monitoring
        self.stats = {
            "total_queued": 0,
            "total_scored": 0,
            "total_errors": 0,
            "queue_size": 0
        }

    async def start(self):
        """
        Start background worker.

        Initializes the queue and starts the worker coroutine.
        Safe to call multiple times (idempotent).
        """
        async with self._lock:
            if self.running:
                logger.debug("Async quality scorer already running")
                return

            self.queue = asyncio.Queue()
            self.running = True
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("Async quality scorer started")

    async def stop(self):
        """
        Stop background worker gracefully.

        Waits for current scoring tasks to complete before stopping.
        """
        async with self._lock:
            if not self.running:
                logger.debug("Async quality scorer already stopped")
                return

            self.running = False

            # Wait for worker to finish
            if self._worker_task:
                try:
                    await asyncio.wait_for(self._worker_task, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Worker task did not finish in time, cancelling")
                    self._worker_task.cancel()
                    try:
                        await self._worker_task
                    except asyncio.CancelledError:
                        pass

            self._worker_task = None
            logger.info("Async quality scorer stopped")

    async def score_memory(
        self,
        memory: Memory,
        query: str,
        storage: Optional[any] = None
    ) -> None:
        """
        Queue memory for background scoring.

        Non-blocking - returns immediately.

        Args:
            memory: Memory object to score
            query: Search query for context
            storage: Optional storage instance to persist updated scores
        """
        if not self.running:
            logger.warning("Async quality scorer not running, starting now")
            await self.start()

        # Add to queue
        await self.queue.put((memory, query, storage))
        self.stats["total_queued"] += 1
        self.stats["queue_size"] = self.queue.qsize()

        logger.debug(
            f"Queued memory {memory.content_hash[:8]} for scoring "
            f"(queue size: {self.queue.qsize()})"
        )

    async def _worker(self):
        """
        Background worker that processes scoring queue.

        Runs continuously while self.running is True.
        Scores memories and optionally persists updates to storage.
        """
        logger.info("Background quality scoring worker started")

        while self.running:
            try:
                # Wait for next item with timeout
                try:
                    memory, query, storage = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # No items in queue, continue loop
                    continue

                # Score the memory
                try:
                    logger.debug(f"Scoring memory {memory.content_hash[:8]} in background")

                    # Get AI score (may be None if disabled or fails)
                    ai_score = await self.evaluator.evaluate_quality(query, memory)

                    # Calculate composite quality score
                    quality_score = await self.scorer.calculate_quality_score(
                        memory, query, ai_score
                    )

                    # Update memory metadata (already done by scorer, but ensure it's set)
                    memory.metadata['quality_score'] = quality_score
                    # Provider is already set in memory.metadata by evaluate_quality()
                    quality_provider = memory.metadata.get('quality_provider', 'unknown')

                    # Persist to storage if provided
                    if storage:
                        try:
                            await storage.update_memory_metadata(
                                content_hash=memory.content_hash,
                                updates={
                                    'quality_score': quality_score,
                                    'quality_provider': quality_provider,
                                    'ai_scores': memory.metadata.get('ai_scores', []),
                                    'quality_components': memory.metadata.get('quality_components', {})
                                },
                                preserve_timestamps=True
                            )
                            logger.debug(
                                f"Persisted quality score {quality_score:.3f} for "
                                f"memory {memory.content_hash[:8]}"
                            )
                        except Exception as e:
                            logger.error(f"Failed to persist quality score: {e}")
                            self.stats["total_errors"] += 1

                    self.stats["total_scored"] += 1
                    self.stats["queue_size"] = self.queue.qsize()

                except Exception as e:
                    logger.error(f"Background scoring failed for memory {memory.content_hash[:8]}: {e}")
                    self.stats["total_errors"] += 1

                finally:
                    # Mark task as done
                    self.queue.task_done()

            except asyncio.CancelledError:
                logger.info("Background worker cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in background worker: {e}")
                # Continue running despite errors

        logger.info("Background quality scoring worker stopped")

    def get_stats(self) -> dict:
        """
        Get current statistics for the scoring queue.

        Returns:
            Dictionary with queue statistics:
            - total_queued: Total memories queued since start
            - total_scored: Total memories successfully scored
            - total_errors: Total errors encountered
            - queue_size: Current number of items in queue
            - is_running: Whether worker is currently running
        """
        return {
            **self.stats,
            "is_running": self.running
        }


# Global instance for use across the application
async_scorer = AsyncQualityScorer()
