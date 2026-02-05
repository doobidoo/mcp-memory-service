"""
Write Queue Coordination for SQLite Concurrency Management

Serializes concurrent write operations to prevent SQLite database locking
when multiple MCP clients attempt simultaneous writes. Uses asyncio.Queue
with backpressure (HTTP 429) when queue is full.

Design:
- Single asyncio.Queue (max size 20) for write operation serialization
- FIFO processing with background task coordination
- HTTP 429 response when queue capacity exceeded
- Per-operation error handling without crashing queue processor

Usage:
    from mcp_memory_service.web.write_queue import write_queue

    # In FastAPI endpoint:
    await write_queue.enqueue(storage.store, content=content, tags=tags)
    background_tasks.add_task(write_queue.process_queue)
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class WriteOperation:
    """Represents a single write operation to be queued."""

    def __init__(
        self,
        operation: Callable[..., Coroutine[Any, Any, Any]],
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        self.operation = operation
        self.args = args
        self.kwargs = kwargs or {}
        self.enqueued_at = datetime.now()
        self.future: asyncio.Future = asyncio.Future()

    async def execute(self) -> Any:
        """Execute the write operation and set result on future."""
        try:
            result = await self.operation(*self.args, **self.kwargs)
            self.future.set_result(result)
            return result
        except Exception as e:
            self.future.set_exception(e)
            raise


class WriteQueue:
    """
    Async queue for serializing concurrent write operations to SQLite.

    Prevents database locking by ensuring writes are processed sequentially
    in FIFO order. Provides backpressure via HTTP 429 when queue is full.

    Attributes:
        max_size: Maximum queue depth before rejecting new operations
        queue: asyncio.Queue for write operations
        processing: Flag to prevent concurrent queue processing
    """

    def __init__(self, max_size: int = 20):
        """
        Initialize write queue.

        Args:
            max_size: Maximum number of queued operations (default: 20)
        """
        self.max_size = max_size
        self.queue: asyncio.Queue[WriteOperation] = asyncio.Queue(maxsize=max_size)
        self.processing = False
        self._lock = asyncio.Lock()
        logger.info(f"WriteQueue initialized with max_size={max_size}")

    async def enqueue(
        self,
        operation: Callable[..., Coroutine[Any, Any, Any]],
        *args,
        **kwargs,
    ) -> asyncio.Future:
        """
        Enqueue a write operation for serialized execution.

        Args:
            operation: Async callable to execute (e.g., storage.store)
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            asyncio.Future that will contain the operation result

        Raises:
            HTTPException: 429 if queue is full (backpressure)
        """
        if self.queue.full():
            current_size = self.queue.qsize()
            logger.warning(f"Write queue full ({current_size}/{self.max_size}), rejecting operation")
            raise HTTPException(
                status_code=429,
                detail=f"Write queue full ({current_size}/{self.max_size}), please retry after 5 seconds",
                headers={"Retry-After": "5"},
            )

        write_op = WriteOperation(operation, args, kwargs)
        await self.queue.put(write_op)

        queue_size = self.queue.qsize()
        logger.debug(f"Operation enqueued, queue size: {queue_size}/{self.max_size}")

        return write_op.future

    async def process_queue(self) -> None:
        """
        Process queued write operations sequentially.

        This method should be called as a FastAPI background task. It prevents
        concurrent processing via a lock and processes all queued operations
        in FIFO order. Errors in individual operations are logged but don't
        crash the processor.

        Implementation:
        - Acquires lock to prevent concurrent processing
        - Processes operations until queue is empty
        - Handles per-operation exceptions gracefully
        - Releases lock when queue is drained
        """
        # Prevent concurrent queue processing
        if not self._lock.locked():
            async with self._lock:
                logger.debug("Starting queue processor")
                processed = 0
                errors = 0

                while not self.queue.empty():
                    try:
                        write_op = await asyncio.wait_for(self.queue.get(), timeout=1.0)

                        # Calculate queue latency
                        latency = (datetime.now() - write_op.enqueued_at).total_seconds()
                        if latency > 1.0:
                            logger.warning(f"Write operation queued for {latency:.2f}s")

                        # Execute the write operation
                        await write_op.execute()
                        processed += 1

                        self.queue.task_done()

                    except asyncio.TimeoutError:
                        # Queue became empty while processing
                        break
                    except Exception as e:
                        # Log error but continue processing remaining operations
                        errors += 1
                        logger.error(
                            f"Error processing write operation: {e}",
                            exc_info=True,
                        )

                logger.debug(f"Queue processor finished: {processed} processed, {errors} errors")
        else:
            logger.debug("Queue processor already running, skipping")

    def get_stats(self) -> dict[str, Any]:
        """
        Get current queue statistics.

        Returns:
            Dictionary with queue size and capacity information
        """
        return {
            "queue_size": self.queue.qsize(),
            "max_size": self.max_size,
            "processing": self._lock.locked(),
            "capacity_remaining": self.max_size - self.queue.qsize(),
        }


# Global write queue instance
write_queue = WriteQueue(max_size=20)
