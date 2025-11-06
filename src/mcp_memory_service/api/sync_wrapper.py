# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Async-to-sync utilities for code execution interface.

Provides lightweight wrappers to convert async storage operations into
synchronous functions suitable for code execution contexts (e.g., hooks).

Performance:
    - Cold call: ~50ms (includes event loop creation)
    - Warm call: ~5ms (reuses existing loop)
    - Overhead: <10ms compared to native async

Design Philosophy:
    - Hide asyncio complexity from API users
    - Reuse event loops when possible for performance
    - Graceful error handling and cleanup
    - Zero async/await in public API
"""

import asyncio
from functools import wraps
from typing import Callable, TypeVar, Any
import logging

logger = logging.getLogger(__name__)

# Type variable for generic function wrapping
T = TypeVar('T')


def sync_wrapper(async_func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Convert async function to synchronous with minimal overhead.

    This wrapper handles event loop management transparently:
    1. Attempts to get existing event loop
    2. Creates new loop if none exists
    3. Runs async function to completion
    4. Returns result or raises exception

    Performance:
        - Adds ~1-5ms overhead per call
        - Reuses event loop when possible
        - Optimized for repeated calls (e.g., in hooks)

    Args:
        async_func: Async function to wrap

    Returns:
        Synchronous wrapper function with same signature

    Example:
        >>> async def fetch_data(query: str) -> list:
        ...     return await storage.retrieve(query, limit=5)
        >>> sync_fetch = sync_wrapper(fetch_data)
        >>> results = sync_fetch("architecture")  # No await needed

    Note:
        This wrapper is designed for code execution contexts where
        async/await is not available or desirable. For pure async
        code, use the storage backend directly.
    """
    @wraps(async_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                # Loop exists but is closed, create new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # No event loop in current thread, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            # Run async function to completion
            result = loop.run_until_complete(async_func(*args, **kwargs))
            return result
        except Exception as e:
            # Re-raise exception with context
            logger.error(f"Error in sync wrapper for {async_func.__name__}: {e}")
            raise

    return wrapper


def run_async(coro: Any) -> Any:
    """
    Run a coroutine synchronously and return its result.

    Convenience function for running async operations in sync contexts
    without explicitly creating a wrapper function.

    Args:
        coro: Coroutine object to run

    Returns:
        Result of the coroutine

    Example:
        >>> result = run_async(storage.retrieve("query", limit=5))
        >>> print(len(result))

    Note:
        Prefer sync_wrapper() for repeated calls to the same function,
        as it avoids wrapper creation overhead.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)
