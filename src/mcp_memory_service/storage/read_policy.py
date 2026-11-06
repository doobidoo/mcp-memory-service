"""Request-local error policy for reads preceding destructive maintenance."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_strict_reads = ContextVar("mcp_storage_strict_reads", default=False)


def should_raise_read_error() -> bool:
    """Whether this task must propagate read errors instead of empty fallbacks."""
    return _strict_reads.get()


@contextmanager
def strict_reads() -> Iterator[None]:
    """Prevent read failures looking like empty stores in this task only."""
    token = _strict_reads.set(True)
    try:
        yield
    finally:
        _strict_reads.reset(token)
