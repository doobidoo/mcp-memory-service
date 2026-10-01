"""Lightweight metrics hooks used by core code without web dependencies."""

from __future__ import annotations

from collections.abc import Callable

EmbeddingObserver = Callable[[str, float], None]

_embedding_observer: EmbeddingObserver | None = None


def set_embedding_observer(observer: EmbeddingObserver | None) -> None:
    """Install or remove the process-wide embedding latency observer."""
    global _embedding_observer
    _embedding_observer = observer


def observe_embedding_duration(backend: str, seconds: float) -> None:
    """Record one embedding generation duration when metrics are enabled."""
    observer = _embedding_observer
    if observer is not None:
        observer(backend, seconds)
