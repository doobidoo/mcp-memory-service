"""Delta-sync resolver and event handling."""

from .resolver import EventView, resolve, reduce_events

__all__ = ['EventView', 'resolve', 'reduce_events']