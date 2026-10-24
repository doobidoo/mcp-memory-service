"""Delta-sync protocol capability negotiation helpers.

``sync_protocol`` and ``sync_protocol_min`` describe the wire protocol range;
``KNOWN_SCHEMA_VERSION`` separately describes the event-envelope schema. A
shared protocol version does not imply that peers can safely decode every event.
"""

from typing import Any

from ...config.embedding import EMBEDDING_MODEL_NAME
from .apply import KNOWN_SCHEMA_VERSION

SYNC_PROTOCOL_VERSION = 1
SYNC_PROTOCOL_MIN_VERSION = 1


class SyncProtocolCompatibilityError(ValueError):
    """Raised when a peer cannot safely exchange delta-sync events."""


def resolve_embedding_identity(storage: Any) -> tuple[str, int]:
    """Return the model and dimension used by the storage's actual vector source."""
    model_source = storage
    if (
        "hybrid" in type(storage).__name__.lower()
        and getattr(storage, "primary", None) is not None
    ):
        model_source = storage.primary

    model = getattr(model_source, "embedding_model_name", None) or EMBEDDING_MODEL_NAME
    if not isinstance(model, str) or not model:
        raise SyncProtocolCompatibilityError("Local embedding model is unavailable")

    dimension = getattr(model_source, "embedding_dimension", None)
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        dimension = getattr(storage, "embedding_dimension", None)
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        dimension = 384
    return model, dimension


def negotiate_sync_protocol(capabilities: dict[str, Any]) -> int:
    """Negotiate the highest common protocol version with a peer."""

    def read_version(field: str) -> int:
        value = capabilities.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SyncProtocolCompatibilityError(
                f"Peer advertises an invalid {field} value"
            )
        return value

    peer_max = read_version("sync_protocol")
    peer_min = read_version("sync_protocol_min")
    if peer_min > peer_max:
        raise SyncProtocolCompatibilityError(
            "Peer advertises an invalid sync protocol range"
        )

    lowest_common = max(SYNC_PROTOCOL_MIN_VERSION, peer_min)
    highest_common = min(SYNC_PROTOCOL_VERSION, peer_max)
    if lowest_common > highest_common:
        raise SyncProtocolCompatibilityError(
            "no common sync protocol version is supported by both peers"
        )
    return highest_common


def validate_event_schema(capabilities: dict[str, Any]) -> int:
    """Require the exact event envelope schema this implementation understands."""
    version = capabilities.get("event_schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise SyncProtocolCompatibilityError(
            "Peer advertises an invalid event_schema_version value"
        )
    if version != KNOWN_SCHEMA_VERSION:
        raise SyncProtocolCompatibilityError(
            "Event schema mismatch: local peer supports "
            f"{KNOWN_SCHEMA_VERSION}, remote peer advertises {version}"
        )
    return version
