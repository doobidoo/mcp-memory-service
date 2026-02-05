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

"""Memory-related data models."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Try to import dateutil, but fall back to standard datetime parsing if not available
try:
    from dateutil import parser as dateutil_parser

    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """Represents a single memory entry."""

    content: str
    content_hash: str
    tags: list[str] = field(default_factory=list)
    memory_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    # Timestamp fields with flexible input formats
    # Store as float and ISO8601 string for maximum compatibility
    created_at: float | None = None
    created_at_iso: str | None = None
    updated_at: float | None = None
    updated_at_iso: str | None = None

    # Legacy timestamp field (maintain for backward compatibility)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize timestamps after object creation."""
        # Synchronize the timestamps
        self._sync_timestamps(
            created_at=self.created_at,
            created_at_iso=self.created_at_iso,
            updated_at=self.updated_at,
            updated_at_iso=self.updated_at_iso,
        )

    def _sync_timestamps(self, created_at=None, created_at_iso=None, updated_at=None, updated_at_iso=None):
        """
        Synchronize timestamp fields to ensure all formats are available.
        Handles any combination of inputs and fills in missing values.
        Always uses UTC time.
        """
        now = time.time()

        def iso_to_float(iso_str: str) -> float:
            """Convert ISO string to float timestamp, ensuring UTC interpretation."""
            if DATEUTIL_AVAILABLE:
                # dateutil properly handles timezone info
                parsed_dt = dateutil_parser.isoparse(iso_str)
                return parsed_dt.timestamp()
            else:
                # Fallback to basic ISO parsing with explicit UTC handling
                try:
                    # Handle common ISO formats
                    if iso_str.endswith("Z"):
                        # UTC timezone indicated by 'Z'
                        dt = datetime.fromisoformat(iso_str[:-1])
                        # Treat as UTC and convert to timestamp
                        import calendar

                        return calendar.timegm(dt.timetuple()) + dt.microsecond / 1000000.0
                    elif "+" in iso_str or iso_str.count("-") > 2:
                        # Has timezone info, use fromisoformat in Python 3.7+
                        dt = datetime.fromisoformat(iso_str)
                        return dt.timestamp()
                    else:
                        # No timezone info, assume UTC
                        dt = datetime.fromisoformat(iso_str)
                        return calendar.timegm(dt.timetuple()) + dt.microsecond / 1000000.0
                except (ValueError, TypeError):
                    # Last resort: try strptime and treat as UTC
                    try:
                        dt = datetime.strptime(iso_str[:19], "%Y-%m-%dT%H:%M:%S")
                        return calendar.timegm(dt.timetuple())
                    except (ValueError, TypeError):
                        # If all parsing fails, return current timestamp
                        logging.warning(f"Failed to parse timestamp '{iso_str}', using current time")
                        return datetime.now().timestamp()

        def float_to_iso(ts: float) -> str:
            """Convert float timestamp to ISO string."""
            return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")

        # Handle created_at
        if created_at is not None and created_at_iso is not None:
            # Validate that they represent the same time (with more generous tolerance for timezone issues)
            try:
                iso_ts = iso_to_float(created_at_iso)
                time_diff = abs(created_at - iso_ts)
                # Allow up to 1 second difference for rounding, but reject obvious timezone mismatches
                if time_diff > 1.0 and time_diff < 86400:  # Between 1 second and 24 hours suggests timezone issue
                    logger.info(f"Timezone mismatch detected (diff: {time_diff}s), preferring float timestamp")
                    # Use the float timestamp as authoritative and regenerate ISO
                    self.created_at = created_at
                    self.created_at_iso = float_to_iso(created_at)
                elif time_diff >= 86400:  # More than 24 hours difference suggests data corruption
                    logger.warning(f"Large timestamp difference detected ({time_diff}s), using current time")
                    self.created_at = now
                    self.created_at_iso = float_to_iso(now)
                else:
                    # Small difference, keep both values
                    self.created_at = created_at
                    self.created_at_iso = created_at_iso
            except Exception as e:
                logger.warning(f"Error parsing timestamps: {e}, using float timestamp")
                self.created_at = created_at if created_at is not None else now
                self.created_at_iso = float_to_iso(self.created_at)
        elif created_at is not None:
            self.created_at = created_at
            self.created_at_iso = float_to_iso(created_at)
        elif created_at_iso:
            try:
                self.created_at = iso_to_float(created_at_iso)
                self.created_at_iso = created_at_iso
            except ValueError as e:
                logger.warning(f"Invalid created_at_iso: {e}")
                self.created_at = now
                self.created_at_iso = float_to_iso(now)
        else:
            self.created_at = now
            self.created_at_iso = float_to_iso(now)

        # Handle updated_at
        if updated_at is not None and updated_at_iso is not None:
            # Validate that they represent the same time (with more generous tolerance for timezone issues)
            try:
                iso_ts = iso_to_float(updated_at_iso)
                time_diff = abs(updated_at - iso_ts)
                # Allow up to 1 second difference for rounding, but reject obvious timezone mismatches
                if time_diff > 1.0 and time_diff < 86400:  # Between 1 second and 24 hours suggests timezone issue
                    logger.info(f"Timezone mismatch detected in updated_at (diff: {time_diff}s), preferring float timestamp")
                    # Use the float timestamp as authoritative and regenerate ISO
                    self.updated_at = updated_at
                    self.updated_at_iso = float_to_iso(updated_at)
                elif time_diff >= 86400:  # More than 24 hours difference suggests data corruption
                    logger.warning(f"Large timestamp difference detected in updated_at ({time_diff}s), using current time")
                    self.updated_at = now
                    self.updated_at_iso = float_to_iso(now)
                else:
                    # Small difference, keep both values
                    self.updated_at = updated_at
                    self.updated_at_iso = updated_at_iso
            except Exception as e:
                logger.warning(f"Error parsing updated timestamps: {e}, using float timestamp")
                self.updated_at = updated_at if updated_at is not None else now
                self.updated_at_iso = float_to_iso(self.updated_at)
        elif updated_at is not None:
            self.updated_at = updated_at
            self.updated_at_iso = float_to_iso(updated_at)
        elif updated_at_iso:
            try:
                self.updated_at = iso_to_float(updated_at_iso)
                self.updated_at_iso = updated_at_iso
            except ValueError as e:
                logger.warning(f"Invalid updated_at_iso: {e}")
                self.updated_at = now
                self.updated_at_iso = float_to_iso(now)
        else:
            self.updated_at = now
            self.updated_at_iso = float_to_iso(now)

        # Update legacy timestamp field for backward compatibility
        self.timestamp = datetime.fromtimestamp(self.created_at, timezone.utc)

    def touch(self):
        """Update the updated_at timestamps to the current time."""
        now = time.time()
        self.updated_at = now
        self.updated_at_iso = datetime.fromtimestamp(now, timezone.utc).isoformat().replace("+00:00", "Z")

    def to_dict(self) -> dict[str, Any]:
        """Convert memory to dictionary format for storage."""
        # Ensure timestamps are synchronized
        self._sync_timestamps(
            created_at=self.created_at,
            created_at_iso=self.created_at_iso,
            updated_at=self.updated_at,
            updated_at_iso=self.updated_at_iso,
        )

        return {
            "content": self.content,
            "content_hash": self.content_hash,
            "tags_str": ",".join(self.tags) if self.tags else "",
            "type": self.memory_type,
            # Store timestamps in all formats for better compatibility
            "timestamp": float(self.created_at),  # Changed from int() to preserve precision
            "timestamp_float": self.created_at,  # Legacy timestamp (float)
            "timestamp_str": self.created_at_iso,  # Legacy timestamp (ISO)
            # New timestamp fields
            "created_at": self.created_at,
            "created_at_iso": self.created_at_iso,
            "updated_at": self.updated_at,
            "updated_at_iso": self.updated_at_iso,
            **self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], embedding: list[float] | None = None) -> "Memory":
        """Create a Memory instance from dictionary data."""
        tags = data.get("tags_str", "").split(",") if data.get("tags_str") else []

        # Extract timestamps with different priorities
        # First check new timestamp fields (created_at/updated_at)
        created_at = data.get("created_at")
        created_at_iso = data.get("created_at_iso")
        updated_at = data.get("updated_at")
        updated_at_iso = data.get("updated_at_iso")

        # If new fields are missing, try to get from legacy timestamp fields
        if created_at is None and created_at_iso is None:
            if "timestamp_float" in data:
                created_at = float(data["timestamp_float"])
            elif "timestamp" in data:
                created_at = float(data["timestamp"])

            if "timestamp_str" in data and created_at_iso is None:
                created_at_iso = data["timestamp_str"]

        # Create metadata dictionary without special fields
        metadata = {
            k: v
            for k, v in data.items()
            if k
            not in [
                "content",
                "content_hash",
                "tags_str",
                "type",
                "timestamp",
                "timestamp_float",
                "timestamp_str",
                "created_at",
                "created_at_iso",
                "updated_at",
                "updated_at_iso",
            ]
        }

        # Create memory instance with synchronized timestamps
        return cls(
            content=data["content"],
            content_hash=data["content_hash"],
            tags=[tag for tag in tags if tag],  # Filter out empty tags
            memory_type=data.get("type"),
            metadata=metadata,
            embedding=embedding,
            created_at=created_at,
            created_at_iso=created_at_iso,
            updated_at=updated_at,
            updated_at_iso=updated_at_iso,
        )


@dataclass
class MemoryQueryResult:
    """Represents a memory query result with relevance score and debug information."""

    memory: Memory
    relevance_score: float
    debug_info: dict[str, Any] = field(default_factory=dict)

    @property
    def similarity_score(self) -> float:
        """Alias for relevance_score for backward compatibility."""
        return self.relevance_score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "memory": self.memory.to_dict(),
            "relevance_score": self.relevance_score,
            "similarity_score": self.relevance_score,
            "debug_info": self.debug_info,
        }
