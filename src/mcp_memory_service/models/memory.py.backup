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
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import time
import logging

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
    tags: List[str] = field(default_factory=list)
    memory_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    # Timestamp fields with flexible input formats
    # Store as float and ISO8601 string for maximum compatibility
    created_at: Optional[float] = None
    created_at_iso: Optional[str] = None
    updated_at: Optional[float] = None
    updated_at_iso: Optional[str] = None
    
    # Legacy timestamp field (maintain for backward compatibility)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize timestamps after object creation."""
        # Synchronize the timestamps
        self._sync_timestamps(
            created_at=self.created_at,
            created_at_iso=self.created_at_iso,
            updated_at=self.updated_at,
            updated_at_iso=self.updated_at_iso
        )

    def _sync_timestamps(self, created_at=None, created_at_iso=None, updated_at=None, updated_at_iso=None):
        """
        Synchronize timestamp fields to ensure all formats are available.
        Handles any combination of inputs and fills in missing values.
        Always uses UTC time.
        """
        now = time.time()
        
        def iso_to_float(iso_str: str) -> float:
            """Convert ISO string to float timestamp."""
            if DATEUTIL_AVAILABLE:
                return dateutil_parser.isoparse(iso_str).timestamp()
            else:
                # Fallback to basic ISO parsing
                try:
                    # Handle common ISO formats
                    if iso_str.endswith('Z'):
                        dt = datetime.fromisoformat(iso_str[:-1])
                    elif '+' in iso_str or iso_str.count('-') > 2:
                        # Has timezone info, use fromisoformat in Python 3.7+
                        dt = datetime.fromisoformat(iso_str)
                    else:
                        dt = datetime.fromisoformat(iso_str)
                    return dt.timestamp()
                except:
                    # Last resort: try strptime
                    dt = datetime.strptime(iso_str[:19], "%Y-%m-%dT%H:%M:%S")
                    return dt.timestamp()

        def float_to_iso(ts: float) -> str:
            """Convert float timestamp to ISO string."""
            return datetime.utcfromtimestamp(ts).isoformat() + "Z"

        # Handle created_at
        if created_at is not None and created_at_iso is not None:
            # Validate that they represent the same time
            try:
                iso_ts = iso_to_float(created_at_iso)
                if abs(created_at - iso_ts) > 1e-6:  # Allow for small floating-point differences
                    raise ValueError("created_at and created_at_iso do not match")
                self.created_at = created_at
                self.created_at_iso = created_at_iso
            except ValueError as e:
                logger.warning(f"Invalid created_at or created_at_iso: {e}")
                self.created_at = now
                self.created_at_iso = float_to_iso(now)
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
            # Validate that they represent the same time
            try:
                iso_ts = iso_to_float(updated_at_iso)
                if abs(updated_at - iso_ts) > 1e-6:  # Allow for small floating-point differences
                    raise ValueError("updated_at and updated_at_iso do not match")
                self.updated_at = updated_at
                self.updated_at_iso = updated_at_iso
            except ValueError as e:
                logger.warning(f"Invalid updated_at or updated_at_iso: {e}")
                self.updated_at = now
                self.updated_at_iso = float_to_iso(now)
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
        self.timestamp = datetime.utcfromtimestamp(self.created_at)

    def touch(self):
        """Update the updated_at timestamps to the current time."""
        now = time.time()
        self.updated_at = now
        self.updated_at_iso = datetime.utcfromtimestamp(now).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary format for storage."""
        # Ensure timestamps are synchronized
        self._sync_timestamps(
            created_at=self.created_at,
            created_at_iso=self.created_at_iso,
            updated_at=self.updated_at,
            updated_at_iso=self.updated_at_iso
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
            **self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: Optional[List[float]] = None) -> 'Memory':
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
            k: v for k, v in data.items() 
            if k not in [
                "content", "content_hash", "tags_str", "type",
                "timestamp", "timestamp_float", "timestamp_str",
                "created_at", "created_at_iso", "updated_at", "updated_at_iso"
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
            updated_at_iso=updated_at_iso
        )

@dataclass
class MemoryQueryResult:
    """Represents a memory query result with relevance score and debug information."""
    memory: Memory
    relevance_score: float
    debug_info: Dict[str, Any] = field(default_factory=dict)
