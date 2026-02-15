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
import calendar

from mcp_memory_service.models.ontology import MemoryTypeOntology
from mcp_memory_service.models.tag_taxonomy import TagTaxonomy

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
        """Initialize timestamps and validate ontology after object creation."""
        # Synchronize the timestamps
        self._sync_timestamps(
            created_at=self.created_at,
            created_at_iso=self.created_at_iso,
            updated_at=self.updated_at,
            updated_at_iso=self.updated_at_iso
        )

        # Validate memory_type using ontology
        if self.memory_type is not None:
            if not MemoryTypeOntology.validate_memory_type(self.memory_type):
                logger.warning(
                    f"Invalid memory_type '{self.memory_type}'. "
                    f"Valid types: {', '.join(MemoryTypeOntology.get_all_types()[:5])}... "
                    f"Defaulting to 'observation'."
                )
                self.memory_type = "observation"  # Default to base type

        # Validate tags using taxonomy (soft validation)
        if self.tags:
            invalid_tags = []
            for tag in self.tags:
                # Parse tag to check namespace (parse only once for efficiency)
                namespace, value = TagTaxonomy.parse_tag(tag)
                if namespace is not None:  # Has namespace
                    # Direct check using exposed VALID_NAMESPACES (avoids re-parsing)
                    if namespace not in TagTaxonomy.VALID_NAMESPACES:
                        invalid_tags.append(tag)

            if invalid_tags:
                logger.info(
                    f"Tags with invalid namespaces: {', '.join(invalid_tags)}. "
                    f"Valid namespaces: sys:, q:, proj:, topic:, t:, user:. "
                    f"Legacy tags (no namespace) are still supported."
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
                    if iso_str.endswith('Z'):
                        # UTC timezone indicated by 'Z'
                        dt = datetime.fromisoformat(iso_str[:-1])
                        # Treat as UTC and convert to timestamp
                        return calendar.timegm(dt.timetuple()) + dt.microsecond / 1000000.0
                    elif '+' in iso_str or iso_str.count('-') > 2:
                        # Has timezone info, use fromisoformat in Python 3.7+
                        dt = datetime.fromisoformat(iso_str)
                        return dt.timestamp()
                    else:
                        # No timezone info, assume UTC
                        dt = datetime.fromisoformat(iso_str)
                        return calendar.timegm(dt.timetuple()) + dt.microsecond / 1000000.0
                except (ValueError, TypeError) as e:
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
            return datetime.utcfromtimestamp(ts).isoformat() + "Z"

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
        self.timestamp = datetime.utcfromtimestamp(self.created_at)

    def touch(self):
        """Update the updated_at timestamps to the current time."""
        now = time.time()
        self.updated_at = now
        self.updated_at_iso = datetime.utcfromtimestamp(now).isoformat() + "Z"

    @property
    def quality_score(self) -> float:
        """Get quality score from metadata (default: 0.5)."""
        return self.metadata.get('quality_score', 0.5)

    @property
    def quality_provider(self) -> Optional[str]:
        """Get the provider used for quality scoring."""
        return self.metadata.get('quality_provider')

    @property
    def access_count(self) -> int:
        """Get number of times this memory has been accessed."""
        return self.metadata.get('access_count', 0)

    @property
    def last_accessed_at(self) -> Optional[float]:
        """Get timestamp of last access (Unix timestamp)."""
        return self.metadata.get('last_accessed_at')

    # SHODH Unified API Spec - Source & Trust fields
    @property
    def source_type(self) -> str:
        """Get source type (user, system, api, file, web, ai_generated, inferred)."""
        return self.metadata.get('source_type', 'user')

    @source_type.setter
    def source_type(self, value: str):
        """Set source type."""
        self.metadata['source_type'] = value

    @property
    def credibility(self) -> float:
        """Get credibility score (0.0-1.0)."""
        return self.metadata.get('credibility', 1.0)

    @credibility.setter
    def credibility(self, value: float):
        """Set credibility score."""
        self.metadata['credibility'] = min(max(value, 0.0), 1.0)

    # SHODH Unified API Spec - Emotional Metadata fields
    @property
    def emotion(self) -> Optional[str]:
        """Get emotion label (joy, frustration, surprise, relief, etc.)."""
        return self.metadata.get('emotion')

    @emotion.setter
    def emotion(self, value: Optional[str]):
        """Set emotion label."""
        self.metadata['emotion'] = value

    @property
    def emotional_valence(self) -> Optional[float]:
        """Get emotional valence (-1.0 to 1.0, negative to positive sentiment)."""
        return self.metadata.get('emotional_valence')

    @emotional_valence.setter
    def emotional_valence(self, value: Optional[float]):
        """Set emotional valence."""
        if value is not None:
            self.metadata['emotional_valence'] = min(max(value, -1.0), 1.0)
        else:
            self.metadata['emotional_valence'] = None

    @property
    def emotional_arousal(self) -> Optional[float]:
        """Get emotional arousal (0.0 to 1.0, calm to aroused intensity)."""
        return self.metadata.get('emotional_arousal')

    @emotional_arousal.setter
    def emotional_arousal(self, value: Optional[float]):
        """Set emotional arousal."""
        if value is not None:
            self.metadata['emotional_arousal'] = min(max(value, 0.0), 1.0)
        else:
            self.metadata['emotional_arousal'] = None

    # SHODH Unified API Spec - Episodic Memory fields
    @property
    def episode_id(self) -> Optional[str]:
        """Get episode ID (groups related memories into coherent episodes)."""
        return self.metadata.get('episode_id')

    @episode_id.setter
    def episode_id(self, value: Optional[str]):
        """Set episode ID."""
        self.metadata['episode_id'] = value

    @property
    def sequence_number(self) -> Optional[int]:
        """Get sequence number (order within an episode)."""
        return self.metadata.get('sequence_number')

    @sequence_number.setter
    def sequence_number(self, value: Optional[int]):
        """Set sequence number."""
        self.metadata['sequence_number'] = value

    @property
    def preceding_memory_id(self) -> Optional[str]:
        """Get preceding memory ID (for episodic threading)."""
        return self.metadata.get('preceding_memory_id')

    @preceding_memory_id.setter
    def preceding_memory_id(self, value: Optional[str]):
        """Set preceding memory ID."""
        self.metadata['preceding_memory_id'] = value

    def record_access(self, query: Optional[str] = None):
        """
        Record memory access for implicit signals tracking.

        Args:
            query: Optional query that triggered this access
        """
        now = time.time()
        self.metadata['access_count'] = self.access_count + 1
        self.metadata['last_accessed_at'] = now
        if query:
            # Store recent access queries for analysis (keep last 10)
            access_queries = self.metadata.get('access_queries', [])
            access_queries.append({
                'query': query,
                'timestamp': now
            })
            self.metadata['access_queries'] = access_queries[-10:]

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
            **self.metadata,
            "tags": self.tags,  # override any leaked metadata["tags"]
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
    
    @property
    def similarity_score(self) -> float:
        """Alias for relevance_score for backward compatibility."""
        return self.relevance_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "memory": self.memory.to_dict(),
            "relevance_score": self.relevance_score,
            "similarity_score": self.relevance_score,
            "debug_info": self.debug_info
        }
