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

"""Base classes and interfaces for memory consolidation components."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..models.memory import Memory

logger = logging.getLogger(__name__)

@dataclass
class ConsolidationConfig:
    """Configuration for consolidation operations."""
    
    # Decay settings
    decay_enabled: bool = True
    retention_periods: Dict[str, int] = field(default_factory=lambda: {
        # Base ontology types (from Phase 0 Ontology Foundation)
        'decision': 365,      # Strategic choices, architecture decisions
        'learning': 180,      # Insights, best practices, anti-patterns
        'pattern': 90,        # Recurring issues, code smells, design patterns
        'error': 30,          # Bugs, failures, exceptions
        'observation': 30,    # Code edits, file access, searches, commands

        # Legacy types for backward compatibility (mapped to new types)
        'critical': 365,      # Maps to decision
        'reference': 180,     # Maps to learning
        'standard': 30,       # Maps to observation
        'temporary': 7        # Maps to observation (short-lived)
    })
    
    # Association settings
    associations_enabled: bool = True
    min_similarity: float = 0.3
    max_similarity: float = 0.7
    max_pairs_per_run: int = 100
    
    # Clustering settings
    clustering_enabled: bool = True
    min_cluster_size: int = 5
    clustering_algorithm: str = 'dbscan'  # 'dbscan', 'hierarchical'
    
    # Compression settings
    compression_enabled: bool = True
    max_summary_length: int = 500
    preserve_originals: bool = True
    
    # Forgetting settings
    forgetting_enabled: bool = True
    relevance_threshold: float = 0.1
    access_threshold_days: int = 90
    archive_location: Optional[str] = None

    # Incremental consolidation settings
    batch_size: int = 500  # Memories to process per consolidation run
    incremental_mode: bool = True  # Enable oldest-first batch processing

@dataclass
class ConsolidationReport:
    """Report of consolidation operations performed."""
    time_horizon: str
    start_time: datetime
    end_time: datetime
    memories_processed: int
    associations_discovered: int = 0
    clusters_created: int = 0
    memories_compressed: int = 0
    memories_archived: int = 0
    errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryAssociation:
    """Represents a discovered association between memories."""
    source_memory_hashes: List[str]
    similarity_score: float
    connection_type: str
    discovery_method: str
    discovery_date: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryCluster:
    """Represents a cluster of semantically related memories."""
    cluster_id: str
    memory_hashes: List[str]
    centroid_embedding: List[float]
    coherence_score: float
    created_at: datetime
    theme_keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConsolidationBase(ABC):
    """Abstract base class for consolidation components."""
    
    def __init__(self, config: ConsolidationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def process(self, memories: List[Memory], **kwargs) -> Any:
        """Process the given memories and return results."""
        pass
    
    def _validate_memories(self, memories: List[Memory]) -> bool:
        """Validate that memories list is valid for processing."""
        if not memories:
            self.logger.warning("Empty memories list provided")
            return False
        
        for memory in memories:
            if not hasattr(memory, 'content_hash') or not memory.content_hash:
                self.logger.error(f"Memory missing content_hash: {memory}")
                return False
        
        return True
    
    def _get_memory_age_days(self, memory: Memory, reference_time: Optional[datetime] = None) -> int:
        """Get the age of a memory in days."""
        ref_time = reference_time or datetime.now()
        
        if memory.created_at:
            created_dt = datetime.utcfromtimestamp(memory.created_at)
            return (ref_time - created_dt).days
        elif memory.timestamp:
            return (ref_time - memory.timestamp).days
        else:
            self.logger.warning(f"Memory {memory.content_hash} has no timestamp")
            return 0
    
    def _extract_memory_type(self, memory: Memory) -> str:
        """Extract the memory type, with fallback to 'standard'."""
        return memory.memory_type or 'standard'
    
    def _is_protected_memory(self, memory: Memory) -> bool:
        """Check if a memory is protected from consolidation operations."""
        protected_tags = {'critical', 'important', 'reference', 'permanent'}
        return bool(set(memory.tags).intersection(protected_tags))

class ConsolidationError(Exception):
    """Base exception for consolidation operations."""
    pass

class ConsolidationConfigError(ConsolidationError):
    """Exception raised for configuration errors."""
    pass

class ConsolidationProcessingError(ConsolidationError):
    """Exception raised during processing operations."""
    pass