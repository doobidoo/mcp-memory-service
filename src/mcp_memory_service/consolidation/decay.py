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

"""Exponential decay scoring for memory relevance calculation."""

import math
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass

from .base import ConsolidationBase, ConsolidationConfig
from ..models.memory import Memory

@dataclass
class RelevanceScore:
    """Represents a memory's relevance score with breakdown."""
    memory_hash: str
    total_score: float
    base_importance: float
    decay_factor: float
    connection_boost: float
    access_boost: float
    metadata: Dict[str, Any]

class ExponentialDecayCalculator(ConsolidationBase):
    """
    Calculates memory relevance using exponential decay.
    
    Memories naturally lose relevance over time unless reinforced by:
    - Connections to other memories
    - Recent access patterns  
    - Base importance scores
    - Memory type-specific retention periods
    """
    
    def __init__(self, config: ConsolidationConfig):
        super().__init__(config)
        self.retention_periods = config.retention_periods
        
    async def process(self, memories: List[Memory], **kwargs) -> List[RelevanceScore]:
        """Calculate relevance scores for all memories."""
        if not self._validate_memories(memories):
            return []
        
        reference_time = kwargs.get('reference_time', datetime.now())
        memory_connections = kwargs.get('connections', {})  # hash -> connection_count mapping
        access_patterns = kwargs.get('access_patterns', {})  # hash -> last_accessed mapping
        
        scores = []
        for memory in memories:
            score = await self._calculate_memory_relevance(
                memory, reference_time, memory_connections, access_patterns
            )
            scores.append(score)
        
        self.logger.info(f"Calculated relevance scores for {len(scores)} memories")
        return scores
    
    async def _calculate_memory_relevance(
        self,
        memory: Memory,
        current_time: datetime,
        connections: Dict[str, int],
        access_patterns: Dict[str, datetime]
    ) -> RelevanceScore:
        """
        Calculate memory relevance using exponential decay.
        
        Factors:
        - Age of memory
        - Base importance score (from metadata or tags)
        - Retention period (varies by memory type)
        - Connections to other memories
        - Recent access patterns
        """
        # Get memory age in days
        age_days = self._get_memory_age_days(memory, current_time)
        
        # Extract base importance score
        base_importance = self._get_base_importance(memory)
        
        # Get retention period for memory type
        memory_type = self._extract_memory_type(memory)
        retention_period = self.retention_periods.get(memory_type, 30)
        
        # Calculate exponential decay factor
        decay_factor = math.exp(-age_days / retention_period)
        
        # Calculate connection boost
        connection_count = connections.get(memory.content_hash, 0)
        connection_boost = 1 + (0.1 * connection_count)  # 10% boost per connection
        
        # Calculate access boost
        access_boost = self._calculate_access_boost(memory, access_patterns, current_time)
        
        # Calculate total relevance score
        total_score = base_importance * decay_factor * connection_boost * access_boost
        
        # Ensure protected memories maintain minimum relevance
        if self._is_protected_memory(memory):
            total_score = max(total_score, 0.5)  # Minimum 50% relevance for protected memories
        
        return RelevanceScore(
            memory_hash=memory.content_hash,
            total_score=total_score,
            base_importance=base_importance,
            decay_factor=decay_factor,
            connection_boost=connection_boost,
            access_boost=access_boost,
            metadata={
                'age_days': age_days,
                'memory_type': memory_type,
                'retention_period': retention_period,
                'connection_count': connection_count,
                'is_protected': self._is_protected_memory(memory)
            }
        )
    
    def _get_base_importance(self, memory: Memory) -> float:
        """
        Extract base importance score from memory metadata or tags.
        
        Priority order:
        1. Explicit importance_score in metadata
        2. Importance derived from tags
        3. Default score of 1.0
        """
        # Check for explicit importance score
        if 'importance_score' in memory.metadata:
            try:
                score = float(memory.metadata['importance_score'])
                return max(0.0, min(2.0, score))  # Clamp between 0 and 2
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid importance_score in memory {memory.content_hash}")
        
        # Derive importance from tags
        tag_importance = {
            'critical': 2.0,
            'important': 1.5,
            'reference': 1.3,
            'urgent': 1.4,
            'project': 1.2,
            'personal': 1.1,
            'temporary': 0.7,
            'draft': 0.8,
            'note': 0.9
        }
        
        max_tag_importance = 1.0
        for tag in memory.tags:
            tag_score = tag_importance.get(tag.lower(), 1.0)
            max_tag_importance = max(max_tag_importance, tag_score)
        
        return max_tag_importance
    
    def _calculate_access_boost(
        self,
        memory: Memory,
        access_patterns: Dict[str, datetime],
        current_time: datetime
    ) -> float:
        """
        Calculate boost factor based on recent access patterns.
        
        Recent access increases relevance:
        - Accessed within last day: 1.5x boost
        - Accessed within last week: 1.2x boost  
        - Accessed within last month: 1.1x boost
        - No recent access: 1.0x (no boost)
        """
        last_accessed = access_patterns.get(memory.content_hash)
        
        if not last_accessed:
            # Check memory's own updated_at timestamp
            if memory.updated_at:
                last_accessed = datetime.fromtimestamp(memory.updated_at, tz=timezone.utc)
            else:
                return 1.0  # No access data available
        
        days_since_access = (current_time - last_accessed).days
        
        if days_since_access <= 1:
            return 1.5  # Accessed within last day
        elif days_since_access <= 7:
            return 1.2  # Accessed within last week
        elif days_since_access <= 30:
            return 1.1  # Accessed within last month
        else:
            return 1.0  # No recent access
    
    async def get_low_relevance_memories(
        self,
        scores: List[RelevanceScore],
        threshold: float = 0.1
    ) -> List[RelevanceScore]:
        """Get memories with relevance scores below the threshold."""
        return [score for score in scores if score.total_score < threshold]
    
    async def get_high_relevance_memories(
        self,
        scores: List[RelevanceScore], 
        threshold: float = 1.0
    ) -> List[RelevanceScore]:
        """Get memories with relevance scores above the threshold."""
        return [score for score in scores if score.total_score >= threshold]
    
    async def update_memory_relevance_metadata(
        self,
        memory: Memory,
        score: RelevanceScore
    ) -> Memory:
        """Update memory metadata with calculated relevance score."""
        memory.metadata.update({
            'relevance_score': score.total_score,
            'relevance_calculated_at': datetime.now().isoformat(),
            'decay_factor': score.decay_factor,
            'connection_boost': score.connection_boost,
            'access_boost': score.access_boost
        })
        memory.touch()  # Update the updated_at timestamp
        return memory