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

"""Creative association discovery engine for memory connections."""

import random
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from itertools import combinations
from datetime import datetime
from dataclasses import dataclass
import re

from .base import ConsolidationBase, ConsolidationConfig, MemoryAssociation
from ..models.memory import Memory

@dataclass
class AssociationAnalysis:
    """Analysis results for a potential memory association."""
    memory1_hash: str
    memory2_hash: str
    similarity_score: float
    connection_reasons: List[str]
    shared_concepts: List[str]
    temporal_relationship: Optional[str]
    tag_overlap: List[str]
    confidence_score: float

class CreativeAssociationEngine(ConsolidationBase):
    """
    Discovers creative connections between seemingly unrelated memories.
    
    Similar to how dreams create unexpected associations, this engine randomly
    pairs memories to discover non-obvious connections in the "sweet spot"
    of moderate similarity (0.3-0.7 range).
    """
    
    def __init__(self, config: ConsolidationConfig):
        super().__init__(config)
        self.min_similarity = config.min_similarity
        self.max_similarity = config.max_similarity
        self.max_pairs_per_run = config.max_pairs_per_run
        
        # Compile regex patterns for concept extraction
        self._concept_patterns = {
            'urls': re.compile(r'https?://[^\s]+'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'dates': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'),
            'numbers': re.compile(r'\b\d+\.?\d*\b'),
            'camelCase': re.compile(r'\b[a-z]+[A-Z][a-zA-Z]*\b'),
            'PascalCase': re.compile(r'\b[A-Z][a-z]*[A-Z][a-zA-Z]*\b'),
            'acronyms': re.compile(r'\b[A-Z]{2,}\b')
        }
    
    async def process(self, memories: List[Memory], **kwargs) -> List[MemoryAssociation]:
        """Discover creative associations between memories."""
        if not self._validate_memories(memories) or len(memories) < 2:
            return []
        
        # Get existing associations to avoid duplicates
        existing_associations = kwargs.get('existing_associations', set())
        
        # Sample memory pairs for analysis
        pairs = self._sample_memory_pairs(memories)
        
        associations = []
        for mem1, mem2 in pairs:
            # Skip if association already exists
            pair_key = tuple(sorted([mem1.content_hash, mem2.content_hash]))
            if pair_key in existing_associations:
                continue
            
            # Calculate semantic similarity
            similarity = await self._calculate_semantic_similarity(mem1, mem2)
            
            # Check if similarity is in the "sweet spot" for creative connections
            if self.min_similarity <= similarity <= self.max_similarity:
                analysis = await self._analyze_association(mem1, mem2, similarity)
                
                if analysis.confidence_score > 0.3:  # Minimum confidence threshold
                    association = await self._create_association_memory(analysis)
                    associations.append(association)
        
        self.logger.info(f"Discovered {len(associations)} creative associations from {len(pairs)} pairs")
        return associations
    
    def _sample_memory_pairs(self, memories: List[Memory]) -> List[Tuple[Memory, Memory]]:
        """Sample random pairs of memories for association discovery."""
        # Calculate maximum possible pairs
        total_possible = len(memories) * (len(memories) - 1) // 2
        max_pairs = min(self.max_pairs_per_run, total_possible)
        
        if total_possible <= max_pairs:
            # Return all possible pairs if total is manageable
            return list(combinations(memories, 2))
        else:
            # Randomly sample pairs to prevent combinatorial explosion
            all_pairs = list(combinations(memories, 2))
            return random.sample(all_pairs, max_pairs)
    
    async def _calculate_semantic_similarity(self, mem1: Memory, mem2: Memory) -> float:
        """Calculate semantic similarity between two memories using embeddings."""
        if not mem1.embedding or not mem2.embedding:
            # Fallback to text-based similarity if embeddings unavailable
            return self._calculate_text_similarity(mem1.content, mem2.content)
        
        # Use cosine similarity for embeddings
        embedding1 = np.array(mem1.embedding)
        embedding2 = np.array(mem2.embedding)
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Convert to 0-1 range (cosine similarity can be -1 to 1)
        return (similarity + 1) / 2
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Fallback text similarity using word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def _analyze_association(
        self, 
        mem1: Memory, 
        mem2: Memory, 
        similarity: float
    ) -> AssociationAnalysis:
        """Analyze why two memories might be associated."""
        connection_reasons = []
        shared_concepts = []
        tag_overlap = []
        temporal_relationship = None
        
        # Analyze tag overlap
        tags1 = set(mem1.tags)
        tags2 = set(mem2.tags)
        tag_overlap = list(tags1.intersection(tags2))
        if tag_overlap:
            connection_reasons.append("shared_tags")
        
        # Analyze temporal relationship
        temporal_relationship = self._analyze_temporal_relationship(mem1, mem2)
        if temporal_relationship:
            connection_reasons.append("temporal_proximity")
        
        # Extract and compare concepts
        concepts1 = self._extract_concepts(mem1.content)
        concepts2 = self._extract_concepts(mem2.content)
        shared_concepts = list(concepts1.intersection(concepts2))
        if shared_concepts:
            connection_reasons.append("shared_concepts")
        
        # Analyze content patterns
        if self._has_similar_structure(mem1.content, mem2.content):
            connection_reasons.append("similar_structure")
        
        if self._has_complementary_content(mem1.content, mem2.content):
            connection_reasons.append("complementary_content")
        
        # Calculate confidence score based on multiple factors
        confidence_score = self._calculate_confidence_score(
            similarity, len(connection_reasons), len(shared_concepts), len(tag_overlap)
        )
        
        return AssociationAnalysis(
            memory1_hash=mem1.content_hash,
            memory2_hash=mem2.content_hash,
            similarity_score=similarity,
            connection_reasons=connection_reasons,
            shared_concepts=shared_concepts,
            temporal_relationship=temporal_relationship,
            tag_overlap=tag_overlap,
            confidence_score=confidence_score
        )
    
    def _extract_concepts(self, text: str) -> Set[str]:
        """Extract key concepts from text using various patterns."""
        concepts = set()
        
        # Extract different types of concepts
        for concept_type, pattern in self._concept_patterns.items():
            matches = pattern.findall(text)
            concepts.update(matches)
        
        # Extract capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        concepts.update(capitalized_words)
        
        # Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', text)
        concepts.update(quoted_phrases)
        
        # Extract common important words (filter out common stop words)
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w{4,}\b', text.lower())  # Words with 4+ characters
        important_words = [word for word in words if word not in stop_words]
        concepts.update(important_words[:10])  # Limit to top 10 words
        
        return concepts
    
    def _analyze_temporal_relationship(self, mem1: Memory, mem2: Memory) -> Optional[str]:
        """Analyze temporal relationship between memories."""
        if not (mem1.created_at and mem2.created_at):
            return None
        
        time_diff = abs(mem1.created_at - mem2.created_at)
        days_diff = time_diff / (24 * 3600)  # Convert to days
        
        if days_diff < 1:
            return "same_day"
        elif days_diff < 7:
            return "same_week"
        elif days_diff < 30:
            return "same_month"
        elif days_diff < 365:
            return "same_year"
        else:
            return "different_years"
    
    def _has_similar_structure(self, text1: str, text2: str) -> bool:
        """Check if texts have similar structural patterns."""
        # Check for similar formatting patterns
        patterns = [
            r'\n\s*[-*+]\s+',  # List items
            r'\n\s*\d+\.\s+',  # Numbered lists
            r'\n#{1,6}\s+',    # Headers
            r'```[\s\S]*?```', # Code blocks
            r'\[.*?\]\(.*?\)', # Links
        ]
        
        for pattern in patterns:
            matches1 = len(re.findall(pattern, text1))
            matches2 = len(re.findall(pattern, text2))
            
            if matches1 > 0 and matches2 > 0:
                return True
        
        return False
    
    def _has_complementary_content(self, text1: str, text2: str) -> bool:
        """Check if texts contain complementary information."""
        # Look for question-answer patterns
        has_question1 = bool(re.search(r'\?', text1))
        has_question2 = bool(re.search(r'\?', text2))
        
        # If one has questions and the other doesn't, they might be complementary
        if has_question1 != has_question2:
            return True
        
        # Look for problem-solution patterns
        problem_words = ['problem', 'issue', 'error', 'bug', 'fail', 'wrong']
        solution_words = ['solution', 'fix', 'resolve', 'answer', 'correct', 'solve']
        
        has_problem1 = any(word in text1.lower() for word in problem_words)
        has_solution1 = any(word in text1.lower() for word in solution_words)
        has_problem2 = any(word in text2.lower() for word in problem_words)
        has_solution2 = any(word in text2.lower() for word in solution_words)
        
        # Complementary if one focuses on problems, other on solutions
        if (has_problem1 and has_solution2) or (has_solution1 and has_problem2):
            return True
        
        return False
    
    def _calculate_confidence_score(
        self,
        similarity: float,
        num_reasons: int,
        num_shared_concepts: int,
        num_shared_tags: int
    ) -> float:
        """Calculate confidence score for the association."""
        base_score = similarity
        
        # Boost for multiple connection reasons
        reason_boost = min(0.3, num_reasons * 0.1)
        
        # Boost for shared concepts
        concept_boost = min(0.2, num_shared_concepts * 0.05)
        
        # Boost for shared tags
        tag_boost = min(0.2, num_shared_tags * 0.1)
        
        total_score = base_score + reason_boost + concept_boost + tag_boost
        
        return min(1.0, total_score)
    
    async def _create_association_memory(self, analysis: AssociationAnalysis) -> MemoryAssociation:
        """Create a memory association from analysis results."""
        return MemoryAssociation(
            source_memory_hashes=[analysis.memory1_hash, analysis.memory2_hash],
            similarity_score=analysis.similarity_score,
            connection_type=', '.join(analysis.connection_reasons),
            discovery_method="creative_association",
            discovery_date=datetime.now(),
            metadata={
                "shared_concepts": analysis.shared_concepts,
                "temporal_relationship": analysis.temporal_relationship,
                "tag_overlap": analysis.tag_overlap,
                "confidence_score": analysis.confidence_score,
                "analysis_version": "1.0"
            }
        )
    
    async def filter_high_confidence_associations(
        self,
        associations: List[MemoryAssociation],
        min_confidence: float = 0.5
    ) -> List[MemoryAssociation]:
        """Filter associations by confidence score."""
        return [
            assoc for assoc in associations
            if assoc.metadata.get('confidence_score', 0) >= min_confidence
        ]
    
    async def group_associations_by_type(
        self,
        associations: List[MemoryAssociation]
    ) -> Dict[str, List[MemoryAssociation]]:
        """Group associations by their connection type."""
        groups = {}
        for assoc in associations:
            conn_type = assoc.connection_type
            if conn_type not in groups:
                groups[conn_type] = []
            groups[conn_type].append(assoc)
        
        return groups