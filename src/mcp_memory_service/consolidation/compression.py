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

"""Semantic compression engine for memory cluster summarization."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass
from collections import Counter
import re
import hashlib

from .base import ConsolidationBase, ConsolidationConfig, MemoryCluster
from ..models.memory import Memory

@dataclass
class CompressionResult:
    """Result of compressing a memory cluster."""
    cluster_id: str
    compressed_memory: Memory
    compression_ratio: float
    key_concepts: List[str]
    temporal_span: Dict[str, Any]
    source_memory_count: int
    compression_metadata: Dict[str, Any]

class SemanticCompressionEngine(ConsolidationBase):
    """
    Creates condensed representations of memory clusters for efficient storage.
    
    This creates higher-level abstractions while preserving key information,
    using statistical methods and concept extraction to summarize clusters.
    """
    
    def __init__(self, config: ConsolidationConfig):
        super().__init__(config)
        self.max_summary_length = config.max_summary_length
        self.preserve_originals = config.preserve_originals
        
        # Word importance patterns
        self._important_patterns = {
            'technical_terms': re.compile(r'\b[A-Z][a-z]*[A-Z][a-zA-Z]*\b'),  # CamelCase
            'acronyms': re.compile(r'\b[A-Z]{2,}\b'),
            'numbers': re.compile(r'\b\d+(?:\.\d+)?\b'),
            'urls': re.compile(r'https?://[^\s]+'),
            'file_paths': re.compile(r'[/\\][^\s]+'),
            'quoted_text': re.compile(r'"([^"]*)"'),
            'code_blocks': re.compile(r'```[\s\S]*?```|`[^`]+`')
        }
    
    async def process(self, clusters: List[MemoryCluster], memories: List[Memory], **kwargs) -> List[CompressionResult]:
        """Compress memory clusters into condensed representations."""
        if not clusters:
            return []
        
        # Create memory hash lookup
        memory_lookup = {m.content_hash: m for m in memories}
        
        compression_results = []
        for cluster in clusters:
            # Get memories for this cluster
            cluster_memories = []
            for hash_val in cluster.memory_hashes:
                if hash_val in memory_lookup:
                    cluster_memories.append(memory_lookup[hash_val])
            
            if not cluster_memories:
                continue
            
            # Compress the cluster
            result = await self._compress_cluster(cluster, cluster_memories)
            if result:
                compression_results.append(result)
        
        self.logger.info(f"Compressed {len(compression_results)} clusters")
        return compression_results
    
    async def _compress_cluster(self, cluster: MemoryCluster, memories: List[Memory]) -> Optional[CompressionResult]:
        """Compress a single memory cluster."""
        if len(memories) < 2:
            return None
        
        # Extract key concepts and themes
        key_concepts = await self._extract_key_concepts(memories, cluster.theme_keywords)
        
        # Generate thematic summary
        summary = await self._generate_thematic_summary(memories, key_concepts)
        
        # Calculate temporal information
        temporal_span = self._calculate_temporal_span(memories)
        
        # Aggregate tags and metadata
        aggregated_tags = self._aggregate_tags(memories)
        aggregated_metadata = self._aggregate_metadata(memories)
        
        # Create compressed memory embedding (cluster centroid)
        compressed_embedding = cluster.centroid_embedding
        
        # Calculate compression ratio
        original_size = sum(len(m.content) for m in memories)
        compressed_size = len(summary)
        compression_ratio = compressed_size / original_size if original_size > 0 else 0
        
        # Create content hash for the compressed memory
        content_hash = hashlib.sha256(summary.encode()).hexdigest()
        
        # Create compressed memory object
        compressed_memory = Memory(
            content=summary,
            content_hash=content_hash,
            tags=aggregated_tags,
            memory_type='compressed_cluster',
            metadata={
                **aggregated_metadata,
                'cluster_id': cluster.cluster_id,
                'compression_date': datetime.now().isoformat(),
                'source_memory_count': len(memories),
                'compression_ratio': compression_ratio,
                'key_concepts': key_concepts,
                'temporal_span': temporal_span,
                'theme_keywords': cluster.theme_keywords,
                'coherence_score': cluster.coherence_score,
                'compression_version': '1.0'
            },
            embedding=compressed_embedding,
            created_at=datetime.now().timestamp(),
            created_at_iso=datetime.now().isoformat() + 'Z'
        )
        
        return CompressionResult(
            cluster_id=cluster.cluster_id,
            compressed_memory=compressed_memory,
            compression_ratio=compression_ratio,
            key_concepts=key_concepts,
            temporal_span=temporal_span,
            source_memory_count=len(memories),
            compression_metadata={
                'algorithm': 'semantic_compression_v1',
                'original_total_length': original_size,
                'compressed_length': compressed_size,
                'concept_count': len(key_concepts),
                'theme_keywords': cluster.theme_keywords
            }
        )
    
    async def _extract_key_concepts(self, memories: List[Memory], theme_keywords: List[str]) -> List[str]:
        """Extract key concepts from cluster memories."""
        all_text = ' '.join([m.content for m in memories])
        concepts = set()
        
        # Add theme keywords as primary concepts
        concepts.update(theme_keywords)
        
        # Extract important patterns
        for pattern_name, pattern in self._important_patterns.items():
            matches = pattern.findall(all_text)
            if pattern_name == 'quoted_text':
                # For quoted text, add the content inside quotes
                concepts.update(matches)
            else:
                concepts.update(matches)
        
        # Extract capitalized terms (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', all_text)
        concepts.update(capitalized)
        
        # Extract frequent meaningful words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
        word_counts = Counter(words)
        
        # Filter out common words
        stop_words = {
            'this', 'that', 'with', 'have', 'will', 'from', 'they', 'know',
            'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when',
            'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over',
            'such', 'take', 'than', 'them', 'well', 'were', 'what', 'work',
            'your', 'could', 'should', 'would', 'there', 'their', 'these',
            'about', 'after', 'again', 'before', 'being', 'between', 'during',
            'under', 'where', 'while', 'other', 'through', 'against', 'without'
        }
        
        # Add frequent non-stop words
        for word, count in word_counts.most_common(20):
            if word not in stop_words and count >= 2:  # Must appear at least twice
                concepts.add(word)
        
        # Case-insensitive deduplication
        concept_dict = {}
        for concept in concepts:
            lower_key = concept.lower()
            if lower_key not in concept_dict:
                # Prefer capitalized form (proper nouns)
                concept_dict[lower_key] = concept
            elif concept[0].isupper() and not concept_dict[lower_key][0].isupper():
                # Replace lowercase with capitalized version
                concept_dict[lower_key] = concept

        # Filter SQL keywords and meta-concepts to reduce noise
        SQL_KEYWORDS = {'between', 'select', 'where', 'from', 'join', 'inner', 'outer', 'group', 'order', 'limit'}
        META_CONCEPTS = {'memory', 'memories', 'association', 'similarity', 'storage', 'retrieval'}

        filtered_concepts = [
            c for c in concept_dict.values()
            if c.lower() not in SQL_KEYWORDS and c.lower() not in META_CONCEPTS
        ]

        # Sort by frequency (most common first)
        filtered_concepts.sort(key=lambda x: word_counts.get(x.lower(), 0), reverse=True)

        return filtered_concepts[:15]  # Limit to top 15 thematic concepts
    
    async def _generate_thematic_summary(self, memories: List[Memory], key_concepts: List[str]) -> str:
        """Generate a thematic summary of the memory cluster."""
        # Analyze the memories to identify common themes and patterns
        all_content = [m.content for m in memories]
        
        # Extract representative sentences that contain key concepts
        representative_sentences = []
        concept_coverage = set()
        
        for memory in memories:
            sentences = self._split_into_sentences(memory.content)
            for sentence in sentences:
                sentence_concepts = set()
                sentence_lower = sentence.lower()
                
                # Check which concepts this sentence covers
                for concept in key_concepts:
                    if concept.lower() in sentence_lower:
                        sentence_concepts.add(concept)
                
                # If this sentence covers new concepts, include it
                new_concepts = sentence_concepts - concept_coverage
                if new_concepts and len(sentence) > 20:  # Minimum sentence length
                    representative_sentences.append({
                        'sentence': sentence.strip(),
                        'concepts': sentence_concepts,
                        'new_concepts': new_concepts,
                        'score': len(new_concepts) + len(sentence_concepts) * 0.1
                    })
                    concept_coverage.update(new_concepts)
        
        # Sort by score and select best sentences
        representative_sentences.sort(key=lambda x: x['score'], reverse=True)
        
        # Build summary
        summary_parts = []
        
        # Add cluster overview
        memory_count = len(memories)
        time_span = self._calculate_temporal_span(memories)
        concept_str = ', '.join(key_concepts[:5])
        
        overview = f"Cluster of {memory_count} related memories about {concept_str}"
        if time_span['span_days'] > 0:
            overview += f" spanning {time_span['span_days']} days"
        overview += "."
        
        summary_parts.append(overview)
        
        # Add key insights from representative sentences
        used_length = len(overview)
        remaining_length = self.max_summary_length - used_length - 100  # Reserve space for conclusion
        
        for sent_info in representative_sentences:
            sentence = sent_info['sentence']
            if used_length + len(sentence) < remaining_length:
                summary_parts.append(sentence)
                used_length += len(sentence)
            else:
                break
        
        # Add concept summary if space allows
        if used_length < self.max_summary_length - 50:
            concept_summary = f"Key concepts: {', '.join(key_concepts[:8])}."
            if used_length + len(concept_summary) < self.max_summary_length:
                summary_parts.append(concept_summary)
        
        summary = ' '.join(summary_parts)
        
        # Truncate if still too long
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length - 3] + '...'
        
        return summary
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics."""
        # Simple sentence splitting (could be improved with NLTK)
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Filter out very short sentences and clean up
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum sentence length
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def _calculate_temporal_span(self, memories: List[Memory]) -> Dict[str, Any]:
        """Calculate temporal information for the memory cluster."""
        timestamps = []
        
        for memory in memories:
            if memory.created_at:
                timestamps.append(memory.created_at)
            elif memory.timestamp:
                timestamps.append(memory.timestamp.timestamp())
        
        if not timestamps:
            return {
                'start_time': None,
                'end_time': None,
                'span_days': 0,
                'span_description': 'unknown'
            }
        
        start_time = min(timestamps)
        end_time = max(timestamps)
        span_seconds = end_time - start_time
        span_days = int(span_seconds / (24 * 3600))
        
        # Create human-readable span description
        if span_days == 0:
            span_description = 'same day'
        elif span_days < 7:
            span_description = f'{span_days} days'
        elif span_days < 30:
            weeks = span_days // 7
            span_description = f'{weeks} week{"s" if weeks > 1 else ""}'
        elif span_days < 365:
            months = span_days // 30
            span_description = f'{months} month{"s" if months > 1 else ""}'
        else:
            years = span_days // 365
            span_description = f'{years} year{"s" if years > 1 else ""}'
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'span_days': span_days,
            'span_description': span_description,
            'start_iso': datetime.utcfromtimestamp(start_time).isoformat() + 'Z',
            'end_iso': datetime.utcfromtimestamp(end_time).isoformat() + 'Z'
        }
    
    def _aggregate_tags(self, memories: List[Memory]) -> List[str]:
        """Aggregate tags from cluster memories."""
        all_tags = []
        for memory in memories:
            all_tags.extend(memory.tags)
        
        # Count tag frequency
        tag_counts = Counter(all_tags)
        
        # Include tags that appear in multiple memories or are important
        aggregated_tags = ['cluster', 'compressed']  # Always include these
        
        for tag, count in tag_counts.most_common():
            if count > 1 or tag in {'important', 'critical', 'reference', 'project'}:
                if tag not in aggregated_tags:
                    aggregated_tags.append(tag)
        
        return aggregated_tags[:10]  # Limit to 10 tags
    
    def _aggregate_metadata(self, memories: List[Memory]) -> Dict[str, Any]:
        """Aggregate metadata from cluster memories."""
        aggregated = {
            'source_memory_hashes': [m.content_hash for m in memories]
        }
        
        # Collect unique metadata keys and their values
        all_metadata = {}
        for memory in memories:
            for key, value in memory.metadata.items():
                if key not in all_metadata:
                    all_metadata[key] = []
                all_metadata[key].append(value)
        
        # Aggregate metadata intelligently
        for key, values in all_metadata.items():
            unique_values = list(set(str(v) for v in values))
            
            if len(unique_values) == 1:
                # All memories have the same value
                aggregated[f'common_{key}'] = unique_values[0]
            elif len(unique_values) <= 5:
                # Small number of unique values, list them
                aggregated[f'varied_{key}'] = unique_values
            else:
                # Many unique values, just note the variety
                aggregated[f'{key}_variety_count'] = len(unique_values)
        
        return aggregated
    
    async def estimate_compression_benefit(
        self,
        clusters: List[MemoryCluster],
        memories: List[Memory]
    ) -> Dict[str, Any]:
        """Estimate the benefit of compressing given clusters."""
        memory_lookup = {m.content_hash: m for m in memories}
        
        total_original_size = 0
        total_compressed_size = 0
        compressible_clusters = 0
        
        for cluster in clusters:
            cluster_memories = [memory_lookup[h] for h in cluster.memory_hashes if h in memory_lookup]
            
            if len(cluster_memories) < 2:
                continue
            
            compressible_clusters += 1
            original_size = sum(len(m.content) for m in cluster_memories)
            
            # Estimate compressed size (rough approximation)
            estimated_compressed_size = min(self.max_summary_length, original_size // 3)
            
            total_original_size += original_size
            total_compressed_size += estimated_compressed_size
        
        overall_ratio = total_compressed_size / total_original_size if total_original_size > 0 else 1.0
        savings = total_original_size - total_compressed_size
        
        return {
            'compressible_clusters': compressible_clusters,
            'total_original_size': total_original_size,
            'estimated_compressed_size': total_compressed_size,
            'compression_ratio': overall_ratio,
            'estimated_savings_bytes': savings,
            'estimated_savings_percent': (1 - overall_ratio) * 100
        }