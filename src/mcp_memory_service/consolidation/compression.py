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

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
from collections import Counter
import re
import hashlib

from .base import ConsolidationBase, ConsolidationConfig, MemoryCluster
from ..compat import _sanitize_log_value
from ..models.memory import Memory

# Tags on memories the service or its hooks write without the user: discovered
# associations and auto-captured session summaries (Claude Code and opencode
# hooks). Earlier compression output is matched by COMPRESSION_OUTPUT_TAGS.
SELF_GENERATED_TAGS = frozenset({'association', 'session-consolidation', 'session-summary'})

# _aggregate_tags puts both on every pattern this engine writes. 'compressed'
# alone is not enough: forgetting uses it for shortened user memories too.
COMPRESSION_OUTPUT_TAGS = frozenset({'cluster', 'compressed'})

# Words that come from the layout of stored content, not from its subject:
# session summaries ("Session", "Summary", "User") and dedup reports
# ("copies", "Hash"). They win on frequency alone and say nothing about a cluster.
STRUCTURAL_TOKENS = frozenset({'session', 'summary', 'user', 'copies', 'hash'})

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
        """Compress memory clusters into condensed representations (parallel)."""
        if not clusters:
            return []

        # Create memory hash lookup
        memory_lookup = {m.content_hash: m for m in memories}

        # Build tasks for clusters that have matching memories. Memories the
        # service wrote itself are left out: summarizing them only repeats their
        # template words, and a cluster made mostly of them ends up below the
        # two-memory minimum in _compress_cluster.
        tasks = []
        for cluster in clusters:
            cluster_memories = [
                memory_lookup[h] for h in cluster.memory_hashes
                if h in memory_lookup and not self._is_self_generated(memory_lookup[h])
            ]
            if cluster_memories:
                tasks.append(self._compress_cluster(cluster, cluster_memories))

        # Run all cluster compressions concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        compression_results = []
        for r in results:
            if isinstance(r, Exception):
                self.logger.warning(f"Cluster compression failed: {_sanitize_log_value(str(r))}")
            elif r is not None:
                compression_results.append(r)

        self.logger.info("Compressed %d clusters", len(compression_results))
        return compression_results

    @staticmethod
    def _is_self_generated(memory: Memory) -> bool:
        """True for memories the service or its hooks wrote, not the user.

        Tags are the marker, not memory_type: associations are stored as
        'observation', and so are most auto-captured session summaries.
        """
        tags = set(memory.tags)
        return (
            memory.memory_type == 'session'
            or not SELF_GENERATED_TAGS.isdisjoint(tags)
            or COMPRESSION_OUTPUT_TAGS <= tags
        )

    async def _compress_cluster(self, cluster: MemoryCluster, memories: List[Memory]) -> Optional[CompressionResult]:
        """Compress a single memory cluster."""
        if len(memories) < 2:
            return None
        
        # Extract key concepts and themes
        key_concepts = await self._extract_key_concepts(memories, cluster.theme_keywords)
        
        # Without a sentence that carries a concept, the summary would be the
        # generated overview line alone: a pattern memory with nothing in it.
        sentences = self._select_representative_sentences(memories, key_concepts)
        if not sentences:
            self.logger.debug(
                "Skipping cluster %s: no representative sentence", _sanitize_log_value(cluster.cluster_id)
            )
            return None

        # Generate thematic summary
        summary = await self._generate_thematic_summary(memories, key_concepts, sentences)
        
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
            memory_type='pattern',  # Changed from 'compressed_cluster' to valid ontology type
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
        
        # Add frequent non-stop words. This check is a head start, not the rule:
        # the filter below is authoritative and re-checks every source.
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

        # Filter stop words, SQL keywords, meta-concepts and bare numbers. The
        # stop-word check runs here, over every concept, and not only over the
        # frequent-lowercase-word source above: a stop word arrives capitalized
        # ("With"), through a theme keyword, or through an important-pattern
        # match just as often, and the per-source check missed all of those.
        SQL_KEYWORDS = {'between', 'select', 'where', 'from', 'join', 'inner', 'outer', 'group', 'order', 'limit'}
        META_CONCEPTS = {'memory', 'memories', 'association', 'similarity', 'storage', 'retrieval'}
        PURELY_NUMERIC = re.compile(r'\d+(?:\.\d+)?')

        def is_noise(concept: str) -> bool:
            # Quoted text arrives with its surrounding spaces, so " 42 " has to
            # face the same checks as "42".
            stripped = concept.strip()
            lowered = stripped.lower()
            return (
                lowered in stop_words
                or lowered in STRUCTURAL_TOKENS
                or lowered in SQL_KEYWORDS
                or lowered in META_CONCEPTS
                # Bare numbers carry no theme: "0.682", "42".
                or PURELY_NUMERIC.fullmatch(stripped) is not None
            )

        filtered_concepts = [c for c in concept_dict.values() if not is_noise(c)]

        # Sort by frequency (most common first)
        filtered_concepts.sort(key=lambda x: word_counts.get(x.lower(), 0), reverse=True)

        return filtered_concepts[:15]  # Limit to top 15 thematic concepts
    
    def _select_representative_sentences(self, memories: List[Memory], key_concepts: List[str]) -> List[str]:
        """Pick sentences that cover key concepts, best coverage first."""
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
        return [s['sentence'] for s in representative_sentences]

    async def _generate_thematic_summary(
        self,
        memories: List[Memory],
        key_concepts: List[str],
        representative_sentences: Optional[List[str]] = None,
    ) -> str:
        """Generate a thematic summary of the memory cluster."""
        if representative_sentences is None:
            representative_sentences = self._select_representative_sentences(memories, key_concepts)

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
        
        for sentence in representative_sentences:
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
            'start_iso': datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat().replace('+00:00', 'Z'),
            'end_iso': datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat().replace('+00:00', 'Z')
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
    
    # Internal consolidation metadata keys that are per-memory state and
    # become meaningless after compression.  Keys starting with '_' are also
    # treated as private/internal and are excluded.
    _INTERNAL_METADATA_KEYS = frozenset({
        'connection_boost',
        'decay_factor',
        'last_consolidated_at',
        'access_boost',
        '_is_compressed',
        'consolidation_run',
    })

    @staticmethod
    def _strip_compression_prefixes(key: str) -> str:
        """Remove all leading common_/varied_ prefixes and trailing _variety_count suffix.

        Prevents exponential prefix nesting when _aggregate_metadata() is called
        on memories that were themselves produced by a previous compression run
        (fix for GitHub issue #543).
        """
        # Strip leading prefixes iteratively
        while key.startswith('common_') or key.startswith('varied_'):
            if key.startswith('common_'):
                key = key[7:]   # len('common_') == 7
            elif key.startswith('varied_'):
                key = key[7:]   # len('varied_') == 7
        # Strip trailing _variety_count suffix
        if key.endswith('_variety_count'):
            key = key[:-14]     # len('_variety_count') == 14
        return key

    def _aggregate_metadata(self, memories: List[Memory]) -> Dict[str, Any]:
        """Aggregate metadata from cluster memories.

        Keys inherited from previously-compressed memories already carry
        ``common_``/``varied_`` prefixes.  Those prefixes are stripped before
        new ones are applied so that repeated compression runs do not cause
        exponential key-name growth (GitHub issue #543).
        """
        aggregated = {
            'source_memory_hashes': [m.content_hash for m in memories]
        }

        # Collect unique metadata keys and their values, normalising keys first
        all_metadata: Dict[str, list] = {}
        for memory in memories:
            for raw_key, value in memory.metadata.items():
                # Derive the canonical (prefix-free) key
                clean_key = self._strip_compression_prefixes(raw_key)

                # Skip internal / private keys
                if clean_key in self._INTERNAL_METADATA_KEYS or clean_key.startswith('_'):
                    continue

                if clean_key not in all_metadata:
                    all_metadata[clean_key] = []
                all_metadata[clean_key].append(value)

        # Aggregate metadata intelligently using canonical keys
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