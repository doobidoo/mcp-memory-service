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

"""Controlled forgetting system with archival for memory management."""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
import hashlib

from .base import ConsolidationBase, ConsolidationConfig
from .decay import RelevanceScore
from ..models.memory import Memory

@dataclass
class ForgettingCandidate:
    """A memory candidate for forgetting."""
    memory: Memory
    relevance_score: RelevanceScore
    forgetting_reasons: List[str]
    archive_priority: int  # 1=high, 2=medium, 3=low
    can_be_deleted: bool

@dataclass
class ForgettingResult:
    """Result of forgetting operation."""
    memory_hash: str
    action_taken: str  # 'archived', 'compressed', 'deleted', 'skipped'
    archive_path: Optional[str]
    compressed_version: Optional[Memory]
    metadata: Dict[str, Any]

class ControlledForgettingEngine(ConsolidationBase):
    """
    Implements intelligent forgetting to maintain memory system health.
    
    Rather than deleting memories, this system compresses and archives 
    low-value memories while maintaining audit trails and recovery options.
    """
    
    def __init__(self, config: ConsolidationConfig):
        super().__init__(config)
        self.relevance_threshold = config.relevance_threshold
        self.access_threshold_days = config.access_threshold_days
        self.archive_location = config.archive_location or "~/.mcp_memory_archive"
        
        # Ensure archive directory exists
        self.archive_path = Path(os.path.expanduser(self.archive_location))
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different archive types
        self.daily_archive = self.archive_path / "daily"
        self.compressed_archive = self.archive_path / "compressed" 
        self.metadata_archive = self.archive_path / "metadata"
        
        for archive_dir in [self.daily_archive, self.compressed_archive, self.metadata_archive]:
            archive_dir.mkdir(exist_ok=True)
    
    async def process(self, memories: List[Memory], relevance_scores: List[RelevanceScore], **kwargs) -> List[ForgettingResult]:
        """Identify and process memories for controlled forgetting."""
        if not self._validate_memories(memories):
            return []
        
        # Create score lookup
        score_lookup = {score.memory_hash: score for score in relevance_scores}
        
        # Get access patterns from kwargs
        access_patterns = kwargs.get('access_patterns', {})
        time_horizon = kwargs.get('time_horizon', 'monthly')
        
        # Identify forgetting candidates
        candidates = await self._identify_forgetting_candidates(
            memories, score_lookup, access_patterns, time_horizon
        )
        
        if not candidates:
            self.logger.info("No memories identified for forgetting")
            return []
        
        # Process candidates
        results = []
        for candidate in candidates:
            result = await self._process_forgetting_candidate(candidate)
            results.append(result)
        
        # Log forgetting summary
        actions_summary = {}
        for result in results:
            action = result.action_taken
            actions_summary[action] = actions_summary.get(action, 0) + 1
        
        self.logger.info(f"Forgetting results: {actions_summary}")
        return results
    
    async def _identify_forgetting_candidates(
        self,
        memories: List[Memory],
        score_lookup: Dict[str, RelevanceScore],
        access_patterns: Dict[str, datetime],
        time_horizon: str
    ) -> List[ForgettingCandidate]:
        """
        Identify memories that are candidates for forgetting.

        Uses quality-based retention policy:
        - High quality (≥0.7): Preserve up to 365 days inactive
        - Medium quality (0.5-0.7): Preserve up to 180 days inactive
        - Low quality (<0.5): Archive after 30-90 days inactive (scaled by quality)
        """
        from ..config import (
            MCP_QUALITY_RETENTION_HIGH,
            MCP_QUALITY_RETENTION_MEDIUM,
            MCP_QUALITY_RETENTION_LOW_MIN,
            MCP_QUALITY_RETENTION_LOW_MAX
        )

        candidates = []
        current_time = datetime.now(timezone.utc)

        for memory in memories:
            # Skip protected memories
            if self._is_protected_memory(memory):
                continue

            # Get relevance score
            relevance_score = score_lookup.get(memory.content_hash)
            if not relevance_score:
                continue

            # Check if memory meets forgetting criteria
            forgetting_reasons = []
            can_be_deleted = False
            archive_priority = 3  # Default to low priority

            # Low relevance check
            if relevance_score.total_score < self.relevance_threshold:
                forgetting_reasons.append("low_relevance")
                archive_priority = min(archive_priority, 2)  # Medium priority

            # Access pattern check with quality-based thresholds
            last_accessed = access_patterns.get(memory.content_hash)
            if not last_accessed and memory.updated_at:
                last_accessed = datetime.fromtimestamp(memory.updated_at, tz=timezone.utc)

            if last_accessed:
                if last_accessed.tzinfo is None:
                    last_accessed = last_accessed.replace(tzinfo=timezone.utc)
                days_since_access = (current_time - last_accessed).days

                # Quality-based retention thresholds
                quality_score = memory.quality_score

                if quality_score >= 0.7:
                    # High quality: Keep longer
                    threshold_days = MCP_QUALITY_RETENTION_HIGH
                elif quality_score >= 0.5:
                    # Medium quality: Standard retention
                    threshold_days = MCP_QUALITY_RETENTION_MEDIUM
                else:
                    # Low quality: Aggressive archival
                    # Scale between min and max based on quality score
                    threshold_days = int(
                        MCP_QUALITY_RETENTION_LOW_MIN +
                        (quality_score / 0.5) * (MCP_QUALITY_RETENTION_LOW_MAX - MCP_QUALITY_RETENTION_LOW_MIN)
                    )

                if days_since_access > threshold_days:
                    forgetting_reasons.append("old_access")
                    self.logger.info(
                        f"Archival candidate: {memory.content_hash[:12]} quality={quality_score:.2f}, "
                        f"inactive={days_since_access:.0f}d, threshold={threshold_days}d"
                    )
                    if days_since_access > threshold_days * 2:
                        archive_priority = min(archive_priority, 1)  # High priority
                        can_be_deleted = True  # Can be deleted if very old
            
            # Memory type specific checks
            memory_type = self._extract_memory_type(memory)
            # Check for temporary memories (legacy type or tag-based)
            is_temporary = (
                memory_type == 'temporary' or  # Legacy support
                'temporary' in memory.tags or
                't:temporary' in memory.tags  # New tag taxonomy
            )
            if is_temporary:
                age_days = self._get_memory_age_days(memory, current_time)
                if age_days > 7:  # Temporary memories older than a week
                    forgetting_reasons.append("expired_temporary")
                    can_be_deleted = True
                    archive_priority = 1
            
            # Content quality checks
            if self._is_low_quality_content(memory):
                forgetting_reasons.append("low_quality")
                archive_priority = min(archive_priority, 2)
            
            # Duplicate content check. Only the *redundant* copies are flagged: the
            # best-ranked member of a near-duplicate group is never a candidate, so
            # deduplication always leaves a survivor in storage.
            if self._is_redundant_copy(memory, memories, score_lookup):
                forgetting_reasons.append("potential_duplicate")
                can_be_deleted = True
                archive_priority = 1
            
            # Create candidate if reasons exist
            if forgetting_reasons:
                # Override time horizon restriction for certain types of deletable content
                can_delete_final = can_be_deleted
                if not (time_horizon in ['quarterly', 'yearly']):
                    # Still allow deletion for expired temporary memories and duplicates
                    if not ('expired_temporary' in forgetting_reasons or 'potential_duplicate' in forgetting_reasons):
                        can_delete_final = False
                
                candidate = ForgettingCandidate(
                    memory=memory,
                    relevance_score=relevance_score,
                    forgetting_reasons=forgetting_reasons,
                    archive_priority=archive_priority,
                    can_be_deleted=can_delete_final
                )
                candidates.append(candidate)
        
        # Sort by priority (higher priority = lower number = first in list)
        candidates.sort(key=lambda c: (c.archive_priority, -c.relevance_score.total_score))
        
        self.logger.info(f"Identified {len(candidates)} forgetting candidates")
        return candidates
    
    def _is_low_quality_content(self, memory: Memory) -> bool:
        """Check if memory content appears to be low quality."""
        content = memory.content.strip()
        
        # Very short content
        if len(content) < 10:
            return True
        
        # Mostly punctuation or special characters
        alpha_chars = sum(1 for c in content if c.isalpha())
        if alpha_chars / len(content) < 0.3:  # Less than 30% alphabetic
            return True
        
        # Repetitive content patterns
        if len(set(content.split())) < len(content.split()) * 0.5:  # Less than 50% unique words
            return True
        
        # Common low-value patterns
        low_value_patterns = [
            'test', 'testing', 'hello world', 'lorem ipsum',
            'asdf', 'qwerty', '1234', 'temp', 'temporary'
        ]
        
        content_lower = content.lower()
        for pattern in low_value_patterns:
            if pattern in content_lower and len(content) < 100:
                return True
        
        return False
    
    def _appears_to_be_duplicate(self, memory: Memory, all_memories: List[Memory]) -> bool:
        """Check if memory appears to be a duplicate of another memory.

        Note this is a *symmetric* relation, so it is true for both members of a pair.
        Use :meth:`_is_redundant_copy` to decide what may be deleted.
        """
        return any(self._is_near_duplicate_of(memory, other) for other in all_memories
                   if other.content_hash != memory.content_hash)

    def _is_near_duplicate_of(self, memory: Memory, other_memory: Memory) -> bool:
        """Pairwise near-duplicate test. Symmetric by construction."""
        content = memory.content.strip().lower()

        # Skip very short content for duplicate detection
        if len(content) < 20:
            return False

        other_content = other_memory.content.strip().lower()

        # Exact match
        if content == other_content:
            return True

        # Very similar content (simple check): only worth testing on longer texts
        if min(len(content), len(other_content)) <= 50:
            return False

        # Check if one is a substring of the other with high overlap
        if content in other_content or other_content in content:
            return True

        # Check word overlap
        words1 = set(content.split())
        words2 = set(other_content.split())
        if min(len(words1), len(words2)) <= 5:
            return False

        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return overlap / union > 0.8  # 80% word overlap

    @staticmethod
    def _keep_rank(memory: Memory, score_lookup: Dict[str, "RelevanceScore"]):
        """Sort key for "which copy is worth keeping" - lower sorts first (kept)."""
        score = score_lookup.get(memory.content_hash)
        return (
            -(score.total_score if score else 0.0),   # best relevance first
            memory.created_at or 0.0,                  # then the original, not the copy
            memory.content_hash or "",                 # then a stable tiebreak
        )

    def _is_redundant_copy(
        self,
        memory: Memory,
        all_memories: List[Memory],
        score_lookup: Dict[str, "RelevanceScore"],
    ) -> bool:
        """True only if a near-duplicate exists that OUTRANKS this memory.

        `_appears_to_be_duplicate` is symmetric, so flagging on it deleted *every*
        member of a duplicate group instead of the extras. Ranking each candidate
        against its own duplicates fixes that without needing to build groups first,
        which matters because near-duplicate matching is not transitive: the
        best-ranked member of any group is outranked by nobody, so it is never
        flagged and always survives.
        """
        mine = self._keep_rank(memory, score_lookup)
        for other_memory in all_memories:
            if other_memory.content_hash == memory.content_hash:
                continue
            if not self._is_near_duplicate_of(memory, other_memory):
                continue
            if self._keep_rank(other_memory, score_lookup) < mine:
                return True
        return False
    
    async def _process_forgetting_candidate(self, candidate: ForgettingCandidate) -> ForgettingResult:
        """Process a single forgetting candidate."""
        memory = candidate.memory
        
        try:
            # Determine action based on candidate properties
            if candidate.can_be_deleted and 'potential_duplicate' in candidate.forgetting_reasons:
                # Delete obvious duplicates or expired temporary content
                return await self._delete_memory(candidate)
            
            elif candidate.archive_priority <= 2:
                # Archive high and medium priority candidates
                return await self._archive_memory(candidate)
            
            else:
                # Compress low priority candidates
                return await self._compress_memory(candidate)
        
        except Exception as e:
            self.logger.error(f"Error processing forgetting candidate {memory.content_hash}: {e}")
            return ForgettingResult(
                memory_hash=memory.content_hash,
                action_taken='skipped',
                archive_path=None,
                compressed_version=None,
                metadata={'error': str(e)}
            )
    
    async def _archive_memory(self, candidate: ForgettingCandidate) -> ForgettingResult:
        """Archive a memory to the filesystem."""
        memory = candidate.memory
        
        # Create archive filename with timestamp and hash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_hash = memory.content_hash[:12]
        filename = f"{timestamp}_{short_hash}.json"
        
        # Choose archive directory based on priority
        if candidate.archive_priority == 1:
            archive_dir = self.daily_archive
        else:
            archive_dir = self.compressed_archive
        
        archive_file = archive_dir / filename
        
        # Create archive data
        archive_data = {
            'memory': memory.to_dict(),
            'relevance_score': {
                'total_score': candidate.relevance_score.total_score,
                'base_importance': candidate.relevance_score.base_importance,
                'decay_factor': candidate.relevance_score.decay_factor,
                'connection_boost': candidate.relevance_score.connection_boost,
                'access_boost': candidate.relevance_score.access_boost,
                'metadata': candidate.relevance_score.metadata
            },
            'forgetting_metadata': {
                'reasons': candidate.forgetting_reasons,
                'archive_priority': candidate.archive_priority,
                'archive_date': datetime.now().isoformat(),
                'original_hash': memory.content_hash
            }
        }
        
        # Write to archive
        with open(archive_file, 'w', encoding='utf-8') as f:
            json.dump(archive_data, f, indent=2, ensure_ascii=False)
        
        # Create metadata entry
        await self._create_metadata_entry(memory, archive_file, 'archived')
        
        return ForgettingResult(
            memory_hash=memory.content_hash,
            action_taken='archived',
            archive_path=str(archive_file),
            compressed_version=None,
            metadata={
                'archive_priority': candidate.archive_priority,
                'reasons': candidate.forgetting_reasons,
                'file_size': archive_file.stat().st_size
            }
        )
    
    async def _compress_memory(self, candidate: ForgettingCandidate) -> ForgettingResult:
        """Create a compressed version of the memory."""
        memory = candidate.memory
        original_content = memory.content
        
        # Simple compression: extract key information
        compressed_content = self._create_compressed_content(original_content)
        
        # Create compressed memory
        compressed_hash = hashlib.sha256(compressed_content.encode()).hexdigest()
        compressed_memory = Memory(
            content=compressed_content,
            content_hash=compressed_hash,
            tags=memory.tags + ['compressed'],
            memory_type='pattern',  # Changed from 'compressed' to valid ontology type
            metadata={
                **memory.metadata,
                'original_hash': memory.content_hash,
                'original_length': len(original_content),
                'compressed_length': len(compressed_content),
                'compression_ratio': len(compressed_content) / len(original_content),
                'compression_date': datetime.now().isoformat(),
                'forgetting_reasons': candidate.forgetting_reasons
            },
            embedding=memory.embedding,  # Preserve embedding
            created_at=memory.created_at,
            created_at_iso=memory.created_at_iso
        )
        
        # Archive original for recovery
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_hash = memory.content_hash[:12]
        archive_file = self.compressed_archive / f"original_{timestamp}_{short_hash}.json"
        
        with open(archive_file, 'w', encoding='utf-8') as f:
            json.dump(memory.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Create metadata entry
        await self._create_metadata_entry(memory, archive_file, 'compressed')
        
        return ForgettingResult(
            memory_hash=memory.content_hash,
            action_taken='compressed',
            archive_path=str(archive_file),
            compressed_version=compressed_memory,
            metadata={
                'original_length': len(original_content),
                'compressed_length': len(compressed_content),
                'compression_ratio': len(compressed_content) / len(original_content),
                'reasons': candidate.forgetting_reasons
            }
        )
    
    async def _delete_memory(self, candidate: ForgettingCandidate) -> ForgettingResult:
        """Delete a memory (with metadata backup)."""
        memory = candidate.memory
        
        # Always create a metadata backup before deletion
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_hash = memory.content_hash[:12]
        backup_file = self.metadata_archive / f"deleted_{timestamp}_{short_hash}.json"
        
        backup_data = {
            'memory': memory.to_dict(),
            'deletion_metadata': {
                'reasons': candidate.forgetting_reasons,
                'deletion_date': datetime.now().isoformat(),
                'relevance_score': candidate.relevance_score.total_score
            }
        }
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        return ForgettingResult(
            memory_hash=memory.content_hash,
            action_taken='deleted',
            archive_path=str(backup_file),
            compressed_version=None,
            metadata={
                'reasons': candidate.forgetting_reasons,
                'backup_location': str(backup_file)
            }
        )
    
    def _create_compressed_content(self, original_content: str) -> str:
        """Create compressed version of content preserving key information."""
        # Simple compression strategy: extract key sentences and important terms
        sentences = original_content.split('.')
        
        # Keep first and last sentences if content is long enough
        if len(sentences) > 3:
            key_sentences = [sentences[0].strip(), sentences[-1].strip()]
            middle_content = ' '.join(sentences[1:-1])
            
            # Extract important terms from middle content
            important_terms = self._extract_important_terms(middle_content)
            if important_terms:
                key_sentences.append(f"Key terms: {', '.join(important_terms[:5])}")
            
            compressed = '. '.join(key_sentences)
        else:
            # Content is already short, just clean it up
            compressed = original_content.strip()
        
        # Add compression indicator
        if len(compressed) < len(original_content) * 0.8:
            compressed += " [Compressed]"
        
        return compressed
    
    def _extract_important_terms(self, text: str) -> List[str]:
        """Extract important terms from text."""
        import re
        
        # Extract capitalized words, numbers, and technical terms
        terms = set()
        
        # Capitalized words (potential proper nouns)
        terms.update(re.findall(r'\b[A-Z][a-z]+\b', text))
        
        # Numbers and measurements
        terms.update(re.findall(r'\b\d+(?:\.\d+)?(?:\s*[a-zA-Z]+)?\b', text))
        
        # Words in quotes
        terms.update(re.findall(r'"([^"]*)"', text))
        
        # Technical-looking terms (CamelCase, with underscores, etc.)
        terms.update(re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b', text))
        terms.update(re.findall(r'\b\w+_\w+\b', text))
        
        return list(terms)[:10]  # Limit to 10 terms
    
    async def _create_metadata_entry(self, memory: Memory, archive_path: Path, action: str):
        """Create a metadata entry for tracking archived/compressed memories."""
        metadata_file = self.metadata_archive / "forgetting_log.jsonl"
        
        entry = {
            'memory_hash': memory.content_hash,
            'action': action,
            'archive_path': str(archive_path),
            'timestamp': datetime.now().isoformat(),
            'memory_type': memory.memory_type,
            'tags': memory.tags,
            'content_length': len(memory.content)
        }
        
        # Append to log file
        with open(metadata_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    
    async def recover_memory(self, memory_hash: str) -> Optional[Memory]:
        """Recover a forgotten memory from archives."""
        # Search through archive directories
        for archive_dir in [self.daily_archive, self.compressed_archive]:
            for archive_file in archive_dir.glob("*.json"):
                try:
                    with open(archive_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Check if this is the memory we're looking for
                    if data.get('memory', {}).get('content_hash') == memory_hash:
                        memory_data = data['memory']
                        return Memory.from_dict(memory_data)
                
                except Exception as e:
                    self.logger.warning(f"Error reading archive file {archive_file}: {e}")
        
        return None
    
    async def get_forgetting_statistics(self) -> Dict[str, Any]:
        """Get statistics about forgetting operations."""
        stats = {
            'total_archived': 0,
            'total_compressed': 0,
            'total_deleted': 0,
            'archive_size_bytes': 0,
            'oldest_archive': None,
            'newest_archive': None
        }
        
        # Count files in archive directories
        for archive_dir in [self.daily_archive, self.compressed_archive]:
            if archive_dir.exists():
                files = list(archive_dir.glob("*.json"))
                stats['total_archived'] += len(files)
                
                for file in files:
                    stats['archive_size_bytes'] += file.stat().st_size
        
        # Read forgetting log for detailed stats
        log_file = self.metadata_archive / "forgetting_log.jsonl"
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        action = entry.get('action', 'unknown')
                        
                        if action == 'archived':
                            stats['total_archived'] += 1
                        elif action == 'compressed':
                            stats['total_compressed'] += 1
                        elif action == 'deleted':
                            stats['total_deleted'] += 1
                        
                        # Track date range
                        timestamp = entry.get('timestamp')
                        if timestamp:
                            if not stats['oldest_archive'] or timestamp < stats['oldest_archive']:
                                stats['oldest_archive'] = timestamp
                            if not stats['newest_archive'] or timestamp > stats['newest_archive']:
                                stats['newest_archive'] = timestamp
                                
            except Exception as e:
                self.logger.warning(f"Error reading forgetting log: {e}")
        
        return stats