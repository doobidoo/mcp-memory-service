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

"""Main dream-inspired consolidation orchestrator."""

import asyncio
from typing import List, Dict, Any, Optional, Protocol
from datetime import datetime, timedelta
import logging
import time

from .base import ConsolidationConfig, ConsolidationReport, ConsolidationError
from .decay import ExponentialDecayCalculator
from .associations import CreativeAssociationEngine
from .clustering import SemanticClusteringEngine
from .compression import SemanticCompressionEngine
from .forgetting import ControlledForgettingEngine
from .health import ConsolidationHealthMonitor
from ..models.memory import Memory

# Protocol for storage backend interface
class StorageProtocol(Protocol):
    async def get_all_memories(self) -> List[Memory]: ...
    async def get_memories_by_time_range(self, start_time: float, end_time: float) -> List[Memory]: ...
    async def store_memory(self, memory: Memory) -> bool: ...
    async def update_memory(self, memory: Memory) -> bool: ...
    async def delete_memory(self, content_hash: str) -> bool: ...
    async def get_memory_connections(self) -> Dict[str, int]: ...
    async def get_access_patterns(self) -> Dict[str, datetime]: ...

class DreamInspiredConsolidator:
    """
    Main consolidation engine with biologically-inspired processing.
    
    Orchestrates the full consolidation pipeline including:
    - Exponential decay scoring
    - Creative association discovery
    - Semantic clustering and compression
    - Controlled forgetting with archival
    """
    
    def __init__(self, storage: StorageProtocol, config: ConsolidationConfig):
        self.storage = storage
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize component engines
        self.decay_calculator = ExponentialDecayCalculator(config)
        self.association_engine = CreativeAssociationEngine(config)
        self.clustering_engine = SemanticClusteringEngine(config)
        self.compression_engine = SemanticCompressionEngine(config)
        self.forgetting_engine = ControlledForgettingEngine(config)
        
        # Initialize health monitoring
        self.health_monitor = ConsolidationHealthMonitor(config)
        
        # Performance tracking
        self.last_consolidation_times = {}
        self.consolidation_stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'total_memories_processed': 0,
            'total_associations_created': 0,
            'total_clusters_created': 0,
            'total_memories_compressed': 0,
            'total_memories_archived': 0
        }
    
    async def consolidate(self, time_horizon: str, **kwargs) -> ConsolidationReport:
        """
        Run full consolidation pipeline for given time horizon.
        
        Args:
            time_horizon: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
            **kwargs: Additional parameters for consolidation
        
        Returns:
            ConsolidationReport with results and performance metrics
        """
        start_time = datetime.now()
        report = ConsolidationReport(
            time_horizon=time_horizon,
            start_time=start_time,
            end_time=start_time,  # Will be updated at the end
            memories_processed=0
        )
        
        try:
            self.logger.info(f"Starting {time_horizon} consolidation")
            
            # 1. Retrieve memories for processing
            memories = await self._get_memories_for_horizon(time_horizon, **kwargs)
            report.memories_processed = len(memories)
            
            if not memories:
                self.logger.info(f"No memories to process for {time_horizon} consolidation")
                return self._finalize_report(report, [])
            
            self.logger.info(f"Processing {len(memories)} memories for {time_horizon} consolidation")
            
            # 2. Calculate/update relevance scores
            performance_start = time.time()
            relevance_scores = await self._update_relevance_scores(memories, time_horizon)
            self.logger.debug(f"Relevance scoring took {time.time() - performance_start:.2f}s")
            
            # 3. Cluster by semantic similarity (if enabled and appropriate)
            clusters = []
            if self.config.clustering_enabled and time_horizon in ['weekly', 'monthly', 'quarterly']:
                performance_start = time.time()
                clusters = await self.clustering_engine.process(memories)
                report.clusters_created = len(clusters)
                self.logger.debug(f"Clustering took {time.time() - performance_start:.2f}s, created {len(clusters)} clusters")
            
            # 4. Run creative associations (if enabled and appropriate)
            associations = []
            if self.config.associations_enabled and time_horizon in ['weekly', 'monthly']:
                performance_start = time.time()
                existing_associations = await self._get_existing_associations()
                associations = await self.association_engine.process(
                    memories, existing_associations=existing_associations
                )
                report.associations_discovered = len(associations)
                self.logger.debug(f"Association discovery took {time.time() - performance_start:.2f}s, found {len(associations)} associations")
                
                # Store new associations as memories
                await self._store_associations_as_memories(associations)
            
            # 5. Compress clusters (if enabled and clusters exist)
            compression_results = []
            if self.config.compression_enabled and clusters:
                performance_start = time.time()
                compression_results = await self.compression_engine.process(clusters, memories)
                report.memories_compressed = len(compression_results)
                self.logger.debug(f"Compression took {time.time() - performance_start:.2f}s, compressed {len(compression_results)} clusters")
                
                # Store compressed memories and update originals
                await self._handle_compression_results(compression_results)
            
            # 6. Controlled forgetting (if enabled and appropriate)
            forgetting_results = []
            if self.config.forgetting_enabled and time_horizon in ['monthly', 'quarterly', 'yearly']:
                performance_start = time.time()
                access_patterns = await self._get_access_patterns()
                forgetting_results = await self.forgetting_engine.process(
                    memories, relevance_scores, 
                    access_patterns=access_patterns, 
                    time_horizon=time_horizon
                )
                report.memories_archived = len([r for r in forgetting_results if r.action_taken in ['archived', 'deleted']])
                self.logger.debug(f"Forgetting took {time.time() - performance_start:.2f}s, processed {len(forgetting_results)} candidates")
                
                # Apply forgetting results to storage
                await self._apply_forgetting_results(forgetting_results)
            
            # 7. Update consolidation statistics
            self._update_consolidation_stats(report)
            
            # 8. Finalize report
            return self._finalize_report(report, [])
            
        except ConsolidationError as e:
            # Re-raise configuration and validation errors
            self.logger.error(f"Configuration error during {time_horizon} consolidation: {e}")
            self.health_monitor.record_error('consolidator', e, {'time_horizon': time_horizon})
            raise
        except Exception as e:
            self.logger.error(f"Error during {time_horizon} consolidation: {e}")
            self.health_monitor.record_error('consolidator', e, {'time_horizon': time_horizon})
            report.errors.append(str(e))
            return self._finalize_report(report, [str(e)])
    
    async def _get_memories_for_horizon(self, time_horizon: str, **kwargs) -> List[Memory]:
        """Get memories appropriate for the given time horizon."""
        now = datetime.now()
        
        # Define time ranges for different horizons
        time_ranges = {
            'daily': timedelta(days=1),
            'weekly': timedelta(days=7),
            'monthly': timedelta(days=30),
            'quarterly': timedelta(days=90),
            'yearly': timedelta(days=365)
        }
        
        if time_horizon not in time_ranges:
            raise ConsolidationError(f"Unknown time horizon: {time_horizon}")
        
        # For daily processing, get recent memories
        # For longer horizons, get broader ranges
        if time_horizon == 'daily':
            start_time = (now - timedelta(days=2)).timestamp()
            end_time = now.timestamp()
            memories = await self.storage.get_memories_by_time_range(start_time, end_time)
        else:
            # For longer horizons, process all memories but focus on older ones
            memories = await self.storage.get_all_memories()
            
            # Filter by relevance to time horizon
            if time_horizon in ['quarterly', 'yearly']:
                # For long horizons, focus on older memories that need consolidation
                cutoff_date = now - time_ranges[time_horizon]
                memories = [
                    m for m in memories 
                    if m.created_at and datetime.fromtimestamp(m.created_at, datetime.UTC) < cutoff_date
                ]
        
        return memories
    
    async def _update_relevance_scores(self, memories: List[Memory], time_horizon: str) -> List:
        """Calculate and update relevance scores for memories."""
        # Get connection and access data
        connections = await self._get_memory_connections()
        access_patterns = await self._get_access_patterns()
        
        # Calculate relevance scores
        relevance_scores = await self.decay_calculator.process(
            memories,
            connections=connections,
            access_patterns=access_patterns,
            reference_time=datetime.now()
        )
        
        # Update memory metadata with relevance scores
        for memory in memories:
            score = next((s for s in relevance_scores if s.memory_hash == memory.content_hash), None)
            if score:
                updated_memory = await self.decay_calculator.update_memory_relevance_metadata(memory, score)
                await self.storage.update_memory(updated_memory)
        
        return relevance_scores
    
    async def _get_memory_connections(self) -> Dict[str, int]:
        """Get memory connection counts from storage."""
        try:
            return await self.storage.get_memory_connections()
        except AttributeError:
            # Fallback if storage doesn't implement connection tracking
            self.logger.warning("Storage backend doesn't support connection tracking")
            return {}
    
    async def _get_access_patterns(self) -> Dict[str, datetime]:
        """Get memory access patterns from storage."""
        try:
            return await self.storage.get_access_patterns()
        except AttributeError:
            # Fallback if storage doesn't implement access tracking
            self.logger.warning("Storage backend doesn't support access pattern tracking")
            return {}
    
    async def _get_existing_associations(self) -> set:
        """Get existing memory associations to avoid duplicates."""
        try:
            # Look for existing association memories
            all_memories = await self.storage.get_all_memories()
            associations = set()
            
            for memory in all_memories:
                if memory.memory_type == 'association' and 'source_memory_hashes' in memory.metadata:
                    source_hashes = memory.metadata['source_memory_hashes']
                    if isinstance(source_hashes, list) and len(source_hashes) >= 2:
                        # Create canonical pair representation
                        pair_key = tuple(sorted(source_hashes[:2]))
                        associations.add(pair_key)
            
            return associations
            
        except Exception as e:
            self.logger.warning(f"Error getting existing associations: {e}")
            return set()
    
    async def _store_associations_as_memories(self, associations) -> None:
        """Store discovered associations as first-class memories."""
        for association in associations:
            # Create memory content from association
            source_hashes = association.source_memory_hashes
            similarity = association.similarity_score
            connection_type = association.connection_type
            
            content = f"Association between memories {source_hashes[0][:8]} and {source_hashes[1][:8]}: {connection_type} (similarity: {similarity:.3f})"
            
            # Create association memory
            association_memory = Memory(
                content=content,
                content_hash=f"assoc_{source_hashes[0][:8]}_{source_hashes[1][:8]}",
                tags=['association', 'discovered'] + connection_type.split(', '),
                memory_type='association',
                metadata={
                    'source_memory_hashes': source_hashes,
                    'similarity_score': similarity,
                    'connection_type': connection_type,
                    'discovery_method': association.discovery_method,
                    'discovery_date': association.discovery_date.isoformat(),
                    **association.metadata
                },
                created_at=datetime.now().timestamp(),
                created_at_iso=datetime.now().isoformat() + 'Z'
            )
            
            # Store the association memory
            await self.storage.store_memory(association_memory)
    
    async def _handle_compression_results(self, compression_results) -> None:
        """Handle storage of compressed memories and linking to originals."""
        for result in compression_results:
            # Store compressed memory
            await self.storage.store_memory(result.compressed_memory)
            
            # Update original memories with compression links
            # This could involve adding metadata pointing to the compressed version
            # Implementation depends on how the storage backend handles relationships
            pass
    
    async def _apply_forgetting_results(self, forgetting_results) -> None:
        """Apply forgetting results to the storage backend."""
        for result in forgetting_results:
            if result.action_taken == 'deleted':
                await self.storage.delete_memory(result.memory_hash)
            elif result.action_taken == 'compressed' and result.compressed_version:
                # Replace original with compressed version
                await self.storage.delete_memory(result.memory_hash)
                await self.storage.store_memory(result.compressed_version)
            # 'archived' memories are handled by the forgetting engine
    
    def _update_consolidation_stats(self, report: ConsolidationReport) -> None:
        """Update internal consolidation statistics."""
        self.consolidation_stats['total_runs'] += 1
        if not report.errors:
            self.consolidation_stats['successful_runs'] += 1
        
        self.consolidation_stats['total_memories_processed'] += report.memories_processed
        self.consolidation_stats['total_associations_created'] += report.associations_discovered
        self.consolidation_stats['total_clusters_created'] += report.clusters_created
        self.consolidation_stats['total_memories_compressed'] += report.memories_compressed
        self.consolidation_stats['total_memories_archived'] += report.memories_archived
        
        # Update last consolidation time
        self.last_consolidation_times[report.time_horizon] = report.start_time
    
    def _finalize_report(self, report: ConsolidationReport, errors: List[str]) -> ConsolidationReport:
        """Finalize the consolidation report."""
        report.end_time = datetime.now()
        report.errors.extend(errors)
        
        # Add performance metrics
        duration = (report.end_time - report.start_time).total_seconds()
        success = len(errors) == 0
        report.performance_metrics = {
            'duration_seconds': duration,
            'memories_per_second': report.memories_processed / duration if duration > 0 else 0,
            'success': success
        }
        
        # Record performance in health monitor
        self.health_monitor.record_consolidation_performance(
            time_horizon=report.time_horizon,
            duration=duration,
            memories_processed=report.memories_processed,
            success=success,
            errors=errors
        )
        
        # Log summary
        if errors:
            self.logger.error(f"Consolidation {report.time_horizon} completed with errors: {errors}")
        else:
            self.logger.info(
                f"Consolidation {report.time_horizon} completed successfully: "
                f"{report.memories_processed} memories, {report.associations_discovered} associations, "
                f"{report.clusters_created} clusters, {report.memories_compressed} compressed, "
                f"{report.memories_archived} archived in {duration:.2f}s"
            )
        
        return report
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the consolidation system."""
        return await self.health_monitor.check_overall_health()
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of consolidation system health."""
        return await self.health_monitor.get_health_summary()
    
    def get_error_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent error history."""
        return self.health_monitor.error_history[-limit:]
    
    def get_performance_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent performance history."""
        return self.health_monitor.performance_history[-limit:]
    
    def resolve_health_alert(self, alert_id: str):
        """Resolve a health alert."""
        self.health_monitor.resolve_alert(alert_id)
    
    async def get_consolidation_recommendations(self, time_horizon: str) -> Dict[str, Any]:
        """Get recommendations for consolidation based on current memory state."""
        try:
            memories = await self._get_memories_for_horizon(time_horizon)
            
            if not memories:
                return {
                    'recommendation': 'no_action',
                    'reason': 'No memories to process',
                    'memory_count': 0
                }
            
            # Analyze memory distribution
            memory_types = {}
            total_size = 0
            old_memories = 0
            now = datetime.now()
            
            for memory in memories:
                memory_type = memory.memory_type or 'standard'
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
                total_size += len(memory.content)
                
                if memory.created_at:
                    age_days = (now - datetime.fromtimestamp(memory.created_at, datetime.UTC)).days
                    if age_days > 30:
                        old_memories += 1
            
            # Generate recommendations
            recommendations = []
            
            if len(memories) > 1000:
                recommendations.append("Consider running compression to reduce memory usage")
            
            if old_memories > len(memories) * 0.5:
                recommendations.append("Many old memories present - consider forgetting/archival")
            
            if len(memories) > 100 and time_horizon in ['weekly', 'monthly']:
                recommendations.append("Good candidate for association discovery")
            
            if not recommendations:
                recommendations.append("Memory state looks healthy")
            
            return {
                'recommendation': 'consolidation_beneficial' if len(recommendations) > 1 else 'optional',
                'reasons': recommendations,
                'memory_count': len(memories),
                'memory_types': memory_types,
                'total_size_bytes': total_size,
                'old_memory_percentage': (old_memories / len(memories)) * 100,
                'estimated_duration_seconds': len(memories) * 0.01  # Rough estimate
            }
            
        except Exception as e:
            return {
                'recommendation': 'error',
                'error': str(e),
                'memory_count': 0
            }