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
from typing import List, Dict, Any, Optional, Protocol, Tuple
from datetime import datetime, timedelta, timezone
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
from ..storage.graph import GraphStorage
from ..config import GRAPH_STORAGE_MODE
from .relationship_inference import RelationshipInferenceEngine

logger = logging.getLogger(__name__)


# Protocol for storage backend interface
class StorageProtocol(Protocol):
    async def get_all_memories(self) -> List[Memory]: ...
    async def get_memories_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[Memory]: ...
    async def store(self, memory: Memory) -> Tuple[bool, str]: ...
    async def update_memory(self, memory: Memory) -> bool: ...
    async def delete_memory(self, content_hash: str) -> bool: ...
    async def get_memory_connections(self) -> Dict[str, int]: ...
    async def get_access_patterns(self) -> Dict[str, datetime]: ...


class SyncPauseContext:
    """Context manager for pausing/resuming hybrid backend sync."""

    def __init__(self, storage, logger):
        self.storage = storage
        self.logger = logger
        self.is_hybrid = hasattr(storage, "pause_sync") and hasattr(
            storage, "resume_sync"
        )
        self.sync_paused = False

    async def __aenter__(self):
        if self.is_hybrid:
            self.logger.info("Pausing hybrid backend sync during consolidation")
            await self.storage.pause_sync()
            self.sync_paused = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.sync_paused:
            try:
                self.logger.info("Resuming hybrid backend sync after consolidation")
                await self.storage.resume_sync()
            except Exception as e:
                self.logger.error(f"Failed to resume sync: {e}", exc_info=True)


def check_horizon_requirements(
    time_horizon: str, phase_name: str, enabled_phases: Dict[str, List[str]]
) -> bool:
    """Check if a consolidation phase should run for the given horizon.

    Args:
        time_horizon: Current time horizon (daily, weekly, etc.)
        phase_name: Phase identifier (clustering, associations, etc.)
        enabled_phases: Dict mapping phase names to applicable horizons

    Returns:
        bool: True if phase should run
    """
    applicable_horizons = enabled_phases.get(phase_name, [])
    return time_horizon in applicable_horizons


# Horizon configuration
HORIZON_CONFIGS = {
    "daily": {"delta": timedelta(days=1), "cutoff_days": 2},
    "weekly": {"delta": timedelta(days=7), "cutoff_days": None},
    "monthly": {"delta": timedelta(days=30), "cutoff_days": None},
    "quarterly": {"delta": timedelta(days=90), "cutoff_days": 90},
    "yearly": {"delta": timedelta(days=365), "cutoff_days": 365},
}


def filter_memories_by_age(
    memories: List[Memory], cutoff_date: datetime
) -> List[Memory]:
    """Filter memories created before the cutoff date.

    Args:
        memories: List of Memory objects
        cutoff_date: Only keep memories older than this

    Returns:
        Filtered list of memories
    """
    return [
        m
        for m in memories
        if m.created_at and datetime.utcfromtimestamp(m.created_at) < cutoff_date
    ]


class DreamInspiredConsolidator:
    """
    Main consolidation engine with biologically-inspired processing.

    Orchestrates the full consolidation pipeline including:
    - Exponential decay scoring
    - Creative association discovery
    - Semantic clustering and compression
    - Controlled forgetting with archival
    """

    # Phase enablement configuration
    ENABLED_PHASES = {
        "clustering": ["weekly", "monthly", "quarterly"],
        "associations": ["weekly", "monthly"],
        "compression": ["weekly", "monthly", "quarterly"],
        "forgetting": ["monthly", "quarterly", "yearly"],
    }

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
        self.relationship_inference = RelationshipInferenceEngine(
            min_confidence=getattr(config, "relationship_confidence_threshold", 0.6)
        )

        # Initialize health monitoring
        self.health_monitor = ConsolidationHealthMonitor(config)

        # Initialize graph storage for associations
        self._init_graph_storage()

        # Performance tracking
        self.last_consolidation_times = {}
        self.consolidation_stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "total_memories_processed": 0,
            "total_associations_created": 0,
            "total_clusters_created": 0,
            "total_memories_compressed": 0,
            "total_memories_archived": 0,
        }

    def _init_graph_storage(self) -> None:
        """Initialize GraphStorage with appropriate db_path from storage backend."""
        try:
            # Try to get db_path from storage backend
            # Hybrid backend: storage.primary.db_path
            # SQLite-vec backend: storage.db_path
            if hasattr(self.storage, "primary") and hasattr(
                self.storage.primary, "db_path"
            ):
                # Hybrid backend
                db_path = self.storage.primary.db_path
                self.logger.info(
                    f"Initialized GraphStorage with hybrid backend: {db_path}"
                )
            elif hasattr(self.storage, "db_path"):
                # SQLite-vec backend
                db_path = self.storage.db_path
                self.logger.info(
                    f"Initialized GraphStorage with SQLite backend: {db_path}"
                )
            else:
                # Cloudflare-only or unsupported backend
                self.logger.warning(
                    "Storage backend does not support graph storage (no db_path)"
                )
                self.graph_storage = None
                return

            self.graph_storage = GraphStorage(db_path)
            self.logger.info(f"Graph storage mode: {GRAPH_STORAGE_MODE}")

        except Exception as e:
            self.logger.warning(f"Failed to initialize GraphStorage: {e}")
            self.graph_storage = None

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
            memories_processed=0,
        )

        try:
            self.logger.info(
                f"Starting {time_horizon} consolidation - this may take several minutes depending on memory count..."
            )

            # Use context manager for sync pause/resume
            async with SyncPauseContext(self.storage, self.logger):
                # 1. Retrieve memories for processing
                memories = await self._get_memories_for_horizon(time_horizon, **kwargs)
                report.memories_processed = len(memories)

                if not memories:
                    self.logger.info(
                        f"No memories to process for {time_horizon} consolidation"
                    )
                    return self._finalize_report(report, [])

                self.logger.info(f"✓ Found {len(memories)} memories to process")

                # 2. Calculate/update relevance scores
                self.logger.info(
                    f"📊 Phase 1/6: Calculating relevance scores for {len(memories)} memories..."
                )
                performance_start = time.time()
                relevance_scores = await self._update_relevance_scores(
                    memories, time_horizon
                )
                self.logger.info(
                    f"✓ Relevance scoring completed in {time.time() - performance_start:.1f}s"
                )

                # 3. Cluster by semantic similarity (if enabled and appropriate)
                clusters = []
                if self.config.clustering_enabled and check_horizon_requirements(
                    time_horizon, "clustering", self.ENABLED_PHASES
                ):
                    self.logger.info(
                        f"🔗 Phase 2/6: Clustering memories by semantic similarity..."
                    )
                    performance_start = time.time()
                    clusters = await self.clustering_engine.process(memories)
                    report.clusters_created = len(clusters)
                    self.logger.info(
                        f"✓ Clustering completed in {time.time() - performance_start:.1f}s, created {len(clusters)} clusters"
                    )

                # 4. Run creative associations (if enabled and appropriate)
                associations = []
                if self.config.associations_enabled and check_horizon_requirements(
                    time_horizon, "associations", self.ENABLED_PHASES
                ):
                    self.logger.info(
                        f"🧠 Phase 3/6: Discovering creative associations..."
                    )
                    performance_start = time.time()
                    existing_associations = await self._get_existing_associations()
                    associations = await self.association_engine.process(
                        memories, existing_associations=existing_associations
                    )
                    report.associations_discovered = len(associations)
                    self.logger.info(
                        f"✓ Association discovery completed in {time.time() - performance_start:.1f}s, found {len(associations)} associations"
                    )

                    # Store new associations
                    await self._store_associations(associations)

                # 5. Compress clusters (if enabled and clusters exist)
                compression_results = []
                if (
                    self.config.compression_enabled
                    and clusters
                    and check_horizon_requirements(
                        time_horizon, "compression", self.ENABLED_PHASES
                    )
                ):
                    self.logger.info(f"🗜️ Phase 4/6: Compressing memory clusters...")
                    performance_start = time.time()
                    compression_results = await self.compression_engine.process(
                        clusters, memories
                    )
                    report.memories_compressed = len(compression_results)
                    self.logger.info(
                        f"✓ Compression completed in {time.time() - performance_start:.1f}s, compressed {len(compression_results)} clusters"
                    )

                    # Store compressed memories and update originals
                    await self._handle_compression_results(compression_results)

                # 6. Controlled forgetting (if enabled and appropriate)
                forgetting_results = []
                if self.config.forgetting_enabled and check_horizon_requirements(
                    time_horizon, "forgetting", self.ENABLED_PHASES
                ):
                    self.logger.info(f"🗂️ Phase 5/6: Applying controlled forgetting...")
                    performance_start = time.time()
                    access_patterns = await self._get_access_patterns()
                    forgetting_results = await self.forgetting_engine.process(
                        memories,
                        relevance_scores,
                        access_patterns=access_patterns,
                        time_horizon=time_horizon,
                    )
                    report.memories_archived = len(
                        [
                            r
                            for r in forgetting_results
                            if r.action_taken in ["archived", "deleted"]
                        ]
                    )
                    self.logger.info(
                        f"✓ Forgetting completed in {time.time() - performance_start:.1f}s, processed {len(forgetting_results)} candidates"
                    )

                    # Apply forgetting results to storage
                    await self._apply_forgetting_results(forgetting_results)

                # 7. Update consolidation statistics
                self._update_consolidation_stats(report)

                # 8. Track consolidation timestamp for incremental mode
                if self.config.incremental_mode:
                    await self._update_consolidation_timestamps(memories)

                # 9. Finalize report
                return self._finalize_report(report, [])

        except ConsolidationError as e:
            # Re-raise configuration and validation errors
            self.logger.error(
                f"Configuration error during {time_horizon} consolidation: {e}"
            )
            self.health_monitor.record_error(
                "consolidator", e, {"time_horizon": time_horizon}
            )
            raise
        except Exception as e:
            self.logger.error(f"Error during {time_horizon} consolidation: {e}")
            self.health_monitor.record_error(
                "consolidator", e, {"time_horizon": time_horizon}
            )
            report.errors.append(str(e))
            return self._finalize_report(report, [str(e)])

    async def _get_memories_for_horizon(
        self, time_horizon: str, **kwargs
    ) -> List[Memory]:
        """Get memories appropriate for the given time horizon.

        With incremental mode enabled, returns oldest-first batch of memories
        that haven't been recently consolidated.
        """
        now = datetime.now(timezone.utc)

        # Validate time horizon
        if time_horizon not in HORIZON_CONFIGS:
            raise ConsolidationError(f"Unknown time horizon: {time_horizon}")

        config = HORIZON_CONFIGS[time_horizon]

        # For daily processing, get recent memories (no change - already efficient)
        if time_horizon == "daily":
            cutoff_days = config.get("cutoff_days", 2)
            start_time = (now - timedelta(days=cutoff_days)).timestamp()
            end_time = now.timestamp()
            memories = await self.storage.get_memories_by_time_range(
                start_time, end_time
            )
        else:
            # For longer horizons: incremental oldest-first processing
            memories = await self.storage.get_all_memories()

            # Filter by relevance to time horizon (quarterly/yearly still focus on old memories)
            cutoff_days = config.get("cutoff_days")
            if cutoff_days is not None:
                cutoff_date = now - timedelta(days=cutoff_days)
                memories = filter_memories_by_age(memories, cutoff_date)

            # Incremental mode: Sort oldest-first and batch
            if self.config.incremental_mode:
                # Sort by last_consolidated_at (oldest first), fallback to created_at
                def get_consolidation_sort_key(memory: Memory) -> float:
                    # Check metadata for last consolidation timestamp
                    if memory.metadata and "last_consolidated_at" in memory.metadata:
                        return float(memory.metadata["last_consolidated_at"])
                    # Fall back to created_at (treat never-consolidated as oldest)
                    return memory.created_at if memory.created_at else 0.0

                memories.sort(key=get_consolidation_sort_key)

                # Limit to batch size
                batch_size = self.config.batch_size
                if len(memories) > batch_size:
                    self.logger.info(
                        f"Incremental mode: Processing {batch_size} oldest memories (out of {len(memories)} total)"
                    )
                    memories = memories[:batch_size]

        return memories

    async def _update_relevance_scores(
        self, memories: List[Memory], time_horizon: str
    ) -> List:
        """Calculate and update relevance scores for memories."""
        # Get connection and access data
        connections = await self._get_memory_connections()
        access_patterns = await self._get_access_patterns()

        # Calculate relevance scores
        relevance_scores = await self.decay_calculator.process(
            memories,
            connections=connections,
            access_patterns=access_patterns,
            reference_time=datetime.now(),
        )

        # Update memory metadata with relevance scores (v8.47.1 - batch optimization)
        # Collect all memories to update, then use single batch operation for 50-100x speedup
        memories_to_update = []
        for memory in memories:
            score = next(
                (s for s in relevance_scores if s.memory_hash == memory.content_hash),
                None,
            )
            if score:
                updated_memory = (
                    await self.decay_calculator.update_memory_relevance_metadata(
                        memory, score
                    )
                )
                memories_to_update.append(updated_memory)

        # Single batch transaction instead of 500+ sequential calls
        if memories_to_update:
            await self.storage.update_memories_batch(memories_to_update)

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
            self.logger.warning(
                "Storage backend doesn't support access pattern tracking"
            )
            return {}

    async def _get_existing_associations(self) -> set:
        """Get existing memory associations to avoid duplicates."""
        try:
            # Look for existing association memories
            all_memories = await self.storage.get_all_memories()
            associations = set()

            for memory in all_memories:
                if (
                    memory.memory_type == "association"
                    and "source_memory_hashes" in memory.metadata
                ):
                    source_hashes = memory.metadata["source_memory_hashes"]
                    if isinstance(source_hashes, list) and len(source_hashes) >= 2:
                        # Create canonical pair representation
                        pair_key = tuple(sorted(source_hashes[:2]))
                        associations.add(pair_key)

            return associations

        except Exception as e:
            self.logger.warning(f"Error getting existing associations: {e}")
            return set()

    async def _store_associations(self, associations) -> None:
        """
        Store discovered associations using configured graph storage mode.

        Supports three modes:
        - memories_only: Store as Memory objects (legacy, backward compatible)
        - dual_write: Store in BOTH memories and graph table (transition mode, default)
        - graph_only: Only store in graph table (recommended, modern)
        """
        if not associations:
            return

        self.logger.info(
            f"Storing {len(associations)} associations using mode: {GRAPH_STORAGE_MODE}"
        )

        # Store in memories table if enabled
        if GRAPH_STORAGE_MODE in ["memories_only", "dual_write"]:
            await self._store_associations_in_memories(associations)

        # Store in graph table if enabled
        if GRAPH_STORAGE_MODE in ["dual_write", "graph_only"]:
            await self._store_associations_in_graph_table(associations)

    async def _store_associations_in_memories(self, associations) -> None:
        """Store associations as Memory objects (legacy method)."""
        stored_count = 0
        failed_count = 0

        for association in associations:
            try:
                # Create memory content from association
                source_hashes = association.source_memory_hashes
                similarity = association.similarity_score
                connection_type = association.connection_type

                content = f"Association between memories {source_hashes[0][:8]} and {source_hashes[1][:8]}: {connection_type} (similarity: {similarity:.3f})"

                # Create association memory
                association_memory = Memory(
                    content=content,
                    content_hash=f"assoc_{source_hashes[0][:8]}_{source_hashes[1][:8]}",
                    tags=["association", "discovered"] + connection_type.split(", "),
                    memory_type="association",
                    metadata={
                        "source_memory_hashes": source_hashes,
                        "similarity_score": similarity,
                        "connection_type": connection_type,
                        "discovery_method": association.discovery_method,
                        "discovery_date": association.discovery_date.isoformat(),
                        **association.metadata,
                    },
                    created_at=datetime.now().timestamp(),
                    created_at_iso=datetime.now().isoformat() + "Z",
                )

                # Store the association memory
                success, _ = await self.storage.store(association_memory)
                if success:
                    stored_count += 1
                else:
                    failed_count += 1
                    self.logger.warning(
                        f"Failed to store association memory for {source_hashes[0][:8]} <-> {source_hashes[1][:8]}"
                    )

            except Exception as e:
                failed_count += 1
                # Try to extract hashes for better debugging context
                try:
                    hashes = association.source_memory_hashes
                    hash_info = f"{hashes[0][:8]} <-> {hashes[1][:8]} "
                except (AttributeError, IndexError):
                    hash_info = ""
                self.logger.warning(
                    f"Error storing association {hash_info}as memory: {e}"
                )

        self.logger.info(
            f"Stored {stored_count} associations as memories ({failed_count} failed)"
            if failed_count > 0
            else f"Stored {stored_count} associations as memories"
        )

    async def _store_associations_in_graph_table(self, associations) -> None:
        """Store associations in graph table using GraphStorage."""
        if self.graph_storage is None:
            self.logger.warning(
                "GraphStorage not available, skipping graph table storage"
            )
            return

        stored_count = 0
        failed_count = 0

        # Build hash -> memory lookup map for efficiency
        all_memories = await self.storage.get_all_memories()
        memory_map = {m.content_hash: m for m in all_memories}

        for association in associations:
            try:
                source_hashes = association.source_memory_hashes
                if len(source_hashes) < 2:
                    self.logger.warning(
                        f"Invalid association: less than 2 source hashes"
                    )
                    failed_count += 1
                    continue

                source_hash = source_hashes[0]
                target_hash = source_hashes[1]

                # Get source and target memories for relationship inference
                source_memory = memory_map.get(source_hash)
                target_memory = memory_map.get(target_hash)

                # Convert connection_type string to list
                connection_types = [
                    ct.strip() for ct in association.connection_type.split(",")
                ]

                # Infer relationship type if both memories available
                relationship_type = "related"
                if source_memory and target_memory:
                    try:
                        (
                            inferred_rel,
                            confidence,
                        ) = await self.relationship_inference.infer_relationship_type(
                            source_type=source_memory.memory_type,
                            target_type=target_memory.memory_type,
                            source_content=source_memory.content,
                            target_content=target_memory.content,
                            source_timestamp=source_memory.created_at,
                            target_timestamp=target_memory.created_at,
                            source_tags=source_memory.tags,
                            target_tags=target_memory.tags,
                        )
                        if inferred_rel != "related":
                            relationship_type = inferred_rel
                            self.logger.debug(
                                f"Inferred relationship '{inferred_rel}' (confidence: {confidence:.2f}) "
                                f"for {source_hash[:8]} <-> {target_hash[:8]}"
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to infer relationship type for {source_hash[:8]} <-> {target_hash[:8]}: {e}"
                        )

                # Prepare metadata
                metadata = {
                    "discovery_method": association.discovery_method,
                    "discovery_date": association.discovery_date.isoformat(),
                    **association.metadata,
                }

                # Store in graph table with inferred relationship type
                success = await self.graph_storage.store_association(
                    source_hash=source_hash,
                    target_hash=target_hash,
                    similarity=association.similarity_score,
                    connection_types=connection_types,
                    metadata=metadata,
                    relationship_type=relationship_type,
                )

                if success:
                    stored_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                failed_count += 1
                self.logger.warning(f"Failed to store association in graph table: {e}")

        self.logger.info(
            f"Stored {stored_count} associations in graph table ({failed_count} failed)"
            if failed_count > 0
            else f"Stored {stored_count} associations in graph table"
        )

    async def _handle_compression_results(self, compression_results) -> None:
        """Handle storage of compressed memories and linking to originals."""
        for result in compression_results:
            # Store compressed memory
            success, _ = await self.storage.store(result.compressed_memory)
            if not success:
                logger.warning(f"Failed to store compressed memory")

            # Update original memories with compression links
            # This could involve adding metadata pointing to the compressed version
            # Implementation depends on how the storage backend handles relationships
            pass

    async def _apply_forgetting_results(self, forgetting_results) -> None:
        """Apply forgetting results to the storage backend."""
        for result in forgetting_results:
            if result.action_taken == "deleted":
                await self.storage.delete_memory(result.memory_hash)
            elif result.action_taken == "compressed" and result.compressed_version:
                # Replace original with compressed version
                await self.storage.delete_memory(result.memory_hash)
                success, _ = await self.storage.store(result.compressed_version)
                if not success:
                    logger.warning(
                        f"Failed to store compressed version for {result.memory_hash}"
                    )
            # 'archived' memories are handled by the forgetting engine

    async def _update_consolidation_timestamps(self, memories: List[Memory]) -> None:
        """Mark memories with last_consolidated_at timestamp for incremental mode using batch updates."""
        consolidation_time = datetime.now().timestamp()

        self.logger.info(
            f"Marking {len(memories)} memories with consolidation timestamp (batch mode)"
        )

        # Update all memories in-place
        for memory in memories:
            if memory.metadata is None:
                memory.metadata = {}
            memory.metadata["last_consolidated_at"] = consolidation_time

        # Use batch update for optimal performance
        try:
            results = await self.storage.update_memories_batch(memories)
            success_count = sum(results)
            self.logger.info(
                f"Consolidation timestamps updated: {success_count}/{len(memories)} memories"
            )

            if success_count < len(memories):
                failed_count = len(memories) - success_count
                self.logger.warning(
                    f"{failed_count} memories failed to update during timestamp marking"
                )

        except Exception as e:
            self.logger.error(f"Batch timestamp update failed: {e}")
            # Fallback to individual updates if batch fails
            self.logger.info("Falling back to individual timestamp updates")
            success_count = 0
            for memory in memories:
                try:
                    success = await self.storage.update_memory(memory)
                    if success:
                        success_count += 1
                except Exception as mem_error:
                    self.logger.warning(
                        f"Failed to update consolidation timestamp for {memory.content_hash}: {mem_error}"
                    )

            self.logger.info(
                f"Fallback completed: {success_count}/{len(memories)} memories updated"
            )

    def _update_consolidation_stats(self, report: ConsolidationReport) -> None:
        """Update internal consolidation statistics."""
        self.consolidation_stats["total_runs"] += 1
        if not report.errors:
            self.consolidation_stats["successful_runs"] += 1

        self.consolidation_stats["total_memories_processed"] += (
            report.memories_processed
        )
        self.consolidation_stats["total_associations_created"] += (
            report.associations_discovered
        )
        self.consolidation_stats["total_clusters_created"] += report.clusters_created
        self.consolidation_stats["total_memories_compressed"] += (
            report.memories_compressed
        )
        self.consolidation_stats["total_memories_archived"] += report.memories_archived

        # Update last consolidation time
        self.last_consolidation_times[report.time_horizon] = report.start_time

    def _finalize_report(
        self, report: ConsolidationReport, errors: List[str]
    ) -> ConsolidationReport:
        """Finalize the consolidation report."""
        report.end_time = datetime.now()
        report.errors.extend(errors)

        # Add performance metrics
        duration = (report.end_time - report.start_time).total_seconds()
        success = len(errors) == 0
        report.performance_metrics = {
            "duration_seconds": duration,
            "memories_per_second": report.memories_processed / duration
            if duration > 0
            else 0,
            "success": success,
        }

        # Record performance in health monitor
        self.health_monitor.record_consolidation_performance(
            time_horizon=report.time_horizon,
            duration=duration,
            memories_processed=report.memories_processed,
            success=success,
            errors=errors,
        )

        # Log summary
        if errors:
            self.logger.error(
                f"Consolidation {report.time_horizon} completed with errors: {errors}"
            )
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

    async def get_consolidation_recommendations(
        self, time_horizon: str
    ) -> Dict[str, Any]:
        """Get recommendations for consolidation based on current memory state."""
        try:
            memories = await self._get_memories_for_horizon(time_horizon)

            if not memories:
                return {
                    "recommendation": "no_action",
                    "reason": "No memories to process",
                    "memory_count": 0,
                }

            # Analyze memory distribution
            memory_types = {}
            total_size = 0
            old_memories = 0
            now = datetime.now()

            for memory in memories:
                memory_type = memory.memory_type or "standard"
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
                total_size += len(memory.content)

                if memory.created_at:
                    age_days = (now - datetime.utcfromtimestamp(memory.created_at)).days
                    if age_days > 30:
                        old_memories += 1

            # Generate recommendations
            recommendations = []

            if len(memories) > 1000:
                recommendations.append(
                    "Consider running compression to reduce memory usage"
                )

            if old_memories > len(memories) * 0.5:
                recommendations.append(
                    "Many old memories present - consider forgetting/archival"
                )

            if len(memories) > 100 and time_horizon in ["weekly", "monthly"]:
                recommendations.append("Good candidate for association discovery")

            if not recommendations:
                recommendations.append("Memory state looks healthy")

            return {
                "recommendation": "consolidation_beneficial"
                if len(recommendations) > 1
                else "optional",
                "reasons": recommendations,
                "memory_count": len(memories),
                "memory_types": memory_types,
                "total_size_bytes": total_size,
                "old_memory_percentage": (old_memories / len(memories)) * 100,
                "estimated_duration_seconds": len(memories) * 0.01,  # Rough estimate
            }

        except Exception as e:
            return {"recommendation": "error", "error": str(e), "memory_count": 0}
