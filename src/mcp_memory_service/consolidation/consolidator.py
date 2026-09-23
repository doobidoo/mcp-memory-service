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

import inspect
from typing import List, Dict, Any, Optional, Protocol, Sequence, Tuple
from datetime import datetime, timedelta, timezone
from pathlib import Path
import asyncio
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
from ..config import GRAPH_STORAGE_MODE, CONSOLIDATION_STORE_ASSOCIATIONS, TYPED_EDGES_ENABLED
from .relationship_inference import RelationshipInferenceEngine
from .run_tracker import RunTracker
from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)


# Protocol for storage backend interface
class StorageProtocol(Protocol):
    async def get_all_memories(self) -> List[Memory]: pass
    async def get_memories_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[Memory]:
        pass

    async def search_by_tag(
        self, tags: List[str], time_start: Optional[float] = None
    ) -> List[Memory]:
        pass

    async def store(self, memory: Memory) -> Tuple[bool, str]:
        pass

    async def update_memory(self, memory: Memory) -> bool:
        pass

    async def delete_memory(self, content_hash: str) -> bool:
        pass

    async def get_memory_connections(self) -> Dict[str, int]:
        pass

    async def get_access_patterns(
        self, content_hashes: Optional[Sequence[str]] = None
    ) -> Dict[str, datetime]:
        pass


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
                self.logger.error("Failed to resume sync: %s", _sanitize_log_value(e), exc_info=True)


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


# Horizon configuration.
#
# `window` is how far back a horizon reaches: a run considers memories created
# within the last `window`, which is what the `memory_consolidate` tool
# description promises for every horizon. `incremental` carries a window too,
# but only as the bootstrap span for its first run -- afterwards it advances a
# cursor from the last recorded run.
HORIZON_CONFIGS = {
    "daily": {"window": timedelta(days=1)},
    "weekly": {"window": timedelta(days=7)},
    "monthly": {"window": timedelta(days=30)},
    "quarterly": {"window": timedelta(days=90)},
    "yearly": {"window": timedelta(days=365)},
    "incremental": {"window": timedelta(days=1)},
}


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
        "clustering": ["weekly", "monthly", "quarterly", "incremental"],
        "associations": ["weekly", "monthly", "incremental"],
        "compression": ["weekly", "monthly", "quarterly", "incremental"],
        "forgetting": ["monthly", "quarterly", "yearly"],
    }

    def __init__(self, storage: StorageProtocol, config: ConsolidationConfig):
        self.storage = storage
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.plugin_registry = None  # Set externally to fire on_consolidate hook

        # Initialize component engines
        self.decay_calculator = ExponentialDecayCalculator(config)
        self.association_engine = CreativeAssociationEngine(config)
        self.clustering_engine = SemanticClusteringEngine(config)
        self.compression_engine = SemanticCompressionEngine(config)
        self.forgetting_engine = ControlledForgettingEngine(config)
        self.relationship_inference = RelationshipInferenceEngine(
            min_confidence=getattr(config, "relationship_confidence_threshold", 0.6),
            typed_edges_enabled=TYPED_EDGES_ENABLED,
        )

        # Initialize health monitoring (pass self so checks inspect live components)
        self.health_monitor = ConsolidationHealthMonitor(config, consolidator=self)

        # Initialize run tracker for incremental consolidation
        self.run_tracker: Optional[RunTracker] = None

        # Graph storage initialized lazily in consolidate() to avoid
        # blocking I/O in __init__ (Milvus backend needs async init).
        self.graph_storage = None
        self._graph_storage_initialized = False
        self._graph_storage_lock = asyncio.Lock()

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

    async def _init_graph_storage(self) -> None:
        """Initialize graph storage with appropriate backend.

        Supports SQLite-vec, Hybrid (via SQLite primary), and Milvus backends.
        Cloudflare-only backend does not support graph storage.
        """
        try:
            # Try to get db_path from storage backend
            # Hybrid backend: storage.primary.db_path
            # SQLite-vec backend: storage.db_path
            if hasattr(self.storage, "primary") and hasattr(
                self.storage.primary, "db_path"
            ):
                # Hybrid backend
                db_path = self.storage.primary.db_path
                self.graph_storage = GraphStorage(db_path)
                self.logger.info(
                    f"Initialized GraphStorage with hybrid backend: {db_path}"
                )
            elif hasattr(self.storage, "db_path"):
                # SQLite-vec backend
                db_path = self.storage.db_path
                self.graph_storage = GraphStorage(db_path)
                self.logger.info(
                    f"Initialized GraphStorage with SQLite backend: {db_path}"
                )
            elif hasattr(self.storage, "uri") and hasattr(self.storage, "collection_name"):
                # Milvus backend — use MilvusGraphStorage with async init
                try:
                    from ..storage.milvus_graph import MilvusGraphStorage
                    self.graph_storage = MilvusGraphStorage(
                        uri=self.storage.uri,
                        token=getattr(self.storage, "token", None),
                        collection_name=self.storage.collection_name,
                    )
                    await self.graph_storage.initialize()
                    self.logger.info(
                        f"Initialized MilvusGraphStorage for consolidation "
                        f"(uri={self.storage.uri}, collection={self.graph_storage.collection_name})"
                    )
                except ImportError:
                    self.logger.warning(
                        "pymilvus not available, graph storage disabled for Milvus backend"
                    )
                    self.graph_storage = None
                    return
            else:
                # Cloudflare-only or unsupported backend
                self.logger.warning(
                    "Storage backend does not support graph storage (no db_path or uri)"
                )
                self.graph_storage = None
                return

            self.logger.info("Graph storage mode: %s", GRAPH_STORAGE_MODE)

        except Exception as e:
            self.logger.warning("Failed to initialize graph storage: %s", _sanitize_log_value(e))
            self.graph_storage = None

    def _resolve_tracker_db_path(self) -> Optional[Path]:
        """Resolve path for the run tracker SQLite DB (next to main memory DB)."""
        db_path = None
        if hasattr(self.storage, "primary") and hasattr(self.storage.primary, "db_path"):
            db_path = getattr(self.storage.primary, "db_path", None)
        if not isinstance(db_path, (str, Path)) and hasattr(self.storage, "db_path"):
            db_path = getattr(self.storage, "db_path", None)
        if isinstance(db_path, (str, Path)):
            return Path(db_path).parent / "consolidation_runs.db"
        return None

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

        is_incremental = time_horizon == "incremental"

        try:
            self.logger.info(
                "Starting %s consolidation - this may take several minutes depending on memory count...",
                _sanitize_log_value(time_horizon)
            )

            # Incremental: initialize run_tracker, concurrency guard, timeout
            if await self._acquire_incremental_slot(time_horizon):
                self.logger.info("Incremental consolidation already in flight, skipping")
                return self._finalize_report(report, ["Skipped: concurrent run in flight"])

            await self._ensure_graph_storage()

            # Use context manager for sync pause/resume
            async with SyncPauseContext(self.storage, self.logger):
                return await self._run_consolidation_pass(
                    report, time_horizon, is_incremental, **kwargs
                )

        except ConsolidationError as e:
            # Re-raise configuration and validation errors
            self.logger.error(
                "Configuration error during %s consolidation: %s",
                _sanitize_log_value(time_horizon), _sanitize_log_value(e)
            )
            self.health_monitor.record_error(
                "consolidator", e, {"time_horizon": time_horizon}
            )
            raise
        except Exception as e:
            self.logger.error("Error during %s consolidation: %s", _sanitize_log_value(time_horizon), _sanitize_log_value(e))
            self.health_monitor.record_error(
                "consolidator", e, {"time_horizon": time_horizon}
            )
            report.errors.append(str(e))
            return self._finalize_report(report, [str(e)])
        finally:
            if is_incremental and self.run_tracker:
                self.run_tracker.release("incremental")

    async def _run_relevance_phase(
        self, memories: List[Memory], time_horizon: str
    ) -> Dict[str, float]:
        """Phase 1/6: score memories by relevance, returning {hash: score}."""
        self.logger.info(
            "📊 Phase 1/6: Calculating relevance scores for %s memories...",
            len(memories),
        )
        performance_start = time.time()
        relevance_scores = await self._update_relevance_scores(
            memories, time_horizon
        )
        self.logger.info(
            "✓ Relevance scoring completed in %.1fs",
            time.time() - performance_start,
        )
        return relevance_scores

    async def _run_clustering_phase(
        self, memories: List[Memory], time_horizon: str
    ) -> List:
        """Phase 2/6: cluster memories by semantic similarity."""
        self.logger.info("🔗 Phase 2/6: Clustering memories by semantic similarity...")
        performance_start = time.time()
        clusters = await self.clustering_engine.process(memories)
        self.logger.info(
            "✓ Clustering completed in %.1fs, created %s clusters",
            time.time() - performance_start,
            len(clusters),
        )
        return clusters

    async def _run_associations_phase(
        self, memories: List[Memory], time_horizon: str
    ) -> List:
        """Phase 3/6: discover and store creative associations."""
        self.logger.info("🧠 Phase 3/6: Discovering creative associations...")
        performance_start = time.time()
        existing_associations = await self._get_existing_associations()
        associations = await self.association_engine.process(
            memories, existing_associations=existing_associations
        )
        self.logger.info(
            "✓ Association discovery completed in %.1fs, found %s associations",
            time.time() - performance_start,
            len(associations),
        )
        await self._store_associations(associations)
        return associations

    async def _run_compression_phase(
        self, clusters: List, memories: List[Memory]
    ) -> List:
        """Phase 4/6: compress clusters and store compressed results."""
        self.logger.info("🗜️ Phase 4/6: Compressing memory clusters...")
        performance_start = time.time()
        compression_results = await self.compression_engine.process(
            clusters, memories
        )
        self.logger.info(
            "✓ Compression completed in %.1fs, compressed %s clusters",
            time.time() - performance_start,
            len(compression_results),
        )
        await self._handle_compression_results(compression_results)
        return compression_results

    async def _run_consolidation_pass(
        self,
        report: ConsolidationReport,
        time_horizon: str,
        is_incremental: bool,
        **kwargs,
    ) -> ConsolidationReport:
        """Run one full pipeline pass inside the sync-pause context."""
        memories = await self._get_memories_for_horizon(time_horizon, **kwargs)
        report.memories_processed = len(memories)

        if not memories:
            self.logger.info(
                "No memories to process for %s consolidation",
                _sanitize_log_value(time_horizon)
            )
            # Record run even on 0 memories to advance timestamp
            if is_incremental and self.run_tracker:
                await self.run_tracker.record_run("incremental", 0)
            return self._finalize_report(report, [])

        self.logger.info("✓ Found %s memories to process", len(memories))

        # 2-6. Run the conditional consolidation phases
        await self._run_phase_schedule(memories, time_horizon, report)

        return await self._finalize_consolidation(
            report, memories, is_incremental
        )

    async def _acquire_incremental_slot(self, time_horizon: str) -> bool:
        """Initialize the run tracker and acquire the incremental slot.

        Returns True when the run should be skipped (another run is in flight).
        """
        if time_horizon != "incremental":
            return False
        if self.run_tracker is None:
            db_path = self._resolve_tracker_db_path()
            db_path = Path(str(db_path)) if db_path else None
            if db_path:
                self.run_tracker = RunTracker(db_path)
        if self.run_tracker and not self.run_tracker.try_acquire("incremental"):
            return True
        return False

    async def _ensure_graph_storage(self) -> None:
        """Lazily initialize graph storage under the double-init lock."""
        async with self._graph_storage_lock:
            if not self._graph_storage_initialized:
                await self._init_graph_storage()
                self._graph_storage_initialized = True

    async def _run_phase_schedule(
        self,
        memories: List[Memory],
        time_horizon: str,
        report: ConsolidationReport,
    ) -> None:
        """Run the conditional consolidation phases (1-6) and update *report*."""
        await self._run_relevance_phase(memories, time_horizon)

        clusters: list = []
        if self.config.clustering_enabled and check_horizon_requirements(
            time_horizon, "clustering", self.ENABLED_PHASES
        ):
            clusters = await self._run_clustering_phase(memories, time_horizon)
            report.clusters_created = len(clusters)

        if self.config.associations_enabled and check_horizon_requirements(
            time_horizon, "associations", self.ENABLED_PHASES
        ):
            associations = await self._run_associations_phase(
                memories, time_horizon
            )
            report.associations_discovered = len(associations)

        if (
            self.config.compression_enabled
            and clusters
            and check_horizon_requirements(
                time_horizon, "compression", self.ENABLED_PHASES
            )
        ):
            compression_results = await self._run_compression_phase(
                clusters, memories
            )
            report.memories_compressed = len(compression_results)

        # Forgetting gets its own candidate selector that reaches beyond the
        # horizon window into the stale tail (Codeberg #325).
        if self.config.forgetting_enabled and check_horizon_requirements(
            time_horizon, "forgetting", self.ENABLED_PHASES
        ):
            await self._run_forgetting_phase(time_horizon, report)

    async def _finalize_consolidation(
        self,
        report: ConsolidationReport,
        memories: List[Memory],
        is_incremental: bool,
    ) -> ConsolidationReport:
        """Post-phase bookkeeping: prune, stats, timestamps, events, report."""
        orphaned = await self._prune_orphaned_graph_edges()
        if orphaned > 0:
            self.logger.info("🧹 Pruned %s orphaned graph edges", orphaned)

        self._update_consolidation_stats(report)

        if self.config.incremental_mode:
            await self._update_consolidation_timestamps(memories)

        report = self._finalize_report(report, [])

        if is_incremental and self.run_tracker:
            await self.run_tracker.record_run(
                "incremental", report.memories_processed
            )

        if self.plugin_registry:
            await self.plugin_registry.fire('on_consolidate', {
                **report.performance_metrics,
                'time_horizon': report.time_horizon,
                'memories_processed': report.memories_processed,
                'associations_discovered': report.associations_discovered,
                'clusters_created': report.clusters_created,
            })
        return report

    async def _get_memories_for_horizon(
        self, time_horizon: str, **kwargs
    ) -> List[Memory]:
        """Get the memories a horizon may touch: those created inside its window.

        Every horizon reaches back exactly as far as its documented window
        (`daily` one day, `weekly` seven, and so on). With incremental mode
        enabled the window is then narrowed to the oldest `batch_size` memories
        inside it, so a run stays bounded on a large store.
        """
        now = datetime.now(timezone.utc)

        # Validate time horizon
        if time_horizon not in HORIZON_CONFIGS:
            raise ConsolidationError(f"Unknown time horizon: {time_horizon}")

        window = HORIZON_CONFIGS[time_horizon]["window"]

        # Incremental: only memories created since last run
        if time_horizon == "incremental":
            last_run = None
            if self.run_tracker:
                last_run = await self.run_tracker.get_last_run_at("incremental")
            # Bootstrap: use the configured window on first run
            if last_run is None:
                last_run = (now - window).timestamp()
            end_time = now.timestamp()
            memories = await self.storage.get_memories_by_time_range(
                last_run, end_time, include_embeddings=True,
            )
            return memories

        memories = await self.storage.get_memories_by_time_range(
            (now - window).timestamp(), now.timestamp(), include_embeddings=True,
        )

        if self.config.incremental_mode:
            memories = self._take_oldest_batch(memories)

        return memories

    async def _get_forgetting_candidates(
        self, time_horizon: str
    ) -> List[Memory]:
        """Get forgetting candidates that reach beyond the horizon window.

        Codeberg #325: every horizon means its documented window, so nothing
        younger than the floor enters the forgetting phase.  This method
        queries everything older than ``forgetting_min_age_days`` (default 365),
        then bounds the read to ``batch_size`` so a run stays finite.

        The ``forgetting_min_age_days`` config acts as a floor:
        nothing younger than that is considered stale enough for archival.
        Set via ``MCP_FORGETTING_MIN_AGE_DAYS`` env var.
        """
        now = datetime.now(timezone.utc)

        window = HORIZON_CONFIGS[time_horizon]["window"]
        min_age_days = max(self.config.forgetting_min_age_days, window.days)
        min_age_cutoff = (now - timedelta(days=min_age_days)).timestamp()

        # Query everything older than the floor (the stale tail)
        candidates = await self.storage.get_memories_by_time_range(
            0.0, min_age_cutoff, include_embeddings=True,
        )

        # Always bound the read so a deployment with thousands of stale
        # memories does not load them all into one run.
        if len(candidates) > self.config.batch_size:
            candidates = self._take_oldest_batch(candidates)

        self.logger.info(
            "Forgetting candidates: %s memories older than %sd",
            len(candidates),
            min_age_days,
        )
        return candidates

    async def _run_forgetting_phase(
        self, time_horizon: str, report: ConsolidationReport
    ) -> list:
        """Run the controlled-forgetting phase and update *report* in place.

        Extracted from :meth:`consolidate` to keep that method's cyclomatic
        complexity under the pre-commit gate.
        """
        self.logger.info("🗂️ Phase 5/6: Applying controlled forgetting...")
        performance_start = time.time()
        forgetting_candidates = await self._get_forgetting_candidates(time_horizon)

        if not forgetting_candidates:
            self.logger.info("No stale-tail candidates for forgetting")
            return []

        forgetting_scores = await self._update_relevance_scores(
            forgetting_candidates, time_horizon
        )
        access_patterns = await self._get_access_patterns(
            [m.content_hash for m in forgetting_candidates]
        )
        forgetting_results = await self.forgetting_engine.process(
            forgetting_candidates,
            forgetting_scores,
            access_patterns=access_patterns,
            time_horizon=time_horizon,
        )

        report.memories_archived = len(
            [r for r in forgetting_results if r.action_taken in ["archived", "deleted"]]
        )
        self.logger.info(
            f"✓ Forgetting completed in {time.time() - performance_start:.1f}s, "
            f"processed {len(forgetting_results)} candidates"
        )

        # Retained rows must sort behind candidates not processed yet.
        await self._update_consolidation_timestamps(forgetting_candidates)
        await self._apply_forgetting_results(forgetting_results)
        return forgetting_results

    def _take_oldest_batch(self, memories: List[Memory]) -> List[Memory]:
        """Narrow a window to the oldest `batch_size` memories in it.

        Least-recently-consolidated first, so repeated runs work through the
        window instead of re-processing the same head every time.
        """
        def sort_key(memory: Memory) -> float:
            # Check metadata for last consolidation timestamp
            if memory.metadata and "last_consolidated_at" in memory.metadata:
                return float(memory.metadata["last_consolidated_at"])
            # Fall back to created_at (treat never-consolidated as oldest)
            return memory.created_at if memory.created_at else 0.0

        memories = sorted(memories, key=sort_key)

        batch_size = self.config.batch_size
        if len(memories) > batch_size:
            self.logger.info(
                f"Processing {batch_size} oldest memories "
                f"(out of {len(memories)} in the window)"
            )
            memories = memories[:batch_size]
        return memories

    async def _update_relevance_scores(
        self, memories: List[Memory], time_horizon: str
    ) -> List:
        """Calculate and update relevance scores for memories."""
        # Get connection and access data
        connections = await self._get_memory_connections()
        access_patterns = await self._get_access_patterns(
            [m.content_hash for m in memories]
        )

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
            await self.storage.update_memories_batch(memories_to_update, preserve_timestamps=True)

        return relevance_scores

    async def _get_memory_connections(self) -> Dict[str, int]:
        """Get memory connection counts from storage."""
        try:
            return await self.storage.get_memory_connections()
        except AttributeError:
            # Fallback if storage doesn't implement connection tracking
            self.logger.warning("Storage backend doesn't support connection tracking")
            return {}

    async def _get_access_patterns(
        self, content_hashes: Optional[Sequence[str]] = None
    ) -> Dict[str, datetime]:
        """Get memory access patterns from storage, scoped to the candidate window.

        Consumers (decay boost, forgetting) only look up hashes that are in the current
        batch, so passing the window keeps per-run memory and latency proportional to
        the batch instead of the whole ever-accessed population.

        Backends that predate the ``content_hashes`` parameter (including third-party
        implementations of the storage protocol) are detected by signature and called
        with no arguments, so they keep working unchanged.
        """
        try:
            getter = self.storage.get_access_patterns
        except AttributeError:
            # Fallback if storage doesn't implement access tracking
            self.logger.warning(
                "Storage backend doesn't support access pattern tracking"
            )
            return {}
        try:
            style = (
                None if content_hashes is None
                else self._access_window_call_style(getter)
            )
            if style == "keyword":
                return await getter(content_hashes=content_hashes)
            if style == "positional":
                return await getter(content_hashes)
            return await getter()
        except AttributeError:
            self.logger.warning(
                "Storage backend doesn't support access pattern tracking"
            )
            return {}

    @staticmethod
    def _access_window_call_style(getter) -> Optional[str]:
        """How to hand the candidate window to a backend's ``get_access_patterns``.

        Returns ``"keyword"`` or ``"positional"`` when the signature can take the
        window, and ``None`` for a backend that predates the parameter (it is then
        called with no arguments).

        Checked by signature rather than by catching ``TypeError``, so a genuine
        ``TypeError`` raised *inside* a backend is not silently downgraded to a
        full-population query. Accepting the window and passing it are decided
        together: a ``**kwargs`` forwarder or a keyword-only ``content_hashes``
        accepts it only by name, a ``*args`` forwarder or a positional-only
        parameter only by position. Keyword is preferred whenever it binds, because
        it cannot land on an unrelated first parameter the way a positional value can.
        """
        try:
            params = inspect.signature(getter).parameters
        except (TypeError, ValueError):
            return None
        named = params.get("content_hashes")
        if named is not None:
            if named.kind is inspect.Parameter.POSITIONAL_ONLY:
                return "positional"
            if named.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                return "keyword"
        kinds = {p.kind for p in params.values()}
        if inspect.Parameter.VAR_KEYWORD in kinds:
            return "keyword"
        if inspect.Parameter.VAR_POSITIONAL in kinds:
            return "positional"
        return None

    async def _get_existing_associations(self) -> set:
        """Get existing memory associations to avoid duplicates."""
        try:
            # Look for existing association memories by tag instead of scanning all
            all_memories = await self.storage.search_by_tag(["association"])
            associations = set()

            for memory in all_memories:
                if "source_memory_hashes" in memory.metadata:
                    source_hashes = memory.metadata["source_memory_hashes"]
                    if isinstance(source_hashes, list) and len(source_hashes) >= 2:
                        pair_key = tuple(sorted(source_hashes[:2]))
                        associations.add(pair_key)

            return associations

        except Exception as e:
            self.logger.warning("Error getting existing associations: %s", _sanitize_log_value(e))
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

        # Store in memories table if enabled (and not suppressed by config)
        if GRAPH_STORAGE_MODE in ["memories_only", "dual_write"] and CONSOLIDATION_STORE_ASSOCIATIONS:
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
                    memory_type="observation",
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

                # Store the association memory; skip semantic dedup because
                # all association memories share very similar templated content.
                success, reason = await self.storage.store(
                    association_memory, skip_semantic_dedup=True
                )
                if success:
                    stored_count += 1
                else:
                    failed_count += 1
                    self.logger.warning(
                        f"Failed to store association memory for {source_hashes[0][:8]} <-> {source_hashes[1][:8]}: {reason}"
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
        supersede_pairs = []

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
                confidence = 0.0
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

                    # Collect contradiction pairs for batch superseding (#732)
                    if (
                        relationship_type == "contradicts"
                        and confidence >= 0.75
                        and source_memory
                        and target_memory
                    ):
                        source_ts = source_memory.created_at or 0.0
                        target_ts = target_memory.created_at or 0.0
                        if source_ts >= target_ts:
                            supersede_pairs.append((source_memory.content_hash, target_memory.content_hash))
                        else:
                            supersede_pairs.append((target_memory.content_hash, source_memory.content_hash))
                else:
                    failed_count += 1

            except Exception as e:
                failed_count += 1
                self.logger.warning("Failed to store association in graph table: %s", _sanitize_log_value(e))

        # Batch-mark superseded memories in a single transaction (#732)
        if supersede_pairs:
            storage = getattr(self.storage, "primary_storage", None) or self.storage
            if hasattr(storage, 'mark_superseded_batch'):
                marked = await storage.mark_superseded_batch(supersede_pairs)
                self.logger.info(
                    f"Auto-superseded {marked} memories on contradiction detection"
                )

        self.logger.info(
            f"Stored {stored_count} associations in graph table ({failed_count} failed)"
            if failed_count > 0
            else f"Stored {stored_count} associations in graph table"
        )

    async def _handle_compression_results(self, compression_results) -> None:
        """Handle storage of compressed memories — batched for efficiency."""
        if not compression_results:
            return

        compressed_memories = [r.compressed_memory for r in compression_results]

        # Use store_batch if available, fall back to sequential store
        if hasattr(self.storage, 'store_batch'):
            results = await self.storage.store_batch(compressed_memories)
        else:
            results = []
            for mem in compressed_memories:
                results.append(await self.storage.store(mem))

        for (success, msg), result in zip(results, compression_results):
            if not success:
                logger.warning(
                    f"Failed to store compressed memory for cluster "
                    f"{result.cluster_id}: {msg}"
                )

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

    async def _prune_orphaned_graph_edges(self) -> int:
        """Remove graph edges referencing deleted or non-existent memories (#632).

        Note: has_entity edges use target_hash for entity *names* (not content
        hashes), so the target-existence check must exclude them. Source-orphan
        pruning still applies to has_entity (if the memory is deleted, its
        entity links should go too).
        """
        try:
            conn = getattr(self.storage, 'conn', None)
            # For hybrid backend, access the primary (sqlite) storage
            if conn is None:
                primary = getattr(self.storage, 'primary_storage', None)
                if primary:
                    conn = getattr(primary, 'conn', None)
            if conn is None:
                return 0

            cursor = conn.execute("""
                DELETE FROM memory_graph
                WHERE NOT EXISTS (
                    SELECT 1 FROM memories m
                    WHERE m.content_hash = memory_graph.source_hash
                    AND m.deleted_at IS NULL
                )
                OR (
                    memory_graph.relationship_type != 'has_entity'
                    AND NOT EXISTS (
                        SELECT 1 FROM memories m
                        WHERE m.content_hash = memory_graph.target_hash
                        AND m.deleted_at IS NULL
                    )
                )
            """)
            conn.commit()
            return cursor.rowcount
        except Exception as e:
            self.logger.warning("Failed to prune orphaned graph edges: %s", _sanitize_log_value(e))
            return 0

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
            results = await self.storage.update_memories_batch(memories, preserve_timestamps=True)
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
            self.logger.error("Batch timestamp update failed: %s", _sanitize_log_value(e))
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
        health = await self.health_monitor.check_overall_health()
        # check_overall_health() always returns an empty 'statistics' dict —
        # it has no reference to this consolidator's run counters. Merge them
        # in here so `memory_consolidate status` reports real totals.
        health['statistics'] = dict(self.consolidation_stats)
        return health

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
            now = datetime.now(timezone.utc)

            for memory in memories:
                memory_type = memory.memory_type or "standard"
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
                total_size += len(memory.content)

                if memory.created_at:
                    age_days = (now - datetime.fromtimestamp(memory.created_at, tz=timezone.utc)).days
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
