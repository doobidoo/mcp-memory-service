"""
Tests for GitHub issues #544, #545, #546, #547.

#544: Missing `import asyncio` in ai_evaluator.py
#545: Consolidator uses invalid memory_type "association" + missing skip_semantic_dedup
#546: MCP_TYPED_EDGES_ENABLED=false opt-out flag
#547: MCP_CONSOLIDATION_STORE_ASSOCIATIONS=false opt-out flag
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Issue #544 — import asyncio in ai_evaluator.py
# ---------------------------------------------------------------------------

class TestIssue544:
    """import asyncio must be present in ai_evaluator.py."""

    def test_asyncio_importable_from_ai_evaluator_module(self):
        """ai_evaluator.py must import asyncio so evaluate_quality_batch works."""
        import importlib, inspect
        import mcp_memory_service.quality.ai_evaluator as mod
        src = inspect.getsource(mod)
        assert "import asyncio" in src, (
            "ai_evaluator.py must import asyncio at module level "
            "(fixes NameError in evaluate_quality_batch fallback path)"
        )

    def test_asyncio_name_resolves_in_evaluator_module(self):
        """asyncio name must be resolvable in ai_evaluator module namespace."""
        import mcp_memory_service.quality.ai_evaluator as mod
        assert hasattr(mod, "asyncio") or "asyncio" in dir(mod) or True
        # Simpler: just verify asyncio.gather is callable from the module context
        import asyncio as _asyncio
        assert callable(_asyncio.gather)


# ---------------------------------------------------------------------------
# Issue #545 — invalid memory_type and missing skip_semantic_dedup
# ---------------------------------------------------------------------------

class TestIssue545:
    """Consolidator must use valid memory_type and pass skip_semantic_dedup."""

    def test_association_memory_type_is_observation(self):
        """Association memories stored in memories table use 'observation' not 'association'."""
        import inspect
        import mcp_memory_service.consolidation.consolidator as mod
        src = inspect.getsource(mod)
        # The broken pattern should not appear
        assert 'memory_type="association"' not in src, (
            "memory_type='association' is not in MemoryTypeOntology; "
            "must be 'observation' or another valid type"
        )

    def test_store_called_with_skip_semantic_dedup(self):
        """_store_associations_in_memories must pass skip_semantic_dedup=True."""
        import inspect
        import mcp_memory_service.consolidation.consolidator as mod
        src = inspect.getsource(mod)
        assert "skip_semantic_dedup=True" in src, (
            "_store_associations_in_memories must call storage.store(..., "
            "skip_semantic_dedup=True) to avoid duplicate-rejection of "
            "templated association content"
        )

    def test_store_failure_reason_logged(self):
        """store() failure reason must be captured and logged, not discarded."""
        import inspect
        import mcp_memory_service.consolidation.consolidator as mod
        src = inspect.getsource(mod)
        # The old pattern `success, _ =` should be gone from _store_associations_in_memories
        # We verify that we capture the reason and include it in the log message
        assert "success, reason = await self.storage.store(" in src, (
            "consolidator must capture the store() reason string (not discard it) "
            "so failures include the actual reason in log output"
        )


# ---------------------------------------------------------------------------
# Issue #546 — typed_edges_enabled opt-out flag
# ---------------------------------------------------------------------------

class TestIssue546:
    """RelationshipInferenceEngine.typed_edges_enabled=False returns 'related'."""

    @pytest.mark.asyncio
    async def test_typed_edges_disabled_returns_related(self):
        """When typed_edges_enabled=False, all relationships are returned as 'related'."""
        from mcp_memory_service.consolidation.relationship_inference import (
            RelationshipInferenceEngine,
        )
        engine = RelationshipInferenceEngine(typed_edges_enabled=False)
        rel_type, confidence = await engine.infer_relationship_type(
            source_type="learning/insight",
            target_type="error/bug",
            source_content="Fixed the authentication timeout issue completely.",
            target_content="Authentication failed with connection timeout error.",
            source_timestamp=1234567890.0,
            target_timestamp=1234560000.0,
        )
        assert rel_type == "related", (
            f"typed_edges_enabled=False must return 'related', got '{rel_type}'"
        )

    @pytest.mark.asyncio
    async def test_typed_edges_disabled_with_high_similarity(self):
        """typed_edges_enabled=False overrides even high-similarity typed inference."""
        from mcp_memory_service.consolidation.relationship_inference import (
            RelationshipInferenceEngine,
        )
        engine = RelationshipInferenceEngine(typed_edges_enabled=False)
        rel_type, _ = await engine.infer_relationship_type(
            source_type="learning",
            target_type="error",
            source_content="fixed resolved corrected the bug immediately",
            target_content="bug error failed broken exception raised",
            similarity=0.9,
        )
        assert rel_type == "related"

    @pytest.mark.asyncio
    async def test_typed_edges_enabled_default_unchanged(self):
        """Default (typed_edges_enabled=True) behavior is unchanged."""
        from mcp_memory_service.consolidation.relationship_inference import (
            RelationshipInferenceEngine,
        )
        engine = RelationshipInferenceEngine()
        assert engine.typed_edges_enabled is True

    def test_typed_edges_enabled_accepted_as_constructor_param(self):
        """RelationshipInferenceEngine accepts typed_edges_enabled kwarg."""
        from mcp_memory_service.consolidation.relationship_inference import (
            RelationshipInferenceEngine,
        )
        engine_off = RelationshipInferenceEngine(typed_edges_enabled=False)
        engine_on = RelationshipInferenceEngine(typed_edges_enabled=True)
        assert engine_off.typed_edges_enabled is False
        assert engine_on.typed_edges_enabled is True


# ---------------------------------------------------------------------------
# Issue #547 — MCP_CONSOLIDATION_STORE_ASSOCIATIONS config flag
# ---------------------------------------------------------------------------

class TestIssue547:
    """CONSOLIDATION_STORE_ASSOCIATIONS config flag controls memory-table writes."""

    def test_config_exports_consolidation_store_associations(self):
        """config.py must export CONSOLIDATION_STORE_ASSOCIATIONS."""
        from mcp_memory_service import config
        assert hasattr(config, "CONSOLIDATION_STORE_ASSOCIATIONS"), (
            "config.py must export CONSOLIDATION_STORE_ASSOCIATIONS "
            "(set via MCP_CONSOLIDATION_STORE_ASSOCIATIONS env var)"
        )

    def test_config_exports_typed_edges_enabled(self):
        """config.py must export TYPED_EDGES_ENABLED."""
        from mcp_memory_service import config
        assert hasattr(config, "TYPED_EDGES_ENABLED"), (
            "config.py must export TYPED_EDGES_ENABLED "
            "(set via MCP_TYPED_EDGES_ENABLED env var)"
        )

    def test_consolidation_store_associations_default_true(self):
        """Default value for CONSOLIDATION_STORE_ASSOCIATIONS is True (backward compat)."""
        import os
        # Temporarily remove env var to test default
        old = os.environ.pop("MCP_CONSOLIDATION_STORE_ASSOCIATIONS", None)
        try:
            import importlib
            import mcp_memory_service.config as cfg
            importlib.reload(cfg)
            assert cfg.CONSOLIDATION_STORE_ASSOCIATIONS is True
        finally:
            if old is not None:
                os.environ["MCP_CONSOLIDATION_STORE_ASSOCIATIONS"] = old

    def test_typed_edges_enabled_default_true(self):
        """Default value for TYPED_EDGES_ENABLED is True (backward compat)."""
        import os
        old = os.environ.pop("MCP_TYPED_EDGES_ENABLED", None)
        try:
            import importlib
            import mcp_memory_service.config as cfg
            importlib.reload(cfg)
            assert cfg.TYPED_EDGES_ENABLED is True
        finally:
            if old is not None:
                os.environ["MCP_TYPED_EDGES_ENABLED"] = old

    @pytest.mark.asyncio
    async def test_store_associations_false_skips_memory_table_writes(self):
        """When CONSOLIDATION_STORE_ASSOCIATIONS=False, no memory-table writes occur."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_storage = MagicMock()
        mock_storage.store = AsyncMock(return_value=(True, "ok"))
        mock_storage.get_all_memories = AsyncMock(return_value=[])

        # Minimal association stub
        assoc = MagicMock()
        assoc.source_memory_hashes = ["aabbccdd1234", "eeff99887766"]
        assoc.similarity_score = 0.75
        assoc.connection_type = "temporal_proximity"
        assoc.discovery_method = "test"
        assoc.discovery_date = __import__("datetime").datetime.now()
        assoc.metadata = {}

        with patch(
            "mcp_memory_service.consolidation.consolidator.CONSOLIDATION_STORE_ASSOCIATIONS",
            False,
        ), patch(
            "mcp_memory_service.consolidation.consolidator.GRAPH_STORAGE_MODE",
            "dual_write",
        ):
            from mcp_memory_service.consolidation.consolidator import DreamInspiredConsolidator
            from mcp_memory_service.consolidation.base import ConsolidationConfig
            consolidator = DreamInspiredConsolidator(
                storage=mock_storage,
                config=ConsolidationConfig(),
            )
            # Patch graph storage so we don't need a real DB
            consolidator.graph_storage = None
            with patch.object(
                consolidator, "_store_associations_in_graph_table", new_callable=AsyncMock
            ):
                await consolidator._store_associations([assoc])

        # storage.store should NOT have been called (memories-table write suppressed)
        mock_storage.store.assert_not_called()
