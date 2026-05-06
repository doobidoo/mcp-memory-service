# Copyright 2026 Claudio Ferreira Filho
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Tests for plugin hook system."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_memory_service.plugins.context import PluginContext
from mcp_memory_service.plugins.registry import PluginRegistry


@pytest.fixture
def ctx():
    """Create a PluginContext with mock storage/service."""
    return PluginContext(storage=MagicMock(), service=MagicMock())


@pytest.fixture
def registry(ctx):
    return PluginRegistry(ctx)


class TestPluginContext:
    def test_subscribe_hook(self, ctx):
        fn = AsyncMock()
        ctx.on("on_store", fn)
        assert "on_store" in ctx._hooks
        assert fn in ctx._hooks["on_store"]

    def test_multiple_hooks(self, ctx):
        fn1 = AsyncMock()
        fn2 = AsyncMock()
        ctx.on("on_store", fn1)
        ctx.on("on_store", fn2)
        assert len(ctx._hooks["on_store"]) == 2

    def test_different_hook_types(self, ctx):
        fn1 = AsyncMock()
        fn2 = AsyncMock()
        ctx.on("on_store", fn1)
        ctx.on("on_delete", fn2)
        assert len(ctx._hooks["on_store"]) == 1
        assert len(ctx._hooks["on_delete"]) == 1


class TestPluginRegistry:
    @pytest.mark.asyncio
    async def test_fire_no_handlers(self, registry):
        result = await registry.fire("on_store", {"content": "test"})
        assert result == {"content": "test"}

    @pytest.mark.asyncio
    async def test_fire_on_store(self, registry):
        handler = AsyncMock()
        registry.ctx.on("on_store", handler)
        await registry.fire("on_store", {"content": "test"})
        handler.assert_called_once_with({"content": "test"})

    @pytest.mark.asyncio
    async def test_fire_on_retrieve_reranks(self, registry):
        async def reranker(query, results):
            return list(reversed(results))

        registry.ctx.on("on_retrieve", reranker)
        result = await registry.fire("on_retrieve", "test query", [1, 2, 3])
        assert result == [3, 2, 1]

    @pytest.mark.asyncio
    async def test_fire_on_retrieve_chaining(self, registry):
        """Subsequent handlers receive modified results from previous."""

        async def add_item(query, results):
            return results + [99]

        async def reverse_it(query, results):
            return list(reversed(results))

        registry.ctx.on("on_retrieve", add_item)
        registry.ctx.on("on_retrieve", reverse_it)
        result = await registry.fire("on_retrieve", "q", [1, 2, 3])
        assert result == [99, 3, 2, 1]

    @pytest.mark.asyncio
    async def test_fire_handler_error_doesnt_crash(self, registry):
        async def bad_handler(*args):
            raise ValueError("boom")

        good_handler = AsyncMock()
        registry.ctx.on("on_store", bad_handler)
        registry.ctx.on("on_store", good_handler)

        await registry.fire("on_store", {"content": "test"})
        good_handler.assert_called_once()

    def test_discover_no_plugins(self, registry):
        with patch("mcp_memory_service.plugins.registry.entry_points") as mock_eps:
            mock_eps.return_value.select.return_value = []
            registry.discover_and_register()
        assert registry.loaded_plugins == []

    def test_discover_with_plugin(self, registry):
        mock_register = MagicMock()
        mock_ep = MagicMock()
        mock_ep.name = "test_plugin"
        mock_ep.load.return_value = mock_register

        with patch("mcp_memory_service.plugins.registry.entry_points") as mock_eps:
            mock_eps.return_value.select.return_value = [mock_ep]
            registry.discover_and_register()

        mock_register.assert_called_once_with(registry.ctx)
        assert registry.loaded_plugins == ["test_plugin"]

    def test_discover_plugin_error_doesnt_crash(self, registry):
        mock_ep = MagicMock()
        mock_ep.name = "bad_plugin"
        mock_ep.load.side_effect = ImportError("no module")

        with patch("mcp_memory_service.plugins.registry.entry_points") as mock_eps:
            mock_eps.return_value.select.return_value = [mock_ep]
            registry.discover_and_register()

        assert registry.loaded_plugins == []
