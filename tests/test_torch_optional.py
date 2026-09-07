"""
RED tests for §11: torch optional, ONNX-first fallback chain.

These tests validate that:
- pyproject.toml declares torch/sentence-transformers as OPTIONAL (not required)
- The service starts without torch/sentence-transformers installed
- Fallback chain: External API → ONNX → SentenceTransformer → Hash
- Appropriate warnings are emitted when falling back
"""

import importlib
import logging
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Load pyproject.toml
REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib


def _load_pyproject():
    with open(PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


class TestPyprojectStructure:
    """Validate that pyproject.toml declares torch/sentence-transformers as optional."""

    def test_torch_not_in_required_dependencies(self):
        """torch must NOT appear in [project.dependencies]."""
        data = _load_pyproject()
        deps = data["project"]["dependencies"]
        torch_deps = [d for d in deps if d.startswith("torch")]
        assert torch_deps == [], f"torch found in required deps: {torch_deps}"

    def test_sentence_transformers_not_in_required_dependencies(self):
        """sentence-transformers must NOT appear in [project.dependencies]."""
        data = _load_pyproject()
        deps = data["project"]["dependencies"]
        st_deps = [d for d in deps if "sentence-transformers" in d]
        assert st_deps == [], f"sentence-transformers found in required deps: {st_deps}"

    def test_torch_in_optional_ml_group(self):
        """torch must be in [project.optional-dependencies.ml]."""
        data = _load_pyproject()
        ml_deps = data["project"]["optional-dependencies"]["ml"]
        torch_found = any("torch" in d for d in ml_deps)
        assert torch_found, f"torch not found in [ml] optional deps: {ml_deps}"

    def test_sentence_transformers_in_optional_ml_group(self):
        """sentence-transformers must be in [project.optional-dependencies.ml]."""
        data = _load_pyproject()
        ml_deps = data["project"]["optional-dependencies"]["ml"]
        st_found = any("sentence-transformers" in d for d in ml_deps)
        assert st_found, f"sentence-transformers not in [ml] optional deps: {ml_deps}"

    def test_full_includes_ml(self):
        """[project.optional-dependencies.full] must include the ml extra."""
        data = _load_pyproject()
        full_deps = data["project"]["optional-dependencies"]["full"]
        ml_ref = any("ml" in d for d in full_deps)
        assert ml_ref, f"'ml' not referenced in [full] optional deps: {full_deps}"


class TestFallbackChainNoTorch:
    """When torch AND onnxruntime are both absent, service falls back to hash embeddings."""

    @pytest.fixture
    def _hide_torch_and_onnx(self, monkeypatch):
        """Simulate environment where torch, sentence_transformers, onnxruntime are missing."""
        import builtins

        real_import = builtins.__import__

        blocked = {"torch", "sentence_transformers", "onnxruntime"}

        def fake_import(name, *args, **kwargs):
            if name in blocked or any(name.startswith(b + ".") for b in blocked):
                raise ImportError(f"Mocked: No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # Also disable ONNX via env var and patch SentenceTransformer to None
        # to handle the case where modules are already loaded
        monkeypatch.setenv('MCP_MEMORY_USE_ONNX', '0')
        monkeypatch.setattr(
            "mcp_memory_service.storage.mixins.embeddings.SentenceTransformer", None
        )

    def test_service_starts_without_importerror(self, _hide_torch_and_onnx):
        """Service module should be importable without torch/onnx installed."""
        # Save modules before manipulation
        saved_modules = {k: v for k, v in sys.modules.items() if k.startswith('mcp_memory_service')}
        try:
            # Remove cached modules to force re-import
            for key in list(sys.modules.keys()):
                if key.startswith('mcp_memory_service'):
                    del sys.modules[key]

            # This should NOT raise ImportError
            from mcp_memory_service.storage.mixins.embeddings import EmbeddingsMixin  # noqa: F401
        finally:
            # Restore: remove any newly-loaded modules, then restore originals
            for key in list(sys.modules.keys()):
                if key.startswith('mcp_memory_service'):
                    del sys.modules[key]
            sys.modules.update(saved_modules)

    def test_fallback_to_hash_embeddings_when_nothing_available(self, _hide_torch_and_onnx):
        """With no backends available, must fall back to hash embeddings."""
        from mcp_memory_service.storage.mixins.embeddings import EmbeddingsMixin, _HashEmbeddingModel

        mixin = EmbeddingsMixin()
        mixin.embedding_model_name = "all-MiniLM-L6-v2"
        mixin.embedding_dimension = 384
        mixin.embedding_model = None
        mixin.enable_cache = False
        mixin.conn = None
        mixin._run_in_thread = AsyncMock(return_value=None)

        import asyncio
        asyncio.run(mixin._initialize_embedding_model())

        assert isinstance(mixin.embedding_model, _HashEmbeddingModel)

    def test_hash_fallback_logs_warning(self, _hide_torch_and_onnx, caplog):
        """Falling back to hash embeddings must log a WARNING about reduced quality."""
        from mcp_memory_service.storage.mixins.embeddings import EmbeddingsMixin

        mixin = EmbeddingsMixin()
        mixin.embedding_model_name = "all-MiniLM-L6-v2"
        mixin.embedding_dimension = 384
        mixin.embedding_model = None
        mixin.enable_cache = False
        mixin.conn = None
        mixin._run_in_thread = AsyncMock(return_value=None)

        import asyncio
        with caplog.at_level(logging.WARNING):
            asyncio.run(mixin._initialize_embedding_model())

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("quality" in msg.lower() or "reduced" in msg.lower() for msg in warning_messages), (
            f"Expected warning about reduced quality, got: {warning_messages}"
        )


class TestFallbackChainOnnxOnly:
    """When torch is absent but onnxruntime is available, service uses ONNX."""

    @pytest.fixture
    def _hide_torch_only(self, monkeypatch):
        """Simulate environment where torch/sentence_transformers are missing but onnx is available."""
        import builtins

        real_import = builtins.__import__

        blocked = {"torch", "sentence_transformers"}

        def fake_import(name, *args, **kwargs):
            if name in blocked or any(name.startswith(b + ".") for b in blocked):
                raise ImportError(f"Mocked: No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    def test_onnx_used_when_torch_absent(self, _hide_torch_only, monkeypatch):
        """When torch is missing but ONNX is available, ONNX backend is selected."""
        from mcp_memory_service.storage.mixins.embeddings import EmbeddingsMixin

        mock_onnx_model = MagicMock()
        mock_onnx_model.embedding_dimension = 384
        mock_onnx_model.encode = MagicMock(return_value=[[0.1] * 384])

        monkeypatch.setattr(
            "mcp_memory_service.storage.mixins.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE",
            False,
        )

        # The spec says ONNX should be tried automatically (not only when MCP_MEMORY_USE_ONNX=1)
        # This tests the NEW behavior: auto-fallback to ONNX when torch absent
        mixin = EmbeddingsMixin()
        mixin.embedding_model_name = "all-MiniLM-L6-v2"
        mixin.embedding_dimension = 384
        mixin.embedding_model = None
        mixin.enable_cache = False
        mixin.conn = None
        mixin._run_in_thread = AsyncMock(return_value=None)

        with patch(
            "mcp_memory_service.embeddings.get_onnx_embedding_model",
            return_value=mock_onnx_model,
        ):
            import asyncio
            asyncio.run(mixin._initialize_embedding_model())

        # Should have chosen ONNX, not hash fallback
        assert mixin.embedding_model is mock_onnx_model

    def test_onnx_fallback_logs_info(self, _hide_torch_only, monkeypatch, caplog):
        """ONNX fallback should log an informational message."""
        from mcp_memory_service.storage.mixins.embeddings import EmbeddingsMixin

        mock_onnx_model = MagicMock()
        mock_onnx_model.embedding_dimension = 384

        monkeypatch.setattr(
            "mcp_memory_service.storage.mixins.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE",
            False,
        )

        mixin = EmbeddingsMixin()
        mixin.embedding_model_name = "all-MiniLM-L6-v2"
        mixin.embedding_dimension = 384
        mixin.embedding_model = None
        mixin.enable_cache = False
        mixin.conn = None
        mixin._run_in_thread = AsyncMock(return_value=None)

        with caplog.at_level(logging.INFO):
            with patch(
                "mcp_memory_service.embeddings.get_onnx_embedding_model",
                return_value=mock_onnx_model,
            ):
                import asyncio
                asyncio.run(mixin._initialize_embedding_model())

        log_text = " ".join(r.message for r in caplog.records)
        assert "onnx" in log_text.lower(), f"Expected ONNX log message, got: {log_text}"


class TestFallbackChainPriority:
    """Validate the full fallback chain priority order:
    External API → ONNX → SentenceTransformer → Hash
    """

    def test_external_api_takes_priority_over_onnx(self, monkeypatch):
        """When MCP_EXTERNAL_EMBEDDING_URL is set, external API is used over ONNX."""
        from mcp_memory_service.storage.mixins.embeddings import EmbeddingsMixin

        monkeypatch.setenv("MCP_EXTERNAL_EMBEDDING_URL", "http://localhost:11434")

        mock_ext_model = MagicMock()
        mock_ext_model.embedding_dimension = 768

        mixin = EmbeddingsMixin()
        mixin.embedding_model_name = "all-MiniLM-L6-v2"
        mixin.embedding_dimension = 384
        mixin.embedding_model = None
        mixin.enable_cache = False
        mixin.conn = None
        mixin._run_in_thread = AsyncMock(return_value=None)

        with patch(
            "mcp_memory_service.storage.mixins.embeddings._MODEL_CACHE", {}
        ), patch(
            "mcp_memory_service.storage.mixins.embeddings._DIMENSION_CACHE", {}
        ), patch(
            "mcp_memory_service.embeddings.external_api.get_external_embedding_model",
            return_value=mock_ext_model,
        ):
            import asyncio
            asyncio.run(mixin._initialize_embedding_model())

        assert mixin.embedding_model is mock_ext_model

    def test_onnx_takes_priority_over_sentence_transformer(self, monkeypatch):
        """When both ONNX and SentenceTransformer are available, ONNX is preferred."""
        from mcp_memory_service.storage.mixins.embeddings import EmbeddingsMixin

        mock_onnx_model = MagicMock()
        mock_onnx_model.embedding_dimension = 384

        # §11 spec: ONNX is tried BEFORE SentenceTransformer (without needing env var)
        mixin = EmbeddingsMixin()
        mixin.embedding_model_name = "all-MiniLM-L6-v2"
        mixin.embedding_dimension = 384
        mixin.embedding_model = None
        mixin.enable_cache = False
        mixin.conn = None
        mixin._run_in_thread = AsyncMock(return_value=None)

        monkeypatch.delenv("MCP_EXTERNAL_EMBEDDING_URL", raising=False)
        monkeypatch.delenv("MCP_MEMORY_USE_ONNX", raising=False)

        with patch(
            "mcp_memory_service.storage.mixins.embeddings._MODEL_CACHE", {}
        ), patch(
            "mcp_memory_service.storage.mixins.embeddings._DIMENSION_CACHE", {}
        ), patch(
            "mcp_memory_service.embeddings.get_onnx_embedding_model",
            return_value=mock_onnx_model,
        ):
            import asyncio
            asyncio.run(mixin._initialize_embedding_model())

        # ONNX should be preferred over SentenceTransformer
        assert mixin.embedding_model is mock_onnx_model

    def test_sentence_transformer_used_when_onnx_unavailable(self, monkeypatch):
        """When ONNX fails/absent but torch is available, SentenceTransformer is used."""
        from mcp_memory_service.storage.mixins.embeddings import EmbeddingsMixin

        mock_st_model = MagicMock()
        mock_st_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_model.encode.return_value = MagicMock(shape=(1, 384))

        import numpy as np
        mock_st_model.encode.return_value = np.zeros((1, 384))

        mixin = EmbeddingsMixin()
        mixin.embedding_model_name = "all-MiniLM-L6-v2"
        mixin.embedding_dimension = 384
        mixin.embedding_model = None
        mixin.enable_cache = False
        mixin.conn = None
        mixin._run_in_thread = AsyncMock(return_value=None)
        mixin._is_docker_environment = MagicMock(return_value=False)

        monkeypatch.delenv("MCP_EXTERNAL_EMBEDDING_URL", raising=False)
        monkeypatch.delenv("MCP_MEMORY_USE_ONNX", raising=False)

        with patch(
            "mcp_memory_service.storage.mixins.embeddings._MODEL_CACHE", {}
        ), patch(
            "mcp_memory_service.storage.mixins.embeddings._DIMENSION_CACHE", {}
        ), patch(
            "mcp_memory_service.embeddings.get_onnx_embedding_model",
            return_value=None,  # ONNX not available
        ), patch(
            "mcp_memory_service.storage.mixins.embeddings.SentenceTransformer",
            return_value=mock_st_model,
        ):
            import asyncio
            asyncio.run(mixin._initialize_embedding_model())

        assert mixin.embedding_model is mock_st_model


class TestRuntimeWarning:
    """Verify warning messages with install instructions when using fallback."""

    def test_hash_fallback_warning_includes_install_instructions(self, monkeypatch, caplog):
        """When falling back to hash embeddings, warning must include pip install instructions."""
        from mcp_memory_service.storage.mixins.embeddings import EmbeddingsMixin

        monkeypatch.setattr(
            "mcp_memory_service.storage.mixins.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE",
            False,
        )
        monkeypatch.setattr(
            "mcp_memory_service.storage.mixins.embeddings.SentenceTransformer", None
        )
        monkeypatch.delenv("MCP_EXTERNAL_EMBEDDING_URL", raising=False)
        monkeypatch.setenv("MCP_MEMORY_USE_ONNX", "0")

        mixin = EmbeddingsMixin()
        mixin.embedding_model_name = "all-MiniLM-L6-v2"
        mixin.embedding_dimension = 384
        mixin.embedding_model = None
        mixin.enable_cache = False
        mixin.conn = None
        mixin._run_in_thread = AsyncMock(return_value=None)

        with caplog.at_level(logging.WARNING):
            import asyncio
            asyncio.run(mixin._initialize_embedding_model())

        warning_messages = " ".join(
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        )
        # Must include install instructions per spec
        assert "pip install" in warning_messages or "install" in warning_messages.lower(), (
            f"Warning must include install instructions, got: {warning_messages}"
        )

    def test_warning_emitted_only_on_first_init(self, monkeypatch, caplog):
        """Warning about fallback should be emitted only once, not on every embedding call."""
        from mcp_memory_service.storage.mixins.embeddings import EmbeddingsMixin

        monkeypatch.setattr(
            "mcp_memory_service.storage.mixins.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE",
            False,
        )
        monkeypatch.setattr(
            "mcp_memory_service.storage.mixins.embeddings.SentenceTransformer", None
        )
        monkeypatch.delenv("MCP_EXTERNAL_EMBEDDING_URL", raising=False)
        monkeypatch.setenv("MCP_MEMORY_USE_ONNX", "0")

        mixin = EmbeddingsMixin()
        mixin.embedding_model_name = "all-MiniLM-L6-v2"
        mixin.embedding_dimension = 384
        mixin.embedding_model = None
        mixin.enable_cache = False
        mixin.conn = None
        mixin._run_in_thread = AsyncMock(return_value=None)

        import asyncio

        with caplog.at_level(logging.WARNING):
            asyncio.run(mixin._initialize_embedding_model())

        first_warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]

        caplog.clear()

        # Second init should NOT emit the same warning
        with caplog.at_level(logging.WARNING):
            asyncio.run(mixin._initialize_embedding_model())

        second_warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]

        # Spec: warning on FIRST init only
        assert len(first_warnings) > 0, "Should warn on first init"
        quality_warnings_second = [
            w for w in second_warnings
            if "quality" in w.message.lower() or "hash" in w.message.lower()
        ]
        assert quality_warnings_second == [], (
            f"Should not repeat fallback warning on second init, got: {quality_warnings_second}"
        )

    def test_warning_emitted_once_per_process_not_once_per_instance(self, monkeypatch, caplog):
        """A second, freshly built mixin must not repeat the fallback warning.

        The module comment promises the warning fires once per process, but the
        guard only consulted the per-instance flag, so the module-level
        `_HASH_FALLBACK_WARNED` was written and never read. Production never
        noticed, because `_STORAGE_CACHE` yields one instance per process --
        this pins the documented behaviour for the case where it does not.
        """
        from mcp_memory_service.storage.mixins.embeddings import EmbeddingsMixin

        monkeypatch.setattr(
            "mcp_memory_service.storage.mixins.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE",
            False,
        )
        monkeypatch.setattr(
            "mcp_memory_service.storage.mixins.embeddings.SentenceTransformer", None
        )
        monkeypatch.delenv("MCP_EXTERNAL_EMBEDDING_URL", raising=False)
        monkeypatch.setenv("MCP_MEMORY_USE_ONNX", "0")

        def build_mixin():
            mixin = EmbeddingsMixin()
            mixin.embedding_model_name = "all-MiniLM-L6-v2"
            mixin.embedding_dimension = 384
            mixin.embedding_model = None
            mixin.enable_cache = False
            mixin.conn = None
            mixin._run_in_thread = AsyncMock(return_value=None)
            return mixin

        import asyncio

        def fallback_warnings():
            return [
                r for r in caplog.records
                if r.levelno >= logging.WARNING
                and ("hash" in r.message.lower() or "quality" in r.message.lower())
            ]

        with caplog.at_level(logging.WARNING):
            asyncio.run(build_mixin()._initialize_embedding_model())
        assert fallback_warnings(), "Should warn on the first instance"

        caplog.clear()

        # A different instance, with no per-instance flag set on it.
        with caplog.at_level(logging.WARNING):
            asyncio.run(build_mixin()._initialize_embedding_model())

        assert fallback_warnings() == [], (
            "Fallback warning must not repeat for a second instance in the same "
            f"process, got: {fallback_warnings()}"
        )
