"""
Integration tests for lightweight ONNX quality scoring without transformers.

Tests the complete workflow of:
1. ONNX model loading with tokenizers package only
2. Quality scoring with both classifier and cross-encoder models
3. Auto quality scoring integration in memory service
4. Fallback behavior when dependencies unavailable

Author: Generated for PR #337
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Skip all tests if ONNX Runtime not available
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Check tokenizers availability
try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="Requires ONNX Runtime")
class TestLightweightONNXSetup:
    """Test ONNX quality scoring without transformers dependency."""

    MODEL_NAME = "nvidia-quality-classifier-deberta"
    MODEL_PATH = Path.home() / ".cache" / "mcp_memory" / "onnx_models" / MODEL_NAME

    @pytest.fixture
    def mock_onnx_model(self):
        """Mock ONNX model for testing without actual inference."""
        mock_model = Mock()
        # Mock classifier output (3 classes: high, medium, low)
        mock_model.run.return_value = [np.array([[2.0, 0.5, -1.0]])]  # High quality
        mock_model.get_providers.return_value = ['CPUExecutionProvider']
        return mock_model

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizers package tokenizer."""
        mock_tok = Mock()
        mock_encoding = Mock()
        mock_encoding.ids = [101] + [1234] * 50 + [102]  # CLS + tokens + SEP
        mock_encoding.attention_mask = [1] * 52
        mock_encoding.type_ids = [0] * 52
        mock_tok.encode.return_value = mock_encoding
        mock_tok.enable_truncation = Mock()
        mock_tok.enable_padding = Mock()
        return mock_tok

    def test_onnx_model_exists(self):
        """Verify ONNX model files exist in cache."""
        onnx_path = self.MODEL_PATH / "model.onnx"
        tokenizer_json = self.MODEL_PATH / "tokenizer.json"

        # If model doesn't exist, skip test (not an error, just not downloaded yet)
        if not onnx_path.exists():
            pytest.skip(f"ONNX model not downloaded yet: {onnx_path}")

        assert onnx_path.exists(), "model.onnx should exist"
        assert onnx_path.stat().st_size > 0, "model.onnx should not be empty"

        # tokenizer.json is required for lightweight setup
        if not tokenizer_json.exists():
            pytest.skip(f"tokenizer.json not found, transformers fallback will be used")

        assert tokenizer_json.exists(), "tokenizer.json should exist for lightweight setup"

    @pytest.mark.skipif(not TOKENIZERS_AVAILABLE, reason="Requires tokenizers package")
    def test_tokenizers_package_loading(self):
        """Test loading tokenizer using tokenizers package (not transformers)."""
        tokenizer_json = self.MODEL_PATH / "tokenizer.json"

        if not tokenizer_json.exists():
            pytest.skip(f"tokenizer.json not found at {tokenizer_json}")

        # Load tokenizer using tokenizers package
        tokenizer = Tokenizer.from_file(str(tokenizer_json))

        # Test single text encoding
        text = "This is a high quality memory."
        encoded = tokenizer.encode(text)

        assert hasattr(encoded, 'ids'), "Should have ids attribute"
        assert hasattr(encoded, 'attention_mask'), "Should have attention_mask attribute"
        assert len(encoded.ids) > 0, "Should produce token IDs"
        assert len(encoded.ids) == len(encoded.attention_mask), "IDs and mask should match length"

    @pytest.mark.skipif(not TOKENIZERS_AVAILABLE, reason="Requires tokenizers package")
    def test_tokenizers_pair_encoding(self):
        """Test text pair encoding for cross-encoder models."""
        tokenizer_json = self.MODEL_PATH / "tokenizer.json"

        if not tokenizer_json.exists():
            pytest.skip(f"tokenizer.json not found at {tokenizer_json}")

        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        tokenizer.enable_truncation(max_length=512)
        tokenizer.enable_padding(length=512)

        # Encode query-document pair
        query = "python async patterns"
        document = "Async/await enables concurrent I/O operations."
        encoded = tokenizer.encode((query, document))

        assert hasattr(encoded, 'type_ids'), "Should have type_ids for pairs"
        assert len(encoded.ids) == 512, "Should pad/truncate to 512 tokens"
        assert len(encoded.attention_mask) == 512, "Attention mask should be 512"
        assert len(encoded.type_ids) == 512, "Type IDs should be 512"

        # Verify token type IDs separate query (0) from document (1)
        assert 0 in encoded.type_ids, "Should have query tokens (type 0)"
        assert 1 in encoded.type_ids, "Should have document tokens (type 1)"

    @pytest.mark.xfail(reason="Needs refactoring: tests mock internal implementation details that changed during refactoring. Should test behavior, not implementation.")
    @patch('mcp_memory_service.quality.onnx_ranker.ort.InferenceSession')
    @patch('tokenizers.Tokenizer')
    def test_onnx_ranker_initialization_without_transformers(
        self, mock_tokenizer_class, mock_inference_session, mock_tokenizer, mock_onnx_model
    ):
        """Test ONNXRankerModel initializes correctly without transformers."""
        from mcp_memory_service.quality.onnx_ranker import ONNXRankerModel

        # Setup mocks
        mock_tokenizer_class.from_file.return_value = mock_tokenizer
        mock_inference_session.return_value = mock_onnx_model

        # Create model path mock
        with patch('mcp_memory_service.quality.onnx_ranker.Path') as mock_path:
            mock_model_path = Mock()
            mock_model_path.exists.return_value = True
            mock_onnx_path = Mock()
            mock_onnx_path.exists.return_value = True
            mock_tokenizer_json = Mock()
            mock_tokenizer_json.exists.return_value = True

            mock_path.home.return_value = Mock()
            mock_path.home.return_value.__truediv__.return_value = mock_model_path
            mock_model_path.__truediv__.side_effect = lambda x: {
                'model.onnx': mock_onnx_path,
                'tokenizer.json': mock_tokenizer_json
            }.get(x, Mock())

            # Initialize model
            with patch('mcp_memory_service.quality.onnx_ranker.TRANSFORMERS_AVAILABLE', False):
                ranker = ONNXRankerModel(model_name=self.MODEL_NAME, device="cpu")

            # Verify tokenizers package was used (not transformers)
            mock_tokenizer_class.from_file.assert_called_once()
            assert ranker._use_fast_tokenizer is True, "Should use fast tokenizer"

    @pytest.mark.xfail(reason="Needs refactoring: tests mock internal implementation details that changed during refactoring. Should test behavior, not implementation.")
    @patch('mcp_memory_service.quality.onnx_ranker.ort.InferenceSession')
    @patch('tokenizers.Tokenizer')
    def test_quality_scoring_with_classifier(
        self, mock_tokenizer_class, mock_inference_session, mock_tokenizer, mock_onnx_model
    ):
        """Test quality scoring using classifier model (DeBERTa)."""
        from mcp_memory_service.quality.onnx_ranker import ONNXRankerModel

        # Setup mocks
        mock_tokenizer_class.from_file.return_value = mock_tokenizer
        mock_inference_session.return_value = mock_onnx_model

        with patch('mcp_memory_service.quality.onnx_ranker.Path') as mock_path:
            # Mock path setup (same as above)
            mock_model_path = Mock()
            mock_model_path.exists.return_value = True
            mock_onnx_path = Mock()
            mock_onnx_path.exists.return_value = True
            mock_tokenizer_json = Mock()
            mock_tokenizer_json.exists.return_value = True

            mock_path.home.return_value = Mock()
            mock_path.home.return_value.__truediv__.return_value = mock_model_path
            mock_model_path.__truediv__.side_effect = lambda x: {
                'model.onnx': mock_onnx_path,
                'tokenizer.json': mock_tokenizer_json
            }.get(x, Mock())

            with patch('mcp_memory_service.quality.onnx_ranker.TRANSFORMERS_AVAILABLE', False):
                ranker = ONNXRankerModel(model_name=self.MODEL_NAME, device="cpu")

            # Score quality
            memory_content = "This is a high quality memory about Python async patterns."
            score = ranker.score_quality(query="", memory_content=memory_content)

            # Verify score is valid
            assert 0.0 <= score <= 1.0, f"Score {score} should be between 0 and 1"
            assert score > 0.5, "High quality logits should produce high score"

            # Verify tokenizer was called
            mock_tokenizer.encode.assert_called_once()

    @pytest.mark.xfail(reason="Needs refactoring: tests mock internal implementation details that changed during refactoring. Should test behavior, not implementation.")
    @patch('mcp_memory_service.quality.onnx_ranker.ort.InferenceSession')
    @patch('tokenizers.Tokenizer')
    def test_quality_scoring_with_cross_encoder(
        self, mock_tokenizer_class, mock_inference_session, mock_tokenizer
    ):
        """Test quality scoring using cross-encoder model (MS-MARCO)."""
        from mcp_memory_service.quality.onnx_ranker import ONNXRankerModel

        # Mock cross-encoder output (binary classification)
        mock_model = Mock()
        mock_model.run.return_value = [np.array([[1.5]])]  # Positive logit = high relevance
        mock_model.get_providers.return_value = ['CPUExecutionProvider']

        # Setup mocks
        mock_tokenizer_class.from_file.return_value = mock_tokenizer
        mock_inference_session.return_value = mock_model

        with patch('mcp_memory_service.quality.onnx_ranker.Path') as mock_path:
            # Mock path setup
            mock_model_path = Mock()
            mock_model_path.exists.return_value = True
            mock_onnx_path = Mock()
            mock_onnx_path.exists.return_value = True
            mock_tokenizer_json = Mock()
            mock_tokenizer_json.exists.return_value = True

            mock_path.home.return_value = Mock()
            mock_path.home.return_value.__truediv__.return_value = mock_model_path
            mock_model_path.__truediv__.side_effect = lambda x: {
                'model.onnx': mock_onnx_path,
                'tokenizer.json': mock_tokenizer_json
            }.get(x, Mock())

            # Override model config to cross-encoder
            with patch('mcp_memory_service.quality.onnx_ranker.validate_model_selection') as mock_validate:
                mock_validate.return_value = {
                    'name': 'ms-marco-cross-encoder',
                    'type': 'cross-encoder',
                    'repo': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                    'onnx_file': 'model.onnx'
                }

                with patch('mcp_memory_service.quality.onnx_ranker.TRANSFORMERS_AVAILABLE', False):
                    ranker = ONNXRankerModel(model_name='ms-marco-cross-encoder', device="cpu")

                # Score with query
                query = "python async patterns"
                document = "Async/await enables concurrent I/O operations."
                score = ranker.score_quality(query=query, memory_content=document)

                # Verify score is valid
                assert 0.0 <= score <= 1.0, f"Score {score} should be between 0 and 1"

                # Verify pair encoding was used (tokenizer.encode((query, document)))
                # The mock should have been called with a tuple
                call_args = mock_tokenizer.encode.call_args
                assert call_args is not None, "Tokenizer should have been called"

    @pytest.mark.parametrize("logits_output,expected_logit", [
        (np.array([[1.5]]), 1.5),       # Shape (1, 1) — the bug from issue #764
        (np.array([1.5]), 1.5),         # Shape (1,) — historical path
        (np.array([[-0.7]]), -0.7),     # Shape (1, 1), negative logit
        (np.array([[[2.0]]]), 2.0),     # Shape (1, 1, 1), defensive
    ])
    def test_cross_encoder_scalar_extraction_shape_agnostic(self, logits_output, expected_logit):
        """Regression test for issue #764: cross-encoder must handle (1, 1) logits.

        Previously, ``float(logits)`` on a non-zero-dim ndarray raised TypeError, which
        was swallowed by the outer handler and pinned every score to 0.5. This test
        exercises the scalar extraction path directly with multiple shapes.
        """
        from mcp_memory_service.quality.onnx_ranker import ONNXRankerModel

        # Bypass heavy __init__; we only need to drive score_quality()
        ranker = ONNXRankerModel.__new__(ONNXRankerModel)
        ranker.model_name = 'ms-marco-cross-encoder'
        ranker.model_config = {
            'name': 'ms-marco-cross-encoder',
            'type': 'cross-encoder',
            'repo': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'onnx_file': 'model.onnx',
        }
        ranker._use_fast_tokenizer = True

        # Tokenizer mock for cross-encoder pair encoding
        mock_encoded = Mock()
        mock_encoded.ids = [0] * 8
        mock_encoded.attention_mask = [1] * 8
        mock_encoded.type_ids = [0] * 8
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = mock_encoded
        mock_tokenizer.enable_truncation = Mock()
        mock_tokenizer.enable_padding = Mock()
        ranker._tokenizer = mock_tokenizer

        # ORT session returns the parametrized shape
        mock_model = Mock()
        mock_model.run.return_value = [logits_output]
        ranker._model = mock_model

        score = ranker.score_quality(query="q", memory_content="doc")

        # Score must be the sigmoid of the embedded logit, NOT the 0.5 placeholder
        expected_score = 1.0 / (1.0 + np.exp(-expected_logit))
        assert 0.0 <= score <= 1.0
        assert abs(score - expected_score) < 1e-6, (
            f"Expected sigmoid({expected_logit})={expected_score:.4f}, got {score:.4f}. "
            "If this is 0.5, the (1,1) shape regression has returned."
        )

    @pytest.mark.xfail(reason="Needs refactoring: mock storage doesn't properly simulate real storage behavior. Rewrite to use actual test storage.")
    @pytest.mark.asyncio
    async def test_auto_quality_scoring_after_store(self):
        """Test automatic quality scoring is triggered after memory store."""
        from mcp_memory_service.services.memory_service import MemoryService
        from mcp_memory_service.models.memory import Memory

        # Mock storage
        mock_storage = AsyncMock()
        mock_storage.store.return_value = (True, "Success")

        # Mock async scorer
        with patch('mcp_memory_service.services.memory_service.async_scorer') as mock_scorer:
            mock_scorer.score_memory = AsyncMock()

            # Enable quality boost
            with patch('mcp_memory_service.services.memory_service.MCP_QUALITY_BOOST_ENABLED', True):
                service = MemoryService(storage=mock_storage)

                # Store memory
                result = await service.store_memory(
                    content="Test memory content",
                    tags=["test"],
                    memory_type="note"
                )

                # Verify store succeeded
                assert result["success"] is True

                # Verify async scorer was called
                mock_scorer.score_memory.assert_called_once()
                call_args = mock_scorer.score_memory.call_args
                assert call_args[1]['storage'] == mock_storage, "Should pass storage to scorer"

    @pytest.mark.xfail(reason="Needs refactoring: mock storage doesn't properly simulate real storage behavior. Rewrite to use actual test storage.")
    @pytest.mark.asyncio
    async def test_auto_quality_scoring_after_retrieve(self):
        """Test automatic quality scoring is triggered after memory retrieval."""
        from mcp_memory_service.services.memory_service import MemoryService
        from mcp_memory_service.models.memory import Memory
        from mcp_memory_service.storage.base import SearchResult

        # Mock storage with retrieve results
        mock_storage = AsyncMock()
        mock_memory = Memory(
            content="Test memory",
            content_hash="abc123",
            tags=["test"],
            memory_type="note"
        )
        mock_storage.retrieve.return_value = [
            SearchResult(memory=mock_memory, relevance_score=0.9)
        ]

        # Mock async scorer
        with patch('mcp_memory_service.services.memory_service.async_scorer') as mock_scorer:
            mock_scorer.score_memory = AsyncMock()

            # Enable quality boost
            with patch('mcp_memory_service.services.memory_service.MCP_QUALITY_BOOST_ENABLED', True):
                service = MemoryService(storage=mock_storage)

                # Retrieve memories
                result = await service.retrieve_memories(
                    query="test query",
                    n_results=5
                )

                # Verify retrieve succeeded
                assert len(result["memories"]) == 1

                # Verify async scorer was called
                mock_scorer.score_memory.assert_called_once()
                call_args = mock_scorer.score_memory.call_args
                assert call_args[0][0] == mock_memory, "Should score retrieved memory"
                assert call_args[1]['query'] == "test query", "Should pass query to scorer"

    @pytest.mark.xfail(reason="Needs refactoring: mock storage doesn't properly simulate real storage behavior. Rewrite to use actual test storage.")
    @pytest.mark.asyncio
    async def test_quality_scoring_silent_failure(self):
        """Test quality scoring failures don't break memory operations."""
        from mcp_memory_service.services.memory_service import MemoryService

        # Mock storage
        mock_storage = AsyncMock()
        mock_storage.store.return_value = (True, "Success")

        # Mock async scorer to raise exception
        with patch('mcp_memory_service.services.memory_service.async_scorer') as mock_scorer:
            mock_scorer.score_memory = AsyncMock(side_effect=Exception("Scorer failed"))

            # Enable quality boost
            with patch('mcp_memory_service.services.memory_service.MCP_QUALITY_BOOST_ENABLED', True):
                service = MemoryService(storage=mock_storage)

                # Store memory should still succeed even if scoring fails
                result = await service.store_memory(
                    content="Test memory content",
                    tags=["test"],
                    memory_type="note"
                )

                # Verify store succeeded despite scorer failure
                assert result["success"] is True, "Store should succeed even if quality scoring fails"

    @pytest.mark.xfail(reason="Needs refactoring: tests mock internal implementation details that changed during refactoring. Should test behavior, not implementation.")
    def test_fallback_to_transformers(self):
        """Test graceful fallback to transformers when tokenizers unavailable."""
        from mcp_memory_service.quality.onnx_ranker import ONNXRankerModel

        # Mock tokenizers import failure
        with patch('mcp_memory_service.quality.onnx_ranker.TOKENIZERS_AVAILABLE', False):
            with patch('mcp_memory_service.quality.onnx_ranker.TRANSFORMERS_AVAILABLE', True):
                with patch('mcp_memory_service.quality.onnx_ranker.AutoTokenizer') as mock_auto_tokenizer:
                    with patch('mcp_memory_service.quality.onnx_ranker.ort.InferenceSession'):
                        with patch('mcp_memory_service.quality.onnx_ranker.Path') as mock_path:
                            # Mock path setup
                            mock_model_path = Mock()
                            mock_model_path.exists.return_value = True
                            mock_onnx_path = Mock()
                            mock_onnx_path.exists.return_value = True
                            mock_tokenizer_json = Mock()
                            mock_tokenizer_json.exists.return_value = False  # No tokenizer.json

                            mock_path.home.return_value = Mock()
                            mock_path.home.return_value.__truediv__.return_value = mock_model_path
                            mock_model_path.__truediv__.side_effect = lambda x: {
                                'model.onnx': mock_onnx_path,
                                'tokenizer.json': mock_tokenizer_json
                            }.get(x, Mock())

                            ranker = ONNXRankerModel(model_name=self.MODEL_NAME, device="cpu")

                            # Verify transformers was used instead of tokenizers
                            mock_auto_tokenizer.from_pretrained.assert_called_once()
                            assert ranker._use_fast_tokenizer is False, "Should use slow tokenizer"

    @pytest.mark.xfail(reason="Needs refactoring: tests mock internal implementation details that changed during refactoring. Should test behavior, not implementation.")
    def test_error_on_missing_dependencies(self):
        """Test error when neither tokenizers nor transformers available."""
        from mcp_memory_service.quality.onnx_ranker import ONNXRankerModel

        # Mock both packages unavailable
        with patch('mcp_memory_service.quality.onnx_ranker.TOKENIZERS_AVAILABLE', False):
            with patch('mcp_memory_service.quality.onnx_ranker.TRANSFORMERS_AVAILABLE', False):
                with patch('mcp_memory_service.quality.onnx_ranker.Path') as mock_path:
                    # Mock ONNX model already exists
                    mock_model_path = Mock()
                    mock_model_path.exists.return_value = True
                    mock_onnx_path = Mock()
                    mock_onnx_path.exists.return_value = True
                    mock_tokenizer_json = Mock()
                    mock_tokenizer_json.exists.return_value = False

                    mock_path.home.return_value = Mock()
                    mock_path.home.return_value.__truediv__.return_value = mock_model_path
                    mock_model_path.__truediv__.side_effect = lambda x: {
                        'model.onnx': mock_onnx_path,
                        'tokenizer.json': mock_tokenizer_json
                    }.get(x, Mock())

                    # Should raise ImportError
                    with pytest.raises(ImportError, match="Neither tokenizers nor transformers available"):
                        with patch('mcp_memory_service.quality.onnx_ranker.ort.InferenceSession'):
                            ranker = ONNXRankerModel(model_name=self.MODEL_NAME, device="cpu")


@pytest.mark.integration
class TestLightweightONNXEndToEnd:
    """End-to-end integration tests requiring actual model files."""

    @pytest.mark.xfail(reason="Integration test requires ONNX models to be downloaded. Models may not be available in all test environments.")
    @pytest.mark.skipif(
        not ONNX_AVAILABLE or not TOKENIZERS_AVAILABLE,
        reason="Requires ONNX Runtime and tokenizers package"
    )
    def test_real_onnx_inference(self):
        """Test actual ONNX inference with real model (if available)."""
        from mcp_memory_service.quality.onnx_ranker import get_onnx_ranker_model

        # Try to load real model
        ranker = get_onnx_ranker_model(device="cpu")

        if ranker is None:
            pytest.skip("ONNX model not available (not downloaded yet)")

        # Run real inference
        memory_content = (
            "This is a detailed explanation of Python's asyncio library. "
            "It covers event loops, coroutines, tasks, and futures. "
            "Best practices for async/await patterns are included."
        )

        score = ranker.score_quality(query="", memory_content=memory_content)

        # Verify valid score
        assert 0.0 <= score <= 1.0, f"Score {score} should be between 0 and 1"
        assert score > 0.3, "Detailed content should score reasonably well"

    @pytest.mark.skipif(
        not ONNX_AVAILABLE or not TOKENIZERS_AVAILABLE,
        reason="Requires ONNX Runtime and tokenizers package"
    )
    def test_disk_usage_reduction(self):
        """Verify lightweight setup doesn't require transformers installation."""
        import sys

        # Check if transformers is installed
        transformers_installed = 'transformers' in sys.modules or \
                                  any('transformers' in str(p) for p in sys.path)

        # If transformers is installed, this test can't verify lightweight setup
        if transformers_installed:
            pytest.skip("Transformers already installed, can't verify lightweight setup")

        # Try to use ONNX ranker without transformers
        from mcp_memory_service.quality.onnx_ranker import get_onnx_ranker_model

        ranker = get_onnx_ranker_model(device="cpu")

        if ranker is None:
            pytest.skip("ONNX model not available")

        # Should work without transformers
        assert ranker._use_fast_tokenizer is True, "Should use tokenizers package"
        assert ranker._tokenizer is not None, "Should have loaded tokenizer"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
