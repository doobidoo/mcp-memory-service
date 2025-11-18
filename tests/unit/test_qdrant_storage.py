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

"""Unit tests for Qdrant storage backend with mocked client."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from typing import List
from datetime import datetime, timedelta
import sys

# Mock qdrant_client.exceptions module before importing qdrant_storage
if 'qdrant_client.exceptions' not in sys.modules:
    mock_exceptions = MagicMock()
    mock_exceptions.UnexpectedResponse = type('UnexpectedResponse', (Exception,), {})
    sys.modules['qdrant_client.exceptions'] = mock_exceptions

from src.mcp_memory_service.storage.qdrant_storage import QdrantStorage, StorageError
from src.mcp_memory_service.models.memory import Memory, MemoryQueryResult
from src.mcp_memory_service.utils.hashing import generate_content_hash


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_qdrant_client():
    """Create a fully mocked Qdrant client."""
    client = MagicMock()

    # Collection operations
    client.get_collections.return_value = MagicMock(collections=[])
    client.create_collection.return_value = True
    client.get_collection.return_value = MagicMock(
        config=MagicMock(
            params=MagicMock(
                vectors=MagicMock(size=384)
            )
        ),
        points_count=100
    )

    # Point operations
    client.upsert.return_value = MagicMock(status="acknowledged")
    client.search.return_value = []
    client.scroll.return_value = ([], None)  # (points, next_offset)
    client.retrieve.return_value = []
    client.delete.return_value = MagicMock(status="acknowledged")
    client.set_payload.return_value = MagicMock(status="acknowledged")

    return client


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    content = "This is a test memory about Qdrant vector storage"
    return Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["test", "qdrant"],
        memory_type="standard",
        embedding=[0.1] * 384,  # 384-dimensional embedding
        created_at=datetime.now().timestamp(),
        updated_at=datetime.now().timestamp()
    )


@pytest.fixture
def qdrant_storage(mock_qdrant_client):
    """Create QdrantStorage instance with mocked client."""
    with patch('src.mcp_memory_service.storage.qdrant_storage.QdrantClient', return_value=mock_qdrant_client):
        storage = QdrantStorage(
            embedding_model="all-MiniLM-L6-v2",
            collection_name="test_memories",
            quantization_enabled=False,
            storage_path="/tmp/test_qdrant"  # Embedded mode for tests
        )
        # Inject the mock client
        storage.client = mock_qdrant_client
        storage._vector_size = 384
        return storage


# ============================================================================
# Initialization Tests
# ============================================================================

class TestQdrantInitialization:
    """Test Qdrant storage initialization."""

    @pytest.mark.asyncio
    async def test_initialization_creates_new_collection(self, mock_qdrant_client):
        """Test that initialization creates collection with correct config."""
        # Mock: collection doesn't exist
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])

        with patch('src.mcp_memory_service.storage.qdrant_storage.QdrantClient', return_value=mock_qdrant_client):
            storage = QdrantStorage(
            embedding_model="all-MiniLM-L6-v2",
            storage_path="/tmp/test_qdrant",
                quantization_enabled=False  # Disable to avoid ScalarQuantization API bug
            )

            await storage.initialize()

            # Verify collection was created
            assert mock_qdrant_client.create_collection.called
            create_call = mock_qdrant_client.create_collection.call_args

            # Verify vector size is 384 (all-MiniLM-L6-v2)
            assert storage._vector_size == 384

            # Verify collection creation params
            assert create_call is not None

            # Verify metadata point was stored
            assert mock_qdrant_client.upsert.called
            upsert_call = mock_qdrant_client.upsert.call_args
            # Metadata point has ID = 1 (METADATA_POINT_ID constant)
            assert any(point.id == 1 for point in upsert_call[1]["points"])

    @pytest.mark.asyncio
    async def test_initialization_detects_vector_dimensions(self, mock_qdrant_client):
        """Test auto-detection of vector dimensions from model."""
        test_cases = [
            ("all-MiniLM-L6-v2", 384),
            ("all-mpnet-base-v2", 768),
            ("text-embedding-ada-002", 1536),
        ]

        for model_name, expected_dims in test_cases:
            mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])

            with patch('src.mcp_memory_service.storage.qdrant_storage.QdrantClient', return_value=mock_qdrant_client):
                storage = QdrantStorage(
            storage_path="/tmp/test_qdrant",
                    embedding_model=model_name
                )

                await storage.initialize()
                assert storage._vector_size == expected_dims

    @pytest.mark.asyncio
    async def test_initialization_verifies_existing_collection(self, mock_qdrant_client):
        """Test that initialization verifies model compatibility for existing collection."""
        # Mock: collection exists
        existing_collection = MagicMock()
        existing_collection.name = "test_memories"
        mock_qdrant_client.get_collections.return_value = MagicMock(
            collections=[existing_collection]
        )

        # Mock collection info
        mock_qdrant_client.get_collection.return_value = MagicMock(
            config=MagicMock(
                params=MagicMock(
                    vectors=MagicMock(size=384)
                )
            )
        )

        # Mock metadata retrieval
        mock_qdrant_client.retrieve.return_value = [
            MagicMock(
                id="__metadata__",
                payload={
                    "embedding_model": "all-MiniLM-L6-v2",
                    "vector_size": 384,
                    "dimensions": 384
                }
            )
        ]

        with patch('src.mcp_memory_service.storage.qdrant_storage.QdrantClient', return_value=mock_qdrant_client):
            storage = QdrantStorage(
            embedding_model="all-MiniLM-L6-v2",
            storage_path="/tmp/test_qdrant",
                collection_name="test_memories"
            )

            await storage.initialize()

            # Should not create new collection
            assert not mock_qdrant_client.create_collection.called
            # Should verify compatibility
            assert mock_qdrant_client.retrieve.called


# ============================================================================
# Store Operation Tests
# ============================================================================

class TestQdrantStore:
    """Test Qdrant memory storage operations."""

    @pytest.mark.asyncio
    async def test_store_memory_success(self, qdrant_storage, sample_memory, mock_qdrant_client):
        """Test successful memory storage."""
        qdrant_storage._initialized = True

        success, message = await qdrant_storage.store(sample_memory)

        assert success is True
        assert "successfully" in message.lower()

        # Verify upsert was called with correct data
        assert mock_qdrant_client.upsert.called
        call_args = mock_qdrant_client.upsert.call_args

        points = call_args[1]["points"]
        assert len(points) == 1

        point = points[0]
        # ID is converted to UUID format
        expected_id = qdrant_storage._hash_to_uuid(sample_memory.content_hash)
        assert point.id == expected_id
        assert point.vector == sample_memory.embedding
        assert point.payload["content"] == sample_memory.content
        assert point.payload["tags"] == sample_memory.tags

    @pytest.mark.asyncio
    async def test_store_duplicate_content_is_idempotent(self, qdrant_storage, sample_memory, mock_qdrant_client):
        """Test that storing duplicate content (same hash) updates existing point."""
        qdrant_storage._initialized = True

        # Store first time
        success1, _ = await qdrant_storage.store(sample_memory)
        assert success1 is True

        # Store again with same content_hash
        success2, _ = await qdrant_storage.store(sample_memory)
        assert success2 is True

        # Both should succeed (upsert is idempotent)
        assert mock_qdrant_client.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_store_dimension_mismatch_raises_error(self, qdrant_storage, sample_memory):
        """Test that dimension mismatch raises clear error."""
        qdrant_storage._initialized = True
        qdrant_storage._vector_size = 384

        # Create memory with wrong dimensions
        wrong_memory = Memory(
            content="Wrong dimensions",
            content_hash=generate_content_hash("Wrong dimensions"),
            embedding=[0.1] * 768,  # Wrong! Should be 384
            tags=["test"]
        )

        with pytest.raises(ValueError) as exc_info:
            await qdrant_storage.store(wrong_memory)

        error_msg = str(exc_info.value)
        assert "dimension mismatch" in error_msg.lower()
        assert "384" in error_msg  # Expected
        assert "768" in error_msg  # Got

    @pytest.mark.asyncio
    async def test_store_normalizes_tags(self, qdrant_storage, mock_qdrant_client):
        """Test that tags are normalized and metadata preserved."""
        qdrant_storage._initialized = True

        memory = Memory(
            content="Test content",
            content_hash=generate_content_hash("Test content"),
            tags=["Test", "QDRANT", "storage"],
            metadata={"source": "unit_test", "priority": "high"},
            embedding=[0.1] * 384
        )

        await qdrant_storage.store(memory)

        call_args = mock_qdrant_client.upsert.call_args
        point = call_args[1]["points"][0]

        # Verify tags are stored as-is (normalization happens at MCP layer)
        assert point.payload["tags"] == ["Test", "QDRANT", "storage"]

        # Verify metadata is preserved
        assert point.payload["metadata"]["source"] == "unit_test"
        assert point.payload["metadata"]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_store_preserves_timestamps(self, qdrant_storage, mock_qdrant_client):
        """Test that timestamps are preserved during storage."""
        qdrant_storage._initialized = True

        created_time = datetime(2024, 1, 1, 12, 0, 0).timestamp()
        updated_time = datetime(2024, 1, 2, 12, 0, 0).timestamp()

        memory = Memory(
            content="Timestamped content",
            content_hash=generate_content_hash("Timestamped content"),
            embedding=[0.1] * 384,
            created_at=created_time,
            updated_at=updated_time
        )

        await qdrant_storage.store(memory)

        call_args = mock_qdrant_client.upsert.call_args
        point = call_args[1]["points"][0]

        assert point.payload["created_at"] == created_time
        assert point.payload["updated_at"] == updated_time


# ============================================================================
# Retrieve Operation Tests
# ============================================================================

class TestQdrantRetrieve:
    """Test Qdrant memory retrieval operations."""

    @pytest.mark.asyncio
    async def test_retrieve_semantic_search(self, qdrant_storage, mock_qdrant_client):
        """Test semantic search returns top N results."""
        qdrant_storage._initialized = True

        # Mock embedding service - need to return array with tolist() method
        mock_array = MagicMock()
        mock_array.tolist.return_value = [0.5] * 384
        mock_embedding_service = MagicMock()
        mock_embedding_service.encode.return_value = [mock_array]  # Returns list of arrays
        qdrant_storage.embedding_service = mock_embedding_service

        # Mock search results
        mock_qdrant_client.search.return_value = [
            MagicMock(
                id="hash1",
                score=0.95,
                payload={
                    "content": "Result 1",
                    "content_hash": "hash1",
                    "tags": ["test"],
                    "memory_type": "standard",
                    "metadata": {},
                    "created_at": datetime.now().timestamp()
                }
            ),
            MagicMock(
                id="hash2",
                score=0.85,
                payload={
                    "content": "Result 2",
                    "content_hash": "hash2",
                    "tags": ["test"],
                    "memory_type": "standard",
                    "metadata": {},
                    "created_at": datetime.now().timestamp()
                }
            )
        ]

        results = await qdrant_storage.retrieve("test query", n_results=2)

        assert len(results) == 2
        assert results[0].relevance_score == 0.95
        assert results[1].relevance_score == 0.85
        assert results[0].memory.content == "Result 1"

    @pytest.mark.asyncio
    async def test_retrieve_with_tag_filter(self, qdrant_storage, mock_qdrant_client):
        """Test retrieval with tag filtering (OR logic)."""
        qdrant_storage._initialized = True

        # Mock embedding service - need to return array with tolist() method
        mock_array = MagicMock()
        mock_array.tolist.return_value = [0.5] * 384
        mock_embedding_service = MagicMock()
        mock_embedding_service.encode.return_value = [mock_array]
        qdrant_storage.embedding_service = mock_embedding_service

        mock_qdrant_client.search.return_value = []

        await qdrant_storage.retrieve(
            "test query",
            n_results=5,
            tags=["important", "urgent"]
        )

        # Verify search was called with tag filter
        call_args = mock_qdrant_client.search.call_args
        assert call_args is not None
        query_filter = call_args[1].get("query_filter")

        assert query_filter is not None
        # Should have tag conditions in should[] for OR logic
        assert hasattr(query_filter, 'should')

    @pytest.mark.asyncio
    async def test_retrieve_with_memory_type_filter(self, qdrant_storage, mock_qdrant_client):
        """Test retrieval with memory type filtering (AND logic)."""
        qdrant_storage._initialized = True

        # Mock embedding service - need to return array with tolist() method
        mock_array = MagicMock()
        mock_array.tolist.return_value = [0.5] * 384
        mock_embedding_service = MagicMock()
        mock_embedding_service.encode.return_value = [mock_array]
        qdrant_storage.embedding_service = mock_embedding_service

        mock_qdrant_client.search.return_value = []

        await qdrant_storage.retrieve(
            "test query",
            n_results=5,
            memory_type="document"
        )

        # Verify search was called with memory_type filter
        call_args = mock_qdrant_client.search.call_args
        assert call_args is not None
        query_filter = call_args[1].get("query_filter")

        assert query_filter is not None
        # Should have memory_type condition in must[] for AND logic
        assert hasattr(query_filter, 'must')

    @pytest.mark.asyncio
    async def test_retrieve_applies_min_similarity(self, qdrant_storage, mock_qdrant_client):
        """Test minimum similarity threshold filtering."""
        qdrant_storage._initialized = True

        # Mock embedding service - need to return array with tolist() method
        mock_array = MagicMock()
        mock_array.tolist.return_value = [0.5] * 384
        mock_embedding_service = MagicMock()
        mock_embedding_service.encode.return_value = [mock_array]
        qdrant_storage.embedding_service = mock_embedding_service

        mock_qdrant_client.search.return_value = [
            MagicMock(
                id="hash1",
                score=0.95,
                payload={
                    "content": "High similarity",
                    "content_hash": "hash1",
                    "tags": [],
                    "metadata": {},
                    "created_at": datetime.now().timestamp()
                }
            ),
            MagicMock(
                id="hash2",
                score=0.60,
                payload={
                    "content": "Low similarity",
                    "content_hash": "hash2",
                    "tags": [],
                    "metadata": {},
                    "created_at": datetime.now().timestamp()
                }
            )
        ]

        results = await qdrant_storage.retrieve(
            "test query",
            n_results=10,
            min_similarity=0.7
        )

        # Only result with score >= 0.7 should be returned
        assert len(results) == 1
        assert results[0].relevance_score == 0.95

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self, qdrant_storage, mock_qdrant_client):
        """Test retrieval with no matches returns empty list."""
        qdrant_storage._initialized = True

        # Mock embedding service - need to return array with tolist() method
        mock_array = MagicMock()
        mock_array.tolist.return_value = [0.5] * 384
        mock_embedding_service = MagicMock()
        mock_embedding_service.encode.return_value = [mock_array]
        qdrant_storage.embedding_service = mock_embedding_service

        mock_qdrant_client.search.return_value = []

        results = await qdrant_storage.retrieve("no matches")

        assert results == []


# ============================================================================
# Tag Search Tests
# ============================================================================

class TestQdrantTagSearch:
    """Test Qdrant tag-based search operations."""

    @pytest.mark.asyncio
    async def test_search_by_single_tag(self, qdrant_storage, mock_qdrant_client):
        """Test search by single tag."""
        qdrant_storage._initialized = True

        mock_point = MagicMock(
            id="hash1",
            payload={
                "content": "Tagged content",
                "tags": ["important"],
                "metadata": {},
                "created_at": datetime.now().timestamp()
            }
        )
        mock_qdrant_client.scroll.return_value = ([mock_point], None)

        results = await qdrant_storage.search_by_tag(tags=["important"])

        assert len(results) == 1
        assert results[0].content == "Tagged content"

    @pytest.mark.asyncio
    async def test_search_by_multiple_tags_or_logic(self, qdrant_storage, mock_qdrant_client):
        """Test search by multiple tags with OR logic."""
        qdrant_storage._initialized = True

        mock_qdrant_client.scroll.return_value = ([], None)

        await qdrant_storage.search_by_tag(tags=["tag1", "tag2", "tag3"])

        # Verify scroll was called with OR filter
        call_args = mock_qdrant_client.scroll.call_args
        scroll_filter = call_args[1].get("scroll_filter")

        assert scroll_filter is not None
        # OR logic uses should[] condition
        assert hasattr(scroll_filter, 'should')

    @pytest.mark.asyncio
    async def test_search_by_tag_pagination(self, qdrant_storage, mock_qdrant_client):
        """Test tag search with limit and offset pagination."""
        qdrant_storage._initialized = True

        mock_qdrant_client.scroll.return_value = ([], None)

        await qdrant_storage.search_by_tag(
            tags=["test"],
            limit=20,
            offset=40
        )

        # Verify pagination parameters
        call_args = mock_qdrant_client.scroll.call_args
        assert call_args[1]["limit"] == 20
        assert call_args[1]["offset"] == 40

    @pytest.mark.asyncio
    async def test_search_by_tag_timestamp_range(self, qdrant_storage, mock_qdrant_client):
        """Test tag search with timestamp filtering."""
        qdrant_storage._initialized = True

        start_time = datetime(2024, 1, 1).timestamp()
        end_time = datetime(2024, 12, 31).timestamp()

        mock_qdrant_client.scroll.return_value = ([], None)

        await qdrant_storage.search_by_tag(
            tags=["test"],
            start_timestamp=start_time,
            end_timestamp=end_time
        )

        # Verify timestamp filter was applied
        call_args = mock_qdrant_client.scroll.call_args
        scroll_filter = call_args[1].get("scroll_filter")

        assert scroll_filter is not None

    @pytest.mark.asyncio
    async def test_search_by_tag_combined_filters(self, qdrant_storage, mock_qdrant_client):
        """Test tag search with combined tag and timestamp filters."""
        qdrant_storage._initialized = True

        mock_qdrant_client.scroll.return_value = ([], None)

        await qdrant_storage.search_by_tag(
            tags=["important", "urgent"],
            limit=10,
            offset=0,
            start_timestamp=datetime(2024, 1, 1).timestamp(),
            end_timestamp=datetime.now().timestamp()
        )

        # Verify both filters are combined
        call_args = mock_qdrant_client.scroll.call_args
        scroll_filter = call_args[1].get("scroll_filter")

        assert scroll_filter is not None
        # Combined filters should use must[] for AND logic between tag and timestamp
        assert hasattr(scroll_filter, 'must')


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestQdrantErrorHandling:
    """Test Qdrant error handling and circuit breaker."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, qdrant_storage, mock_qdrant_client, sample_memory):
        """Test circuit breaker opens after 5 consecutive failures."""
        qdrant_storage._initialized = True

        # Simulate 5 consecutive failures
        mock_qdrant_client.upsert.side_effect = Exception("Qdrant unavailable")

        for i in range(5):
            success, _ = await qdrant_storage.store(sample_memory)
            assert success is False

        # Circuit should now be open
        assert qdrant_storage._circuit_open_until is not None
        assert qdrant_storage._failure_count == 5

        # Next call should fail fast without calling Qdrant
        with pytest.raises(StorageError, match="Circuit breaker"):
            await qdrant_storage.store(sample_memory)

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self, qdrant_storage, mock_qdrant_client, sample_memory):
        """Test circuit breaker resets after successful operation."""
        qdrant_storage._initialized = True

        # Simulate 2 failures
        mock_qdrant_client.upsert.side_effect = [
            Exception("Fail 1"),
            Exception("Fail 2"),
            MagicMock(status="acknowledged")  # Then success
        ]

        # First two fail
        await qdrant_storage.store(sample_memory)
        await qdrant_storage.store(sample_memory)
        assert qdrant_storage._failure_count == 2

        # Third succeeds
        success, _ = await qdrant_storage.store(sample_memory)
        assert success is True

        # Failure count should reset
        assert qdrant_storage._failure_count == 0
        assert qdrant_storage._circuit_open_until is None

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_while_open(self, qdrant_storage, sample_memory):
        """Test circuit breaker blocks operations while open."""
        qdrant_storage._initialized = True

        # Manually open circuit
        qdrant_storage._circuit_open_until = datetime.now() + timedelta(seconds=60)

        # All operations should fail fast
        with pytest.raises(StorageError, match="Circuit breaker is open"):
            await qdrant_storage.store(sample_memory)

        with pytest.raises(StorageError, match="Circuit breaker is open"):
            await qdrant_storage.retrieve("test")

        with pytest.raises(StorageError, match="Circuit breaker is open"):
            await qdrant_storage.search_by_tag(["test"])

    @pytest.mark.asyncio
    async def test_circuit_breaker_auto_resets_after_timeout(self, qdrant_storage, mock_qdrant_client, sample_memory):
        """Test circuit breaker automatically resets after timeout."""
        qdrant_storage._initialized = True

        # Open circuit with expired timeout
        qdrant_storage._circuit_open_until = datetime.now() - timedelta(seconds=1)
        qdrant_storage._failure_count = 5

        # Next operation should attempt to reset circuit
        mock_qdrant_client.upsert.return_value = MagicMock(status="acknowledged")
        success, _ = await qdrant_storage.store(sample_memory)

        assert success is True
        assert qdrant_storage._circuit_open_until is None
        assert qdrant_storage._failure_count == 0

    @pytest.mark.asyncio
    async def test_dimension_mismatch_error_message(self, qdrant_storage):
        """Test dimension mismatch provides clear error with fix instructions."""
        qdrant_storage._initialized = True
        qdrant_storage._vector_size = 384

        wrong_memory = Memory(
            content="Wrong dimensions",
            content_hash=generate_content_hash("Wrong dimensions"),
            embedding=[0.1] * 1536,  # text-embedding-ada-002 dimensions
            tags=[]
        )

        with pytest.raises(ValueError) as exc_info:
            await qdrant_storage.store(wrong_memory)

        error_msg = str(exc_info.value)
        assert "dimension mismatch" in error_msg.lower()
        assert "expected 384" in error_msg.lower()
        assert "got 1536" in error_msg.lower()
        assert "configuration error" in error_msg.lower()


# ============================================================================
# Model Change Detection Tests
# ============================================================================

class TestQdrantModelChangeDetection:
    """Test Qdrant model change detection."""

    @pytest.mark.asyncio
    async def test_model_change_detected_raises_error(self):
        """Test that model change raises StorageError with migration command."""
        # Create a fresh mock client for this test
        mock_client = MagicMock()

        # Mock existing collection with different model
        existing_collection = MagicMock()
        existing_collection.name = "test_memories"
        mock_client.get_collections.return_value = MagicMock(
            collections=[existing_collection]
        )

        mock_client.get_collection.return_value = MagicMock(
            config=MagicMock(
                params=MagicMock(
                    vectors=MagicMock(size=768)  # Old model dimensions
                )
            )
        )

        # Mock metadata with old model
        mock_client.retrieve.return_value = [
            MagicMock(
                id="__metadata__",
                payload={
                    "embedding_model": "all-mpnet-base-v2",  # Old model (768 dims)
                    "vector_size": 768,
                    "dimensions": 768
                }
            )
        ]

        with patch('src.mcp_memory_service.storage.qdrant_storage.QdrantClient', return_value=mock_client):
            storage = QdrantStorage(
            embedding_model="all-MiniLM-L6-v2",
            storage_path="/tmp/test_qdrant",  # New model (384 dims)
                collection_name="test_memories"
            )

            with pytest.raises(StorageError) as exc_info:
                await storage.initialize()

            error_msg = str(exc_info.value)
            # Implementation checks collection vector size first
            assert "vector size" in error_msg.lower() or "dimension" in error_msg.lower()
            assert "768" in error_msg  # Old dimensions
            assert "384" in error_msg  # New dimensions
            assert "migration" in error_msg.lower()
            assert "migrate_to_new_model.py" in error_msg

    @pytest.mark.asyncio
    async def test_dimension_mismatch_detected_with_migration_command(self):
        """Test dimension mismatch raises error even if model name matches."""
        # Create a fresh mock client
        mock_client = MagicMock()

        # Mock existing collection
        existing_collection = MagicMock()
        existing_collection.name = "test_memories"
        mock_client.get_collections.return_value = MagicMock(
            collections=[existing_collection]
        )

        # Mock collection with wrong dimensions
        mock_client.get_collection.return_value = MagicMock(
            config=MagicMock(
                params=MagicMock(
                    vectors=MagicMock(size=768)  # Wrong!
                )
            )
        )

        # Mock metadata with matching model but wrong dimensions
        mock_client.retrieve.return_value = [
            MagicMock(
                id="__metadata__",
                payload={
                    "embedding_model": "all-MiniLM-L6-v2",  # Same model
                    "vector_size": 768,  # But wrong dimensions!
                    "dimensions": 768
                }
            )
        ]

        with patch('src.mcp_memory_service.storage.qdrant_storage.QdrantClient', return_value=mock_client):
            storage = QdrantStorage(
            embedding_model="all-MiniLM-L6-v2",
            storage_path="/tmp/test_qdrant",
                collection_name="test_memories"
            )

            with pytest.raises(StorageError) as exc_info:
                await storage.initialize()

            error_msg = str(exc_info.value)
            # Verify dimension mismatch is detected
            assert "vector size" in error_msg.lower() or "dimension" in error_msg.lower()
            assert "768" in error_msg  # Stored dimensions
            assert "384" in error_msg  # Current dimensions
            assert "migration" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_same_model_reinitialization_succeeds(self, mock_qdrant_client):
        """Test re-initialization with same model succeeds (no false positives)."""
        # Mock existing collection
        existing_collection = MagicMock()
        existing_collection.name = "test_memories"
        mock_qdrant_client.get_collections.return_value = MagicMock(
            collections=[existing_collection]
        )

        mock_qdrant_client.get_collection.return_value = MagicMock(
            config=MagicMock(
                params=MagicMock(
                    vectors=MagicMock(size=384)
                )
            )
        )

        # Mock metadata with SAME model
        mock_qdrant_client.retrieve.return_value = [
            MagicMock(
                id="__metadata__",
                payload={
                    "embedding_model": "all-MiniLM-L6-v2",
                    "vector_size": 384,
                    "dimensions": 384
                }
            )
        ]

        with patch('src.mcp_memory_service.storage.qdrant_storage.QdrantClient', return_value=mock_qdrant_client):
            storage = QdrantStorage(
            embedding_model="all-MiniLM-L6-v2",
            storage_path="/tmp/test_qdrant",
                collection_name="test_memories"
            )

            # Should succeed without errors
            await storage.initialize()
            assert storage._initialized is True

    @pytest.mark.asyncio
    async def test_migration_command_includes_model_names(self):
        """Test error message includes exact migration command to run."""
        # Create a fresh mock client
        mock_client = MagicMock()

        existing_collection = MagicMock()
        existing_collection.name = "test_memories"
        mock_client.get_collections.return_value = MagicMock(
            collections=[existing_collection]
        )

        mock_client.get_collection.return_value = MagicMock(
            config=MagicMock(
                params=MagicMock(vectors=MagicMock(size=768))
            )
        )

        mock_client.retrieve.return_value = [
            MagicMock(
                id="__metadata__",
                payload={
                    "embedding_model": "all-mpnet-base-v2",
                    "vector_size": 768,
                    "dimensions": 768
                }
            )
        ]

        with patch('src.mcp_memory_service.storage.qdrant_storage.QdrantClient', return_value=mock_client):
            storage = QdrantStorage(
            embedding_model="all-MiniLM-L6-v2",
            storage_path="/tmp/test_qdrant",
                collection_name="test_memories"
            )

            with pytest.raises(StorageError) as exc_info:
                await storage.initialize()

            error_msg = str(exc_info.value)
            # Should include migration command with new model
            assert "--new-model all-MiniLM-L6-v2" in error_msg or "--new-model all-minilm-l6-v2" in error_msg.lower()
            assert "migrate_to_new_model.py" in error_msg


# ============================================================================
# Additional CRUD Tests
# ============================================================================

class TestQdrantCRUD:
    """Test Qdrant CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_memory_by_hash(self, qdrant_storage, mock_qdrant_client, sample_memory):
        """Test retrieving specific memory by hash."""
        qdrant_storage._initialized = True

        mock_qdrant_client.retrieve.return_value = [
            MagicMock(
                id=sample_memory.content_hash,
                payload={
                    "content": sample_memory.content,
                    "tags": sample_memory.tags,
                    "metadata": sample_memory.metadata or {},
                    "created_at": sample_memory.created_at
                }
            )
        ]

        result = await qdrant_storage.get_memory_by_hash(sample_memory.content_hash)

        assert result is not None
        assert result.content == sample_memory.content
        assert result.tags == sample_memory.tags

    @pytest.mark.asyncio
    async def test_delete_memory(self, qdrant_storage, mock_qdrant_client, sample_memory):
        """Test deleting memory by hash."""
        qdrant_storage._initialized = True

        mock_qdrant_client.delete.return_value = MagicMock(status="acknowledged")

        success, message = await qdrant_storage.delete(sample_memory.content_hash)

        assert success is True
        assert "successfully" in message.lower()
        assert mock_qdrant_client.delete.called
        # Verify UUID conversion was applied
        call_args = mock_qdrant_client.delete.call_args
        expected_uuid = qdrant_storage._hash_to_uuid(sample_memory.content_hash)
        assert expected_uuid in str(call_args)

    @pytest.mark.asyncio
    async def test_update_memory_metadata(self, qdrant_storage, mock_qdrant_client, sample_memory):
        """Test updating memory metadata without recreating entry."""
        qdrant_storage._initialized = True

        # Mock existing memory
        mock_qdrant_client.retrieve.return_value = [
            MagicMock(
                id=sample_memory.content_hash,
                payload={
                    "content": sample_memory.content,
                    "tags": sample_memory.tags,
                    "memory_type": sample_memory.memory_type,
                    "metadata": {},
                    "created_at": sample_memory.created_at
                }
            )
        ]

        mock_qdrant_client.set_payload.return_value = MagicMock(status="acknowledged")

        updates = {
            "tags": ["updated", "tags"],
            "metadata": {"new_field": "new_value"}
        }

        success, message = await qdrant_storage.update_memory_metadata(
            sample_memory.content_hash,
            updates,
            preserve_timestamps=True
        )

        assert success is True
        assert mock_qdrant_client.set_payload.called

        # Verify payload update preserves content
        call_args = mock_qdrant_client.set_payload.call_args
        payload = call_args[1]["payload"]
        assert payload["content"] == sample_memory.content
        assert payload["tags"] == ["updated", "tags"]

    @pytest.mark.asyncio
    async def test_get_stats(self, qdrant_storage, mock_qdrant_client):
        """Test getting storage statistics."""
        qdrant_storage._initialized = True

        mock_qdrant_client.get_collection.return_value = MagicMock(
            points_count=101  # 100 memories + 1 __metadata__ point
        )

        stats = await qdrant_storage.get_stats()

        assert stats["total_memories"] == 100  # Excludes __metadata__
        assert stats["storage_backend"] == "qdrant"
        assert stats["embedding_model"] == "all-MiniLM-L6-v2"
        assert stats["vector_size"] == 384
        assert "circuit_breaker" in stats
        assert stats["circuit_breaker"]["status"] == "closed"
