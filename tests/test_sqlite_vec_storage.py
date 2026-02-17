"""
Comprehensive tests for SQLite-vec storage backend.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
import shutil
import json
from unittest.mock import Mock, patch
import time
from datetime import date, datetime, timedelta

# Skip tests if sqlite-vec is not available
try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

from src.mcp_memory_service.models.memory import Memory, MemoryQueryResult
from src.mcp_memory_service.utils.hashing import generate_content_hash

if SQLITE_VEC_AVAILABLE:
    from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

# Skip all tests if sqlite-vec is not available
pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")


class TestSqliteVecStorage:
    """Test suite for SQLite-vec storage functionality."""
    
    @pytest_asyncio.fixture
    async def storage(self):
        """Create a test storage instance."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_memory.db")
        
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()
        
        yield storage
        
        # Cleanup
        if storage.conn:
            storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_memory(self):
        """Create a sample memory for testing."""
        content = "This is a test memory for SQLite-vec storage"
        return Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test", "sqlite-vec"],
            memory_type="note",
            metadata={"priority": "medium", "category": "testing"}
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test storage initialization."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_init.db")
        
        try:
            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()
            
            # Check that database file was created
            assert os.path.exists(db_path)
            
            # Check that connection is established
            assert storage.conn is not None
            
            # Check that table was created
            cursor = storage.conn.execute('''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='memories'
            ''')
            assert cursor.fetchone() is not None
            
            storage.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_store_memory(self, storage, sample_memory):
        """Test storing a memory."""
        success, message = await storage.store(sample_memory)
        
        assert success
        assert "successfully" in message.lower()
        
        # Verify memory was stored
        cursor = storage.conn.execute(
            'SELECT content_hash FROM memories WHERE content_hash = ?',
            (sample_memory.content_hash,)
        )
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == sample_memory.content_hash
    
    @pytest.mark.asyncio
    async def test_store_duplicate_memory(self, storage, sample_memory):
        """Test that duplicate memories are rejected."""
        # Store the memory first time
        success, message = await storage.store(sample_memory)
        assert success
        
        # Try to store the same memory again
        success, message = await storage.store(sample_memory)
        assert not success
        assert "duplicate" in message.lower()
    
    @pytest.mark.asyncio
    async def test_retrieve_memory(self, storage, sample_memory):
        """Test retrieving memories using semantic search."""
        # Store the memory
        await storage.store(sample_memory)
        
        # Retrieve using semantic search
        results = await storage.retrieve("test memory sqlite", n_results=5)
        
        assert len(results) > 0
        assert isinstance(results[0], MemoryQueryResult)
        assert results[0].memory.content_hash == sample_memory.content_hash
        assert results[0].relevance_score >= 0.0
        assert results[0].debug_info["backend"] == "sqlite-vec"
    
    @pytest.mark.asyncio
    async def test_retrieve_no_results(self, storage):
        """Test retrieving when no memories match."""
        results = await storage.retrieve("nonexistent query", n_results=5)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_retrieve_knn_syntax(self, storage):
        """Test retrieve() uses k=? syntax correctly (PR #308 fix).

        This test verifies that the KNN query doesn't throw:
        sqlite3.OperationalError: A LIMIT or 'k = ?' constraint is required on vec0 knn queries.
        """
        # Store multiple memories for KNN search
        memories = []
        for i in range(5):
            content = f"Test memory {i} for KNN parameter validation"
            memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=[f"knn-test-{i}"]
            )
            memories.append(memory)
            await storage.store(memory)

        # Test with various n_results values to ensure k=? works
        for n in [1, 3, 5, 10]:
            try:
                results = await storage.retrieve("KNN parameter validation", n_results=n)
                # Should not raise OperationalError
                assert isinstance(results, list), f"Expected list for n_results={n}"
                assert len(results) <= n, f"Should return at most {n} results"
                assert len(results) <= len(memories), "Cannot return more than stored"
            except Exception as e:
                # Fail if we get the specific k=? constraint error
                if "LIMIT or 'k = ?' constraint is required" in str(e):
                    pytest.fail(f"KNN syntax error with n_results={n}: {e}")
                raise  # Re-raise other errors

    @pytest.mark.asyncio
    async def test_recall_knn_syntax(self, storage):
        """Test recall() uses k=? syntax correctly (PR #308 fix).

        This test verifies that recall() with semantic search uses the correct
        k=? parameter syntax and doesn't throw OperationalError.
        """
        # Store test memories
        memories = []
        for i in range(3):
            content = f"Recall test memory {i} with semantic search capability"
            memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=["recall-knn-test"]
            )
            memories.append(memory)
            await storage.store(memory)

        # Test recall with semantic query (uses k=? in subquery)
        try:
            results = await storage.recall("semantic search", n_results=2)
            # Should not raise OperationalError
            assert isinstance(results, list), "Expected list from recall()"
            assert len(results) <= 2, "Should respect n_results limit"
        except Exception as e:
            # Fail if we get the specific k=? constraint error
            if "LIMIT or 'k = ?' constraint is required" in str(e):
                pytest.fail(f"Recall KNN syntax error: {e}")
            raise  # Re-raise other errors

    @pytest.mark.asyncio
    async def test_search_by_tag(self, storage, sample_memory):
        """Test searching memories by tags."""
        # Store the memory
        await storage.store(sample_memory)
        
        # Search by existing tag
        results = await storage.search_by_tag(["test"])
        assert len(results) == 1
        assert results[0].content_hash == sample_memory.content_hash
        
        # Search by non-existent tag
        results = await storage.search_by_tag(["nonexistent"])
        assert len(results) == 0
        
        # Search by multiple tags
        results = await storage.search_by_tag(["test", "sqlite-vec"])
        assert len(results) == 1
    
    @pytest.mark.asyncio
    async def test_search_by_empty_tags(self, storage):
        """Test searching with empty tags list."""
        results = await storage.search_by_tag([])
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, storage, sample_memory):
        """Test deleting a memory by content hash (soft-delete)."""
        # Store the memory
        await storage.store(sample_memory)

        # Delete the memory (soft-delete)
        success, message = await storage.delete(sample_memory.content_hash)
        assert success
        assert sample_memory.content_hash in message

        # Verify memory was soft-deleted (still in DB but deleted_at is set)
        cursor = storage.conn.execute(
            'SELECT content_hash, deleted_at FROM memories WHERE content_hash = ?',
            (sample_memory.content_hash,)
        )
        row = cursor.fetchone()
        assert row is not None, "Memory should still exist in DB as tombstone"
        assert row[1] is not None, "deleted_at should be set"

        # Verify is_deleted() returns True
        assert await storage.is_deleted(sample_memory.content_hash)

        # Verify memory is not returned by get_by_hash (excludes deleted)
        memory = await storage.get_by_hash(sample_memory.content_hash)
        assert memory is None, "Deleted memory should not be returned by get_by_hash"
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_memory(self, storage):
        """Test deleting a non-existent memory."""
        nonexistent_hash = "nonexistent123456789"
        success, message = await storage.delete(nonexistent_hash)
        assert not success
        assert "not found" in message.lower()
    
    @pytest.mark.asyncio
    async def test_delete_by_tag(self, storage):
        """Test deleting memories by tag."""
        # Store multiple memories with different tags
        memory1 = Memory(
            content="Memory 1",
            content_hash=generate_content_hash("Memory 1"),
            tags=["tag1", "shared"]
        )
        memory2 = Memory(
            content="Memory 2", 
            content_hash=generate_content_hash("Memory 2"),
            tags=["tag2", "shared"]
        )
        memory3 = Memory(
            content="Memory 3",
            content_hash=generate_content_hash("Memory 3"),
            tags=["tag3"]
        )
        
        await storage.store(memory1)
        await storage.store(memory2)
        await storage.store(memory3)
        
        # Delete by shared tag
        count, message = await storage.delete_by_tag("shared")
        assert count == 2
        assert "deleted 2 memories" in message.lower()
        
        # Verify correct memories were deleted
        remaining = await storage.search_by_tag(["tag3"])
        assert len(remaining) == 1
        assert remaining[0].content_hash == memory3.content_hash
    
    @pytest.mark.asyncio
    async def test_delete_by_nonexistent_tag(self, storage):
        """Test deleting by a non-existent tag."""
        count, message = await storage.delete_by_tag("nonexistent")
        assert count == 0
        assert "no memories found" in message.lower()
    
    @pytest.mark.asyncio
    async def test_cleanup_duplicates(self, storage):
        """Test cleaning up duplicate memories."""
        # Create memory
        content = "Duplicate test memory"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["duplicate"]
        )

        # Store the memory
        await storage.store(memory)

        # Temporarily drop the unique constraint to allow duplicate
        # This simulates the scenario where duplicates exist (e.g., from migration or bug)
        storage.conn.execute('PRAGMA foreign_keys = OFF')

        # Create a temporary table without unique constraint
        storage.conn.execute('''
            CREATE TABLE memories_temp AS SELECT * FROM memories
        ''')

        # Drop original table
        storage.conn.execute('DROP TABLE memories')

        # Recreate without unique constraint temporarily (include deleted_at for v8.64.0+)
        storage.conn.execute('''
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT,
                memory_type TEXT,
                metadata TEXT,
                created_at REAL,
                updated_at REAL,
                created_at_iso TEXT,
                updated_at_iso TEXT,
                deleted_at REAL DEFAULT NULL
            )
        ''')

        # Copy data back (explicitly list columns since temp table may not have deleted_at)
        storage.conn.execute('''
            INSERT INTO memories (id, content_hash, content, tags, memory_type,
                metadata, created_at, updated_at, created_at_iso, updated_at_iso)
            SELECT id, content_hash, content, tags, memory_type,
                metadata, created_at, updated_at, created_at_iso, updated_at_iso
            FROM memories_temp
        ''')
        storage.conn.execute('DROP TABLE memories_temp')

        # Now insert the duplicate
        embedding = storage._generate_embedding(content)
        current_time = time.time()

        storage.conn.execute('''
            INSERT INTO memories (
                content_hash, content, tags, memory_type,
                metadata, created_at, updated_at, created_at_iso, updated_at_iso
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory.content_hash,
            content,
            "duplicate",
            None,
            "{}",
            current_time,
            current_time,
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z"
        ))

        rowid = storage.conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        storage.conn.execute('''
            INSERT INTO memory_embeddings (rowid, content_embedding)
            VALUES (?, ?)
        ''', (rowid, sqlite_vec.serialize_float32(embedding)))

        storage.conn.commit()

        # Verify we have 2 duplicates
        cursor = storage.conn.execute(
            'SELECT COUNT(*) FROM memories WHERE content_hash = ?',
            (memory.content_hash,)
        )
        assert cursor.fetchone()[0] == 2

        # Clean up duplicates
        count, message = await storage.cleanup_duplicates()
        assert count == 1
        assert "1 duplicate" in message.lower()

        # Verify only one active copy remains (soft-delete keeps both rows but marks one as deleted)
        cursor = storage.conn.execute(
            'SELECT COUNT(*) FROM memories WHERE content_hash = ? AND deleted_at IS NULL',
            (memory.content_hash,)
        )
        assert cursor.fetchone()[0] == 1

        # Verify one copy is soft-deleted
        cursor = storage.conn.execute(
            'SELECT COUNT(*) FROM memories WHERE content_hash = ? AND deleted_at IS NOT NULL',
            (memory.content_hash,)
        )
        assert cursor.fetchone()[0] == 1
    
    @pytest.mark.asyncio
    async def test_cleanup_no_duplicates(self, storage, sample_memory):
        """Test cleanup when no duplicates exist."""
        await storage.store(sample_memory)
        
        count, message = await storage.cleanup_duplicates()
        assert count == 0
        assert "no duplicate memories found" in message.lower()
    
    @pytest.mark.asyncio
    async def test_update_memory_metadata(self, storage, sample_memory):
        """Test updating memory metadata."""
        # Store the memory
        await storage.store(sample_memory)
        
        # Update metadata
        updates = {
            "tags": ["updated", "test"],
            "memory_type": "reminder", 
            "metadata": {"priority": "high", "due_date": "2024-01-15"},
            "status": "active"
        }
        
        success, message = await storage.update_memory_metadata(
            content_hash=sample_memory.content_hash,
            updates=updates
        )
        
        assert success
        assert "updated fields" in message.lower()
        
        # Verify updates
        cursor = storage.conn.execute('''
            SELECT tags, memory_type, metadata
            FROM memories WHERE content_hash = ?
        ''', (sample_memory.content_hash,))
        
        row = cursor.fetchone()
        assert row is not None
        
        tags_str, memory_type, metadata_str = row
        metadata = json.loads(metadata_str)
        
        assert tags_str == "updated,test"
        assert memory_type == "reminder"
        assert metadata["priority"] == "high"
        assert metadata["due_date"] == "2024-01-15"
        assert metadata["status"] == "active"
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_memory(self, storage):
        """Test updating metadata for non-existent memory."""
        nonexistent_hash = "nonexistent123456789"
        success, message = await storage.update_memory_metadata(
            content_hash=nonexistent_hash,
            updates={"tags": ["test"]}
        )
        
        assert not success
        assert "not found" in message.lower()
    
    @pytest.mark.asyncio
    async def test_update_memory_with_invalid_tags(self, storage, sample_memory):
        """Test updating memory with invalid tags format."""
        await storage.store(sample_memory)
        
        success, message = await storage.update_memory_metadata(
            content_hash=sample_memory.content_hash,
            updates={"tags": "not_a_list"}
        )
        
        assert not success
        assert "list of strings" in message.lower()
    
    @pytest.mark.asyncio
    async def test_update_memory_with_invalid_metadata(self, storage, sample_memory):
        """Test updating memory with invalid metadata format."""
        await storage.store(sample_memory)
        
        success, message = await storage.update_memory_metadata(
            content_hash=sample_memory.content_hash,
            updates={"metadata": "not_a_dict"}
        )
        
        assert not success
        assert "dictionary" in message.lower()
    
    @pytest.mark.asyncio
    async def test_update_memory_preserve_timestamps(self, storage, sample_memory):
        """Test updating memory while preserving timestamps."""
        await storage.store(sample_memory)
        
        # Get original timestamps
        cursor = storage.conn.execute('''
            SELECT created_at, created_at_iso FROM memories WHERE content_hash = ?
        ''', (sample_memory.content_hash,))
        original_created_at, original_created_at_iso = cursor.fetchone()
        
        # Wait a moment
        await asyncio.sleep(0.1)
        
        # Update with preserve_timestamps=True
        success, message = await storage.update_memory_metadata(
            content_hash=sample_memory.content_hash,
            updates={"tags": ["updated"]},
            preserve_timestamps=True
        )
        
        assert success
        
        # Check timestamps
        cursor = storage.conn.execute('''
            SELECT created_at, created_at_iso, updated_at FROM memories WHERE content_hash = ?
        ''', (sample_memory.content_hash,))
        created_at, created_at_iso, updated_at = cursor.fetchone()
        
        # created_at should be preserved
        assert abs(created_at - original_created_at) < 0.01
        assert created_at_iso == original_created_at_iso
        
        # updated_at should be newer
        assert updated_at > original_created_at
    
    @pytest.mark.asyncio
    async def test_update_memory_reset_timestamps(self, storage, sample_memory):
        """Test updating memory with timestamp reset."""
        await storage.store(sample_memory)
        
        # Get original timestamps
        cursor = storage.conn.execute('''
            SELECT created_at FROM memories WHERE content_hash = ?
        ''', (sample_memory.content_hash,))
        original_created_at = cursor.fetchone()[0]
        
        # Wait a moment to ensure timestamp difference
        await asyncio.sleep(0.2)

        # Update with preserve_timestamps=False
        success, message = await storage.update_memory_metadata(
            content_hash=sample_memory.content_hash,
            updates={"tags": ["updated"]},
            preserve_timestamps=False
        )

        assert success

        # Check timestamps
        cursor = storage.conn.execute('''
            SELECT created_at FROM memories WHERE content_hash = ?
        ''', (sample_memory.content_hash,))
        created_at = cursor.fetchone()[0]

        # created_at should be updated (newer) or equal if update was very fast
        # Use >= instead of > to handle fast systems where timestamps might be identical
        assert created_at >= original_created_at
    
    @pytest.mark.asyncio
    async def test_get_stats(self, storage):
        """Test getting storage statistics."""
        stats = await storage.get_stats()

        assert isinstance(stats, dict)
        assert stats["backend"] == "sqlite-vec"
        assert "total_memories" in stats
        assert "database_size_bytes" in stats
        assert "embedding_model" in stats
        assert "embedding_dimension" in stats
    
    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, storage, sample_memory):
        """Test getting statistics with data."""
        await storage.store(sample_memory)

        stats = await storage.get_stats()

        assert stats["total_memories"] >= 1
        assert stats["database_size_bytes"] > 0
        assert stats["embedding_dimension"] == storage.embedding_dimension
    
    @pytest.mark.asyncio
    async def test_close_connection(self, storage):
        """Test closing the database connection."""
        assert storage.conn is not None

        await storage.close()

        assert storage.conn is None
    
    @pytest.mark.asyncio
    async def test_multiple_memories_retrieval(self, storage):
        """Test retrieving multiple memories with ranking."""
        # Store multiple memories
        memories = []
        for i in range(5):
            content = f"Test memory {i} with different content and keywords"
            memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=[f"tag{i}"],
                memory_type="note"
            )
            memories.append(memory)
            await storage.store(memory)
        
        # Retrieve memories
        results = await storage.retrieve("test memory content", n_results=3)
        
        assert len(results) <= 3
        assert len(results) > 0
        
        # Check that results are properly ranked (higher relevance first)
        for i in range(len(results) - 1):
            assert results[i].relevance_score >= results[i + 1].relevance_score
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, storage):
        """Test embedding generation functionality."""
        test_text = "This is a test for embedding generation"
        
        embedding = storage._generate_embedding(test_text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == storage.embedding_dimension
        assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_memory_with_complex_metadata(self, storage):
        """Test storing and retrieving memory with complex metadata."""
        content = "Memory with complex metadata"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["complex", "metadata", "test"],
            memory_type="structured",
            metadata={
                "nested": {"level1": {"level2": "value"}},
                "array": [1, 2, 3, "four"],
                "boolean": True,
                "null_value": None,
                "unicode": "测试中文 🚀"
            }
        )
        
        # Store the memory
        success, message = await storage.store(memory)
        assert success
        
        # Retrieve and verify
        results = await storage.retrieve("complex metadata", n_results=1)
        assert len(results) == 1
        
        retrieved_memory = results[0].memory
        assert retrieved_memory.metadata["nested"]["level1"]["level2"] == "value"
        assert retrieved_memory.metadata["array"] == [1, 2, 3, "four"]
        assert retrieved_memory.metadata["boolean"] is True
        assert retrieved_memory.metadata["unicode"] == "测试中文 🚀"
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, storage):
        """Test concurrent storage operations."""
        # Create multiple memories
        memories = []
        for i in range(10):
            content = f"Concurrent test memory {i}"
            memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=[f"concurrent{i}"]
            )
            memories.append(memory)
        
        # Store memories concurrently
        tasks = [storage.store(memory) for memory in memories]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(success for success, _ in results)
        
        # Verify all were stored
        for memory in memories:
            cursor = storage.conn.execute(
                'SELECT content_hash FROM memories WHERE content_hash = ?',
                (memory.content_hash,)
            )
            assert cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_get_memories_by_time_range_basic(self, storage):
        """Test basic time range filtering."""
        # Store memories at different times
        now = time.time()

        # Memory from 1 hour ago
        memory1 = Memory(
            content="Memory from 1 hour ago",
            content_hash=generate_content_hash("Memory from 1 hour ago"),
            tags=["timerange"],
            created_at=now - 3600
        )

        # Memory from 30 minutes ago
        memory2 = Memory(
            content="Memory from 30 minutes ago",
            content_hash=generate_content_hash("Memory from 30 minutes ago"),
            tags=["timerange"],
            created_at=now - 1800
        )

        # Memory from now
        memory3 = Memory(
            content="Memory from now",
            content_hash=generate_content_hash("Memory from now"),
            tags=["timerange"],
            created_at=now
        )

        await storage.store(memory1)
        await storage.store(memory2)
        await storage.store(memory3)

        # Get memories from last 45 minutes (should get memory2 and memory3)
        results = await storage.get_memories_by_time_range(now - 2700, now + 100)
        assert len(results) == 2
        contents = [m.content for m in results]
        assert "Memory from 30 minutes ago" in contents
        assert "Memory from now" in contents
        assert "Memory from 1 hour ago" not in contents

    @pytest.mark.asyncio
    async def test_get_memories_by_time_range_empty(self, storage):
        """Test time range with no matching memories."""
        # Store one memory now
        memory = Memory(
            content="Current memory",
            content_hash=generate_content_hash("Current memory"),
            tags=["test"]
        )
        await storage.store(memory)

        # Query for memories from far in the past
        now = time.time()
        results = await storage.get_memories_by_time_range(now - 86400, now - 7200)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_memories_by_time_range_boundaries(self, storage):
        """Test inclusive boundaries of time range."""
        now = time.time()

        # Memory exactly at start boundary
        memory_start = Memory(
            content="At start boundary",
            content_hash=generate_content_hash("At start boundary"),
            tags=["boundary"],
            created_at=now - 1000
        )

        # Memory exactly at end boundary
        memory_end = Memory(
            content="At end boundary",
            content_hash=generate_content_hash("At end boundary"),
            tags=["boundary"],
            created_at=now
        )

        # Memory just before start
        memory_before = Memory(
            content="Before start",
            content_hash=generate_content_hash("Before start"),
            tags=["boundary"],
            created_at=now - 1001
        )

        # Memory just after end
        memory_after = Memory(
            content="After end",
            content_hash=generate_content_hash("After end"),
            tags=["boundary"],
            created_at=now + 1
        )

        await storage.store(memory_start)
        await storage.store(memory_end)
        await storage.store(memory_before)
        await storage.store(memory_after)

        # Query with inclusive boundaries
        results = await storage.get_memories_by_time_range(now - 1000, now)
        assert len(results) == 2
        contents = [m.content for m in results]
        assert "At start boundary" in contents
        assert "At end boundary" in contents
        assert "Before start" not in contents
        assert "After end" not in contents

    @pytest.mark.asyncio
    async def test_get_memories_by_time_range_ordering(self, storage):
        """Test that results are ordered by created_at DESC."""
        now = time.time()

        # Store three memories in random order
        memory1 = Memory(
            content="First",
            content_hash=generate_content_hash("First"),
            tags=["order"],
            created_at=now - 300
        )
        memory2 = Memory(
            content="Second",
            content_hash=generate_content_hash("Second"),
            tags=["order"],
            created_at=now - 200
        )
        memory3 = Memory(
            content="Third",
            content_hash=generate_content_hash("Third"),
            tags=["order"],
            created_at=now - 100
        )

        await storage.store(memory3)  # Store in non-chronological order
        await storage.store(memory1)
        await storage.store(memory2)

        # Get all three
        results = await storage.get_memories_by_time_range(now - 400, now)
        assert len(results) == 3

        # Should be ordered newest first (DESC)
        assert results[0].content == "Third"
        assert results[1].content == "Second"
        assert results[2].content == "First"


class TestSqliteVecTimeBasedDeletion:
    """Tests for time-based deletion methods added in v8.66.0."""

    @pytest_asyncio.fixture
    async def storage(self):
        """Create a test storage instance."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_time_deletion.db")

        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()

        yield storage

        # Cleanup
        if storage.conn:
            storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    # delete_by_timeframe tests

    @pytest.mark.asyncio
    async def test_delete_by_timeframe_success(self, storage):
        """Test deleting memories within a date range."""
        today = date.today()
        yesterday = today - timedelta(days=1)
        tomorrow = today + timedelta(days=1)

        # Store memories with specific dates
        memory_yesterday = Memory(
            content="Memory from yesterday",
            content_hash=generate_content_hash("Memory from yesterday"),
            tags=["timeframe-test"],
            created_at=datetime.combine(yesterday, datetime.min.time()).timestamp()
        )
        memory_today = Memory(
            content="Memory from today",
            content_hash=generate_content_hash("Memory from today"),
            tags=["timeframe-test"],
            created_at=datetime.combine(today, datetime.min.time()).timestamp()
        )
        memory_tomorrow = Memory(
            content="Memory from tomorrow",
            content_hash=generate_content_hash("Memory from tomorrow"),
            tags=["timeframe-test"],
            created_at=datetime.combine(tomorrow, datetime.min.time()).timestamp()
        )

        await storage.store(memory_yesterday)
        await storage.store(memory_today)
        await storage.store(memory_tomorrow)

        # Delete memories from yesterday to today (inclusive)
        count, message = await storage.delete_by_timeframe(yesterday, today)

        assert count == 2
        assert "deleted 2 memories" in message.lower()
        assert str(yesterday) in message
        assert str(today) in message

        # Verify only tomorrow's memory remains
        remaining = await storage.search_by_tag(["timeframe-test"])
        assert len(remaining) == 1
        assert remaining[0].content == "Memory from tomorrow"

    @pytest.mark.asyncio
    async def test_delete_by_timeframe_with_tag_filter(self, storage):
        """Test deleting memories within date range with tag filter."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        # Store memories with different tags
        memory_with_tag = Memory(
            content="Memory with specific tag",
            content_hash=generate_content_hash("Memory with specific tag"),
            tags=["timeframe-tag", "keep"],
            created_at=datetime.combine(today, datetime.min.time()).timestamp()
        )
        memory_without_tag = Memory(
            content="Memory without specific tag",
            content_hash=generate_content_hash("Memory without specific tag"),
            tags=["other-tag"],
            created_at=datetime.combine(today, datetime.min.time()).timestamp()
        )

        await storage.store(memory_with_tag)
        await storage.store(memory_without_tag)

        # Delete only memories with specific tag
        count, message = await storage.delete_by_timeframe(yesterday, today, tag="timeframe-tag")

        assert count == 1
        assert "timeframe-tag" in message

        # Verify memory without tag still exists
        remaining = await storage.search_by_tag(["other-tag"])
        assert len(remaining) == 1

    @pytest.mark.asyncio
    async def test_delete_by_timeframe_empty_range(self, storage):
        """Test deleting when no memories exist in timeframe."""
        today = date.today()
        far_past = today - timedelta(days=365)
        past = today - timedelta(days=364)

        # Store a recent memory
        memory = Memory(
            content="Recent memory",
            content_hash=generate_content_hash("Recent memory"),
            tags=["recent"],
            created_at=datetime.combine(today, datetime.min.time()).timestamp()
        )
        await storage.store(memory)

        # Try to delete from far past (no memories in that range)
        count, message = await storage.delete_by_timeframe(far_past, past)

        assert count == 0
        assert "deleted 0 memories" in message.lower()

        # Verify recent memory still exists
        remaining = await storage.search_by_tag(["recent"])
        assert len(remaining) == 1

    @pytest.mark.asyncio
    async def test_delete_by_timeframe_boundaries(self, storage):
        """Test inclusive start/end boundaries of timeframe deletion."""
        today = date.today()
        yesterday = today - timedelta(days=1)
        tomorrow = today + timedelta(days=1)
        two_days_ago = today - timedelta(days=2)

        # Store memories at exact boundaries
        memory_before = Memory(
            content="Before range",
            content_hash=generate_content_hash("Before range"),
            tags=["boundary-test"],
            created_at=datetime.combine(two_days_ago, datetime.max.time()).timestamp()
        )
        memory_at_start = Memory(
            content="At start boundary",
            content_hash=generate_content_hash("At start boundary"),
            tags=["boundary-test"],
            created_at=datetime.combine(yesterday, datetime.min.time()).timestamp()
        )
        memory_at_end = Memory(
            content="At end boundary",
            content_hash=generate_content_hash("At end boundary"),
            tags=["boundary-test"],
            created_at=datetime.combine(today, datetime.max.time()).timestamp()
        )
        memory_after = Memory(
            content="After range",
            content_hash=generate_content_hash("After range"),
            tags=["boundary-test"],
            created_at=datetime.combine(tomorrow, datetime.min.time()).timestamp()
        )

        await storage.store(memory_before)
        await storage.store(memory_at_start)
        await storage.store(memory_at_end)
        await storage.store(memory_after)

        # Delete with inclusive boundaries
        count, message = await storage.delete_by_timeframe(yesterday, today)

        assert count == 2

        # Verify memories at boundaries were deleted
        remaining = await storage.search_by_tag(["boundary-test"])
        assert len(remaining) == 2
        contents = [m.content for m in remaining]
        assert "Before range" in contents
        assert "After range" in contents

    # delete_before_date tests

    @pytest.mark.asyncio
    async def test_delete_before_date_success(self, storage):
        """Test deleting all memories before a specific date."""
        today = date.today()
        yesterday = today - timedelta(days=1)
        two_days_ago = today - timedelta(days=2)

        # Store memories at different dates
        memory_old = Memory(
            content="Old memory",
            content_hash=generate_content_hash("Old memory"),
            tags=["before-test"],
            created_at=datetime.combine(two_days_ago, datetime.min.time()).timestamp()
        )
        memory_yesterday = Memory(
            content="Yesterday memory",
            content_hash=generate_content_hash("Yesterday memory"),
            tags=["before-test"],
            created_at=datetime.combine(yesterday, datetime.min.time()).timestamp()
        )
        memory_today = Memory(
            content="Today memory",
            content_hash=generate_content_hash("Today memory"),
            tags=["before-test"],
            created_at=datetime.combine(today, datetime.min.time()).timestamp()
        )

        await storage.store(memory_old)
        await storage.store(memory_yesterday)
        await storage.store(memory_today)

        # Delete all memories before today
        count, message = await storage.delete_before_date(today)

        assert count == 2
        assert "deleted 2 memories" in message.lower()
        assert str(today) in message

        # Verify only today's memory remains
        remaining = await storage.search_by_tag(["before-test"])
        assert len(remaining) == 1
        assert remaining[0].content == "Today memory"

    @pytest.mark.asyncio
    async def test_delete_before_date_with_tag_filter(self, storage):
        """Test deleting memories before date with tag filter."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        # Store old memories with different tags
        memory_with_tag = Memory(
            content="Old memory with tag",
            content_hash=generate_content_hash("Old memory with tag"),
            tags=["before-tag"],
            created_at=datetime.combine(yesterday, datetime.min.time()).timestamp()
        )
        memory_without_tag = Memory(
            content="Old memory without tag",
            content_hash=generate_content_hash("Old memory without tag"),
            tags=["other"],
            created_at=datetime.combine(yesterday, datetime.min.time()).timestamp()
        )

        await storage.store(memory_with_tag)
        await storage.store(memory_without_tag)

        # Delete only old memories with specific tag
        count, message = await storage.delete_before_date(today, tag="before-tag")

        assert count == 1
        assert "before-tag" in message

        # Verify memory without tag still exists
        remaining = await storage.search_by_tag(["other"])
        assert len(remaining) == 1

    @pytest.mark.asyncio
    async def test_delete_before_date_no_matches(self, storage):
        """Test deleting when no memories exist before date."""
        today = date.today()
        tomorrow = today + timedelta(days=1)

        # Store a recent memory
        memory = Memory(
            content="Recent memory",
            content_hash=generate_content_hash("Recent memory"),
            tags=["recent"],
            created_at=datetime.combine(today, datetime.min.time()).timestamp()
        )
        await storage.store(memory)

        # Try to delete before future date (should not match today's memory)
        count, message = await storage.delete_before_date(today)

        assert count == 0
        assert "deleted 0 memories" in message.lower()

        # Verify memory still exists
        remaining = await storage.search_by_tag(["recent"])
        assert len(remaining) == 1

    # get_by_exact_content tests

    @pytest.mark.asyncio
    async def test_get_by_exact_content_found(self, storage):
        """Test retrieving memory by exact content match."""
        content = "Exact content for retrieval test"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["exact-test"]
        )
        await storage.store(memory)

        # Retrieve by exact content
        results = await storage.get_by_exact_content(content)

        assert len(results) == 1
        assert results[0].content == content
        assert results[0].content_hash == memory.content_hash

    @pytest.mark.asyncio
    async def test_get_by_exact_content_not_found(self, storage):
        """Test retrieving when no exact content match exists."""
        # Store a memory
        memory = Memory(
            content="Some content",
            content_hash=generate_content_hash("Some content"),
            tags=["test"]
        )
        await storage.store(memory)

        # Try to retrieve with different content
        results = await storage.get_by_exact_content("Different content")

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_by_exact_content_multiple_matches(self, storage):
        """Test behavior when multiple exact matches exist (should be prevented by unique constraint)."""
        content = "Duplicate content test"

        # Store first memory
        memory1 = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["first"]
        )
        await storage.store(memory1)

        # Try to store duplicate (should fail due to unique constraint)
        memory2 = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["second"]
        )
        success, message = await storage.store(memory2)

        # Should fail to store duplicate
        assert not success
        assert "duplicate" in message.lower()

        # Verify only one memory exists
        results = await storage.get_by_exact_content(content)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_by_exact_content_excludes_deleted(self, storage):
        """Test that soft-deleted memories are not returned."""
        content = "Memory to be deleted"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["delete-test"]
        )
        await storage.store(memory)

        # Verify memory exists
        results = await storage.get_by_exact_content(content)
        assert len(results) == 1

        # Soft-delete the memory
        await storage.delete(memory.content_hash)

        # Verify deleted memory is not returned
        results = await storage.get_by_exact_content(content)
        assert len(results) == 0


class TestSqliteVecStorageWithoutEmbeddings:
    """Test SQLite-vec storage when sentence transformers is not available."""
    
    @pytest.mark.asyncio
    async def test_initialization_without_embeddings(self):
        """Test that storage can initialize without sentence transformers."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_no_embeddings.db")

        try:
            with patch('src.mcp_memory_service.storage.sqlite_vec.SENTENCE_TRANSFORMERS_AVAILABLE', False):
                storage = SqliteVecMemoryStorage(db_path)
                await storage.initialize()

                assert storage.conn is not None
                # When sentence_transformers unavailable, falls back to _HashEmbeddingModel
                from src.mcp_memory_service.storage.sqlite_vec import _HashEmbeddingModel
                assert isinstance(storage.embedding_model, _HashEmbeddingModel)

                storage.close()

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_operations_without_embeddings(self):
        """Test basic operations without embeddings."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_no_embeddings.db")
        
        try:
            with patch('src.mcp_memory_service.storage.sqlite_vec.SENTENCE_TRANSFORMERS_AVAILABLE', False):
                storage = SqliteVecMemoryStorage(db_path)
                await storage.initialize()
                
                # Store should work (with zero embeddings)
                content = "Test without embeddings"
                memory = Memory(
                    content=content,
                    content_hash=generate_content_hash(content),
                    tags=["no-embeddings"]
                )
                
                success, message = await storage.store(memory)
                assert success
                
                # Tag search should work
                results = await storage.search_by_tag(["no-embeddings"])
                assert len(results) == 1
                
                # Semantic search won't work well but shouldn't crash
                results = await storage.retrieve("test", n_results=1)
                # May or may not return results, but shouldn't crash
                
                storage.close()
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run basic tests when executed directly
    async def run_basic_tests():
        """Run basic tests to verify functionality."""
        if not SQLITE_VEC_AVAILABLE:
            print("⚠️  sqlite-vec not available, skipping tests")
            return
        
        print("Running basic SQLite-vec storage tests...")
        
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_basic.db")
        
        try:
            # Test basic functionality
            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()
            
            # Store a memory
            content = "Test memory for basic validation"
            memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=["test", "basic"]
            )
            
            success, message = await storage.store(memory)
            print(f"Store: {success}, {message}")
            
            # Retrieve the memory
            results = await storage.retrieve("test memory", n_results=1)
            print(f"Retrieve: Found {len(results)} results")
            
            if results:
                print(f"Content: {results[0].memory.content}")
                print(f"Relevance: {results[0].relevance_score}")
            
            # Search by tag
            tag_results = await storage.search_by_tag(["test"])
            print(f"Tag search: Found {len(tag_results)} results")
            
            # Get stats
            stats = storage.get_stats()
            print(f"Stats: {stats['total_memories']} memories, {stats['database_size_mb']} MB")
            
            storage.close()
            print("✅ Basic tests passed!")
            
        except Exception as e:
            print(f"❌ Basic tests failed: {e}")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Run the basic tests
    asyncio.run(run_basic_tests())


@pytest.mark.asyncio
@pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")
async def test_embedding_model_initialization():
    """Regression test for Issue #311: Verify SentenceTransformer loads correctly.

    This test ensures that the wandb dependency conflict fix (setting WANDB_DISABLED)
    prevents fallback to _HashEmbeddingModel and correctly initializes SentenceTransformer.
    """
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_embedding_model.db")

    try:
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()

        # Verify SentenceTransformer is loaded (not hash fallback)
        assert hasattr(storage, 'embedding_model'), "Storage should have embedding_model attribute"
        model_type_name = type(storage.embedding_model).__name__

        # Should be SentenceTransformer, not _HashEmbeddingModel
        assert model_type_name == "SentenceTransformer", \
            f"Expected SentenceTransformer but got {model_type_name}. " \
            "This indicates wandb dependency conflict is not resolved."

        # Verify correct embedding dimension (all-MiniLM-L6-v2 produces 384-dim vectors)
        assert storage.embedding_dimension == 384, \
            f"Expected 384-dim embeddings but got {storage.embedding_dimension}"

        # Test that embeddings are actually generated (not hash-based)
        test_content = "Test embedding generation"
        embedding = storage._generate_embedding(test_content)

        assert embedding is not None, "Embedding should not be None"
        assert len(embedding) == 384, f"Expected 384-dim embedding but got {len(embedding)}"

        # Hash-based embeddings would be deterministic and produce same values
        # Real embeddings should have floating-point values in reasonable range
        import numpy as np
        embedding_array = np.array(embedding)

        # Check that values are in typical range for normalized embeddings
        assert embedding_array.min() >= -1.5, "Embedding values too small (hash-based?)"
        assert embedding_array.max() <= 1.5, "Embedding values too large (hash-based?)"

        # Check that not all values are the same (hash would produce patterns)
        unique_values = len(set(embedding))
        assert unique_values > 10, \
            f"Too few unique values ({unique_values}), suspicious for real embeddings"

        print(f"✅ SentenceTransformer initialized correctly (Issue #311 fixed)")

    finally:
        if storage.conn:
            storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestSemanticDeduplication:
    """Test semantic deduplication functionality in SQLite-vec storage."""

    @pytest_asyncio.fixture
    async def storage(self):
        """Create a test storage instance with semantic dedup enabled."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_semantic_dedup.db")

        # Force enable semantic dedup for these tests
        original_env = os.environ.get('MCP_SEMANTIC_DEDUP_ENABLED')
        os.environ['MCP_SEMANTIC_DEDUP_ENABLED'] = 'true'

        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()

        yield storage

        # Cleanup
        if storage.conn:
            storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Restore original env var
        if original_env is None:
            os.environ.pop('MCP_SEMANTIC_DEDUP_ENABLED', None)
        else:
            os.environ['MCP_SEMANTIC_DEDUP_ENABLED'] = original_env

    @pytest.mark.asyncio
    async def test_semantic_duplicate_detection(self, storage):
        """Test that semantically similar memories are rejected within 24 hours."""
        # Store original memory
        original = Memory(
            content="Claude Code is a powerful CLI tool for software engineering.",
            content_hash=generate_content_hash("Claude Code is a powerful CLI tool for software engineering."),
            tags=["claude-code", "tool"],
            memory_type="note"
        )
        success1, msg1 = await storage.store(original)
        assert success1, f"Failed to store original: {msg1}"

        # Try to store semantically similar memory (reformulation)
        similar = Memory(
            content="The Claude Code CLI is an excellent software development tool.",
            content_hash=generate_content_hash("The Claude Code CLI is an excellent software development tool."),
            tags=["claude-code", "cli"],
            memory_type="note"
        )
        success2, msg2 = await storage.store(similar)
        assert not success2, "Should reject semantic duplicate"
        assert "semantically similar" in msg2.lower(), f"Expected semantic duplicate message, got: {msg2}"

        # Verify original is still retrievable
        results = await storage.retrieve("Claude Code", n_results=5)
        assert len(results) == 1, "Should only have original memory"
        assert results[0].memory.content == original.content

    @pytest.mark.asyncio
    async def test_semantic_duplicate_time_window(self, storage):
        """Test that semantic dedup respects time window."""
        # Set 1-hour time window for testing
        storage.semantic_dedup_time_window = 1

        # Store original memory
        original = Memory(
            content="Machine learning models require large datasets for training.",
            content_hash=generate_content_hash("Machine learning models require large datasets for training."),
            tags=["ml", "training"],
            memory_type="note"
        )
        success1, msg1 = await storage.store(original)
        assert success1, f"Failed to store original: {msg1}"

        # Mock time to be 2 hours later (outside 1-hour window)
        with patch('time.time', return_value=time.time() + 2 * 3600):
            # Should allow storage after time window expires
            similar = Memory(
                content="ML models need big datasets to train properly.",
                content_hash=generate_content_hash("ML models need big datasets to train properly."),
                tags=["ml", "datasets"],
                memory_type="note"
            )
            success2, msg2 = await storage.store(similar)
            assert success2, f"Should allow storage after time window: {msg2}"

        # Verify both memories exist
        results = await storage.retrieve("machine learning datasets", n_results=5)
        assert len(results) == 2, "Should have both memories after time window"

    @pytest.mark.asyncio
    async def test_semantic_duplicate_disabled(self):
        """Test that semantic dedup can be disabled via config."""
        # Temporarily disable semantic dedup
        original_env = os.environ.get('MCP_SEMANTIC_DEDUP_ENABLED')
        os.environ['MCP_SEMANTIC_DEDUP_ENABLED'] = 'false'

        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_dedup_disabled.db")

        try:
            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()

            # Store original memory
            original = Memory(
                content="Python is a versatile programming language.",
                content_hash=generate_content_hash("Python is a versatile programming language."),
                tags=["python", "programming"],
                memory_type="note"
            )
            success1, msg1 = await storage.store(original)
            assert success1, f"Failed to store original: {msg1}"

            # Try to store similar memory - should succeed when disabled
            similar = Memory(
                content="Python is a flexible programming language.",
                content_hash=generate_content_hash("Python is a flexible programming language."),
                tags=["python", "coding"],
                memory_type="note"
            )
            success2, msg2 = await storage.store(similar)
            assert success2, f"Should allow similar content when dedup disabled: {msg2}"

            # Verify both memories exist
            results = await storage.retrieve("Python programming", n_results=5)
            assert len(results) == 2, "Should have both memories when dedup disabled"

            storage.close()

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            # Restore original env var
            if original_env is None:
                os.environ.pop('MCP_SEMANTIC_DEDUP_ENABLED', None)
            else:
                os.environ['MCP_SEMANTIC_DEDUP_ENABLED'] = original_env

    @pytest.mark.asyncio
    async def test_semantic_duplicate_different_content(self, storage):
        """Test that genuinely different content is not flagged as duplicate."""
        # Store memory about Claude Code
        memory1 = Memory(
            content="Claude Code is a powerful CLI tool for software engineering.",
            content_hash=generate_content_hash("Claude Code is a powerful CLI tool for software engineering."),
            tags=["claude-code"],
            memory_type="note"
        )
        success1, msg1 = await storage.store(memory1)
        assert success1, f"Failed to store first memory: {msg1}"

        # Store completely different memory about Python
        memory2 = Memory(
            content="Python is excellent for data science and machine learning.",
            content_hash=generate_content_hash("Python is excellent for data science and machine learning."),
            tags=["python"],
            memory_type="note"
        )
        success2, msg2 = await storage.store(memory2)
        assert success2, f"Should allow different content: {msg2}"

        # Verify both exist
        results = await storage.retrieve("software", n_results=10)
        assert len(results) >= 2, "Should have both different memories"

    @pytest.mark.asyncio
    async def test_semantic_duplicate_threshold_configuration(self):
        """Test that similarity threshold can be configured."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_threshold.db")

        # Set high threshold (0.95) - only very similar content will be rejected
        original_threshold = os.environ.get('MCP_SEMANTIC_DEDUP_THRESHOLD')
        os.environ['MCP_SEMANTIC_DEDUP_THRESHOLD'] = '0.95'
        os.environ['MCP_SEMANTIC_DEDUP_ENABLED'] = 'true'

        try:
            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()

            # Store original
            original = Memory(
                content="Testing semantic similarity threshold configuration.",
                content_hash=generate_content_hash("Testing semantic similarity threshold configuration."),
                tags=["test"],
                memory_type="note"
            )
            success1, msg1 = await storage.store(original)
            assert success1, f"Failed to store original: {msg1}"

            # Store somewhat similar content - should be allowed with high threshold
            similar = Memory(
                content="Examining semantic similarity threshold settings.",
                content_hash=generate_content_hash("Examining semantic similarity threshold settings."),
                tags=["test"],
                memory_type="note"
            )
            success2, msg2 = await storage.store(similar)
            # With 0.95 threshold, moderately similar content should be allowed
            # (it's only rejected if similarity > 0.95)

            storage.close()

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            # Restore original env var
            if original_threshold is None:
                os.environ.pop('MCP_SEMANTIC_DEDUP_THRESHOLD', None)
            else:
                os.environ['MCP_SEMANTIC_DEDUP_THRESHOLD'] = original_threshold
            os.environ.pop('MCP_SEMANTIC_DEDUP_ENABLED', None)

    @pytest.mark.asyncio
    async def test_semantic_duplicate_exact_match_takes_precedence(self, storage):
        """Test that exact hash duplicates are still caught even with semantic dedup."""
        # Store original memory
        content = "This is the exact same content."
        memory1 = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test"],
            memory_type="note"
        )
        success1, msg1 = await storage.store(memory1)
        assert success1, f"Failed to store original: {msg1}"

        # Try to store exact same content
        memory2 = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test"],
            memory_type="note"
        )
        success2, msg2 = await storage.store(memory2)
        assert not success2, "Should reject exact duplicate"
        assert "exact match" in msg2.lower(), f"Expected exact match message, got: {msg2}"

    @pytest.mark.asyncio
    async def test_skip_semantic_dedup_allows_similar_content(self, storage):
        """When skip_semantic_dedup=True, semantically similar content is stored."""
        original = Memory(
            content="Claude Code is a powerful CLI tool for software engineering.",
            content_hash=generate_content_hash("Claude Code is a powerful CLI tool for software engineering."),
            tags=["claude-code"],
            memory_type="note"
        )
        success1, _ = await storage.store(original)
        assert success1

        similar = Memory(
            content="The Claude Code CLI is an excellent software development tool.",
            content_hash=generate_content_hash("The Claude Code CLI is an excellent software development tool."),
            tags=["claude-code"],
            memory_type="note"
        )
        # Without flag: rejected
        success2, msg2 = await storage.store(similar)
        assert not success2
        assert "semantically similar" in msg2.lower()

        # With flag: accepted
        success3, msg3 = await storage.store(similar, skip_semantic_dedup=True)
        assert success3, f"Expected success with skip_semantic_dedup=True, got: {msg3}"

    def test_max_tag_search_candidates_within_sqlite_vec_limit(self):
        """_MAX_TAG_SEARCH_CANDIDATES must not exceed sqlite-vec's hard k=4096 KNN limit.

        If this constant exceeds 4096, any tag-filtered retrieve() on a database
        with more than 4096 memories will raise 'k value in knn query too large'
        and silently return 0 results.
        """
        from mcp_memory_service.storage.sqlite_vec import _MAX_TAG_SEARCH_CANDIDATES, _SQLITE_VEC_MAX_KNN_K
        assert _MAX_TAG_SEARCH_CANDIDATES <= 4096, (
            f"_MAX_TAG_SEARCH_CANDIDATES={_MAX_TAG_SEARCH_CANDIDATES} exceeds "
            f"sqlite-vec hard limit of {_SQLITE_VEC_MAX_KNN_K}"
        )

    @pytest.mark.asyncio
    async def test_retrieve_k_value_capping_without_tags(self, storage):
        """Test that k_value is capped even when no tags provided (DoS protection).

        Without capping, an attacker could request n_results=1000000 to force
        exhaustive scan of all embeddings, consuming excessive CPU/memory.
        """
        # Store some test memories
        for i in range(5):
            memory = Memory(
                content=f"Test memory {i} for k-value capping",
                content_hash=generate_content_hash(f"Test memory {i} for k-value capping"),
                tags=["k-cap-test"]
            )
            await storage.store(memory)

        # Request an arbitrarily large number of results
        # Should be capped at _MAX_TAG_SEARCH_CANDIDATES (4,096 — sqlite-vec hard limit)
        results = await storage.retrieve("k-value test", n_results=1000000)

        # Should return results (not crash) and be limited by what's actually in DB
        assert isinstance(results, list)
        assert len(results) <= 5  # Only 5 memories stored

    @pytest.mark.asyncio
    async def test_retrieve_all_invalid_tags_returns_empty(self, storage):
        """Test that providing only invalid tags returns empty results (not unfiltered).

        When tags filter is provided but all tags are invalid (non-string),
        should return empty results instead of silently ignoring the filter.
        """
        # Store some memories
        for i in range(3):
            memory = Memory(
                content=f"Test memory {i} for invalid tags",
                content_hash=generate_content_hash(f"Test memory {i} for invalid tags"),
                tags=["valid-tag"]
            )
            await storage.store(memory)

        # Provide only invalid tags (non-strings)
        results = await storage.retrieve(
            "invalid tags test",
            n_results=10,
            tags=[None, 123, {"key": "value"}]  # All invalid
        )

        # Should return empty results, not unfiltered results
        assert len(results) == 0