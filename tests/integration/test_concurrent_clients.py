"""
Integration tests for concurrent MCP client access to SQLite-vec backend.
"""

import asyncio
import os
import tempfile

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


class TestConcurrentClients:
    """Test suite for concurrent client access scenarios."""

    @pytest_asyncio.fixture
    async def db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            path = tmp.name
        yield path
        # Cleanup
        for ext in ["", "-wal", "-shm"]:
            try:
                os.unlink(path + ext)
            except:
                pass

    async def create_client(self, db_path: str, client_id: str) -> SqliteVecMemoryStorage:
        """Create a storage client instance."""
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()
        return storage

    async def client_writer(self, db_path: str, client_id: str, num_memories: int) -> list[tuple]:
        """Simulate a client writing memories."""
        storage = await self.create_client(db_path, client_id)
        results = []

        try:
            for i in range(num_memories):
                memory = Memory(
                    content=f"Memory from {client_id} - {i}",
                    tags=[client_id, "concurrent"],
                    memory_type="test",
                    metadata={"client_id": client_id, "index": i},
                )

                success, msg = await storage.store(memory)
                results.append((success, msg, memory.content_hash))

                # Small delay to simulate real-world timing
                await asyncio.sleep(0.01)
        finally:
            storage.close()

        return results

    async def client_reader(self, db_path: str, client_id: str, num_reads: int) -> list[int]:
        """Simulate a client reading memories."""
        storage = await self.create_client(db_path, client_id)
        counts = []

        try:
            for _i in range(num_reads):
                # Get all memories with tag "concurrent"
                memories = await storage.search_by_tag(["concurrent"])
                counts.append(len(memories))

                # Small delay between reads
                await asyncio.sleep(0.02)
        finally:
            storage.close()

        return counts

    @pytest.mark.asyncio
    async def test_two_clients_concurrent_write(self, db_path):
        """Test two clients writing memories concurrently."""
        # Run two clients concurrently
        results = await asyncio.gather(
            self.client_writer(db_path, "client1", 10), self.client_writer(db_path, "client2", 10), return_exceptions=True
        )

        # Check results
        assert len(results) == 2
        assert not isinstance(results[0], Exception), f"Client 1 failed: {results[0]}"
        assert not isinstance(results[1], Exception), f"Client 2 failed: {results[1]}"

        client1_results, client2_results = results

        # Count successful writes
        client1_success = sum(1 for success, _, _ in client1_results if success)
        client2_success = sum(1 for success, _, _ in client2_results if success)

        # Both clients should have written their memories
        assert client1_success == 10, f"Client 1 only wrote {client1_success}/10 memories"
        assert client2_success == 10, f"Client 2 only wrote {client2_success}/10 memories"

        # Verify total memories in database
        storage = await self.create_client(db_path, "verifier")
        try:
            all_memories = await storage.search_by_tag(["concurrent"])
            assert len(all_memories) == 20, f"Expected 20 memories, found {len(all_memories)}"
        finally:
            storage.close()

    @pytest.mark.asyncio
    async def test_reader_writer_concurrent(self, db_path):
        """Test one client reading while another writes."""
        # Start with some initial data
        initial_storage = await self.create_client(db_path, "initial")
        for i in range(5):
            memory = Memory(content=f"Initial memory {i}", tags=["concurrent", "initial"], memory_type="test")
            await initial_storage.store(memory)
        initial_storage.close()

        # Run reader and writer concurrently
        results = await asyncio.gather(
            self.client_reader(db_path, "reader", 10), self.client_writer(db_path, "writer", 5), return_exceptions=True
        )

        assert not isinstance(results[0], Exception), f"Reader failed: {results[0]}"
        assert not isinstance(results[1], Exception), f"Writer failed: {results[1]}"

        read_counts, write_results = results

        # Reader should see increasing counts as writer adds memories
        assert read_counts[0] >= 5, "Reader should see initial memories"
        assert read_counts[-1] >= read_counts[0], "Read count should not decrease"

        # Writer should succeed
        write_success = sum(1 for success, _, _ in write_results if success)
        assert write_success == 5, f"Writer only wrote {write_success}/5 memories"

    @pytest.mark.asyncio
    async def test_multiple_readers_one_writer(self, db_path):
        """Test multiple readers accessing while one writer updates."""

        # Create writer and multiple readers
        async def run_scenario():
            tasks = [
                self.client_writer(db_path, "writer", 10),
                self.client_reader(db_path, "reader1", 5),
                self.client_reader(db_path, "reader2", 5),
                self.client_reader(db_path, "reader3", 5),
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = await run_scenario()

        # Check all operations completed without exceptions
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Task {i} failed: {result}"

        write_results = results[0]
        write_success = sum(1 for success, _, _ in write_results if success)
        assert write_success == 10, f"Writer only wrote {write_success}/10 memories"

        # All readers should have successfully read
        for reader_counts in results[1:]:
            assert len(reader_counts) == 5, "Reader should complete all reads"
            assert all(isinstance(count, int) for count in reader_counts)

    @pytest.mark.asyncio
    async def test_rapid_concurrent_access(self, db_path):
        """Test rapid concurrent access with minimal delays."""

        async def rapid_writer(client_id: str):
            storage = await self.create_client(db_path, client_id)
            try:
                for i in range(20):
                    memory = Memory(content=f"Rapid {client_id}-{i}", tags=["rapid"], memory_type="test")
                    await storage.store(memory)
                    # No delay - rapid fire
            finally:
                storage.close()

        async def rapid_reader(client_id: str):
            storage = await self.create_client(db_path, client_id)
            try:
                for _i in range(20):
                    await storage.search_by_tag(["rapid"])
                    # No delay - rapid fire
            finally:
                storage.close()

        # Run multiple clients with rapid access
        tasks = [
            rapid_writer("w1"),
            rapid_writer("w2"),
            rapid_reader("r1"),
            rapid_reader("r2"),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check no exceptions occurred
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"

    @pytest.mark.asyncio
    async def test_database_consistency(self, db_path):
        """Test that database remains consistent under concurrent access."""

        # Write unique memories from multiple clients
        async def write_unique_memories(client_id: str, start_idx: int):
            storage = await self.create_client(db_path, client_id)
            hashes = []
            try:
                for i in range(10):
                    memory = Memory(
                        content=f"Unique memory {start_idx + i}", tags=["consistency", client_id], memory_type="test"
                    )
                    success, msg = await storage.store(memory)
                    if success:
                        hashes.append(memory.content_hash)
            finally:
                storage.close()
            return hashes

        # Run concurrent writes
        results = await asyncio.gather(
            write_unique_memories("client1", 0),
            write_unique_memories("client2", 100),
            write_unique_memories("client3", 200),
        )

        all_hashes = []
        for client_hashes in results:
            all_hashes.extend(client_hashes)

        # Verify all memories are in database and no duplicates
        storage = await self.create_client(db_path, "verifier")
        try:
            memories = await storage.search_by_tag(["consistency"])

            # Check count matches
            assert len(memories) == len(all_hashes), f"Memory count mismatch: {len(memories)} vs {len(all_hashes)}"

            # Check no duplicates
            db_hashes = {m.content_hash for m in memories}
            assert len(db_hashes) == len(memories), "Duplicate memories found"

            # Check all written memories are present
            for hash_val in all_hashes:
                assert hash_val in db_hashes, f"Memory {hash_val} not found in database"
        finally:
            storage.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
