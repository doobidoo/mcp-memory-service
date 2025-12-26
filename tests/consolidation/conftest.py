"""Test fixtures for consolidation tests."""

import pytest
import tempfile
import shutil
import os
from datetime import datetime, timedelta
from typing import List
import numpy as np
from unittest.mock import AsyncMock

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.consolidation.base import ConsolidationConfig


@pytest.fixture
def temp_archive_path():
    """Create a temporary directory for consolidation archives."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def consolidation_config(temp_archive_path):
    """Create a test consolidation configuration."""
    return ConsolidationConfig(
        # Decay settings
        decay_enabled=True,
        retention_periods={
            'critical': 365,
            'reference': 180,
            'standard': 30,
            'temporary': 7
        },
        
        # Association settings
        associations_enabled=True,
        min_similarity=0.3,
        max_similarity=0.7,
        max_pairs_per_run=50,  # Smaller for tests
        
        # Clustering settings
        clustering_enabled=True,
        min_cluster_size=3,  # Smaller for tests
        clustering_algorithm='simple',  # Use simple for tests (no sklearn dependency)
        
        # Compression settings
        compression_enabled=True,
        max_summary_length=200,  # Shorter for tests
        preserve_originals=True,
        
        # Forgetting settings
        forgetting_enabled=True,
        relevance_threshold=0.1,
        access_threshold_days=30,  # Shorter for tests
        archive_location=temp_archive_path
    )


@pytest.fixture
def sample_memories():
    """Create a sample set of memories for testing."""
    base_time = datetime.now().timestamp()
    
    memories = [
        # Recent critical memory
        Memory(
            content="Critical system configuration backup completed successfully",
            content_hash="hash001",
            tags=["critical", "backup", "system"],
            memory_type="critical",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5] * 64,  # 320-dim embedding
            metadata={"importance_score": 2.0},
            created_at=base_time - 86400,  # 1 day ago
            created_at_iso=datetime.fromtimestamp(base_time - 86400).isoformat() + 'Z'
        ),
        
        # Related system memory
        Memory(
            content="System configuration updated with new security settings",
            content_hash="hash002",
            tags=["system", "security", "config"],
            memory_type="standard",
            embedding=[0.15, 0.25, 0.35, 0.45, 0.55] * 64,  # Similar embedding
            metadata={},
            created_at=base_time - 172800,  # 2 days ago
            created_at_iso=datetime.fromtimestamp(base_time - 172800).isoformat() + 'Z'
        ),
        
        # Unrelated old memory
        Memory(
            content="Weather is nice today, went for a walk in the park",
            content_hash="hash003",
            tags=["personal", "weather"],
            memory_type="temporary",
            embedding=[0.9, 0.8, 0.7, 0.6, 0.5] * 64,  # Different embedding
            metadata={},
            created_at=base_time - 259200,  # 3 days ago
            created_at_iso=datetime.fromtimestamp(base_time - 259200).isoformat() + 'Z'
        ),
        
        # Reference memory
        Memory(
            content="Python documentation: List comprehensions provide concise syntax",
            content_hash="hash004",
            tags=["reference", "python", "documentation"],
            memory_type="reference",
            embedding=[0.2, 0.3, 0.4, 0.5, 0.6] * 64,
            metadata={"importance_score": 1.5},
            created_at=base_time - 604800,  # 1 week ago
            created_at_iso=datetime.fromtimestamp(base_time - 604800).isoformat() + 'Z'
        ),
        
        # Related programming memory
        Memory(
            content="Python best practices: Use list comprehensions for simple transformations",
            content_hash="hash005",
            tags=["python", "best-practices", "programming"],
            memory_type="standard",
            embedding=[0.25, 0.35, 0.45, 0.55, 0.65] * 64,  # Related to reference
            metadata={},
            created_at=base_time - 691200,  # 8 days ago
            created_at_iso=datetime.fromtimestamp(base_time - 691200).isoformat() + 'Z'
        ),
        
        # Old low-quality memory
        Memory(
            content="test test test",
            content_hash="hash006",
            tags=["test"],
            memory_type="temporary",
            embedding=[0.1, 0.1, 0.1, 0.1, 0.1] * 64,
            metadata={},
            created_at=base_time - 2592000,  # 30 days ago
            created_at_iso=datetime.fromtimestamp(base_time - 2592000).isoformat() + 'Z'
        ),
        
        # Another programming memory for clustering
        Memory(
            content="JavaScript arrow functions provide cleaner syntax for callbacks",
            content_hash="hash007",
            tags=["javascript", "programming", "syntax"],
            memory_type="standard",
            embedding=[0.3, 0.4, 0.5, 0.6, 0.7] * 64,  # Related to other programming
            metadata={},
            created_at=base_time - 777600,  # 9 days ago
            created_at_iso=datetime.fromtimestamp(base_time - 777600).isoformat() + 'Z'
        ),
        
        # Duplicate-like memory
        Memory(
            content="test test test duplicate",
            content_hash="hash008",
            tags=["test", "duplicate"],
            memory_type="temporary",
            embedding=[0.11, 0.11, 0.11, 0.11, 0.11] * 64,  # Very similar to hash006
            metadata={},
            created_at=base_time - 2678400,  # 31 days ago
            created_at_iso=datetime.fromtimestamp(base_time - 2678400).isoformat() + 'Z'
        )
    ]
    
    return memories


@pytest.fixture
def mock_storage(sample_memories):
    """Create a mock storage backend for testing."""
    
    class MockStorage:
        def __init__(self):
            self.memories = {mem.content_hash: mem for mem in sample_memories}
            self.connections = {
                "hash001": 2,  # Critical memory has connections
                "hash002": 1,  # System memory has some connections
                "hash004": 3,  # Reference memory is well-connected
                "hash005": 2,  # Programming memory has connections
                "hash007": 1,  # JavaScript memory has some connections
            }
            self.access_patterns = {
                "hash001": datetime.now() - timedelta(hours=6),  # Recently accessed
                "hash004": datetime.now() - timedelta(days=2),   # Accessed 2 days ago
                "hash002": datetime.now() - timedelta(days=5),   # Accessed 5 days ago
            }


        async def get_all_memories(self) -> List[Memory]:
            return list(self.memories.values())

        async def get_memories_by_time_range(self, start_time: float, end_time: float) -> List[Memory]:
            return [
                mem for mem in self.memories.values()
                if mem.created_at and start_time <= mem.created_at <= end_time
            ]

        async def store_memory(self, memory: Memory) -> bool:
            self.memories[memory.content_hash] = memory
            return True

        async def store(self, memory: Memory):
            """Store method that returns tuple (success, hash) for consolidator."""
            self.memories[memory.content_hash] = memory
            return (True, memory.content_hash)

        async def update_memory(self, memory: Memory) -> bool:
            if memory.content_hash in self.memories:
                self.memories[memory.content_hash] = memory
                return True
            return False

        async def update_memories_batch(self, memories: List[Memory]) -> List[bool]:
            """Batch update memories and return list of success statuses."""
            results = []
            for memory in memories:
                success = await self.update_memory(memory)
                results.append(success)
            return results

        async def delete_memory(self, content_hash: str) -> bool:
            if content_hash in self.memories:
                del self.memories[content_hash]
                return True
            return False

        async def get_memory_connections(self):
            return self.connections

        async def get_access_patterns(self):
            return self.access_patterns
    
    return MockStorage()


@pytest.fixture
def large_memory_set():
    """Create a larger set of memories for performance testing."""
    base_time = datetime.now().timestamp()
    memories = []
    
    # Create 100 memories with various patterns
    for i in range(100):
        # Create embeddings with some clustering patterns
        if i < 30:  # First cluster - technical content
            base_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
            tags = ["technical", "programming"]
            memory_type = "reference" if i % 5 == 0 else "standard"
        elif i < 60:  # Second cluster - personal content  
            base_embedding = [0.6, 0.7, 0.8, 0.9, 1.0]
            tags = ["personal", "notes"]
            memory_type = "standard"
        elif i < 90:  # Third cluster - work content
            base_embedding = [0.2, 0.4, 0.6, 0.8, 1.0]
            tags = ["work", "project"]
            memory_type = "standard"
        else:  # Outliers
            base_embedding = [np.random.random() for _ in range(5)]
            tags = ["misc"]
            memory_type = "temporary"
        
        # Add noise to embeddings
        embedding = []
        for val in base_embedding * 64:  # 320-dim
            noise = np.random.normal(0, 0.1)
            embedding.append(max(0, min(1, val + noise)))
        
        memory = Memory(
            content=f"Test memory content {i} with some meaningful text about the topic",
            content_hash=f"hash{i:03d}",
            tags=tags + [f"item{i}"],
            memory_type=memory_type,
            embedding=embedding,
            metadata={"test_id": i},
            created_at=base_time - (i * 3600),  # Spread over time
            created_at_iso=datetime.fromtimestamp(base_time - (i * 3600)).isoformat() + 'Z'
        )
        memories.append(memory)
    
    return memories


@pytest.fixture
def mock_large_storage(large_memory_set):
    """Create a mock storage with large memory set."""
    
    class MockLargeStorage:
        def __init__(self):
            self.memories = {mem.content_hash: mem for mem in large_memory_set}
            # Generate some random connections
            self.connections = {}
            for mem in large_memory_set[:50]:  # Half have connections
                self.connections[mem.content_hash] = np.random.randint(0, 5)

            # Generate random access patterns
            self.access_patterns = {}
            for mem in large_memory_set[:30]:  # Some have recent access
                days_ago = np.random.randint(1, 30)
                self.access_patterns[mem.content_hash] = datetime.now() - timedelta(days=days_ago)

        async def get_all_memories(self) -> List[Memory]:
            return list(self.memories.values())

        async def get_memories_by_time_range(self, start_time: float, end_time: float) -> List[Memory]:
            return [
                mem for mem in self.memories.values()
                if mem.created_at and start_time <= mem.created_at <= end_time
            ]

        async def store_memory(self, memory: Memory) -> bool:
            self.memories[memory.content_hash] = memory
            return True

        async def store(self, memory: Memory):
            """Store method that returns tuple (success, hash) for consolidator."""
            self.memories[memory.content_hash] = memory
            return (True, memory.content_hash)

        async def update_memory(self, memory: Memory) -> bool:
            if memory.content_hash in self.memories:
                self.memories[memory.content_hash] = memory
                return True
            return False

        async def update_memories_batch(self, memories: List[Memory]) -> List[bool]:
            """Batch update memories and return list of success statuses."""
            results = []
            for memory in memories:
                success = await self.update_memory(memory)
                results.append(success)
            return results

        async def delete_memory(self, content_hash: str) -> bool:
            if content_hash in self.memories:
                del self.memories[content_hash]
                return True
            return False

        async def get_memory_connections(self):
            return self.connections

        async def get_access_patterns(self):
            return self.access_patterns
    
    return MockLargeStorage()