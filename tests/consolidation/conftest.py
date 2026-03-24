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
            # Base ontology types (from Phase 0 Ontology Foundation)
            'decision': 365,      # Strategic choices, architecture decisions
            'learning': 180,      # Insights, best practices, anti-patterns
            'pattern': 90,        # Recurring issues, code smells, design patterns
            'error': 30,          # Bugs, failures, exceptions
            'observation': 30,    # Code edits, file access, searches, commands

            # Legacy types for backward compatibility
            'critical': 365,      # Maps to decision
            'reference': 180,     # Maps to learning
            'standard': 30,       # Maps to observation
            'temporary': 7        # Maps to observation (short-lived)
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
    
    def _mem_ts(offset):
        """Helper: return created_at/updated_at kwargs for a memory `offset` seconds old."""
        ts = base_time - offset
        iso = datetime.fromtimestamp(ts).isoformat() + 'Z'
        return dict(created_at=ts, created_at_iso=iso, updated_at=ts, updated_at_iso=iso)

    memories = [
        # Recent critical memory
        Memory(
            content="Critical system configuration backup completed successfully",
            content_hash="hash001",
            tags=["critical", "backup", "system"],
            memory_type="decision",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5] * 64,
            metadata={"importance_score": 2.0},
            **_mem_ts(86400),  # 1 day ago
        ),

        # Related system memory
        Memory(
            content="System configuration updated with new security settings",
            content_hash="hash002",
            tags=["system", "security", "config"],
            memory_type="observation",
            embedding=[0.15, 0.25, 0.35, 0.45, 0.55] * 64,
            metadata={},
            **_mem_ts(172800),  # 2 days ago
        ),

        # Unrelated old memory
        Memory(
            content="Weather is nice today, went for a walk in the park",
            content_hash="hash003",
            tags=["personal", "weather"],
            memory_type="observation",
            embedding=[0.9, 0.8, 0.7, 0.6, 0.5] * 64,
            metadata={},
            **_mem_ts(259200),  # 3 days ago
        ),

        # Reference memory
        Memory(
            content="Python documentation: List comprehensions provide concise syntax",
            content_hash="hash004",
            tags=["reference", "python", "documentation"],
            memory_type="learning",
            embedding=[0.2, 0.3, 0.4, 0.5, 0.6] * 64,
            metadata={"importance_score": 1.5},
            **_mem_ts(604800),  # 1 week ago
        ),

        # Related programming memory
        Memory(
            content="Python best practices: Use list comprehensions for simple transformations",
            content_hash="hash005",
            tags=["python", "best-practices", "programming"],
            memory_type="observation",
            embedding=[0.25, 0.35, 0.45, 0.55, 0.65] * 64,
            metadata={},
            **_mem_ts(691200),  # 8 days ago
        ),

        # Old low-quality memory
        Memory(
            content="test test test",
            content_hash="hash006",
            tags=["test"],
            memory_type="observation",
            embedding=[0.1, 0.1, 0.1, 0.1, 0.1] * 64,
            metadata={},
            **_mem_ts(2592000),  # 30 days ago
        ),

        # Another programming memory for clustering
        Memory(
            content="JavaScript arrow functions provide cleaner syntax for callbacks",
            content_hash="hash007",
            tags=["javascript", "programming", "syntax"],
            memory_type="observation",
            embedding=[0.3, 0.4, 0.5, 0.6, 0.7] * 64,
            metadata={},
            **_mem_ts(777600),  # 9 days ago
        ),

        # Duplicate-like memory
        Memory(
            content="test test test duplicate",
            content_hash="hash008",
            tags=["test", "duplicate"],
            memory_type="observation",
            embedding=[0.11, 0.11, 0.11, 0.11, 0.11] * 64,
            metadata={},
            **_mem_ts(2678400),  # 31 days ago
        ),

        # Very old memory to ensure low relevance score for testing
        Memory(
            content="Old observation from long ago",
            content_hash="hash009",
            tags=["old", "archived"],
            memory_type="observation",
            embedding=[0.05, 0.05, 0.05, 0.05, 0.05] * 64,
            metadata={},
            **_mem_ts(7776000),  # 90 days ago
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

        async def update_memories_batch(self, memories: List[Memory], preserve_timestamps: bool = False) -> List[bool]:
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
            memory_type = "learning" if i % 5 == 0 else "observation"  # Changed to valid ontology types
        elif i < 60:  # Second cluster - personal content  
            base_embedding = [0.6, 0.7, 0.8, 0.9, 1.0]
            tags = ["personal", "notes"]
            memory_type = "observation"  # Changed to valid ontology type
        elif i < 90:  # Third cluster - work content
            base_embedding = [0.2, 0.4, 0.6, 0.8, 1.0]
            tags = ["work", "project"]
            memory_type = "observation"  # Changed to valid ontology type
        else:  # Outliers
            base_embedding = [np.random.random() for _ in range(5)]
            tags = ["misc"]
            memory_type = "observation"  # Changed to valid ontology type
        
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

        async def update_memories_batch(self, memories: List[Memory], preserve_timestamps: bool = False) -> List[bool]:
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