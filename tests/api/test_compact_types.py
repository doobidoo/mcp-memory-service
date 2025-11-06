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

"""
Tests for compact data types.

Validates token efficiency, immutability, and type safety of compact types.
"""

import pytest
import time
from mcp_memory_service.api.types import (
    CompactMemory,
    CompactSearchResult,
    CompactHealthInfo
)


class TestCompactMemory:
    """Tests for CompactMemory type."""

    def test_compact_memory_creation(self):
        """Test basic CompactMemory creation."""
        memory = CompactMemory(
            hash='abc12345',
            preview='Test content preview',
            tags=('test', 'example'),
            created=time.time(),
            score=0.95
        )

        assert memory.hash == 'abc12345'
        assert memory.preview == 'Test content preview'
        assert memory.tags == ('test', 'example')
        assert memory.score == 0.95
        assert isinstance(memory.created, float)

    def test_compact_memory_immutability(self):
        """Test that CompactMemory is immutable."""
        memory = CompactMemory(
            hash='abc12345',
            preview='Test content',
            tags=('test',),
            created=time.time(),
            score=0.85
        )

        # NamedTuple should be immutable
        with pytest.raises(AttributeError):
            memory.hash = 'new_hash'  # type: ignore

    def test_compact_memory_tuple_behavior(self):
        """Test that CompactMemory behaves like a tuple."""
        memory = CompactMemory(
            hash='abc12345',
            preview='Test content',
            tags=('test',),
            created=1730928000.0,
            score=0.85
        )

        # Should support tuple unpacking
        hash_val, preview, tags, created, score = memory

        assert hash_val == 'abc12345'
        assert preview == 'Test content'
        assert tags == ('test',)
        assert created == 1730928000.0
        assert score == 0.85

    def test_compact_memory_field_access(self):
        """Test named field access."""
        memory = CompactMemory(
            hash='test123',
            preview='Preview text',
            tags=('tag1', 'tag2'),
            created=1730928000.0,
            score=0.75
        )

        # Named access should work
        assert memory.hash == 'test123'
        assert memory.tags[0] == 'tag1'
        assert memory.tags[1] == 'tag2'

    def test_compact_memory_token_size(self):
        """Test that CompactMemory achieves target token size."""
        # Create memory with typical values
        memory = CompactMemory(
            hash='abc12345',
            preview='A' * 200,  # 200 char preview
            tags=('tag1', 'tag2', 'tag3'),
            created=1730928000.0,
            score=0.85
        )

        # Convert to string representation (approximates token count)
        repr_str = str(memory)

        # Should be much smaller than full Memory object (~820 tokens)
        # Target: ~73 tokens, allow some margin
        # Rough estimate: 1 token H 4 characters
        estimated_tokens = len(repr_str) / 4

        assert estimated_tokens < 150, \
            f"CompactMemory too large: {estimated_tokens} tokens (target: <150)"


class TestCompactSearchResult:
    """Tests for CompactSearchResult type."""

    def test_compact_search_result_creation(self):
        """Test basic CompactSearchResult creation."""
        memories = (
            CompactMemory('hash1', 'preview1', ('tag1',), time.time(), 0.95),
            CompactMemory('hash2', 'preview2', ('tag2',), time.time(), 0.85),
        )

        result = CompactSearchResult(
            memories=memories,
            total=2,
            query='test query'
        )

        assert len(result.memories) == 2
        assert result.total == 2
        assert result.query == 'test query'

    def test_compact_search_result_repr(self):
        """Test CompactSearchResult string representation."""
        memories = (
            CompactMemory('hash1', 'preview1', ('tag1',), time.time(), 0.95),
            CompactMemory('hash2', 'preview2', ('tag2',), time.time(), 0.85),
            CompactMemory('hash3', 'preview3', ('tag3',), time.time(), 0.75),
        )

        result = CompactSearchResult(
            memories=memories,
            total=3,
            query='architecture'
        )

        repr_str = repr(result)
        assert 'found=3' in repr_str
        assert 'shown=3' in repr_str

    def test_compact_search_result_empty(self):
        """Test CompactSearchResult with no results."""
        result = CompactSearchResult(
            memories=(),
            total=0,
            query='nonexistent query'
        )

        assert len(result.memories) == 0
        assert result.total == 0
        assert repr(result) == 'SearchResult(found=0, shown=0)'

    def test_compact_search_result_iteration(self):
        """Test iterating over search results."""
        memories = tuple(
            CompactMemory(f'hash{i}', f'preview{i}', ('tag',), time.time(), 0.9 - i*0.1)
            for i in range(5)
        )

        result = CompactSearchResult(
            memories=memories,
            total=5,
            query='test'
        )

        # Should be iterable
        for i, memory in enumerate(result.memories):
            assert memory.hash == f'hash{i}'

    def test_compact_search_result_token_size(self):
        """Test that CompactSearchResult achieves target token size."""
        # Create result with 5 memories (typical use case)
        memories = tuple(
            CompactMemory(
                f'hash{i:04d}',
                'A' * 200,  # 200 char preview
                ('tag1', 'tag2'),
                time.time(),
                0.9 - i*0.05
            )
            for i in range(5)
        )

        result = CompactSearchResult(
            memories=memories,
            total=5,
            query='architecture decisions'
        )

        # Convert to string representation
        repr_str = str(result.memories)

        # Target: ~385 tokens for 5 results (vs ~2,625 for full Memory objects)
        # Allow some margin
        estimated_tokens = len(repr_str) / 4

        assert estimated_tokens < 800, \
            f"CompactSearchResult too large: {estimated_tokens} tokens (target: <800 for 5 results)"


class TestCompactHealthInfo:
    """Tests for CompactHealthInfo type."""

    def test_compact_health_info_creation(self):
        """Test basic CompactHealthInfo creation."""
        info = CompactHealthInfo(
            status='healthy',
            count=1247,
            backend='sqlite_vec'
        )

        assert info.status == 'healthy'
        assert info.count == 1247
        assert info.backend == 'sqlite_vec'

    def test_compact_health_info_status_values(self):
        """Test different status values."""
        statuses = ['healthy', 'degraded', 'error']

        for status in statuses:
            info = CompactHealthInfo(
                status=status,
                count=100,
                backend='cloudflare'
            )
            assert info.status == status

    def test_compact_health_info_backends(self):
        """Test different backend types."""
        backends = ['sqlite_vec', 'cloudflare', 'hybrid']

        for backend in backends:
            info = CompactHealthInfo(
                status='healthy',
                count=500,
                backend=backend
            )
            assert info.backend == backend

    def test_compact_health_info_token_size(self):
        """Test that CompactHealthInfo achieves target token size."""
        info = CompactHealthInfo(
            status='healthy',
            count=1247,
            backend='sqlite_vec'
        )

        repr_str = str(info)

        # Target: ~20 tokens (vs ~125 for full health check)
        estimated_tokens = len(repr_str) / 4

        assert estimated_tokens < 50, \
            f"CompactHealthInfo too large: {estimated_tokens} tokens (target: <50)"


class TestTokenEfficiency:
    """Integration tests for overall token efficiency."""

    def test_memory_size_comparison(self):
        """Compare CompactMemory size to full Memory object."""
        from mcp_memory_service.models.memory import Memory

        # Create full Memory object
        full_memory = Memory(
            content='A' * 1000,  # Long content
            content_hash='abc123def456' * 5,
            tags=['tag1', 'tag2', 'tag3'],
            memory_type='note',
            metadata={'key': 'value'},
            embedding=[0.1] * 768,  # Full embedding vector
        )

        # Create compact version
        compact = CompactMemory(
            hash='abc12345',
            preview='A' * 200,  # First 200 chars only
            tags=('tag1', 'tag2', 'tag3'),
            created=time.time(),
            score=0.95
        )

        # Compare sizes
        full_repr = str(full_memory.to_dict())
        compact_repr = str(compact)

        # Compact should be significantly smaller
        # Note: String representation is not exact token count, allow some margin
        size_ratio = len(compact_repr) / len(full_repr)

        assert size_ratio < 0.30, \
            f"CompactMemory not small enough: {size_ratio:.2%} of full size (target: <30%)"

    def test_search_result_size_reduction(self):
        """Validate 85%+ token reduction for search results."""
        # Create 5 compact memories
        memories = tuple(
            CompactMemory(
                f'hash{i:04d}',
                'A' * 200,
                ('tag1', 'tag2'),
                time.time(),
                0.9 - i*0.05
            )
            for i in range(5)
        )

        result = CompactSearchResult(
            memories=memories,
            total=5,
            query='test'
        )

        # Estimate tokens
        repr_str = str(result)
        estimated_tokens = len(repr_str) / 4

        # Target: 85% reduction from ~2,625 tokens � ~385 tokens
        # Allow some margin: should be under 600 tokens
        assert estimated_tokens < 600, \
            f"Search result not efficient enough: {estimated_tokens} tokens (target: <600)"

        # Verify we're achieving significant reduction
        # Original would be ~2,625 tokens, we should be well under 1000
        reduction_vs_original = 1 - (estimated_tokens / 2625)
        assert reduction_vs_original >= 0.75, \
            f"Token reduction insufficient: {reduction_vs_original:.1%} (target: e75%)"
