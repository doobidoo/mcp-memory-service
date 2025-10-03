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
Tests for content splitting and backend-specific length limits.

Tests cover:
- Content splitting utility functions
- Backend limit enforcement
- Automatic chunking with metadata
- Boundary preservation (sentences, paragraphs, code blocks)
- Overlap between chunks for context preservation
"""

import pytest
from src.mcp_memory_service.utils.content_splitter import (
    split_content,
    estimate_chunks_needed,
    validate_chunk_lengths,
    _find_best_split_point
)


class TestContentSplitter:
    """Test the content_splitter utility module."""

    def test_split_short_content(self):
        """Content shorter than max_length should not be split."""
        content = "This is a short sentence."
        chunks = split_content(content, max_length=100)

        assert len(chunks) == 1
        assert chunks[0] == content

    def test_split_long_content_character_mode(self):
        """Test character-based splitting without boundary preservation."""
        content = "a" * 500
        chunks = split_content(content, max_length=100, preserve_boundaries=False, overlap=10)

        # Should create multiple chunks
        assert len(chunks) > 1
        # All chunks should be <= max_length
        assert all(len(chunk) <= 100 for chunk in chunks)
        # Should have overlap
        assert chunks[1].startswith(chunks[0][-10:])

    def test_split_preserves_paragraphs(self):
        """Test that paragraph boundaries are preferred for splitting."""
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = split_content(content, max_length=30, preserve_boundaries=True)

        # Should split at paragraph boundaries
        assert len(chunks) >= 2
        # Each chunk should end cleanly (no mid-paragraph cuts)
        for chunk in chunks[:-1]:  # Check all but last chunk
            assert chunk.strip().endswith('.') or '\n\n' in chunk

    def test_split_preserves_sentences(self):
        """Test that sentence boundaries are preferred when paragraphs don't fit."""
        content = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = split_content(content, max_length=40, preserve_boundaries=True)

        # Should split at sentence boundaries
        assert len(chunks) >= 2
        # Most chunks should end with period
        period_endings = sum(1 for chunk in chunks if chunk.strip().endswith('.'))
        assert period_endings >= len(chunks) - 1

    def test_split_preserves_words(self):
        """Test that word boundaries are preferred when sentences don't fit."""
        content = "word1 word2 word3 word4 word5 word6 word7 word8"
        chunks = split_content(content, max_length=25, preserve_boundaries=True)

        # Should split at word boundaries
        assert len(chunks) >= 2
        # No chunk should end mid-word (except possibly last)
        for chunk in chunks[:-1]:
            # Should not end with partial word (will end with space or be complete)
            assert chunk.endswith(' ') or chunk == chunks[-1]

    def test_split_overlap(self):
        """Test that chunks have proper overlap for context."""
        content = "The quick brown fox jumps over the lazy dog. " * 10
        chunks = split_content(content, max_length=100, preserve_boundaries=True, overlap=20)

        assert len(chunks) > 1
        # Check that consecutive chunks have overlap
        for i in range(len(chunks) - 1):
            # The next chunk should contain some content from the end of current chunk
            current_end = chunks[i][-20:]
            assert any(word in chunks[i+1] for word in current_end.split()[:3])

    def test_estimate_chunks_needed(self):
        """Test chunk estimation function."""
        # Basic cases without overlap
        assert estimate_chunks_needed(0, 100) == 0
        assert estimate_chunks_needed(100, 100) == 1
        assert estimate_chunks_needed(200, 100) == 2
        assert estimate_chunks_needed(250, 100) == 3

        # Cases with overlap
        assert estimate_chunks_needed(100, 100, overlap=10) == 1  # Fits in one chunk
        assert estimate_chunks_needed(150, 100, overlap=10) == 2  # First chunk 100, second chunk covers remaining 50
        assert estimate_chunks_needed(200, 100, overlap=50) == 3  # Effective chunk size is 50

        # Edge cases
        assert estimate_chunks_needed(100, 100, overlap=100) == 1  # Invalid overlap, fallback to simple division
        assert estimate_chunks_needed(100, 100, overlap=150) == 1  # Invalid overlap larger than max_length

    def test_validate_chunk_lengths(self):
        """Test chunk length validation."""
        valid_chunks = ["short", "also short", "still short"]
        invalid_chunks = ["short", "this is way too long for the limit", "short"]

        assert validate_chunk_lengths(valid_chunks, max_length=50) is True
        assert validate_chunk_lengths(invalid_chunks, max_length=20) is False

    def test_find_best_split_point_paragraph(self):
        """Test that paragraph breaks are prioritized."""
        text = "First para.\n\nSecond para.\n\nThird para."
        split_point = _find_best_split_point(text, max_length=25)

        # Should split at first paragraph break
        assert text[split_point-2:split_point] == '\n\n'

    def test_find_best_split_point_sentence(self):
        """Test that sentence boundaries are used when no paragraph breaks."""
        text = "First sentence. Second sentence. Third sentence."
        split_point = _find_best_split_point(text, max_length=30)

        # Should split at sentence boundary
        assert '. ' in text[:split_point]

    def test_split_empty_content(self):
        """Test handling of empty content."""
        chunks = split_content("", max_length=100)
        assert chunks == []

    def test_split_exact_length(self):
        """Test content exactly at max_length."""
        content = "a" * 100
        chunks = split_content(content, max_length=100)

        assert len(chunks) == 1
        assert chunks[0] == content

    def test_split_code_blocks(self):
        """Test that code blocks are handled reasonably."""
        content = """def function_one():
    return True

def function_two():
    return False

def function_three():
    return None"""

        chunks = split_content(content, max_length=60, preserve_boundaries=True)

        # Should split at paragraph/function boundaries
        assert len(chunks) >= 2
        # Each chunk should contain complete functions ideally
        for chunk in chunks:
            # Count function definitions
            if 'def ' in chunk:
                # If it has a def, it should have a return (complete function)
                assert 'return' in chunk or chunk == chunks[-1]


class TestBackendLimits:
    """Test backend-specific content length limits."""

    def test_cloudflare_limit(self):
        """Test that Cloudflare backend uses config constant."""
        from src.mcp_memory_service.storage.cloudflare import CloudflareStorage
        from src.mcp_memory_service.config import CLOUDFLARE_MAX_CONTENT_LENGTH

        # Verify the class constant matches config
        assert CloudflareStorage._MAX_CONTENT_LENGTH == CLOUDFLARE_MAX_CONTENT_LENGTH

    def test_chromadb_limit(self):
        """Test that ChromaDB backend uses config constant."""
        from src.mcp_memory_service.storage.chroma import ChromaMemoryStorage
        from src.mcp_memory_service.config import CHROMADB_MAX_CONTENT_LENGTH

        assert ChromaMemoryStorage._MAX_CONTENT_LENGTH == CHROMADB_MAX_CONTENT_LENGTH

    def test_sqlitevec_unlimited(self):
        """Test that SQLite-vec backend uses config constant."""
        from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        from src.mcp_memory_service.config import SQLITEVEC_MAX_CONTENT_LENGTH

        # Create a mock instance to check property
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteVecMemoryStorage(db_path=db_path)

            # Should return configured value (default: None/unlimited)
            assert storage.max_content_length == SQLITEVEC_MAX_CONTENT_LENGTH
            assert storage.supports_chunking is True

    def test_hybrid_follows_config(self):
        """Test that Hybrid backend uses config constant."""
        from src.mcp_memory_service.storage.hybrid import HybridMemoryStorage
        from src.mcp_memory_service.config import HYBRID_MAX_CONTENT_LENGTH
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = HybridMemoryStorage(
                sqlite_db_path=db_path,
                cloudflare_config=None  # No cloud sync for this test
            )

            # Should match configured hybrid limit
            assert storage.max_content_length == HYBRID_MAX_CONTENT_LENGTH
            assert storage.supports_chunking is True


class TestConfigurationConstants:
    """Test configuration constants for content limits."""

    def test_config_constants_exist(self):
        """Test that all content limit constants are defined."""
        from src.mcp_memory_service.config import (
            CLOUDFLARE_MAX_CONTENT_LENGTH,
            CHROMADB_MAX_CONTENT_LENGTH,
            SQLITEVEC_MAX_CONTENT_LENGTH,
            HYBRID_MAX_CONTENT_LENGTH,
            ENABLE_AUTO_SPLIT,
            CONTENT_SPLIT_OVERLAP,
            CONTENT_PRESERVE_BOUNDARIES
        )

        assert CLOUDFLARE_MAX_CONTENT_LENGTH == 800
        assert CHROMADB_MAX_CONTENT_LENGTH == 1500
        assert SQLITEVEC_MAX_CONTENT_LENGTH is None  # Unlimited
        assert HYBRID_MAX_CONTENT_LENGTH == CLOUDFLARE_MAX_CONTENT_LENGTH
        assert isinstance(ENABLE_AUTO_SPLIT, bool)
        assert isinstance(CONTENT_SPLIT_OVERLAP, int)
        assert isinstance(CONTENT_PRESERVE_BOUNDARIES, bool)

    def test_config_validation(self):
        """Test that config values are sensible."""
        from src.mcp_memory_service.config import (
            CLOUDFLARE_MAX_CONTENT_LENGTH,
            CHROMADB_MAX_CONTENT_LENGTH,
            CONTENT_SPLIT_OVERLAP
        )

        # Limits should be positive
        assert CLOUDFLARE_MAX_CONTENT_LENGTH > 0
        assert CHROMADB_MAX_CONTENT_LENGTH > 0

        # ChromaDB should have higher limit (larger model)
        assert CHROMADB_MAX_CONTENT_LENGTH > CLOUDFLARE_MAX_CONTENT_LENGTH

        # Overlap should be reasonable
        assert 0 <= CONTENT_SPLIT_OVERLAP <= 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
