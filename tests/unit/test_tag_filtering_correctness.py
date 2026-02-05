"""
Comprehensive tag filtering correctness tests.

These tests verify that the relational tag storage correctly handles:
- Exact matching (no false positives from substring matches)
- Special characters in tag names
- Empty tag arrays
- Single vs multiple tag queries
- OR vs AND logic
- Edge cases
"""

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


class TestTagFilteringCorrectness:
    """Verify tag filtering correctness with normalized schema."""

    @pytest.mark.asyncio
    async def test_exact_tag_matching_no_false_positives(self, temp_db_path):
        """
        Verify exact tag matching - no substring matches.

        Old problem: Searching for "test" would match "testing", "test123", etc.
        New solution: Relational storage with exact tag.name matching.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Store memories with similar but distinct tags
        await storage.store(
            Memory(content="Content 1", content_hash=generate_content_hash("Content 1"), tags=["test"], memory_type="note")
        )

        await storage.store(
            Memory(content="Content 2", content_hash=generate_content_hash("Content 2"), tags=["testing"], memory_type="note")
        )

        await storage.store(
            Memory(content="Content 3", content_hash=generate_content_hash("Content 3"), tags=["test123"], memory_type="note")
        )

        await storage.store(
            Memory(content="Content 4", content_hash=generate_content_hash("Content 4"), tags=["my-test"], memory_type="note")
        )

        # Search for exact "test" tag
        results = await storage.search_by_tag(["test"])

        # Should ONLY match memories with exactly "test", not "testing" or "test123"
        assert len(results) == 1, f"Should return exactly 1 result, got {len(results)}"
        assert results[0].content == "Content 1"
        assert results[0].tags == ["test"]

    @pytest.mark.asyncio
    async def test_special_characters_in_tags(self, temp_db_path):
        """
        Verify tags with special characters work correctly.

        Tags can contain: hyphens, underscores, dots, etc.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Store memories with special character tags
        special_tags_cases = [
            (["python-3.11"], "Python version"),
            (["user_authentication"], "Auth module"),
            (["bug-fix"], "Bug fix"),
            (["@annotation"], "Annotation"),
            (["c++"], "C++ language"),
        ]

        for tags, content in special_tags_cases:
            await storage.store(
                Memory(content=content, content_hash=generate_content_hash(content), tags=tags, memory_type="note")
            )

        # Search for each special tag
        results = await storage.search_by_tag(["python-3.11"])
        assert len(results) == 1
        assert results[0].tags == ["python-3.11"]

        results = await storage.search_by_tag(["user_authentication"])
        assert len(results) == 1
        assert results[0].tags == ["user_authentication"]

        results = await storage.search_by_tag(["c++"])
        assert len(results) == 1
        assert results[0].tags == ["c++"]

    @pytest.mark.asyncio
    async def test_empty_tags_handling(self, temp_db_path):
        """
        Verify correct handling of memories with no tags.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Store memory with empty tags
        await storage.store(
            Memory(content="Untagged content", content_hash=generate_content_hash("Untagged"), tags=[], memory_type="note")
        )

        # Store memory with tags
        await storage.store(
            Memory(content="Tagged content", content_hash=generate_content_hash("Tagged"), tags=["python"], memory_type="note")
        )

        # Search for python tag
        results = await storage.search_by_tag(["python"])

        # Should NOT include untagged memory
        assert len(results) == 1
        assert results[0].content == "Tagged content"

    @pytest.mark.asyncio
    async def test_multiple_tags_or_logic(self, temp_db_path):
        """
        Verify OR logic for multiple tags.

        Searching for ["python", "java"] should return memories with python OR java.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        await storage.store(
            Memory(
                content="Python tutorial",
                content_hash=generate_content_hash("Python"),
                tags=["python", "programming"],
                memory_type="note",
            )
        )

        await storage.store(
            Memory(
                content="Java tutorial",
                content_hash=generate_content_hash("Java"),
                tags=["java", "programming"],
                memory_type="note",
            )
        )

        await storage.store(
            Memory(
                content="Rust tutorial",
                content_hash=generate_content_hash("Rust"),
                tags=["rust", "programming"],
                memory_type="note",
            )
        )

        # Search with OR logic (default)
        results = await storage.search_by_tag(["python", "java"])

        # Should return both python and java memories, NOT rust
        assert len(results) == 2
        contents = {r.content for r in results}
        assert "Python tutorial" in contents
        assert "Java tutorial" in contents
        assert "Rust tutorial" not in contents

    @pytest.mark.asyncio
    async def test_multiple_tags_and_logic(self, temp_db_path):
        """
        Verify AND logic for multiple tags.

        Searching for ["python", "advanced"] with AND should only return
        memories that have BOTH tags.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        await storage.store(
            Memory(
                content="Basic Python",
                content_hash=generate_content_hash("Basic Python"),
                tags=["python", "beginner"],
                memory_type="note",
            )
        )

        await storage.store(
            Memory(
                content="Advanced Python",
                content_hash=generate_content_hash("Advanced Python"),
                tags=["python", "advanced"],
                memory_type="note",
            )
        )

        await storage.store(
            Memory(
                content="Advanced Java",
                content_hash=generate_content_hash("Advanced Java"),
                tags=["java", "advanced"],
                memory_type="note",
            )
        )

        # Search with AND logic
        results = await storage.search_by_tags(["python", "advanced"], operation="AND")

        # Should ONLY return "Advanced Python" (has both tags)
        assert len(results) == 1
        assert results[0].content == "Advanced Python"
        assert set(results[0].tags) == {"python", "advanced"}

    @pytest.mark.asyncio
    async def test_case_sensitivity(self, temp_db_path):
        """
        Verify tag matching is case-sensitive.

        "Python" and "python" are different tags.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        await storage.store(
            Memory(
                content="Lowercase python", content_hash=generate_content_hash("Lowercase"), tags=["python"], memory_type="note"
            )
        )

        await storage.store(
            Memory(
                content="Uppercase Python", content_hash=generate_content_hash("Uppercase"), tags=["Python"], memory_type="note"
            )
        )

        # Search for lowercase
        results = await storage.search_by_tag(["python"])
        assert len(results) == 1
        assert results[0].content == "Lowercase python"

        # Search for uppercase
        results = await storage.search_by_tag(["Python"])
        assert len(results) == 1
        assert results[0].content == "Uppercase Python"

    @pytest.mark.asyncio
    async def test_whitespace_handling(self, temp_db_path):
        """
        Verify whitespace in tags is preserved.

        Tags should not be trimmed or normalized automatically.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Note: In practice, tags are usually normalized by the application layer
        # but the storage layer should preserve what it's given
        await storage.store(
            Memory(
                content="With spaces",
                content_hash=generate_content_hash("With spaces"),
                tags=["tag with spaces"],
                memory_type="note",
            )
        )

        await storage.store(
            Memory(
                content="Without spaces",
                content_hash=generate_content_hash("Without spaces"),
                tags=["tagwithoutspaces"],
                memory_type="note",
            )
        )

        # Search for tag with spaces
        results = await storage.search_by_tag(["tag with spaces"])
        assert len(results) == 1
        assert results[0].tags == ["tag with spaces"]

    @pytest.mark.asyncio
    async def test_unicode_tags(self, temp_db_path):
        """
        Verify Unicode tags work correctly.

        Tags can contain non-ASCII characters.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        unicode_cases = [
            (["日本語"], "Japanese content"),
            (["русский"], "Russian content"),
            (["émoji-😀"], "Emoji content"),
        ]

        for tags, content in unicode_cases:
            await storage.store(
                Memory(content=content, content_hash=generate_content_hash(content), tags=tags, memory_type="note")
            )

        # Search for each unicode tag
        results = await storage.search_by_tag(["日本語"])
        assert len(results) == 1
        assert results[0].content == "Japanese content"

        results = await storage.search_by_tag(["русский"])
        assert len(results) == 1
        assert results[0].content == "Russian content"

        results = await storage.search_by_tag(["émoji-😀"])
        assert len(results) == 1
        assert results[0].content == "Emoji content"

    @pytest.mark.asyncio
    async def test_many_tags_per_memory(self, temp_db_path):
        """
        Verify memories with many tags work correctly.

        A single memory can have many tags, and searching for any should find it.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        many_tags = [f"tag{i}" for i in range(50)]

        await storage.store(
            Memory(
                content="Heavily tagged",
                content_hash=generate_content_hash("Heavily tagged"),
                tags=many_tags,
                memory_type="note",
            )
        )

        # Search for first tag
        results = await storage.search_by_tag(["tag0"])
        assert len(results) == 1
        assert len(results[0].tags) == 50

        # Search for middle tag
        results = await storage.search_by_tag(["tag25"])
        assert len(results) == 1

        # Search for last tag
        results = await storage.search_by_tag(["tag49"])
        assert len(results) == 1

        # Search for non-existent tag
        results = await storage.search_by_tag(["tag50"])
        assert len(results) == 0


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide temporary directory for test databases."""
    return tmp_path
