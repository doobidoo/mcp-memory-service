"""
Test exact tag matching - verify no false positives.

This test ensures that tag filtering uses exact comma-delimited matching,
not substring matching. For example, searching for "python" should NOT
match memories tagged with "python3", "cpython", or "jython".
"""

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


class TestExactTagMatching:
    """Test that tag filtering uses exact matching without false positives."""

    @pytest.mark.asyncio
    async def test_exact_tag_match_no_false_positives(self, temp_db_path):
        """
        Verify that searching for "python" does NOT match "python3", "cpython", etc.

        This test validates the comma-boundary technique prevents substring matches.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Store memories with similar but different tags
        test_cases = [
            ("Python programming basics", ["python", "tutorial"]),
            ("Python 3 new features", ["python3", "features"]),
            ("CPython internals guide", ["cpython", "internals"]),
            ("Jython for Java developers", ["jython", "java"]),
            ("My Python project notes", ["my-python-project", "notes"]),
            ("Python coding standards", ["python", "standards"]),
        ]

        for content, tags in test_cases:
            await storage.store(
                Memory(content=content, content_hash=generate_content_hash(content), tags=tags, memory_type="note")
            )

        # Search for exact tag "python" - should only match 2 memories
        results = await storage.retrieve(query="programming", n_results=10, tags=["python"])

        # Verify exact count
        assert len(results) == 2, f"Expected 2 results with exact 'python' tag, got {len(results)}"

        # Verify all results have "python" tag (not python3, cpython, etc.)
        for result in results:
            assert "python" in result.memory.tags, f"Result should have 'python' tag, got {result.memory.tags}"
            # Verify it's NOT one of the false positives
            assert "python3" not in result.memory.tags, f"Should not match 'python3', got {result.memory.tags}"
            assert "cpython" not in result.memory.tags, f"Should not match 'cpython', got {result.memory.tags}"
            assert "jython" not in result.memory.tags, f"Should not match 'jython', got {result.memory.tags}"
            assert (
                "my-python-project" not in result.memory.tags
            ), f"Should not match 'my-python-project', got {result.memory.tags}"

    @pytest.mark.asyncio
    async def test_multi_tag_exact_matching(self, temp_db_path):
        """
        Verify multi-tag OR logic with exact matching.

        Searching for ["python", "java"] should match memories with either tag,
        but NOT "python3" or "javascript".
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        test_cases = [
            ("Python basics", ["python"]),
            ("Java basics", ["java"]),
            ("Python3 features", ["python3"]),
            ("JavaScript tutorial", ["javascript"]),
            ("Python and Java comparison", ["python", "java"]),
        ]

        for content, tags in test_cases:
            await storage.store(
                Memory(content=content, content_hash=generate_content_hash(content), tags=tags, memory_type="note")
            )

        # Search for "python" OR "java"
        results = await storage.retrieve(query="programming", n_results=10, tags=["python", "java"])

        # Should match 3 memories: "Python basics", "Java basics", "Python and Java comparison"
        assert len(results) == 3, f"Expected 3 results with 'python' OR 'java' tags, got {len(results)}"

        # Verify no false positives
        for result in results:
            has_python = "python" in result.memory.tags
            has_java = "java" in result.memory.tags
            has_python3 = "python3" in result.memory.tags
            has_javascript = "javascript" in result.memory.tags

            assert has_python or has_java, f"Result should have 'python' or 'java' tag, got {result.memory.tags}"
            assert not has_python3, f"Should not match 'python3', got {result.memory.tags}"
            assert not has_javascript, f"Should not match 'javascript', got {result.memory.tags}"

    @pytest.mark.asyncio
    async def test_edge_cases_single_tag(self, temp_db_path):
        """Test edge cases: single tag, tags at start/end, empty tags."""
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        test_cases = [
            ("Single tag", ["python"]),
            ("Tag at start", ["python", "coding", "tutorial"]),
            ("Tag at end", ["coding", "tutorial", "python"]),
            ("Tag in middle", ["coding", "python", "tutorial"]),
            ("Empty tags", []),
        ]

        for content, tags in test_cases:
            await storage.store(
                Memory(content=content, content_hash=generate_content_hash(content), tags=tags, memory_type="note")
            )

        # Search for "python"
        results = await storage.retrieve(query="tutorial", n_results=10, tags=["python"])

        # Should match 4 memories (all except "Empty tags")
        assert len(results) == 4, f"Expected 4 results with 'python' tag, got {len(results)}"

        # All results should have "python" tag
        for result in results:
            assert "python" in result.memory.tags, f"Result should have 'python' tag, got {result.memory.tags}"


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide temporary directory for test databases."""
    return tmp_path
