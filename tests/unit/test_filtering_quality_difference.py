"""
Test to demonstrate the difference between database-level filtering vs Python post-filtering.

This test shows that the OLD approach (get N results, then filter) would return
FEWER than N results, while the NEW approach (filter first, then get N) returns
exactly N results with correct semantic ranking within the filtered set.
"""

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


class TestFilteringQualityDifference:
    """Demonstrate the quality improvement from database-level filtering."""

    @pytest.mark.asyncio
    async def test_database_filtering_vs_python_filtering(self, temp_db_path):
        """
        Demonstrate the quality difference between database-level and Python filtering.

        Scenario:
        - 10 memories: 5 about Java (highly relevant to "programming language"),
                       5 about Python (less relevant to that exact phrase)
        - Query: "programming language" with tag filter "python", n_results=5

        OLD behavior (Python post-filter):
        - Get top 5 from ALL memories → likely all 5 Java (more semantically similar)
        - Filter for tag "python" → 0 results (none of the top 5 have python tag)

        NEW behavior (database-level filter):
        - Filter to tag "python" FIRST → 5 Python memories
        - Get top 5 from filtered set → exactly 5 Python results, ranked by relevance
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Store 5 Java memories (very semantically similar to query)
        java_memories = [
            "Java programming language syntax and features",
            "Java programming language for enterprise applications",
            "Object-oriented programming language Java tutorial",
            "Java programming language best practices guide",
            "Learn Java programming language basics",
        ]

        for content in java_memories:
            await storage.store(
                Memory(content=content, content_hash=generate_content_hash(content), tags=["java", "coding"], memory_type="note")
            )

        # Store 5 Python memories (less similar to exact query phrase)
        python_memories = [
            "Python data analysis with pandas",
            "Python web development with Django",
            "Python scripting automation tips",
            "Python machine learning basics",
            "Python debugging techniques",
        ]

        for content in python_memories:
            await storage.store(
                Memory(
                    content=content, content_hash=generate_content_hash(content), tags=["python", "coding"], memory_type="note"
                )
            )

        # Test NEW behavior: database-level filtering
        query = "programming language"
        results_new = await storage.retrieve(query=query, n_results=5, tags=["python"])

        # Verify NEW behavior characteristics
        assert len(results_new) == 5, f"Should return exactly 5 results, got {len(results_new)}"
        for result in results_new:
            assert "python" in result.memory.tags, f"All results should have python tag, got {result.memory.tags}"

        # Results should be ranked by relevance within the filtered set
        scores = [r.relevance_score for r in results_new]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by relevance"

        # To simulate OLD behavior, we'd do:
        # 1. Get top 5 from ALL memories (no filter)
        # 2. Filter in Python for tag "python"
        results_unfiltered = await storage.retrieve(
            query=query,
            n_results=5,
            # No tags filter - simulating old behavior
        )

        # Simulate Python post-filtering
        results_old = [r for r in results_unfiltered if "python" in r.memory.tags]

        # The key insight: OLD approach may return fewer results
        # In this scenario, Java memories are more similar to "programming language"
        # so without filtering, we'd get mostly Java results
        # After Python filtering for "python" tag, we'd get 0-2 results instead of 5

        print("\n=== QUALITY COMPARISON ===")
        print(f"Query: '{query}' with tag='python', n_results=5")
        print(f"OLD (Python post-filter): {len(results_old)} results")
        print(f"NEW (DB-level filter):    {len(results_new)} results")
        print(f"Quality improvement: {len(results_new) - len(results_old)} more relevant results\n")

        # Demonstrate that NEW approach returns more results
        assert len(results_new) > len(
            results_old
        ), f"Database filtering should return more results: {len(results_new)} vs {len(results_old)}"

        # NEW approach returns exactly what was requested
        assert len(results_new) == 5, "Database filtering returns exactly N results as requested"

        # OLD approach returns fewer than requested (the bug we fixed)
        assert len(results_old) < 5, "Python post-filtering returns fewer than N results (the bug)"


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide temporary directory for test databases."""
    return tmp_path
