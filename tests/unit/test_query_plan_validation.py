"""
Test query plan validation - verify indexes are used for tag filtering.

This test uses EXPLAIN QUERY PLAN to verify that tag filtering queries
use the relational tags indexes (idx_tags_name, idx_memory_tags_memory),
not full table scans. This ensures O(log n) performance instead of O(n) at scale.
"""

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


class TestQueryPlanValidation:
    """Verify that tag filtering queries use indexes efficiently."""

    @pytest.mark.asyncio
    async def test_tag_filter_uses_relational_index(self, temp_db_path):
        """
        Verify that tag filtering uses JOIN with idx_tags_name index, not table scan.

        Query plan should show "SEARCH TABLE tags USING INDEX idx_tags_name",
        NOT "SCAN TABLE tags" or "SCAN TABLE memories".
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Store some test data
        for i in range(10):
            await storage.store(
                Memory(
                    content=f"Test memory {i}",
                    content_hash=generate_content_hash(f"Test memory {i}"),
                    tags=["python", "test"] if i % 2 == 0 else ["java", "test"],
                    memory_type="note",
                )
            )

        # Build the same query pattern that retrieve() uses for tag filtering
        query_sql = """
            SELECT id FROM memories
            WHERE id IN (
                SELECT memory_id FROM memory_tags mt
                JOIN tags t ON mt.tag_id = t.id
                WHERE t.name IN (?)
            )
        """
        params = ["python"]

        # Get query plan
        cursor = storage.conn.execute(f"EXPLAIN QUERY PLAN {query_sql}", params)
        plan = cursor.fetchall()

        # Convert plan to string for inspection
        plan_str = " ".join([str(row) for row in plan]).upper()

        print("\n=== TAG FILTER QUERY PLAN ===")
        for row in plan:
            print(row)
        print(f"Plan string: {plan_str}")

        # CRITICAL: Must use SEARCH (index seek), NOT SCAN (table scan)
        assert "SEARCH" in plan_str, f"Query MUST use index (SEARCH), found: {plan_str}"

        # CRITICAL: Should NOT do table scans
        assert "SCAN TABLE TAGS" not in plan_str, f"Query should NOT scan tags table, should use idx_tags_name. Plan: {plan_str}"

        assert (
            "SCAN TABLE MEMORIES" not in plan_str or "COVERING INDEX" in plan_str
        ), f"Query should NOT scan memories table without covering index. Plan: {plan_str}"

    @pytest.mark.asyncio
    async def test_memory_type_filter_uses_index(self, temp_db_path):
        """
        Verify that memory_type filtering uses idx_memory_type index.

        This should use an index since there's no expression involved.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Store test data
        for i in range(10):
            await storage.store(
                Memory(
                    content=f"Test memory {i}",
                    content_hash=generate_content_hash(f"Test memory {i}"),
                    tags=["test"],
                    memory_type="note" if i % 2 == 0 else "reminder",
                )
            )

        # Query with memory_type filter
        query_sql = """
            SELECT id FROM memories
            WHERE memory_type = ?
        """
        params = ["note"]

        # Get query plan
        cursor = storage.conn.execute(f"EXPLAIN QUERY PLAN {query_sql}", params)
        plan = cursor.fetchall()

        plan_str = " ".join([str(row) for row in plan]).upper()

        print("\n=== MEMORY TYPE FILTER PLAN ===")
        for row in plan:
            print(row)

        # Should use idx_memory_type index with SEARCH (not SCAN)
        assert "SEARCH" in plan_str or "INDEX" in plan_str, f"Query should use idx_memory_type index. Plan: {plan_str}"

        assert (
            "SCAN TABLE MEMORIES" not in plan_str or "USING INDEX" in plan_str
        ), f"Query should not do full table scan. Plan: {plan_str}"

    @pytest.mark.asyncio
    async def test_combined_filters_query_plan(self, temp_db_path):
        """
        Verify query plan for combined memory_type + tags filters.

        This tests the actual retrieve() query structure with both filters.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Store test data
        for i in range(20):
            await storage.store(
                Memory(
                    content=f"Test memory {i}",
                    content_hash=generate_content_hash(f"Test memory {i}"),
                    tags=["python"] if i % 2 == 0 else ["java"],
                    memory_type="note" if i % 3 == 0 else "reminder",
                )
            )

        # Build combined filter query (similar to retrieve())
        query_sql = """
            SELECT id FROM memories
            WHERE memory_type = ?
            AND id IN (
                SELECT memory_id FROM memory_tags mt
                JOIN tags t ON mt.tag_id = t.id
                WHERE t.name IN (?)
            )
        """
        params = ["note", "python"]

        # Get query plan
        cursor = storage.conn.execute(f"EXPLAIN QUERY PLAN {query_sql}", params)
        plan = cursor.fetchall()

        plan_str = " ".join([str(row) for row in plan]).upper()

        print("\n=== COMBINED FILTER PLAN ===")
        for row in plan:
            print(row)

        # Verify indexes are used (SEARCH not SCAN)
        assert "SEARCH" in plan_str, f"Query should use indexes (SEARCH), found: {plan_str}"

        # Should NOT do full table scans
        assert "SCAN TABLE TAGS" not in plan_str, f"Tags query should use idx_tags_name, not table scan. Plan: {plan_str}"

    @pytest.mark.asyncio
    async def test_relational_indexes_exist(self, temp_db_path):
        """
        Verify all expected relational indexes exist.

        After migration, we should have:
        - idx_tags_name on tags(name)
        - idx_memory_tags_memory on memory_tags(memory_id)
        - idx_memory_tags_tag on memory_tags(tag_id)
        - Standard indexes on memories table
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Query for all indexes
        cursor = storage.conn.execute("""
            SELECT name, tbl_name FROM sqlite_master
            WHERE type='index'
            ORDER BY tbl_name, name
        """)
        indexes = [(row[0], row[1]) for row in cursor.fetchall()]

        print("\n=== ALL INDEXES ===")
        for name, table in indexes:
            print(f"  {table}.{name}")

        # Extract index names
        index_names = [name for name, _ in indexes]

        # Verify relational tag indexes
        assert "idx_tags_name" in index_names, "idx_tags_name should exist on tags table"

        assert "idx_memory_tags_memory" in index_names, "idx_memory_tags_memory should exist on memory_tags table"

        assert "idx_memory_tags_tag" in index_names, "idx_memory_tags_tag should exist on memory_tags table"

        # Verify standard memory indexes
        assert "idx_content_hash" in index_names, "idx_content_hash should exist on memories table"

        assert "idx_created_at" in index_names, "idx_created_at should exist on memories table"

        assert "idx_memory_type" in index_names, "idx_memory_type should exist on memories table"

    @pytest.mark.asyncio
    async def test_multi_tag_or_query_plan(self, temp_db_path):
        """
        Verify query plan for multiple tags with OR logic.

        Should use idx_tags_name with IN clause.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Store test data
        for i in range(10):
            await storage.store(
                Memory(
                    content=f"Test memory {i}",
                    content_hash=generate_content_hash(f"Test memory {i}"),
                    tags=["python", "code"] if i % 2 == 0 else ["java", "code"],
                    memory_type="note",
                )
            )

        # Query for memories with python OR java tag
        query_sql = """
            SELECT DISTINCT memory_id FROM memory_tags mt
            JOIN tags t ON mt.tag_id = t.id
            WHERE t.name IN (?, ?)
        """
        params = ["python", "java"]

        # Get query plan
        cursor = storage.conn.execute(f"EXPLAIN QUERY PLAN {query_sql}", params)
        plan = cursor.fetchall()

        plan_str = " ".join([str(row) for row in plan]).upper()

        print("\n=== MULTI-TAG OR QUERY PLAN ===")
        for row in plan:
            print(row)

        # Should use SEARCH on idx_tags_name for the IN clause
        assert "SEARCH" in plan_str, f"Query should use index seek (SEARCH). Plan: {plan_str}"

        # Should NOT scan tags table
        assert "SCAN TABLE TAGS" not in plan_str, f"Should use idx_tags_name, not table scan. Plan: {plan_str}"

    @pytest.mark.asyncio
    async def test_multi_tag_and_query_plan(self, temp_db_path):
        """
        Verify query plan for multiple tags with AND logic.

        Uses GROUP BY + HAVING COUNT to ensure all tags present.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        # Store test data
        for i in range(10):
            await storage.store(
                Memory(
                    content=f"Test memory {i}",
                    content_hash=generate_content_hash(f"Test memory {i}"),
                    tags=["python", "test", "code"] if i % 2 == 0 else ["java", "test"],
                    memory_type="note",
                )
            )

        # Query for memories with python AND test tags
        query_sql = """
            SELECT memory_id FROM memory_tags mt
            JOIN tags t ON mt.tag_id = t.id
            WHERE t.name IN (?, ?)
            GROUP BY memory_id
            HAVING COUNT(DISTINCT t.name) = 2
        """
        params = ["python", "test"]

        # Get query plan
        cursor = storage.conn.execute(f"EXPLAIN QUERY PLAN {query_sql}", params)
        plan = cursor.fetchall()

        plan_str = " ".join([str(row) for row in plan]).upper()

        print("\n=== MULTI-TAG AND QUERY PLAN ===")
        for row in plan:
            print(row)

        # Should use SEARCH on indexes
        assert "SEARCH" in plan_str, f"Query should use index seek (SEARCH). Plan: {plan_str}"

        # Should NOT scan tags table without index
        assert "SCAN TABLE TAGS" not in plan_str, f"Should use idx_tags_name, not table scan. Plan: {plan_str}"


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide temporary directory for test databases."""
    return tmp_path
