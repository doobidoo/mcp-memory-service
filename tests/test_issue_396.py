"""
Test for Issue #396: time_expr natural language parsing fails for 'last week', '3 days ago'

This test reproduces the reported issue where time_expr parameter fails for certain
natural language expressions but works for others.
"""
import pytest
import asyncio
from datetime import datetime, timedelta, date
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models import Memory
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.mark.asyncio
async def test_time_expr_parsing_issue_396(temp_db_path):
    """Test that time_expr correctly parses 'last week' and '3 days ago'."""

    # Create storage with temp database
    import os
    db_file = os.path.join(temp_db_path, "test_issue_396.db")
    storage = SqliteVecMemoryStorage(db_path=db_file)
    await storage.initialize()

    # Store test memories across different days
    today = datetime.now()

    # Memory from yesterday (should match "yesterday")
    content_yesterday = "Memory from yesterday"
    yesterday_timestamp = (today - timedelta(days=1)).timestamp()
    memory_yesterday = Memory(
        content=content_yesterday,
        content_hash=generate_content_hash(content_yesterday),
        tags=["test", "__test__"],
        created_at=yesterday_timestamp
    )
    await storage.store(memory_yesterday)

    # Memory from 3 days ago (should match "3 days ago")
    content_3_days = "Memory from 3 days ago"
    three_days_timestamp = (today - timedelta(days=3)).timestamp()
    memory_3_days = Memory(
        content=content_3_days,
        content_hash=generate_content_hash(content_3_days),
        tags=["test", "__test__"],
        created_at=three_days_timestamp
    )
    await storage.store(memory_3_days)

    # Memory from last week (should match "last week")
    content_last_week = "Memory from last week"
    last_week_timestamp = (today - timedelta(days=7)).timestamp()
    memory_last_week = Memory(
        content=content_last_week,
        content_hash=generate_content_hash(content_last_week),
        tags=["test", "__test__"],
        created_at=last_week_timestamp
    )
    await storage.store(memory_last_week)

    # Memory from 2 weeks ago (should NOT match "last week" but should match "3 days ago" range in some searches)
    content_2_weeks = "Memory from 2 weeks ago"
    two_weeks_timestamp = (today - timedelta(days=14)).timestamp()
    memory_2_weeks = Memory(
        content=content_2_weeks,
        content_hash=generate_content_hash(content_2_weeks),
        tags=["test", "__test__"],
        created_at=two_weeks_timestamp
    )
    await storage.store(memory_2_weeks)

    # Wait a bit for storage to process
    await asyncio.sleep(0.1)

    # Test 1: "yesterday" (reported as working)
    result = await storage.search_memories(
        time_expr="yesterday",
        limit=10
    )
    assert result['total'] >= 1, "Should find at least the yesterday memory"
    yesterday_contents = [m.content for m in result['memories']]
    assert any("yesterday" in c.lower() for c in yesterday_contents), "Should find yesterday memory"

    # Test 2: "3 days ago" (reported as failing)
    result = await storage.search_memories(
        time_expr="3 days ago",
        limit=10
    )
    assert result['total'] >= 1, "Should find at least the 3 days ago memory"
    three_days_contents = [m.content for m in result['memories']]
    assert any("3 days ago" in c.lower() for c in three_days_contents), "Should find 3 days ago memory"

    # Test 3: "last week" (reported as failing)
    result = await storage.search_memories(
        time_expr="last week",
        limit=10
    )
    assert result['total'] >= 1, "Should find at least the last week memory"
    last_week_contents = [m.content for m in result['memories']]
    # Last week should return memories from last Monday-Sunday (not including current week)
    # So it should find the memory from 7 days ago
    assert any("last week" in c.lower() for c in last_week_contents), "Should find last week memory"

    # Test 4: Verify workaround with explicit ISO dates works
    three_days_ago_date = (today - timedelta(days=3)).date()
    result = await storage.search_memories(
        after=three_days_ago_date.isoformat(),
        limit=10
    )
    assert result['total'] >= 2, "Should find yesterday and 3 days ago memories"


@pytest.mark.asyncio
async def test_time_expr_edge_cases(temp_db_path):
    """Test additional time_expr edge cases."""

    import os
    db_file = os.path.join(temp_db_path, "test_edge_cases.db")
    storage = SqliteVecMemoryStorage(db_path=db_file)
    await storage.initialize()

    today = datetime.now()

    # Store a test memory
    content = "Test memory for edge cases"
    five_days_timestamp = (today - timedelta(days=5)).timestamp()
    memory = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["test", "__test__"],
        created_at=five_days_timestamp
    )
    await storage.store(memory)

    await asyncio.sleep(0.1)

    # Test various time expressions
    test_cases = [
        "5 days ago",
        "last 5 days",
        "1 week ago",
    ]

    for time_expr in test_cases:
        print(f"\nTesting time_expr='{time_expr}'")
        result = await storage.search_memories(
            time_expr=time_expr,
            limit=10
        )
        print(f"  Found {result['total']} memories")
        # Don't assert on results, just verify no errors
        assert "error" not in result, f"Should not error for time_expr='{time_expr}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
