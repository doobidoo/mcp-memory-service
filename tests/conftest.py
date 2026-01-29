import pytest
import os
import sys
import tempfile
import shutil
import uuid
from typing import Callable, Optional, List

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Disable semantic deduplication during tests to avoid interference with test expectations
# Tests often use similar content patterns (e.g., "Test memory 1", "Test memory 2")
# which would be caught by semantic dedup and cause unexpected failures
os.environ['MCP_SEMANTIC_DEDUP_ENABLED'] = 'false'

# Reserved tag for test memories - enables automatic cleanup
TEST_MEMORY_TAG = "__test__"

@pytest.fixture
def temp_db_path():
    '''Create a temporary directory for database testing.'''
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    shutil.rmtree(temp_dir)

@pytest.fixture
def unique_content() -> Callable[[str], str]:
    """
    Generate unique test content to avoid duplicate content errors.

    Usage:
        def test_example(unique_content):
            content = unique_content("Test memory about authentication")
            hash1 = store(content, tags=["test"])

    Returns:
        A function that takes a base string and returns a unique version.
    """
    def _generator(base: str = "test") -> str:
        return f"{base} [{uuid.uuid4()}]"
    return _generator


@pytest.fixture
def test_store():
    """
    Store function that auto-tags memories with TEST_MEMORY_TAG for cleanup.

    All memories created with this fixture will be automatically deleted
    at the end of the test session via pytest_sessionfinish hook.

    Usage:
        def test_example(test_store, unique_content):
            hash1 = test_store(unique_content("Test memory"), tags=["auth"])
            # Memory will have tags: ["__test__", "auth"]
    """
    from mcp_memory_service.api import store

    def _store(content: str, tags: Optional[List[str]] = None, **kwargs):
        all_tags = [TEST_MEMORY_TAG] + (tags or [])
        return store(content, tags=all_tags, **kwargs)

    return _store


def pytest_sessionfinish(session, exitstatus):
    """
    Cleanup all test memories at end of test session.

    SAFETY: Only cleans test databases, never production.
    Deletes all memories tagged with TEST_MEMORY_TAG.
    """
    try:
        from mcp_memory_service.api import delete_by_tag
        from mcp_memory_service.config import DATABASE_PATH
        import tempfile

        # SAFETY CHECK 1: Verify database path is in temp directory
        db_path = str(DATABASE_PATH)
        temp_prefix = tempfile.gettempdir()

        if not db_path.startswith(temp_prefix):
            # Check if it's a production path
            production_indicators = [
                "Library/Application Support/mcp-memory",
                ".mcp-memory",
                "mcp-memory-service/data"
            ]

            if any(indicator in db_path for indicator in production_indicators):
                print(f"\n[Test Cleanup] ⚠️  SAFETY ABORT: Refusing to clean production database!")
                print(f"[Test Cleanup] Database path: {db_path}")
                print(f"[Test Cleanup] Tests must use temp_db_path fixture for isolation.")
                return

        # SAFETY CHECK 2: Log which database we're cleaning
        print(f"\n[Test Cleanup] Cleaning test database: {db_path}")

        # Perform cleanup
        # Use allow_production=True since we already validated path above
        result = delete_by_tag([TEST_MEMORY_TAG], allow_production=True)
        deleted = result.get('deleted', 0) if isinstance(result, dict) else 0

        if deleted > 0:
            print(f"[Test Cleanup] ✅ Deleted {deleted} test memories tagged with '{TEST_MEMORY_TAG}'")
        else:
            print(f"[Test Cleanup] ✅ No test memories to clean")

    except Exception as e:
        # Make errors VISIBLE instead of silent
        print(f"\n[Test Cleanup] ❌ ERROR during cleanup: {e}")
        print(f"[Test Cleanup] This may indicate a configuration problem.")
        # Still don't fail the test session
