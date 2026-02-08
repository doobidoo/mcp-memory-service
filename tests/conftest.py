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

# CRITICAL: Force sqlite_vec backend for tests to prevent accidental Cloudflare operations
# This prevents tests from soft-deleting production memories in Cloudflare D1
#
# To explicitly test Cloudflare/Hybrid backends, set:
#   MCP_TEST_ALLOW_CLOUD_BACKEND=true pytest tests/integration/test_cloudflare_*.py
#
_original_backend = os.environ.get('MCP_MEMORY_STORAGE_BACKEND')
_allow_cloud = os.environ.get('MCP_TEST_ALLOW_CLOUD_BACKEND', 'false').lower() == 'true'

if not _allow_cloud:
    os.environ['MCP_MEMORY_STORAGE_BACKEND'] = 'sqlite_vec'
    if _original_backend and _original_backend != 'sqlite_vec':
        print(f"\n[Test Safety] ⚠️  Overriding MCP_MEMORY_STORAGE_BACKEND from '{_original_backend}' to 'sqlite_vec'")
        print(f"[Test Safety] This prevents tests from modifying production Cloudflare data.")
        print(f"[Test Safety] To allow cloud backends: MCP_TEST_ALLOW_CLOUD_BACKEND=true pytest ...")

# CRITICAL: Force test database path to prevent production database access
# This ensures tests NEVER touch production databases even if they use the wrong fixtures
_test_db_dir = tempfile.mkdtemp(prefix='mcp-test-db-')
os.environ['MCP_MEMORY_SQLITE_PATH'] = os.path.join(_test_db_dir, 'test.db')
print(f"\n[Test Safety] 🔒 Test database path: {os.environ['MCP_MEMORY_SQLITE_PATH']}")
print(f"[Test Safety] Production databases are isolated and protected.")

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

    SAFETY: Uses TRIPLE verification to prevent production database deletion.
    Only cleans test databases, never production.
    """
    try:
        from mcp_memory_service.config import SQLITE_VEC_PATH, STORAGE_BACKEND
        import tempfile

        # Get database path based on backend
        if STORAGE_BACKEND == 'hybrid':
            # Hybrid uses SQLite-vec as primary
            db_path = str(SQLITE_VEC_PATH) if SQLITE_VEC_PATH else None
        elif STORAGE_BACKEND == 'sqlite_vec':
            db_path = str(SQLITE_VEC_PATH) if SQLITE_VEC_PATH else None
        else:
            # Cloudflare or other backends - skip cleanup (no local DB)
            print(f"\n[Test Cleanup] ℹ️  Backend '{STORAGE_BACKEND}' has no local database to clean")
            return

        if not db_path:
            print(f"\n[Test Cleanup] ⚠️  No database path configured, skipping cleanup")
            return

        # === TRIPLE SAFETY CHECK SYSTEM ===

        # SAFETY CHECK 1: Must be in temp directory
        temp_prefix = tempfile.gettempdir()
        if not db_path.startswith(temp_prefix):
            print(f"\n[Test Cleanup] 🛑 SAFETY ABORT: Database not in temp directory!")
            print(f"[Test Cleanup] Database path: {db_path}")
            print(f"[Test Cleanup] Expected prefix: {temp_prefix}")
            print(f"[Test Cleanup] This indicates a configuration error - tests must use isolated databases.")
            return

        # SAFETY CHECK 2: Must NOT contain production indicators
        production_indicators = [
            "Library/Application Support/mcp-memory",
            ".mcp-memory",
            "mcp-memory-service/data",
            "/Users/",  # Any user home directory
            "/home/",   # Linux home directories
            "C:\\Users\\",  # Windows user directories
        ]

        for indicator in production_indicators:
            if indicator in db_path:
                print(f"\n[Test Cleanup] 🛑 SAFETY ABORT: Production path indicator detected!")
                print(f"[Test Cleanup] Database path: {db_path}")
                print(f"[Test Cleanup] Detected indicator: '{indicator}'")
                print(f"[Test Cleanup] Tests MUST use temp_db_path fixture for isolation.")
                return

        # SAFETY CHECK 3: Must contain test-specific markers
        test_markers = [
            'mcp-test-',  # Our custom test prefix
            'pytest-',    # pytest temp directories
            '/tmp/',      # Unix temp
            '/var/tmp/',  # Unix alternative temp
            'Temp\\',     # Windows temp
        ]

        has_test_marker = any(marker in db_path for marker in test_markers)
        if not has_test_marker:
            print(f"\n[Test Cleanup] 🛑 SAFETY ABORT: No test marker found in database path!")
            print(f"[Test Cleanup] Database path: {db_path}")
            print(f"[Test Cleanup] Expected one of: {test_markers}")
            return

        # === ALL SAFETY CHECKS PASSED ===
        print(f"\n[Test Cleanup] ✅ Safety checks passed")
        print(f"[Test Cleanup] Database path: {db_path}")
        print(f"[Test Cleanup] Cleaning test database...")

        # Import delete function
        from mcp_memory_service.api import delete_by_tag

        # Perform cleanup - DO NOT use allow_production flag
        # The delete_by_tag function has its own safety checks
        # If our checks above passed, the delete_by_tag checks should also pass
        result = delete_by_tag([TEST_MEMORY_TAG], allow_production=False)
        deleted = result.get('deleted', 0) if isinstance(result, dict) else 0

        if deleted > 0:
            print(f"[Test Cleanup] ✅ Deleted {deleted} test memories tagged with '{TEST_MEMORY_TAG}'")
        else:
            print(f"[Test Cleanup] ✅ No test memories to clean")

    except ValueError as e:
        # Safety check from delete_by_tag - this is GOOD, it means safety worked!
        if "production database" in str(e).lower():
            print(f"\n[Test Cleanup] ✅ Safety check prevented production deletion!")
            print(f"[Test Cleanup] Error: {e}")
            print(f"[Test Cleanup] This is the safety system working correctly.")
        else:
            print(f"\n[Test Cleanup] ❌ Unexpected ValueError: {e}")
    except Exception as e:
        # Make errors VISIBLE instead of silent
        print(f"\n[Test Cleanup] ❌ ERROR during cleanup: {e}")
        print(f"[Test Cleanup] This may indicate a configuration problem.")
        # Still don't fail the test session


def pytest_sessionstart(session):
    """
    Verify test isolation before running any tests.

    This hook runs BEFORE any tests execute to ensure proper configuration.
    """
    print("\n" + "="*80)
    print("🔒 TEST ISOLATION VERIFICATION")
    print("="*80)

    try:
        from mcp_memory_service.config import SQLITE_VEC_PATH, STORAGE_BACKEND
        import tempfile

        # Check backend
        if STORAGE_BACKEND not in ['sqlite_vec', 'cloudflare']:
            if not _allow_cloud:
                print(f"⚠️  WARNING: Backend '{STORAGE_BACKEND}' may access production!")
                print(f"⚠️  Tests should use 'sqlite_vec' backend for isolation.")
                print(f"⚠️  Set MCP_TEST_ALLOW_CLOUD_BACKEND=true to explicitly allow this.")

        # Check database path
        if SQLITE_VEC_PATH:
            db_path = str(SQLITE_VEC_PATH)
            temp_prefix = tempfile.gettempdir()

            # Verify it's in temp directory
            if db_path.startswith(temp_prefix):
                print(f"✅ Database in temp directory: {db_path}")
            else:
                print(f"❌ CRITICAL: Database NOT in temp directory!")
                print(f"   Path: {db_path}")
                print(f"   Expected prefix: {temp_prefix}")
                print(f"   Tests may affect production data!")

                # Check for production indicators
                production_indicators = [
                    "Library/Application Support/mcp-memory",
                    ".mcp-memory",
                    "mcp-memory-service/data"
                ]

                for indicator in production_indicators:
                    if indicator in db_path:
                        print(f"   🛑 PRODUCTION PATH DETECTED: '{indicator}'")
                        print(f"   🛑 ABORTING TEST RUN FOR SAFETY")
                        pytest.exit("Test configuration would affect production database", returncode=1)

        print(f"✅ Storage backend: {STORAGE_BACKEND}")
        print(f"✅ Cloud backends: {'ALLOWED' if _allow_cloud else 'BLOCKED'}")
        print("="*80 + "\n")

    except Exception as e:
        print(f"❌ Error during isolation verification: {e}")
        print("="*80 + "\n")


# Cleanup temp test database directory at session end
def pytest_unconfigure(config):
    """Clean up global test database directory."""
    try:
        if os.path.exists(_test_db_dir):
            shutil.rmtree(_test_db_dir, ignore_errors=True)
            print(f"\n[Test Cleanup] 🗑️  Removed test database directory: {_test_db_dir}")
    except Exception as e:
        print(f"\n[Test Cleanup] ⚠️  Failed to remove test database directory: {e}")
