#!/usr/bin/env python3
"""
Quick diagnostic script to test the health check fixes
"""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_storage_initialization():
    """Test if storage initialization works properly."""
    try:
        print("[DEBUG] Testing Storage Initialization...")

        from mcp_memory_service.storage.chroma import ChromaMemoryStorage

        # Test database path
        test_db_path = os.path.join(os.path.dirname(__file__), "test_diagnostic_db")

        # Test initialization
        storage = ChromaMemoryStorage(test_db_path, preload_model=True)

        # Check initialization status
        if hasattr(storage, "is_initialized"):
            is_init = storage.is_initialized()
            print(f"[SUCCESS] Storage initialization status: {is_init}")

            if hasattr(storage, "get_initialization_status"):
                status = storage.get_initialization_status()
                print(f"[INFO] Detailed status: {status}")
        else:
            print("[WARNING] Storage doesn't have initialization status methods")

        # Test basic operations
        if storage.collection is not None:
            count = storage.collection.count()
            print(f"[SUCCESS] Collection accessible, count: {count}")
        else:
            print("[ERROR] Collection is None")

        if storage.embedding_function is not None:
            print("[SUCCESS] Embedding function initialized")
        else:
            print("[ERROR] Embedding function is None")

        return storage

    except Exception as e:
        print(f"[ERROR] Storage initialization failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def test_health_check(storage):
    """Test the health check functionality."""
    try:
        print("\nTesting Health Check...")

        import asyncio

        from mcp_memory_service.utils.db_utils import get_database_stats, validate_database

        async def run_health_check():
            # Test validation
            is_valid, message = await validate_database(storage)
            print(f"Validation result: {is_valid} - {message}")

            # Test stats
            stats = get_database_stats(storage)
            print(f"Stats: {stats}")

            return is_valid, stats

        return asyncio.run(run_health_check())

    except Exception as e:
        print(f"Health check failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False, {}


def cleanup_test_db(test_db_path):
    """Clean up test database."""
    try:
        import shutil

        if os.path.exists(test_db_path):
            shutil.rmtree(test_db_path)
            print(f"Cleaned up test database: {test_db_path}")
    except Exception as e:
        print(f"Could not clean up test database: {e}")


if __name__ == "__main__":
    print("ChromaDB Health Check Diagnostic Script")
    print("=" * 50)

    test_db_path = os.path.join(os.path.dirname(__file__), "test_diagnostic_db")

    try:
        # Test storage initialization
        storage = test_storage_initialization()

        if storage is not None:
            # Test health check
            is_valid, stats = test_health_check(storage)

            print("\nSummary:")
            print(f"• Storage initialized: {' Yes' if storage else ' No'}")
            print(f"• Health check passed: {' Yes' if is_valid else ' No'}")
            print(f"• Stats available: {' Yes' if stats.get('status') == 'healthy' else ' No'}")

            if is_valid:
                print("\nAll tests passed! The health check fixes are working.")
            else:
                print("\nSome issues remain. Check the output above for details.")
        else:
            print("\nStorage initialization failed. Cannot proceed with health check.")

    finally:
        cleanup_test_db(test_db_path)
