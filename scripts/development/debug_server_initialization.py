#!/usr/bin/env python3
"""Enhanced diagnostic script to debug server initialization and Cloudflare backend issues"""

import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path

# Setup logging to see detailed information
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))
os.chdir(Path(__file__).parent)

# Load environment
try:
    from mcp_memory_service import env_loader
except ImportError:
    # env_loader might not be available in newer versions
    pass
from mcp_memory_service.config import CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, STORAGE_BACKEND

print("=" * 80)
print("ENHANCED MCP MEMORY SERVICE CLOUDFLARE BACKEND DIAGNOSTIC")
print("=" * 80)

print("\n📋 Configuration Check:")
print(f"  Storage Backend: {STORAGE_BACKEND}")
print(f"  API Token: {'SET' if CLOUDFLARE_API_TOKEN else 'NOT SET'}")
print(f"  Account ID: {'SET' if CLOUDFLARE_ACCOUNT_ID else 'NOT SET'}")


async def test_server_initialization():
    """Test the actual server initialization flow"""
    print("\n🚀 Testing Server Initialization Flow:")

    try:
        from mcp_memory_service.server import MemoryServer

        print("  ✅ MemoryServer import successful")
        server = MemoryServer()
        print("  ✅ MemoryServer instance created")

        # Test the eager initialization directly
        print("\n⚡ Testing Eager Storage Initialization:")

        try:
            success = await server._initialize_storage_with_timeout()
            print(f"  📊 Eager init result: {'SUCCESS' if success else 'FAILED'}")

            if success and hasattr(server, "storage") and server.storage:
                storage_type = server.storage.__class__.__name__
                print(f"  📦 Storage type: {storage_type}")

                # Test storage initialization
                if hasattr(server.storage, "initialize"):
                    print("  🔧 Testing storage.initialize()...")
                    await server.storage.initialize()
                    print("  ✅ Storage initialization complete")

            else:
                print("  ❌ No storage object created or eager init failed")

        except Exception as eager_error:
            print(f"  ❌ Eager initialization error: {str(eager_error)}")
            print("  📝 Traceback:")
            traceback.print_exc()

        # Test the lazy initialization path
        print("\n🔄 Testing Lazy Storage Initialization:")

        # Reset state to test lazy initialization
        server.storage = None
        server._storage_initialized = False

        try:
            storage = await server._ensure_storage_initialized()
            if storage:
                storage_type = storage.__class__.__name__
                print(f"  ✅ Lazy init successful, storage type: {storage_type}")
            else:
                print("  ❌ Lazy init returned None")

        except Exception as lazy_error:
            print(f"  ❌ Lazy initialization error: {str(lazy_error)}")
            print("  📝 Traceback:")
            traceback.print_exc()

        # Test health check
        print("\n🏥 Testing Health Check:")

        try:
            result = await server.handle_check_database_health({})
            health_text = result[0].text if result and len(result) > 0 else "No result"
            print("  📊 Health check result:")
            # Parse and pretty print the health check result
            import json

            try:
                health_data = json.loads(health_text.replace("Database Health Check Results:\n", ""))
                backend = health_data.get("statistics", {}).get("backend", "unknown")
                status = health_data.get("validation", {}).get("status", "unknown")
                print(f"    Backend: {backend}")
                print(f"    Status: {status}")
                if "error" in health_data.get("statistics", {}):
                    print(f"    Error: {health_data['statistics']['error']}")
            except json.JSONDecodeError:
                print(f"    Raw result: {health_text[:200]}...")

        except Exception as health_error:
            print(f"  ❌ Health check error: {str(health_error)}")
            print("  📝 Traceback:")
            traceback.print_exc()

    except Exception as server_error:
        print(f"❌ Server creation error: {str(server_error)}")
        print("📝 Traceback:")
        traceback.print_exc()


async def test_cloudflare_storage_directly():
    """Test Cloudflare storage initialization directly"""
    print("\n☁️  Testing Cloudflare Storage Directly:")

    try:
        from mcp_memory_service.config import (
            CLOUDFLARE_ACCOUNT_ID,
            CLOUDFLARE_API_TOKEN,
            CLOUDFLARE_BASE_DELAY,
            CLOUDFLARE_D1_DATABASE_ID,
            CLOUDFLARE_EMBEDDING_MODEL,
            CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
            CLOUDFLARE_MAX_RETRIES,
            CLOUDFLARE_R2_BUCKET,
            CLOUDFLARE_VECTORIZE_INDEX,
        )
        from mcp_memory_service.storage.cloudflare import CloudflareStorage

        print("  📋 Creating CloudflareStorage instance...")
        storage = CloudflareStorage(
            api_token=CLOUDFLARE_API_TOKEN,
            account_id=CLOUDFLARE_ACCOUNT_ID,
            vectorize_index=CLOUDFLARE_VECTORIZE_INDEX,
            d1_database_id=CLOUDFLARE_D1_DATABASE_ID,
            r2_bucket=CLOUDFLARE_R2_BUCKET,
            embedding_model=CLOUDFLARE_EMBEDDING_MODEL,
            large_content_threshold=CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
            max_retries=CLOUDFLARE_MAX_RETRIES,
            base_delay=CLOUDFLARE_BASE_DELAY,
        )
        print("  ✅ CloudflareStorage instance created")

        print("  🔧 Testing initialize() method...")
        await storage.initialize()
        print("  ✅ CloudflareStorage.initialize() completed")

        print("  📊 Testing get_stats() method...")
        stats = await storage.get_stats()
        print(f"  ✅ Statistics retrieved: {stats}")

    except Exception as direct_error:
        print(f"  ❌ Direct Cloudflare storage error: {str(direct_error)}")
        print("  📝 Traceback:")
        traceback.print_exc()


async def main():
    """Run all diagnostic tests"""
    await test_cloudflare_storage_directly()
    await test_server_initialization()

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
