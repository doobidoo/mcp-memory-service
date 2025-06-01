#!/usr/bin/env python
"""
EchoVault Connectivity Test with Proper Path Setup
Run from project root with: python test_echovault_connectivity.py
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Fix Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("echovault_connectivity")

# Import EchoVault modules
try:
    from mcp_memory_service.storage.neon_client import NeonClient
    from mcp_memory_service.storage.vector_store import VectorStoreClient
    from mcp_memory_service.storage.blob_store import BlobStoreClient
    logger.info("Successfully imported EchoVault modules")
except ImportError as e:
    logger.error(f"Failed to import EchoVault modules: {e}")
    # Try importing individual modules to see what's missing
    try:
        import asyncpg
        logger.info("asyncpg is available")
    except ImportError:
        logger.error("asyncpg is NOT installed")
    
    try:
        import qdrant_client
        logger.info("qdrant_client is available")
    except ImportError:
        logger.error("qdrant_client is NOT installed")
    
    try:
        import boto3
        logger.info("boto3 is available")
    except ImportError:
        logger.error("boto3 is NOT installed")
    
    sys.exit(1)

async def test_neon_connection() -> dict:
    """Test connection to Neon PostgreSQL."""
    result = {
        "service": "Neon PostgreSQL",
        "status": "❌ FAIL",
        "details": {}
    }
    
    logger.info("Testing Neon PostgreSQL connection...")
    
    # Check if Neon DSN is configured
    dsn = os.environ.get("NEON_DSN")
    if not dsn:
        result["details"]["error"] = "NEON_DSN environment variable is not set"
        return result
    
    try:
        # Initialize Neon client
        neon_client = NeonClient()
        await neon_client.initialize()
        
        # Get database stats
        stats = await neon_client.get_memory_stats()
        
        result["status"] = "✅ PASS"
        result["details"]["connection"] = "Successfully connected"
        result["details"]["memory_count"] = stats.get('memory_count', 0)
        result["details"]["vector_count"] = stats.get('vector_count', 0)
        
        # Test pgvector functionality
        try:
            # Simple query to verify pgvector is installed
            async with neon_client.pool.acquire() as conn:
                version = await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                if version:
                    result["details"]["pgvector_version"] = version
                else:
                    result["details"]["pgvector"] = "NOT INSTALLED"
        except Exception as e:
            result["details"]["pgvector_check"] = f"Error: {str(e)}"
        
        # Close connection
        await neon_client.close()
        
    except Exception as e:
        result["details"]["error"] = str(e)
        result["details"]["error_type"] = type(e).__name__
    
    return result

async def test_qdrant_connection() -> dict:
    """Test connection to Qdrant."""
    result = {
        "service": "Qdrant Cloud",
        "status": "❌ FAIL",
        "details": {}
    }
    
    logger.info("Testing Qdrant connection...")
    
    # Check if Qdrant is enabled
    use_qdrant = os.environ.get("USE_QDRANT", "false").lower() in ("true", "1", "yes")
    if not use_qdrant:
        result["status"] = "⏭️ SKIP"
        result["details"]["reason"] = "USE_QDRANT not set to true"
        return result
    
    # Check if Qdrant URL is configured
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    
    if not qdrant_url:
        result["details"]["error"] = "QDRANT_URL environment variable is not set"
        return result
    
    try:
        from qdrant_client import QdrantClient
        
        # Initialize Qdrant client
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        # Test connection by getting collections
        collections = client.get_collections()
        
        result["status"] = "✅ PASS"
        result["details"]["connection"] = "Successfully connected"
        result["details"]["collections"] = [c.name for c in collections.collections]
        result["details"]["collection_count"] = len(collections.collections)
        
    except Exception as e:
        result["details"]["error"] = str(e)
        result["details"]["error_type"] = type(e).__name__
    
    return result

async def test_r2_connection() -> dict:
    """Test connection to Cloudflare R2."""
    result = {
        "service": "Cloudflare R2",
        "status": "❌ FAIL",
        "details": {}
    }
    
    logger.info("Testing Cloudflare R2 connection...")
    
    # Check if R2 credentials are configured
    r2_endpoint = os.environ.get("R2_ENDPOINT")
    r2_access_key = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    r2_bucket = os.environ.get("R2_BUCKET")
    
    if not all([r2_endpoint, r2_access_key, r2_secret_key, r2_bucket]):
        missing = []
        if not r2_endpoint: missing.append("R2_ENDPOINT")
        if not r2_access_key: missing.append("R2_ACCESS_KEY_ID")
        if not r2_secret_key: missing.append("R2_SECRET_ACCESS_KEY")
        if not r2_bucket: missing.append("R2_BUCKET")
        result["details"]["error"] = f"Missing credentials: {', '.join(missing)}"
        return result
    
    try:
        # Initialize blob store client
        blob_store = BlobStoreClient()
        await blob_store.initialize()
        
        if not blob_store.is_configured():
            result["details"]["error"] = "Blob store client failed to initialize"
            return result
        
        # Test bucket access
        result["details"]["bucket"] = r2_bucket
        
        # Try to list objects in the bucket
        try:
            # List up to 10 objects
            objects = await blob_store.list_objects(prefix="", max_keys=10)
            result["details"]["object_count"] = len(objects) if objects else 0
            result["status"] = "✅ PASS"
            result["details"]["connection"] = "Successfully connected"
        except Exception as e:
            # Try generating a presigned URL as fallback test
            test_key = "test/connectivity_test.txt"
            presigned_url = await blob_store.generate_presigned_url(test_key)
            
            if presigned_url:
                result["status"] = "⚠️ PARTIAL"
                result["details"]["connection"] = "Connected but limited access"
                result["details"]["presigned_url"] = "Generated successfully"
            else:
                result["details"]["error"] = str(e)
                
    except Exception as e:
        result["details"]["error"] = str(e)
        result["details"]["error_type"] = type(e).__name__
    
    return result

async def test_vector_store() -> dict:
    """Test unified vector store client."""
    result = {
        "service": "Vector Store Client",
        "status": "❌ FAIL",
        "details": {}
    }
    
    logger.info("Testing unified vector store client...")
    
    try:
        vector_store = VectorStoreClient()
        await vector_store.initialize()
        
        stats = await vector_store.get_stats()
        
        result["status"] = "✅ PASS"
        result["details"]["initialized"] = "Successfully"
        result["details"]["stats"] = stats
        
        # Get active providers
        providers = []
        if hasattr(vector_store, 'neon_client') and vector_store.neon_client:
            providers.append("Neon")
        if hasattr(vector_store, 'qdrant_client') and vector_store.qdrant_client:
            providers.append("Qdrant")
        result["details"]["active_providers"] = providers
        
        # Close connections
        await vector_store.close()
        
    except Exception as e:
        result["details"]["error"] = str(e)
        result["details"]["error_type"] = type(e).__name__
    
    return result

async def run_all_tests():
    """Run all connectivity tests."""
    print("\n" + "="*60)
    print("EchoVault Connectivity Test")
    print(f"Test Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("="*60 + "\n")
    
    # Test all services
    results = []
    
    # Test Neon
    neon_result = await test_neon_connection()
    results.append(neon_result)
    
    # Test Qdrant
    qdrant_result = await test_qdrant_connection()
    results.append(qdrant_result)
    
    # Test R2
    r2_result = await test_r2_connection()
    results.append(r2_result)
    
    # Test Vector Store
    vector_result = await test_vector_store()
    results.append(vector_result)
    
    # Print results
    for result in results:
        print(f"\n### {result['service']}")
        print(f"- **Status**: {result['status']}")
        for key, value in result['details'].items():
            print(f"- **{key.replace('_', ' ').title()}**: {value}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    passed = sum(1 for r in results if "✅" in r["status"])
    failed = sum(1 for r in results if "❌" in r["status"])
    skipped = sum(1 for r in results if "⏭️" in r["status"])
    partial = sum(1 for r in results if "⚠️" in r["status"])
    
    print(f"\n✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏭️ Skipped: {skipped}")
    print(f"⚠️ Partial: {partial}")
    
    if failed == 0 and passed > 0:
        print("\n🎉 All active services are connected successfully!")
    else:
        print("\n⚠️ Some services failed to connect. Check the details above.")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(run_all_tests())
    
    # Write results to markdown file
    with open("REAL_CONNECTION_TEST.md", "w", encoding="utf-8") as f:
        f.write("# Real Connection Test Results\n\n")
        f.write(f"## Test Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")
        
        for result in results:
            f.write(f"### {result['service']}\n")
            f.write(f"- **Status**: {result['status']}\n")
            for key, value in result['details'].items():
                f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
            f.write("\n")
        
        f.write("## Summary\n\n")
        f.write("### What Works:\n")
        for r in results:
            if "✅" in r["status"]:
                f.write(f"- ✅ {r['service']}\n")
        
        f.write("\n### What Needs Fixing:\n")
        for r in results:
            if "❌" in r["status"]:
                f.write(f"- ❌ {r['service']}: {r['details'].get('error', 'Unknown error')}\n")
        
        f.write("\n### Skipped:\n")
        for r in results:
            if "⏭️" in r["status"]:
                f.write(f"- ⏭️ {r['service']}: {r['details'].get('reason', 'Unknown reason')}\n")
    
    print(f"\n📄 Results saved to REAL_CONNECTION_TEST.md") 