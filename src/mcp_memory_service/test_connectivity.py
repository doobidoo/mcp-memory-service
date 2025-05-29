"""
Connectivity Test for EchoVault Memory Service
Copyright (c) 2025 EchoVault
Licensed under the MIT License.

This script tests connections to all required services:
- Neon PostgreSQL
- Qdrant Vector Database
- Cloudflare R2 Object Storage
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("connectivity_test")

# Import EchoVault modules
try:
    from storage.neon_client import NeonClient
    from storage.vector_store import VectorStoreClient
    from storage.blob_store import BlobStoreClient
except ImportError as e:
    logger.error(f"Failed to import EchoVault modules: {e}")
    sys.exit(1)

async def test_neon_connection() -> bool:
    """
    Test connection to Neon PostgreSQL.
    
    Returns:
        True if connection successful
    """
    logger.info("Testing Neon PostgreSQL connection...")
    
    # Check if Neon DSN is configured
    dsn = os.environ.get("NEON_DSN")
    if not dsn:
        logger.error("NEON_DSN environment variable is not set")
        return False
    
    try:
        # Initialize Neon client
        neon_client = NeonClient()
        await neon_client.initialize()
        
        # Get database stats
        stats = await neon_client.get_memory_stats()
        logger.info(f"Connected to Neon PostgreSQL successfully")
        logger.info(f"Memory count: {stats.get('memory_count', 0)}")
        
        # Close connection
        await neon_client.close()
        
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Neon PostgreSQL: {e}")
        return False

async def test_qdrant_connection() -> bool:
    """
    Test connection to Qdrant.
    
    Returns:
        True if connection successful
    """
    logger.info("Testing Qdrant connection...")
    
    # Check if Qdrant is enabled
    use_qdrant = os.environ.get("USE_QDRANT", "").lower() in ("true", "1", "yes")
    if not use_qdrant:
        logger.info("Qdrant is disabled, skipping test")
        return True
    
    # Check if Qdrant URL is configured
    qdrant_url = os.environ.get("QDRANT_URL")
    if not qdrant_url:
        logger.error("QDRANT_URL environment variable is not set")
        return False
    
    try:
        # Import Qdrant client
        try:
            import qdrant_client
            from qdrant_client import QdrantClient
        except ImportError:
            logger.error("qdrant-client package is not installed")
            return False
        
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=os.environ.get("QDRANT_API_KEY")
        )
        
        # Test connection by getting collections
        collections = qdrant_client.get_collections()
        logger.info(f"Connected to Qdrant successfully")
        logger.info(f"Collections: {[c.name for c in collections.collections]}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return False

async def test_r2_connection() -> bool:
    """
    Test connection to Cloudflare R2.
    
    Returns:
        True if connection successful
    """
    logger.info("Testing Cloudflare R2 connection...")
    
    # Check if R2 credentials are configured
    r2_endpoint = os.environ.get("R2_ENDPOINT")
    r2_access_key = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    r2_bucket = os.environ.get("R2_BUCKET")
    
    if not all([r2_endpoint, r2_access_key, r2_secret_key, r2_bucket]):
        logger.error("R2 credentials not fully configured")
        return False
    
    try:
        # Import boto3
        try:
            import boto3
        except ImportError:
            logger.error("boto3 package is not installed")
            return False
        
        # Initialize blob store client
        blob_store = BlobStoreClient()
        await blob_store.initialize()
        
        if not blob_store.is_configured():
            logger.error("Blob store client failed to initialize")
            return False
        
        # Test connection by generating a presigned URL
        test_key = "test/connectivity_test.txt"
        presigned_url = await blob_store.generate_presigned_url(test_key)
        
        if presigned_url:
            logger.info(f"Connected to Cloudflare R2 successfully")
            logger.info(f"Generated presigned URL: {presigned_url}")
            return True
        else:
            logger.error("Failed to generate presigned URL")
            return False
    except Exception as e:
        logger.error(f"Failed to connect to Cloudflare R2: {e}")
        return False

async def test_all_connections():
    """Test connections to all services."""
    logger.info("Starting EchoVault connectivity tests")
    
    # Test Neon connection
    neon_success = await test_neon_connection()
    
    # Test Qdrant connection
    qdrant_success = await test_qdrant_connection()
    
    # Test R2 connection
    r2_success = await test_r2_connection()
    
    # Print summary
    print("\n=== EchoVault Connectivity Test Results ===")
    print(f"Neon PostgreSQL: {'✅ Connected' if neon_success else '❌ Failed'}")
    print(f"Qdrant Vector DB: {'✅ Connected' if qdrant_success else '❌ Failed'}")
    print(f"Cloudflare R2: {'✅ Connected' if r2_success else '❌ Failed'}")
    print("===========================================")
    
    # Create a unified vector store client test
    if neon_success or qdrant_success:
        try:
            logger.info("Testing unified vector store client...")
            vector_store = VectorStoreClient()
            await vector_store.initialize()
            
            stats = await vector_store.get_stats()
            logger.info(f"Vector store stats: {stats}")
            
            # Close connections
            await vector_store.close()
            
            print(f"Vector Store Client: ✅ Initialized successfully")
        except Exception as e:
            logger.error(f"Failed to test vector store client: {e}")
            print(f"Vector Store Client: ❌ Initialization failed")
    
    # Overall status
    if neon_success and qdrant_success and r2_success:
        print("\n✅ All connections successful!")
    else:
        print("\n⚠️ Some connections failed. See logs for details.")

if __name__ == "__main__":
    print("\nEchoVault Connectivity Test\n" + "=" * 30)
    asyncio.run(test_all_connections())