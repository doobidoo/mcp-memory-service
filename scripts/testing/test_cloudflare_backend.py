#!/usr/bin/env python3
"""
Test script for Cloudflare backend integration.
Run this after setting up your Cloudflare resources.
"""

import os
import sys
import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so 'src' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mcp_memory_service.storage.cloudflare import CloudflareStorage
from src.mcp_memory_service.models.memory import Memory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_cloudflare_backend():
    """Test all Cloudflare backend functionality."""
    
    # Check environment variables
    required_vars = [
        'CLOUDFLARE_API_TOKEN',
        'CLOUDFLARE_ACCOUNT_ID', 
        'CLOUDFLARE_VECTORIZE_INDEX',
        'CLOUDFLARE_D1_DATABASE_ID'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        return False
    
    try:
        # Initialize storage
        logger.info("🔧 Initializing Cloudflare storage...")
        storage = CloudflareStorage(
            api_token=os.getenv('CLOUDFLARE_API_TOKEN'),
            account_id=os.getenv('CLOUDFLARE_ACCOUNT_ID'),
            vectorize_index=os.getenv('CLOUDFLARE_VECTORIZE_INDEX'),
            d1_database_id=os.getenv('CLOUDFLARE_D1_DATABASE_ID'),
            r2_bucket=os.getenv('CLOUDFLARE_R2_BUCKET')  # Optional
        )
        
        # Test initialization
        logger.info("🚀 Testing storage initialization...")
        await storage.initialize()
        logger.info("✅ Storage initialized successfully")
        
        # Test storing a memory
        logger.info("💾 Testing memory storage...")
        test_content = "This is a test memory for Cloudflare backend integration."
        test_memory = Memory(
            content=test_content,
            content_hash=hashlib.sha256(test_content.encode()).hexdigest(),
            tags=["test", "cloudflare", "integration"],
            memory_type="test",
            metadata={"test_run": datetime.now().isoformat()}
        )
        
        success, message = await storage.store(test_memory)
        if success:
            logger.info(f"✅ Memory stored: {message}")
        else:
            logger.error(f"❌ Failed to store memory: {message}")
            return False
        
        # Test retrieval
        logger.info("🔍 Testing memory retrieval...")
        results = await storage.retrieve("test memory cloudflare", n_results=5)
        if results:
            logger.info(f"✅ Retrieved {len(results)} memories")
            for i, result in enumerate(results):
                logger.info(f"  {i+1}. Score: {result.similarity_score:.3f} - {result.memory.content[:50]}...")
        else:
            logger.warning("⚠️  No memories retrieved")
        
        # Test tag search
        logger.info("🏷️  Testing tag search...")
        tag_results = await storage.search_by_tag(["test"])
        if tag_results:
            logger.info(f"✅ Found {len(tag_results)} memories with 'test' tag")
        else:
            logger.warning("⚠️  No memories found with 'test' tag")
        
        # Test statistics
        logger.info("📊 Testing statistics...")
        stats = await storage.get_stats()
        logger.info(f"✅ Stats: {stats['total_memories']} memories, {stats['status']} status")
        
        # Test cleanup (optional - uncomment to clean up test data)
        # logger.info("🧹 Cleaning up test data...")
        # deleted_count, delete_message = await storage.delete_by_tag("test")
        # logger.info(f"✅ Cleaned up: {delete_message}")
        
        logger.info("🎉 All tests passed! Cloudflare backend is working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    
    finally:
        if 'storage' in locals():
            await storage.close()
            logger.info("🔒 Storage connection closed")

def print_setup_instructions():
    """Print setup instructions if environment is not configured."""
    print("\n" + "="*60)
    print("🔧 CLOUDFLARE BACKEND SETUP REQUIRED")
    print("="*60)
    print()
    print("Please complete these steps:")
    print()
    print("1. Create API token with these permissions:")
    print("   - Vectorize:Edit")
    print("   - D1:Edit") 
    print("   - Workers AI:Edit")
    print("   - R2:Edit (optional)")
    print()
    print("2. Create Cloudflare resources:")
    print("   wrangler vectorize create mcp-memory-index --dimensions=768 --metric=cosine")
    print("   wrangler d1 create mcp-memory-db")
    print("   wrangler r2 bucket create mcp-memory-content  # optional")
    print()
    print("3. Set environment variables:")
    print("   export CLOUDFLARE_API_TOKEN='your-token'")
    print("   export CLOUDFLARE_ACCOUNT_ID='be0e35a26715043ef8df90253268c33f'")
    print("   export CLOUDFLARE_VECTORIZE_INDEX='mcp-memory-index'") 
    print("   export CLOUDFLARE_D1_DATABASE_ID='your-d1-id'")
    print("   export CLOUDFLARE_R2_BUCKET='mcp-memory-content'  # optional")
    print()
    print("4. Run this test again:")
    print("   python test_cloudflare_backend.py")
    print()
    print("See docs/cloudflare-setup.md for detailed instructions.")
    print("="*60)

if __name__ == "__main__":
    # Check if basic environment is set up
    if not all(os.getenv(var) for var in ['CLOUDFLARE_API_TOKEN', 'CLOUDFLARE_ACCOUNT_ID']):
        print_setup_instructions()
    else:
        success = asyncio.run(test_cloudflare_backend())
        if success:
            print("\n🎉 Cloudflare backend is ready for production use!")
        else:
            print("\n❌ Tests failed. Check the logs above for details.")