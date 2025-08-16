#!/usr/bin/env python3
"""
Automated Cloudflare resource setup for MCP Memory Service.
This script creates the required Cloudflare resources using the HTTP API.
"""

import os
import sys
import asyncio
import json
import logging
from typing import Dict, Any, Optional
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudflareSetup:
    def __init__(self, api_token: str, account_id: str):
        self.api_token = api_token
        self.account_id = account_id
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        self.client = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self.client is None:
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            self.client = httpx.AsyncClient(headers=headers, timeout=30.0)
        return self.client
    
    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to Cloudflare API."""
        client = await self._get_client()
        response = await client.request(method, url, **kwargs)
        
        if response.status_code not in [200, 201]:
            logger.error(f"API request failed: {response.status_code} {response.text}")
            response.raise_for_status()
        
        return response.json()
    
    async def create_vectorize_index(self, name: str = "mcp-memory-index") -> str:
        """Create Vectorize index and return its ID."""
        logger.info(f"Creating Vectorize index: {name}")
        
        # Check if index already exists
        try:
            url = f"{self.base_url}/vectorize/indexes/{name}"
            result = await self._make_request("GET", url)
            if result.get("success"):
                logger.info(f"Vectorize index {name} already exists")
                return name
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
        
        # Create new index
        url = f"{self.base_url}/vectorize/indexes"
        payload = {
            "name": name,
            "config": {
                "dimensions": 768,
                "metric": "cosine"
            }
        }
        
        result = await self._make_request("POST", url, json=payload)
        if result.get("success"):
            logger.info(f"✅ Created Vectorize index: {name}")
            return name
        else:
            raise ValueError(f"Failed to create Vectorize index: {result}")
    
    async def create_d1_database(self, name: str = "mcp-memory-db") -> str:
        """Create D1 database and return its ID."""
        logger.info(f"Creating D1 database: {name}")
        
        # List existing databases to check if it exists
        url = f"{self.base_url}/d1/database"
        result = await self._make_request("GET", url)
        
        if result.get("success"):
            for db in result.get("result", []):
                if db.get("name") == name:
                    db_id = db.get("uuid")
                    logger.info(f"D1 database {name} already exists with ID: {db_id}")
                    return db_id
        
        # Create new database
        payload = {"name": name}
        result = await self._make_request("POST", url, json=payload)
        
        if result.get("success"):
            db_id = result["result"]["uuid"]
            logger.info(f"✅ Created D1 database: {name} (ID: {db_id})")
            return db_id
        else:
            raise ValueError(f"Failed to create D1 database: {result}")
    
    async def create_r2_bucket(self, name: str = "mcp-memory-content") -> str:
        """Create R2 bucket and return its name."""
        logger.info(f"Creating R2 bucket: {name}")
        
        # Check if bucket already exists
        try:
            url = f"{self.base_url}/r2/buckets/{name}"
            result = await self._make_request("GET", url)
            if result.get("success"):
                logger.info(f"R2 bucket {name} already exists")
                return name
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
        
        # Create new bucket
        url = f"{self.base_url}/r2/buckets"
        payload = {"name": name}
        
        result = await self._make_request("POST", url, json=payload)
        if result.get("success"):
            logger.info(f"✅ Created R2 bucket: {name}")
            return name
        else:
            raise ValueError(f"Failed to create R2 bucket: {result}")
    
    async def verify_workers_ai_access(self) -> bool:
        """Verify Workers AI access and embedding model."""
        logger.info("Verifying Workers AI access...")
        
        # Test embedding generation
        url = f"{self.base_url}/ai/run/@cf/baai/bge-base-en-v1.5"
        payload = {"text": ["test embedding"]}
        
        try:
            result = await self._make_request("POST", url, json=payload)
            if result.get("success"):
                logger.info("✅ Workers AI access verified")
                return True
            else:
                logger.warning(f"Workers AI test failed: {result}")
                return False
        except Exception as e:
            logger.warning(f"Workers AI verification failed: {e}")
            return False
    
    async def close(self):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()

async def main():
    """Main setup routine."""
    print("🚀 Cloudflare Backend Setup for MCP Memory Service")
    print("=" * 55)
    
    # Check for required environment variables
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    
    if not api_token:
        print("❌ CLOUDFLARE_API_TOKEN environment variable not set")
        print("Please create an API token at: https://dash.cloudflare.com/profile/api-tokens")
        print("Required permissions: Vectorize:Edit, D1:Edit, Workers AI:Edit, R2:Edit")
        return False
    
    if not account_id:
        print("❌ CLOUDFLARE_ACCOUNT_ID environment variable not set")
        print("You can find your account ID in the Cloudflare dashboard sidebar")
        return False
    
    setup = CloudflareSetup(api_token, account_id)
    
    try:
        # Create resources
        vectorize_index = await setup.create_vectorize_index()
        d1_database_id = await setup.create_d1_database()
        
        # R2 bucket is optional
        r2_bucket = None
        create_r2 = input("\n🪣 Create R2 bucket for large content storage? (y/N): ").lower().strip()
        if create_r2 in ['y', 'yes']:
            try:
                r2_bucket = await setup.create_r2_bucket()
            except Exception as e:
                logger.warning(f"Failed to create R2 bucket: {e}")
                logger.warning("Continuing without R2 storage...")
        
        # Verify Workers AI
        ai_available = await setup.verify_workers_ai_access()
        
        print("\n🎉 Setup Complete!")
        print("=" * 20)
        print(f"Vectorize Index: {vectorize_index}")
        print(f"D1 Database ID: {d1_database_id}")
        print(f"R2 Bucket: {r2_bucket or 'Not configured'}")
        print(f"Workers AI: {'Available' if ai_available else 'Limited access'}")
        
        print("\n📝 Environment Variables:")
        print("=" * 25)
        print(f"export CLOUDFLARE_API_TOKEN=\"{api_token[:10]}...\"")
        print(f"export CLOUDFLARE_ACCOUNT_ID=\"{account_id}\"")
        print(f"export CLOUDFLARE_VECTORIZE_INDEX=\"{vectorize_index}\"")
        print(f"export CLOUDFLARE_D1_DATABASE_ID=\"{d1_database_id}\"")
        if r2_bucket:
            print(f"export CLOUDFLARE_R2_BUCKET=\"{r2_bucket}\"")
        print("export MCP_MEMORY_STORAGE_BACKEND=\"cloudflare\"")
        
        print("\n🧪 Test the setup:")
        print("python test_cloudflare_backend.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False
    
    finally:
        await setup.close()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)