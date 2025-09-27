#!/usr/bin/env python3
"""Integration test for Cloudflare connection and storage initialization."""

import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

# Add the project directory to path for standalone execution
project_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_dir / "src"))

# Load environment from .env file
load_dotenv(project_dir / ".env")


def test_cloudflare_config():
    """Test Cloudflare configuration and API connectivity."""
    print("=== Cloudflare Configuration Test ===")
    print(f"Backend: {os.getenv('MCP_MEMORY_STORAGE_BACKEND')}")
    print(f"API Token: {os.getenv('CLOUDFLARE_API_TOKEN')[:10]}..." if os.getenv('CLOUDFLARE_API_TOKEN') else "API Token: NOT SET")
    print(f"Account ID: {os.getenv('CLOUDFLARE_ACCOUNT_ID')}")
    print(f"D1 Database ID: {os.getenv('CLOUDFLARE_D1_DATABASE_ID')}")
    print(f"Vectorize Index: {os.getenv('CLOUDFLARE_VECTORIZE_INDEX')}")

    # Test Cloudflare API connection
    api_token = os.getenv('CLOUDFLARE_API_TOKEN')
    account_id = os.getenv('CLOUDFLARE_ACCOUNT_ID')

    if api_token and account_id:
        print("\n=== Testing Cloudflare API Connection ===")
        headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }

        # Test API token validity
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                print("✅ Cloudflare API token is valid")
                return True
            else:
                print(f"❌ Cloudflare API error: {response.status_code}")
                print(f"Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {str(e)}")
            return False
    else:
        print("❌ Missing API token or account ID")
        return False


def test_storage_backend_import():
    """Test storage backend import and initialization."""
    print("\n=== Testing Storage Backend Import ===")
    try:
        from mcp_memory_service.config import STORAGE_BACKEND
        print(f"Config says backend is: {STORAGE_BACKEND}")

        if STORAGE_BACKEND == 'cloudflare':
            try:
                from mcp_memory_service.storage.cloudflare import CloudflareStorage
                print("✅ CloudflareStorage import successful")

                # Try to initialize
                print("\n=== Attempting to Initialize Cloudflare Storage ===")
                from mcp_memory_service.config import (
                    CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID,
                    CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID,
                    CLOUDFLARE_R2_BUCKET, CLOUDFLARE_EMBEDDING_MODEL,
                    CLOUDFLARE_LARGE_CONTENT_THRESHOLD, CLOUDFLARE_MAX_RETRIES,
                    CLOUDFLARE_BASE_DELAY
                )
                storage = CloudflareStorage(
                    api_token=CLOUDFLARE_API_TOKEN,
                    account_id=CLOUDFLARE_ACCOUNT_ID,
                    vectorize_index=CLOUDFLARE_VECTORIZE_INDEX,
                    d1_database_id=CLOUDFLARE_D1_DATABASE_ID,
                    r2_bucket=CLOUDFLARE_R2_BUCKET,
                    embedding_model=CLOUDFLARE_EMBEDDING_MODEL,
                    large_content_threshold=CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
                    max_retries=CLOUDFLARE_MAX_RETRIES,
                    base_delay=CLOUDFLARE_BASE_DELAY
                )
                print("✅ CloudflareStorage initialized")
                return True
            except Exception as e:
                print(f"❌ CloudflareMemoryStorage error: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"⚠️ Backend is set to {STORAGE_BACKEND}, not cloudflare")
            return False

    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Cloudflare connection tests."""
    print("🧪 Running Cloudflare Integration Tests")
    print("=" * 50)

    success = True

    # Test API connectivity
    if not test_cloudflare_config():
        success = False

    # Test storage backend
    if not test_storage_backend_import():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("✅ All Cloudflare connection tests passed!")
        return 0
    else:
        print("❌ Some Cloudflare connection tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())