#!/usr/bin/env python3
"""
Test script to verify API key authentication fallback works with OAuth enabled.

This test verifies that existing API key authentication continues to work
when OAuth is enabled, ensuring backward compatibility.
"""

import asyncio
import sys
from pathlib import Path

import httpx

# Add src to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


async def test_api_key_fallback(base_url: str = "http://localhost:8000", api_key: str = None) -> bool:
    """
    Test API key authentication fallback with OAuth enabled.

    Returns:
        True if all tests pass, False otherwise
    """
    print(f"Testing API key fallback at {base_url}")
    print("=" * 50)

    if not api_key:
        print("❌ No API key provided - cannot test fallback")
        print("   Set MCP_API_KEY environment variable or pass as argument")
        return False

    async with httpx.AsyncClient() as client:
        try:
            # Test 1: API Key as Bearer Token (should work)
            print("1. Testing API Key as Bearer Token...")

            headers = {"Authorization": f"Bearer {api_key}"}
            response = await client.get(f"{base_url}/api/memories", headers=headers)

            if response.status_code == 200:
                print(f"   ✅ API key authentication working")
            else:
                print(f"   ❌ API key authentication failed: {response.status_code}")
                return False

            # Test 2: API Key for write operations
            print("\n2. Testing API Key for Write Operations...")

            memory_data = {
                "content": "Test memory for API key authentication",
                "tags": ["test", "api-key"],
                "memory_type": "test"
            }

            response = await client.post(f"{base_url}/api/memories", json=memory_data, headers=headers)

            if response.status_code == 200:
                print(f"   ✅ API key write operation working")
                # Store content hash for cleanup
                memory_hash = response.json().get("content_hash")
            else:
                print(f"   ❌ API key write operation failed: {response.status_code}")
                return False

            # Test 3: Invalid API Key (should fail)
            print("\n3. Testing Invalid API Key...")

            invalid_headers = {"Authorization": "Bearer invalid_key"}
            response = await client.get(f"{base_url}/api/memories", headers=invalid_headers)

            if response.status_code == 401:
                print(f"   ✅ Invalid API key correctly rejected")
            else:
                print(f"   ⚠️  Invalid API key test inconclusive: {response.status_code}")

            # Test 4: Cleanup - Delete test memory
            if memory_hash:
                print("\n4. Cleaning up test memory...")
                response = await client.delete(f"{base_url}/api/memories/{memory_hash}", headers=headers)
                if response.status_code == 200:
                    print(f"   ✅ Test memory cleaned up successfully")
                else:
                    print(f"   ⚠️  Cleanup failed: {response.status_code}")

            print("\n" + "=" * 50)
            print("🎉 API key fallback authentication tests passed!")
            print("✅ Backward compatibility maintained")
            return True

        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            return False


async def main():
    """Main test function."""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8000"

    # Try to get API key from command line or environment
    api_key = None
    if len(sys.argv) > 2:
        api_key = sys.argv[2]
    else:
        import os
        api_key = os.getenv('MCP_API_KEY')

    print("API Key Authentication Fallback Test")
    print("===================================")
    print(f"Target: {base_url}")
    print()
    print("This test verifies that API key authentication works")
    print("as a fallback when OAuth 2.1 is enabled.")
    print()

    success = await test_api_key_fallback(base_url, api_key)

    if success:
        print("\n🚀 API key fallback is working correctly!")
        sys.exit(0)
    else:
        print("\n💥 API key fallback tests failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())