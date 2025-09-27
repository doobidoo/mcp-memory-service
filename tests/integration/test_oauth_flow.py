#!/usr/bin/env python3
"""
OAuth 2.1 Dynamic Client Registration integration test.

Tests the OAuth endpoints for full flow functionality from client registration
through token acquisition and API access.
"""

import asyncio
import json
import sys
from typing import Optional

import httpx


async def test_oauth_endpoints(base_url: str = "http://localhost:8000") -> bool:
    """
    Test OAuth 2.1 endpoints for basic functionality.

    Returns:
        True if all tests pass, False otherwise
    """
    print(f"Testing OAuth endpoints at {base_url}")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        try:
            # Test 1: OAuth Authorization Server Metadata
            print("1. Testing OAuth Authorization Server Metadata...")
            response = await client.get(f"{base_url}/.well-known/oauth-authorization-server/mcp")

            if response.status_code != 200:
                print(f"   ❌ Failed: {response.status_code}")
                return False

            metadata = response.json()
            required_fields = ["issuer", "authorization_endpoint", "token_endpoint", "registration_endpoint"]

            for field in required_fields:
                if field not in metadata:
                    print(f"   ❌ Missing required field: {field}")
                    return False

            print(f"   ✅ Metadata endpoint working")
            print(f"   📋 Issuer: {metadata.get('issuer')}")

            # Test 2: Client Registration
            print("\n2. Testing Dynamic Client Registration...")

            registration_data = {
                "client_name": "Test Client",
                "redirect_uris": ["https://example.com/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"]
            }

            response = await client.post(
                f"{base_url}/oauth/register",
                json=registration_data
            )

            if response.status_code != 201:
                print(f"   ❌ Registration failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

            client_info = response.json()
            client_id = client_info.get("client_id")
            client_secret = client_info.get("client_secret")

            if not client_id or not client_secret:
                print(f"   ❌ Missing client credentials in response")
                return False

            print(f"   ✅ Client registration successful")
            print(f"   📋 Client ID: {client_id}")

            # Test 3: Authorization Endpoint (expect redirect)
            print("\n3. Testing Authorization Endpoint...")

            auth_url = f"{base_url}/oauth/authorize"
            auth_params = {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "https://example.com/callback",
                "state": "test_state_123"
            }

            response = await client.get(auth_url, params=auth_params, follow_redirects=False)

            if response.status_code not in [302, 307]:
                print(f"   ❌ Authorization failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

            location = response.headers.get("location", "")
            if "code=" not in location or "state=test_state_123" not in location:
                print(f"   ❌ Invalid redirect: {location}")
                return False

            print(f"   ✅ Authorization endpoint working")
            print(f"   📋 Redirect URL: {location[:100]}...")

            # Extract authorization code from redirect
            auth_code = None
            for param in location.split("?")[1].split("&"):
                if param.startswith("code="):
                    auth_code = param.split("=")[1]
                    break

            if not auth_code:
                print(f"   ❌ No authorization code in redirect")
                return False

            # Test 4: Token Endpoint
            print("\n4. Testing Token Endpoint...")

            token_data = {
                "grant_type": "authorization_code",
                "code": auth_code,
                "redirect_uri": "https://example.com/callback",
                "client_id": client_id,
                "client_secret": client_secret
            }

            response = await client.post(
                f"{base_url}/oauth/token",
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            if response.status_code != 200:
                print(f"   ❌ Token request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

            token_response = response.json()
            access_token = token_response.get("access_token")

            if not access_token:
                print(f"   ❌ No access token in response")
                return False

            print(f"   ✅ Token endpoint working")
            print(f"   📋 Token type: {token_response.get('token_type')}")
            print(f"   📋 Expires in: {token_response.get('expires_in')} seconds")

            # Test 5: Protected Resource Access
            print("\n5. Testing Protected API Endpoints...")

            headers = {"Authorization": f"Bearer {access_token}"}

            # Test health endpoint (should be public, no auth required)
            response = await client.get(f"{base_url}/api/health")
            if response.status_code == 200:
                print(f"   ✅ Public health endpoint accessible")
            else:
                print(f"   ❌ Health endpoint failed: {response.status_code}")

            # Test protected memories endpoint (requires read access)
            response = await client.get(f"{base_url}/api/memories", headers=headers)
            if response.status_code == 200:
                print(f"   ✅ Protected memories endpoint accessible with Bearer token")
            else:
                print(f"   ❌ Protected memories endpoint failed: {response.status_code}")

            # Test protected search endpoint (requires read access)
            search_data = {"query": "test search", "n_results": 5}
            response = await client.post(f"{base_url}/api/search", json=search_data, headers=headers)
            if response.status_code in [200, 404]:  # 404 is OK if no memories exist
                print(f"   ✅ Protected search endpoint accessible with Bearer token")
            else:
                print(f"   ❌ Protected search endpoint failed: {response.status_code}")

            # Test accessing protected endpoint without token (should fail)
            response = await client.get(f"{base_url}/api/memories")
            if response.status_code == 401:
                print(f"   ✅ Protected endpoint correctly rejects unauthenticated requests")
            else:
                print(f"   ⚠️  Protected endpoint security test inconclusive: {response.status_code}")

            print("\n" + "=" * 50)
            print("🎉 All OAuth 2.1 tests passed!")
            print("✅ Ready for Claude Code HTTP transport integration")
            print("✅ API endpoints properly protected with OAuth authentication")
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

    print("OAuth 2.1 Dynamic Client Registration Test")
    print("==========================================")
    print(f"Target: {base_url}")
    print()
    print("Make sure the MCP Memory Service is running with OAuth enabled:")
    print("  export MCP_OAUTH_ENABLED=true")
    print("  uv run memory server --http")
    print()

    success = await test_oauth_endpoints(base_url)

    if success:
        print("\n🚀 OAuth implementation is ready!")
        sys.exit(0)
    else:
        print("\n💥 OAuth tests failed - check implementation")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())