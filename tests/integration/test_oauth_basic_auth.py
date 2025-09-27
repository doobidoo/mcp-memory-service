#!/usr/bin/env python3
"""
OAuth 2.1 Basic Authentication Test

Tests both client_secret_basic (HTTP Basic auth) and client_secret_post (form data)
authentication methods for the OAuth token endpoint.
"""

import asyncio
import base64
import sys
from typing import Optional

import httpx


async def test_oauth_basic_auth(base_url: str = "http://localhost:8000") -> bool:
    """
    Test OAuth 2.1 token endpoint with both Basic and form authentication.

    Returns:
        True if all tests pass, False otherwise
    """
    print(f"Testing OAuth Basic Authentication at {base_url}")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        try:
            # Step 1: Register a client first
            print("1. Registering OAuth client...")

            registration_data = {
                "client_name": "Basic Auth Test Client",
                "redirect_uris": ["https://example.com/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"]
            }

            response = await client.post(
                f"{base_url}/oauth/register",
                json=registration_data
            )

            if response.status_code != 201:
                print(f"   ❌ Client registration failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

            client_info = response.json()
            client_id = client_info.get("client_id")
            client_secret = client_info.get("client_secret")

            if not client_id or not client_secret:
                print(f"   ❌ Missing client credentials in response")
                return False

            print(f"   ✅ Client registered successfully")
            print(f"   📋 Client ID: {client_id}")

            # Step 2: Get authorization code
            print("\n2. Getting authorization code...")

            auth_params = {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "https://example.com/callback",
                "state": "test_state_basic_auth"
            }

            response = await client.get(
                f"{base_url}/oauth/authorize",
                params=auth_params,
                follow_redirects=False
            )

            if response.status_code not in [302, 307]:
                print(f"   ❌ Authorization failed: {response.status_code}")
                return False

            location = response.headers.get("location", "")
            if "code=" not in location:
                print(f"   ❌ No authorization code in redirect: {location}")
                return False

            # Extract authorization code
            auth_code = None
            for param in location.split("?")[1].split("&"):
                if param.startswith("code="):
                    auth_code = param.split("=")[1]
                    break

            if not auth_code:
                print(f"   ❌ Could not extract authorization code")
                return False

            print(f"   ✅ Authorization code obtained")

            # Step 3: Test token endpoint with HTTP Basic authentication
            print("\n3. Testing Token Endpoint with HTTP Basic Auth...")

            # Create Basic auth header
            credentials = f"{client_id}:{client_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            basic_auth_header = f"Basic {encoded_credentials}"

            token_data = {
                "grant_type": "authorization_code",
                "code": auth_code,
                "redirect_uri": "https://example.com/callback"
                # Note: client_id and client_secret NOT in form data for Basic auth
            }

            response = await client.post(
                f"{base_url}/oauth/token",
                data=token_data,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Authorization": basic_auth_header
                }
            )

            if response.status_code != 200:
                print(f"   ❌ Basic auth token request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

            basic_token_response = response.json()
            basic_access_token = basic_token_response.get("access_token")

            if not basic_access_token:
                print(f"   ❌ No access token in Basic auth response")
                return False

            print(f"   ✅ HTTP Basic authentication successful")
            print(f"   📋 Token type: {basic_token_response.get('token_type')}")

            # Step 4: Test the access token works for API calls
            print("\n4. Testing Basic auth access token...")

            headers = {"Authorization": f"Bearer {basic_access_token}"}
            response = await client.get(f"{base_url}/api/memories", headers=headers)

            if response.status_code == 200:
                print(f"   ✅ Basic auth access token works for API calls")
            else:
                print(f"   ❌ Basic auth access token failed API call: {response.status_code}")
                return False

            # Step 5: Get a new authorization code for form-based test
            print("\n5. Getting new authorization code for form auth test...")

            auth_params["state"] = "test_state_form_auth"
            response = await client.get(
                f"{base_url}/oauth/authorize",
                params=auth_params,
                follow_redirects=False
            )

            location = response.headers.get("location", "")
            form_auth_code = None
            for param in location.split("?")[1].split("&"):
                if param.startswith("code="):
                    form_auth_code = param.split("=")[1]
                    break

            if not form_auth_code:
                print(f"   ❌ Could not get new authorization code")
                return False

            print(f"   ✅ New authorization code obtained")

            # Step 6: Test token endpoint with form-based authentication
            print("\n6. Testing Token Endpoint with Form-based Auth...")

            token_data = {
                "grant_type": "authorization_code",
                "code": form_auth_code,
                "redirect_uri": "https://example.com/callback",
                "client_id": client_id,
                "client_secret": client_secret
                # Note: credentials in form data, NO Authorization header
            }

            response = await client.post(
                f"{base_url}/oauth/token",
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
                # Note: NO Authorization header
            )

            if response.status_code != 200:
                print(f"   ❌ Form auth token request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

            form_token_response = response.json()
            form_access_token = form_token_response.get("access_token")

            if not form_access_token:
                print(f"   ❌ No access token in form auth response")
                return False

            print(f"   ✅ Form-based authentication successful")
            print(f"   📋 Token type: {form_token_response.get('token_type')}")

            # Step 7: Test the form-based access token works for API calls
            print("\n7. Testing form auth access token...")

            headers = {"Authorization": f"Bearer {form_access_token}"}
            response = await client.get(f"{base_url}/api/memories", headers=headers)

            if response.status_code == 200:
                print(f"   ✅ Form auth access token works for API calls")
            else:
                print(f"   ❌ Form auth access token failed API call: {response.status_code}")
                return False

            print("\n" + "=" * 60)
            print("🎉 All OAuth authentication methods work correctly!")
            print("✅ HTTP Basic authentication (client_secret_basic)")
            print("✅ Form-based authentication (client_secret_post)")
            print("✅ Both access tokens work for protected API endpoints")
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

    print("OAuth 2.1 Basic Authentication Test")
    print("===================================")
    print(f"Target: {base_url}")
    print()
    print("This test verifies both HTTP Basic and form-based authentication")
    print("methods work correctly with the OAuth token endpoint.")
    print()

    success = await test_oauth_basic_auth(base_url)

    if success:
        print("\n🚀 OAuth Basic authentication implementation is working perfectly!")
        sys.exit(0)
    else:
        print("\n💥 OAuth Basic authentication tests failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())