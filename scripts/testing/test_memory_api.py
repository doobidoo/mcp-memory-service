#!/usr/bin/env python3
# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test script for memory CRUD operations via HTTP API.

This script tests the basic memory operations to ensure they work correctly.
"""

import asyncio
import time

import aiohttp

BASE_URL = "http://localhost:8000"


async def test_memory_crud():
    """Test the complete CRUD workflow for memories."""

    async with aiohttp.ClientSession() as session:
        print("Testing Memory CRUD Operations")
        print("=" * 50)

        # Test 1: Health check
        print("\n[1] Testing health check...")
        try:
            async with session.get(f"{BASE_URL}/api/health") as resp:
                if resp.status == 200:
                    health = await resp.json()
                    print(f"[PASS] Health check passed: {health['status']}")
                else:
                    print(f"[FAIL] Health check failed: {resp.status}")
                    return
        except Exception as e:
            print(f"[FAIL] Cannot connect to server: {e}")
            print("NOTE: Make sure the server is running with: python scripts/run_http_server.py")
            return

        # Test 2: Store a memory
        print("\n[2] Testing memory storage...")
        test_memory = {
            "content": "This is a test memory for CRUD operations",
            "tags": ["test", "crud", "api"],
            "memory_type": "test",
            "metadata": {"test_run": time.time(), "importance": "high"},
        }

        try:
            async with session.post(
                f"{BASE_URL}/api/memories", json=test_memory, headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    if result["success"]:
                        content_hash = result["content_hash"]
                        print("[PASS] Memory stored successfully")
                        print(f"   Content hash: {content_hash[:16]}...")
                        print(f"   Message: {result['message']}")
                    else:
                        print(f"[FAIL] Memory storage failed: {result['message']}")
                        return
                else:
                    print(f"[FAIL] Memory storage failed: {resp.status}")
                    error = await resp.text()
                    print(f"   Error: {error}")
                    return
        except Exception as e:
            print(f"[FAIL] Memory storage error: {e}")
            return

        # Test 3: List memories
        print("\n[3] Testing memory listing...")
        try:
            async with session.get(f"{BASE_URL}/api/memories?page=1&page_size=5") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    memories = result["memories"]
                    print(f"✅ Retrieved {len(memories)} memories")
                    print(f"   Total memories: {result['total']}")
                    print(f"   Page: {result['page']}, Has more: {result['has_more']}")

                    if memories:
                        print(f"   First memory preview: {memories[0]['content'][:50]}...")
                else:
                    print(f"❌ Memory listing failed: {resp.status}")
                    error = await resp.text()
                    print(f"   Error: {error}")
        except Exception as e:
            print(f"❌ Memory listing error: {e}")

        # Test 4: Get specific memory
        print("\n4️⃣  Testing specific memory retrieval...")
        try:
            async with session.get(f"{BASE_URL}/api/memories/{content_hash}") as resp:
                if resp.status == 200:
                    memory = await resp.json()
                    print("✅ Retrieved specific memory")
                    print(f"   Content: {memory['content'][:50]}...")
                    print(f"   Tags: {memory['tags']}")
                    print(f"   Type: {memory['memory_type']}")
                elif resp.status == 404:
                    print("⚠️  Memory not found (this might be expected if get-by-hash isn't implemented)")
                else:
                    print(f"❌ Memory retrieval failed: {resp.status}")
                    error = await resp.text()
                    print(f"   Error: {error}")
        except Exception as e:
            print(f"❌ Memory retrieval error: {e}")

        # Test 5: Filter by tag
        print("\n5️⃣  Testing tag filtering...")
        try:
            async with session.get(f"{BASE_URL}/api/memories?tag=test") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    memories = result["memories"]
                    print(f"✅ Retrieved {len(memories)} memories with tag 'test'")
                    if memories:
                        for memory in memories[:2]:  # Show first 2
                            print(f"   - {memory['content'][:40]}... (tags: {memory['tags']})")
                else:
                    print(f"❌ Tag filtering failed: {resp.status}")
        except Exception as e:
            print(f"❌ Tag filtering error: {e}")

        # Test 6: Delete memory
        print("\n6️⃣  Testing memory deletion...")
        try:
            async with session.delete(f"{BASE_URL}/api/memories/{content_hash}") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    if result["success"]:
                        print("✅ Memory deleted successfully")
                        print(f"   Message: {result['message']}")
                    else:
                        print(f"❌ Memory deletion failed: {result['message']}")
                else:
                    print(f"❌ Memory deletion failed: {resp.status}")
                    error = await resp.text()
                    print(f"   Error: {error}")
        except Exception as e:
            print(f"❌ Memory deletion error: {e}")

        # Test 7: Verify deletion
        print("\n7️⃣  Verifying memory deletion...")
        try:
            async with session.get(f"{BASE_URL}/api/memories/{content_hash}") as resp:
                if resp.status == 404:
                    print("✅ Memory successfully deleted (404 as expected)")
                elif resp.status == 200:
                    print("⚠️  Memory still exists after deletion")
                else:
                    print(f"❓ Unexpected response: {resp.status}")
        except Exception as e:
            print(f"❌ Deletion verification error: {e}")

        print("\n" + "=" * 50)
        print("🎉 Memory CRUD testing completed!")


if __name__ == "__main__":
    asyncio.run(test_memory_crud())
