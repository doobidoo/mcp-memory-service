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

"""Test script for search API endpoints."""

import requests

BASE_URL = "http://localhost:8000"


def test_search_functionality():
    """Test all search endpoints."""

    print("Testing Search API Endpoints")
    print("=" * 40)

    # First, check server health
    print("\n[0] Health check...")
    try:
        resp = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if resp.status_code != 200:
            print(f"[FAIL] Server not healthy: {resp.status_code}")
            return
        print("[PASS] Server is healthy")
    except Exception as e:
        print(f"[FAIL] Cannot connect: {e}")
        return

    # Create some test memories for searching
    print("\n[1] Creating test memories...")
    test_memories = [
        {
            "content": "Python programming tutorial for beginners",
            "tags": ["python", "programming", "tutorial"],
            "memory_type": "learning",
            "metadata": {"difficulty": "beginner"},
        },
        {
            "content": "Advanced machine learning algorithms with PyTorch",
            "tags": ["python", "machine-learning", "pytorch"],
            "memory_type": "learning",
            "metadata": {"difficulty": "advanced"},
        },
        {
            "content": "JavaScript async await patterns and best practices",
            "tags": ["javascript", "async", "programming"],
            "memory_type": "reference",
            "metadata": {"language": "js"},
        },
        {
            "content": "Database design principles and normalization",
            "tags": ["database", "design", "sql"],
            "memory_type": "learning",
            "metadata": {"topic": "databases"},
        },
        {
            "content": "Meeting notes from yesterday's project sync",
            "tags": ["meeting", "project", "notes"],
            "memory_type": "note",
            "metadata": {"date": "yesterday"},
        },
    ]

    created_hashes = []
    for i, memory in enumerate(test_memories):
        try:
            resp = requests.post(
                f"{BASE_URL}/api/memories", json=memory, headers={"Content-Type": "application/json"}, timeout=10
            )
            if resp.status_code == 200:
                result = resp.json()
                if result["success"]:
                    created_hashes.append(result["content_hash"])
                    print(f"  Created memory {i + 1}: {memory['content'][:30]}...")
                else:
                    print(f"  [WARN] Memory {i + 1} might already exist")
            else:
                print(f"  [WARN] Failed to create memory {i + 1}: {resp.status_code}")
        except Exception as e:
            print(f"  [WARN] Error creating memory {i + 1}: {e}")

    print(f"[INFO] Created {len(created_hashes)} new memories")

    # Test 2: Semantic search
    print("\n[2] Testing semantic search...")
    search_queries = ["programming tutorial", "machine learning AI", "database SQL design", "meeting project discussion"]

    for query in search_queries:
        try:
            search_request = {"query": query, "n_results": 3, "similarity_threshold": 0.1}

            resp = requests.post(
                f"{BASE_URL}/api/search", json=search_request, headers={"Content-Type": "application/json"}, timeout=15
            )

            if resp.status_code == 200:
                result = resp.json()
                print(f"  Query: '{query}' -> {result['total_found']} results ({result['processing_time_ms']:.1f}ms)")

                for i, search_result in enumerate(result["results"][:2]):  # Show top 2
                    memory = search_result["memory"]
                    score = search_result.get("similarity_score", 0)
                    print(f"    {i + 1}. {memory['content'][:50]}... (score: {score:.3f})")
            else:
                print(f"  [FAIL] Search failed for '{query}': {resp.status_code}")

        except Exception as e:
            print(f"  [FAIL] Search error for '{query}': {e}")

    # Test 3: Tag-based search
    print("\n[3] Testing tag-based search...")
    tag_searches = [
        {"tags": ["python"], "match_all": False},
        {"tags": ["programming", "tutorial"], "match_all": False},
        {"tags": ["python", "programming"], "match_all": True},
    ]

    for search in tag_searches:
        try:
            resp = requests.post(
                f"{BASE_URL}/api/search/by-tag", json=search, headers={"Content-Type": "application/json"}, timeout=10
            )

            if resp.status_code == 200:
                result = resp.json()
                match_type = "ALL" if search["match_all"] else "ANY"
                print(f"  Tags {search['tags']} ({match_type}) -> {result['total_found']} results")

                for i, search_result in enumerate(result["results"][:2]):
                    memory = search_result["memory"]
                    print(f"    {i + 1}. {memory['content'][:40]}... (tags: {memory['tags']})")
            else:
                print(f"  [FAIL] Tag search failed: {resp.status_code}")

        except Exception as e:
            print(f"  [FAIL] Tag search error: {e}")

    # Test 4: Time-based search
    print("\n[4] Testing time-based search...")
    time_queries = ["today", "yesterday", "this week", "last week"]

    for query in time_queries:
        try:
            time_request = {"query": query, "n_results": 5}

            resp = requests.post(
                f"{BASE_URL}/api/search/by-time", json=time_request, headers={"Content-Type": "application/json"}, timeout=10
            )

            if resp.status_code == 200:
                result = resp.json()
                print(f"  Time: '{query}' -> {result['total_found']} results")

                if result["results"]:
                    memory = result["results"][0]["memory"]
                    print(f"    Example: {memory['content'][:40]}...")
            elif resp.status_code == 400:
                print(f"  [INFO] Time query '{query}' not supported yet")
            else:
                print(f"  [FAIL] Time search failed for '{query}': {resp.status_code}")

        except Exception as e:
            print(f"  [FAIL] Time search error for '{query}': {e}")

    # Test 5: Similar memories
    print("\n[5] Testing similar memory search...")
    if created_hashes:
        try:
            content_hash = created_hashes[0]
            resp = requests.get(f"{BASE_URL}/api/search/similar/{content_hash}?n_results=3", timeout=10)

            if resp.status_code == 200:
                result = resp.json()
                print(f"  Similar to first memory -> {result['total_found']} results")

                for i, search_result in enumerate(result["results"][:2]):
                    memory = search_result["memory"]
                    score = search_result.get("similarity_score", 0)
                    print(f"    {i + 1}. {memory['content'][:40]}... (score: {score:.3f})")
            elif resp.status_code == 404:
                print("  [INFO] Memory not found (expected with current get-by-hash implementation)")
            else:
                print(f"  [FAIL] Similar search failed: {resp.status_code}")

        except Exception as e:
            print(f"  [FAIL] Similar search error: {e}")

    # Cleanup: Delete test memories
    print(f"\n[6] Cleaning up {len(created_hashes)} test memories...")
    for content_hash in created_hashes:
        try:
            resp = requests.delete(f"{BASE_URL}/api/memories/{content_hash}", timeout=5)
            if resp.status_code == 200:
                result = resp.json()
                if result["success"]:
                    print(f"  Deleted: {content_hash[:12]}...")
        except Exception as e:
            print(f"  [WARN] Cleanup error: {e}")

    print("\n" + "=" * 40)
    print("Search API testing completed!")


if __name__ == "__main__":
    test_search_functionality()
