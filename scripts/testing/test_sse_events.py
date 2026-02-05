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
Test client for Server-Sent Events functionality.

This script connects to the SSE endpoint and displays real-time events
while performing memory operations to trigger them.
"""

import asyncio
import json

import aiohttp

BASE_URL = "http://10.0.1.30:8000"


class SSETestClient:
    """Simple SSE test client."""

    def __init__(self):
        self.session: aiohttp.ClientSession | None = None
        self.sse_task: asyncio.Task | None = None
        self.running = False

    async def start(self):
        """Start the SSE client."""
        self.session = aiohttp.ClientSession()
        self.running = True

        # Start SSE listening task
        self.sse_task = asyncio.create_task(self._listen_to_events())

        print("SSE Test Client started")
        print("Connecting to SSE stream...")

    async def stop(self):
        """Stop the SSE client."""
        self.running = False

        if self.sse_task:
            self.sse_task.cancel()
            try:
                await self.sse_task
            except asyncio.CancelledError:
                pass

        if self.session:
            await self.session.close()

        print("\nSSE Test Client stopped")

    async def _listen_to_events(self):
        """Listen to SSE events from the server."""
        try:
            async with self.session.get(f"{BASE_URL}/api/events") as response:
                if response.status != 200:
                    print(f"Failed to connect to SSE: {response.status}")
                    return

                print("✅ Connected to SSE stream!")
                print("Listening for events...\n")

                async for line in response.content:
                    if not self.running:
                        break

                    line = line.decode("utf-8").strip()

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])  # Remove 'data: ' prefix
                            self._handle_event(data)
                        except json.JSONDecodeError:
                            print(f"Invalid JSON: {line}")
                    elif line.startswith("event: "):
                        event_type = line[7:]  # Remove 'event: ' prefix
                        print(f"Event type: {event_type}")

        except asyncio.CancelledError:
            print("SSE connection cancelled")
        except Exception as e:
            print(f"SSE connection error: {e}")

    def _handle_event(self, data: dict):
        """Handle incoming SSE events."""
        timestamp = data.get("timestamp", "Unknown")
        event_time = timestamp.split("T")[1][:8] if "T" in timestamp else timestamp

        # Format different event types
        if "connection_id" in data:
            print(f"[{event_time}] 🔌 Connection: {data.get('message', 'Unknown')}")

        elif "content_hash" in data and "memory_stored" in str(data):
            hash_short = data["content_hash"][:12] + "..."
            content_preview = data.get("content_preview", "No preview")
            tags = data.get("tags", [])
            print(f"[{event_time}] 💾 Memory stored: {hash_short}")
            print(f"    Content: {content_preview}")
            if tags:
                print(f"    Tags: {', '.join(tags)}")

        elif "content_hash" in data and "memory_deleted" in str(data):
            hash_short = data["content_hash"][:12] + "..."
            success = data.get("success", False)
            status = "✅" if success else "❌"
            print(f"[{event_time}] 🗑️  Memory deleted {status}: {hash_short}")

        elif "query" in data and "search_completed" in str(data):
            query = data.get("query", "Unknown")
            results_count = data.get("results_count", 0)
            search_type = data.get("search_type", "unknown")
            processing_time = data.get("processing_time_ms", 0)
            print(f"[{event_time}] 🔍 Search completed ({search_type}): '{query}'")
            print(f"    Results: {results_count}, Time: {processing_time:.1f}ms")

        elif "server_status" in data:
            status = data.get("server_status", "unknown")
            connections = data.get("active_connections", 0)
            print(f"[{event_time}] 💓 Heartbeat: {status} ({connections} connections)")

        else:
            # Generic event display
            print(f"[{event_time}] 📨 Event: {json.dumps(data, indent=2)}")

        print()  # Empty line for readability


async def run_memory_operations():
    """Run some memory operations to trigger SSE events."""
    await asyncio.sleep(2)  # Give SSE time to connect

    print("🚀 Starting memory operations to trigger events...\n")

    async with aiohttp.ClientSession() as session:
        # Test 1: Store some memories
        test_memories = [
            {
                "content": "SSE test memory 1 - This is for testing real-time events",
                "tags": ["sse", "test", "realtime"],
                "memory_type": "test",
            },
            {
                "content": "SSE test memory 2 - Another test memory for event streaming",
                "tags": ["sse", "streaming", "demo"],
                "memory_type": "demo",
            },
        ]

        stored_hashes = []

        for i, memory in enumerate(test_memories):
            print(f"Storing memory {i + 1}...")
            try:
                async with session.post(
                    f"{BASE_URL}/api/memories", json=memory, headers={"Content-Type": "application/json"}, timeout=10
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if result["success"]:
                            stored_hashes.append(result["content_hash"])
                            print(f"  ✅ Stored: {result['content_hash'][:12]}...")
                    await asyncio.sleep(1)  # Pause between operations
            except Exception as e:
                print(f"  ❌ Error: {e}")

        await asyncio.sleep(2)

        # Test 2: Perform searches
        print("Performing searches...")
        search_queries = [{"query": "SSE test memory", "n_results": 5}, {"tags": ["sse"], "match_all": False}]

        # Semantic search
        try:
            async with session.post(
                f"{BASE_URL}/api/search", json=search_queries[0], headers={"Content-Type": "application/json"}, timeout=10
            ) as resp:
                if resp.status == 200:
                    print("  ✅ Semantic search completed")
                await asyncio.sleep(1)
        except Exception as e:
            print(f"  ❌ Search error: {e}")

        # Tag search
        try:
            async with session.post(
                f"{BASE_URL}/api/search/by-tag", json=search_queries[1], headers={"Content-Type": "application/json"}, timeout=10
            ) as resp:
                if resp.status == 200:
                    print("  ✅ Tag search completed")
                await asyncio.sleep(1)
        except Exception as e:
            print(f"  ❌ Tag search error: {e}")

        await asyncio.sleep(2)

        # Test 3: Delete memories
        print("Deleting memories...")
        for content_hash in stored_hashes:
            try:
                async with session.delete(f"{BASE_URL}/api/memories/{content_hash}", timeout=10) as resp:
                    if resp.status == 200:
                        print(f"  ✅ Deleted: {content_hash[:12]}...")
                    await asyncio.sleep(1)
            except Exception as e:
                print(f"  ❌ Delete error: {e}")


async def main():
    """Main test function."""
    print("SSE Events Test Client")
    print("=" * 40)

    # Check if server is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/api/health", timeout=5) as resp:
                if resp.status != 200:
                    print("❌ Server is not healthy")
                    return
                print("✅ Server is healthy")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure server is running: python scripts/run_http_server.py")
        return

    print()

    # Create and start SSE client
    client = SSETestClient()
    await client.start()

    try:
        # Run memory operations in parallel with SSE listening
        operations_task = asyncio.create_task(run_memory_operations())

        # Wait for operations to complete
        await operations_task

        # Keep listening for a few more seconds to catch any final events
        print("Waiting for final events...")
        await asyncio.sleep(3)

    finally:
        await client.stop()

    print("\n🎉 SSE test completed!")


if __name__ == "__main__":
    asyncio.run(main())
