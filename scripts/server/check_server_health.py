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
Health check utility for MCP Memory Service in Claude Desktop.
This script sends MCP protocol requests to check the health of the memory service.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# MCP protocol messages
CHECK_HEALTH_REQUEST = {
    "method": "tools/call",
    "params": {"name": "check_database_health", "arguments": {}},
    "jsonrpc": "2.0",
    "id": 1,
}

CHECK_MODEL_REQUEST = {
    "method": "tools/call",
    "params": {"name": "check_embedding_model", "arguments": {}},
    "jsonrpc": "2.0",
    "id": 2,
}

STORE_MEMORY_REQUEST = {
    "method": "tools/call",
    "params": {
        "name": "store_memory",
        "arguments": {
            "content": f"Health check test memory created on {datetime.now().isoformat()}",
            "metadata": {"tags": ["health-check", "test"], "type": "test"},
        },
    },
    "jsonrpc": "2.0",
    "id": 3,
}

RETRIEVE_MEMORY_REQUEST = {
    "method": "tools/call",
    "params": {"name": "retrieve_memory", "arguments": {"query": "health check test", "n_results": 3}},
    "jsonrpc": "2.0",
    "id": 4,
}

SEARCH_TAG_REQUEST = {
    "method": "tools/call",
    "params": {"name": "search_by_tag", "arguments": {"tags": ["health-check"]}},
    "jsonrpc": "2.0",
    "id": 5,
}


async def write_json(writer, data):
    """Write JSON data to the writer."""
    message = json.dumps(data) + "\r\n"
    writer.write(message.encode())
    await writer.drain()


async def read_json(reader):
    """Read JSON data from the reader."""
    line = await reader.readline()
    if not line:
        return None
    return json.loads(line.decode())


async def check_health(reader, writer):
    """Check database health."""
    logger.info("=== Check 1: Database Health ===")
    await write_json(writer, CHECK_HEALTH_REQUEST)
    response = await read_json(reader)

    if response and "result" in response:
        try:
            text = response["result"]["content"][0]["text"]
            logger.info("Health check response received")
            data = json.loads(text.split("\n", 1)[1])

            # Extract relevant information
            validation_status = data.get("validation", {}).get("status", "unknown")
            has_model = data.get("statistics", {}).get("has_embedding_model", False)
            memory_count = data.get("statistics", {}).get("total_memories", 0)

            if validation_status == "healthy":
                logger.info(f"✅ Database validation status: {validation_status}")
            else:
                logger.error(f"❌ Database validation status: {validation_status}")

            if has_model:
                logger.info(f"✅ Embedding model loaded: {has_model}")
            else:
                logger.error("❌ Embedding model not loaded")

            logger.info(f"Total memories: {memory_count}")
            return data
        except Exception as e:
            logger.error(f"❌ Error parsing health check response: {e}")
    else:
        logger.error(f"❌ Invalid response: {response}")
    return None


async def check_embedding_model(reader, writer):
    """Check embedding model status."""
    logger.info("=== Check 2: Embedding Model ===")
    await write_json(writer, CHECK_MODEL_REQUEST)
    response = await read_json(reader)

    if response and "result" in response:
        try:
            text = response["result"]["content"][0]["text"]
            logger.info("Model check response received")
            data = json.loads(text.split("\n", 1)[1])

            status = data.get("status", "unknown")

            if status == "healthy":
                logger.info(f"✅ Embedding model status: {status}")
                logger.info(f"Model name: {data.get('model_name', 'unknown')}")
                logger.info(f"Dimension: {data.get('embedding_dimension', 0)}")
            else:
                logger.error(f"❌ Embedding model status: {status}")
                logger.error(f"Error: {data.get('error', 'unknown')}")

            return data
        except Exception as e:
            logger.error(f"❌ Error parsing model check response: {e}")
    else:
        logger.error(f"❌ Invalid response: {response}")
    return None


async def store_memory(reader, writer):
    """Store a test memory."""
    logger.info("=== Check 3: Memory Storage ===")
    await write_json(writer, STORE_MEMORY_REQUEST)
    response = await read_json(reader)

    if response and "result" in response:
        try:
            text = response["result"]["content"][0]["text"]
            logger.info(f"Store memory response: {text}")

            if "successfully" in text.lower():
                logger.info("✅ Memory stored successfully")
                return True
            else:
                logger.error(f"❌ Memory storage failed: {text}")
                return False
        except Exception as e:
            logger.error(f"❌ Error parsing store memory response: {e}")
    else:
        logger.error(f"❌ Invalid response: {response}")
    return False


async def retrieve_memory(reader, writer):
    """Retrieve memories using semantic search."""
    logger.info("=== Check 4: Semantic Search ===")
    await write_json(writer, RETRIEVE_MEMORY_REQUEST)
    response = await read_json(reader)

    if response and "result" in response:
        try:
            text = response["result"]["content"][0]["text"]
            logger.info("Retrieve memory response received")

            if "no matching memories" in text.lower():
                logger.warning("⚠️ No memories found via semantic search")
                return False
            else:
                logger.info("✅ Semantic search successful")
                logger.info(f"Response: {text}")
                return True
        except Exception as e:
            logger.error(f"❌ Error parsing retrieve memory response: {e}")
    else:
        logger.error(f"❌ Invalid response: {response}")
    return False


async def search_by_tag(reader, writer):
    """Search memories by tag."""
    logger.info("=== Check 5: Tag Search ===")
    await write_json(writer, SEARCH_TAG_REQUEST)
    response = await read_json(reader)

    if response and "result" in response:
        try:
            text = response["result"]["content"][0]["text"]
            logger.info("Tag search response received")

            if "no memories found" in text.lower():
                logger.warning("⚠️ No memories found via tag search")
                return False
            else:
                logger.info("✅ Tag search successful")
                logger.info(f"Response: {text}")
                return True
        except Exception as e:
            logger.error(f"❌ Error parsing tag search response: {e}")
    else:
        logger.error(f"❌ Invalid response: {response}")
    return False


async def run_health_check():
    """Run all health checks."""
    # Start the server
    server_process = subprocess.Popen(
        ["/bin/bash", "/Users/hkr/Documents/GitHub/mcp-memory-service/run_mcp_memory.sh"],
        cwd="/Users/hkr/Documents/GitHub/mcp-memory-service",
        env={
            "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
            "MCP_MEMORY_SQLITE_PATH": os.path.expanduser("~/Library/Application Support/mcp-memory/sqlite_vec.db"),
            "MCP_MEMORY_BACKUPS_PATH": os.path.expanduser("~/Library/Application Support/mcp-memory/backups"),
            "MCP_MEMORY_USE_ONNX": "1",
            "MCP_MEMORY_USE_HOMEBREW_PYTORCH": "1",
            "PYTHONPATH": "/Users/hkr/Documents/GitHub/mcp-memory-service",
            "LOG_LEVEL": "INFO",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    logger.info("Server started, waiting for initialization...")
    await asyncio.sleep(5)  # Wait for server to start

    reader = None
    writer = None
    success = False

    try:
        # Connect to the server
        reader, writer = await asyncio.open_connection("127.0.0.1", 6789)
        logger.info("Connected to server")

        # Initialize the server
        await write_json(
            writer,
            {
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "health-check", "version": "1.0.0"},
                },
                "jsonrpc": "2.0",
                "id": 0,
            },
        )

        init_response = await read_json(reader)
        logger.info(f"Initialization response: {init_response is not None}")

        # Run health checks
        health_data = await check_health(reader, writer)
        model_data = await check_embedding_model(reader, writer)
        store_success = await store_memory(reader, writer)
        search_success = await search_by_tag(reader, writer)
        retrieve_success = await retrieve_memory(reader, writer)

        # Summarize results
        logger.info("=== Health Check Summary ===")
        if health_data and health_data.get("validation", {}).get("status") == "healthy":
            logger.info("✅ Database health: GOOD")
        else:
            logger.error("❌ Database health: FAILED")
            success = False

        if model_data and model_data.get("status") == "healthy":
            logger.info("✅ Embedding model: GOOD")
        else:
            logger.error("❌ Embedding model: FAILED")
            success = False

        if store_success:
            logger.info("✅ Memory storage: GOOD")
        else:
            logger.error("❌ Memory storage: FAILED")
            success = False

        if search_success:
            logger.info("✅ Tag search: GOOD")
        else:
            logger.warning("⚠️ Tag search: NO RESULTS")

        if retrieve_success:
            logger.info("✅ Semantic search: GOOD")
        else:
            logger.warning("⚠️ Semantic search: NO RESULTS")

        success = (
            health_data
            and health_data.get("validation", {}).get("status") == "healthy"
            and model_data
            and model_data.get("status") == "healthy"
            and store_success
        )

        # Shutdown server
        await write_json(writer, {"method": "shutdown", "params": {}, "jsonrpc": "2.0", "id": 99})

        shutdown_response = await read_json(reader)
        logger.info(f"Shutdown response: {shutdown_response is not None}")

    except Exception as e:
        logger.error(f"Error during health check: {e}")
        success = False
    finally:
        # Clean up
        if writer:
            writer.close()
            await writer.wait_closed()

        # Terminate server
        try:
            os.killpg(os.getpgid(server_process.pid), 15)
        except:
            pass

        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except:
            server_process.kill()

    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MCP Memory Service Health Check")
    parser.parse_args()

    logger.info("Starting health check...")
    success = asyncio.run(run_health_check())

    if success:
        logger.info("Health check completed successfully!")
        sys.exit(0)
    else:
        logger.error("Health check failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
