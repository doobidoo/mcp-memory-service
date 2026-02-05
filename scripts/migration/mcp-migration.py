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
Enhanced migration script for MCP Memory Service.
This script handles migration of memories between different ChromaDB instances,
with support for both local and remote migrations.
"""

import os
import sys
import time

import chromadb
from chromadb import HttpClient, Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Import our environment verifier
from verify_environment import EnvironmentVerifier


def verify_environment():
    """Verify the environment before proceeding with migration"""
    verifier = EnvironmentVerifier()
    verifier.run_verifications()
    if not verifier.print_results():
        print("\n⚠️  Environment verification failed! Migration cannot proceed.")
        sys.exit(1)
    print("\n✓ Environment verification passed! Proceeding with migration.")


# Load environment variables
load_dotenv()


def get_claude_desktop_chroma_path():
    """Get ChromaDB path from Claude Desktop config"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, "claude_config", "mcp-memory", "chroma_db")
    print(f"Using ChromaDB path: {config_path}")
    return config_path


def migrate_memories(source_type, source_config, target_type, target_config):
    """
    Migrate memories between ChromaDB instances.

    Args:
        source_type: 'local' or 'remote'
        source_config: For local: path to ChromaDB, for remote: {'host': host, 'port': port}
        target_type: 'local' or 'remote'
        target_config: For local: path to ChromaDB, for remote: {'host': host, 'port': port}
    """
    print(f"Starting migration from {source_type} to {target_type}")

    try:
        # Set up embedding function
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        # Connect to target ChromaDB
        if target_type == "remote":
            target_client = HttpClient(host=target_config["host"], port=target_config["port"])
            print(f"Connected to remote ChromaDB at {target_config['host']}:{target_config['port']}")
        else:
            settings = Settings(
                anonymized_telemetry=False, allow_reset=True, is_persistent=True, persist_directory=target_config
            )
            target_client = chromadb.Client(settings)
            print(f"Connected to local ChromaDB at {target_config}")

        # Get or create collection for imported memories
        try:
            target_collection = target_client.get_collection(name="mcp_imported_memories", embedding_function=embedding_function)
            print("Found existing collection 'mcp_imported_memories' on target")
        except Exception:
            target_collection = target_client.create_collection(
                name="mcp_imported_memories", metadata={"hnsw:space": "cosine"}, embedding_function=embedding_function
            )
            print("Created new collection 'mcp_imported_memories' on target")

        # Connect to source ChromaDB
        if source_type == "remote":
            source_client = HttpClient(host=source_config["host"], port=source_config["port"])
            print(f"Connected to remote ChromaDB at {source_config['host']}:{source_config['port']}")
        else:
            settings = Settings(
                anonymized_telemetry=False, allow_reset=True, is_persistent=True, persist_directory=source_config
            )
            source_client = chromadb.Client(settings)
            print(f"Connected to local ChromaDB at {source_config}")

        # List collections
        collections = source_client.list_collections()
        print(f"Found {len(collections)} collections in source")
        for coll in collections:
            print(f"- {coll.name}")

        # Try to get the memory collection
        try:
            source_collection = source_client.get_collection(name="memory_collection", embedding_function=embedding_function)
            print("Found source memory collection")
        except ValueError as e:
            print(f"Error accessing source collection: {str(e)}")
            return

        # Get all memories from source
        print("Fetching source memories...")
        results = source_collection.get()

        if not results["ids"]:
            print("No memories found in source collection")
            return

        print(f"Found {len(results['ids'])} memories to migrate")

        # Check for existing memories in target to avoid duplicates
        target_existing = target_collection.get()
        existing_ids = set(target_existing["ids"])

        # Filter out already migrated memories
        new_memories = {"ids": [], "documents": [], "metadatas": []}

        for i, memory_id in enumerate(results["ids"]):
            if memory_id not in existing_ids:
                new_memories["ids"].append(memory_id)
                new_memories["documents"].append(results["documents"][i])
                new_memories["metadatas"].append(results["metadatas"][i])

        if not new_memories["ids"]:
            print("All memories are already migrated!")
            return

        print(f"Found {len(new_memories['ids'])} new memories to migrate")

        # Import in batches of 10
        batch_size = 10
        for i in range(0, len(new_memories["ids"]), batch_size):
            batch_end = min(i + batch_size, len(new_memories["ids"]))

            batch_ids = new_memories["ids"][i:batch_end]
            batch_documents = new_memories["documents"][i:batch_end]
            batch_metadatas = new_memories["metadatas"][i:batch_end]

            print(f"Migrating batch {i // batch_size + 1} ({len(batch_ids)} memories)...")

            target_collection.add(documents=batch_documents, metadatas=batch_metadatas, ids=batch_ids)

            # Small delay between batches
            time.sleep(1)

        print("\nMigration complete!")

        # Verify migration
        target_results = target_collection.get()
        print(f"Verification: {len(target_results['ids'])} total memories in target collection")

    except Exception as e:
        print(f"Error during migration: {str(e)}")
        print("Please ensure both ChromaDB instances are running and accessible")


if __name__ == "__main__":
    # First verify the environment
    verify_environment()

    # Example usage:
    # Local to remote migration
    migrate_memories(
        source_type="local",
        source_config=get_claude_desktop_chroma_path(),
        target_type="remote",
        target_config={"host": "16.171.169.46", "port": 8000},
    )

    # Remote to local migration
    # migrate_memories(
    #     source_type='remote',
    #     source_config={'host': '16.171.169.46', 'port': 8000},
    #     target_type='local',
    #     target_config=get_claude_desktop_chroma_path()
    # )
