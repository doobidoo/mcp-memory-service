#!/usr/bin/env python3
"""
Migrate memories from embedded Qdrant to Qdrant server.

This script exports memories from the embedded Qdrant instance (with file locks)
and imports them to the Qdrant server instance for multi-client access.

Usage:
    python scripts/migrate/migrate_to_qdrant_server.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_from_embedded(embedded_path: str, collection_name: str = "memories") -> list[dict[str, Any]]:
    """Export all memories from embedded Qdrant."""
    logger.info(f"Connecting to embedded Qdrant at {embedded_path}")

    # Connect to embedded instance
    client = QdrantClient(path=embedded_path)

    # Check if collection exists
    collections = client.get_collections()
    if collection_name not in [c.name for c in collections.collections]:
        logger.error(f"Collection '{collection_name}' not found in embedded storage")
        logger.info(f"Available collections: {[c.name for c in collections.collections]}")
        return []

    # Get collection info
    collection_info = client.get_collection(collection_name)
    logger.info(f"Collection points count: {collection_info.points_count}")
    logger.info(f"Vector size: {collection_info.config.params.vectors.size}")
    logger.info(f"Distance metric: {collection_info.config.params.vectors.distance}")

    # Export all points with pagination
    all_points = []
    offset = None
    batch_size = 100

    while True:
        # Scroll through points
        result = client.scroll(
            collection_name=collection_name, limit=batch_size, offset=offset, with_payload=True, with_vectors=True
        )

        points, next_offset = result

        if not points:
            break

        # Convert points to serializable format
        for point in points:
            point_data = {"id": point.id, "vector": point.vector, "payload": point.payload}
            all_points.append(point_data)

        logger.info(f"Exported {len(all_points)} points so far...")

        offset = next_offset
        if offset is None:
            break

    logger.info(f"Successfully exported {len(all_points)} points from embedded storage")

    # Close embedded client
    del client

    return all_points


def import_to_server(
    server_url: str,
    points: list[dict[str, Any]],
    collection_name: str = "memories",
    vector_size: int = 384,
    recreate: bool = True,
) -> bool:
    """Import memories to Qdrant server."""
    logger.info(f"Connecting to Qdrant server at {server_url}")

    # Connect to server
    client = QdrantClient(url=server_url)

    # Check server health
    try:
        collections = client.get_collections()
        logger.info(f"Server is healthy. Existing collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant server: {e}")
        return False

    # Recreate or verify collection
    if recreate:
        # Delete existing collection if it exists
        if collection_name in [c.name for c in collections.collections]:
            logger.info(f"Deleting existing collection '{collection_name}'")
            client.delete_collection(collection_name)

        # Create new collection with same parameters
        logger.info(f"Creating collection '{collection_name}' with vector size {vector_size}")
        client.create_collection(
            collection_name=collection_name, vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    else:
        # Verify collection exists and matches
        if collection_name not in [c.name for c in collections.collections]:
            logger.error(f"Collection '{collection_name}' does not exist on server")
            return False

        collection_info = client.get_collection(collection_name)
        if collection_info.config.params.vectors.size != vector_size:
            logger.error(
                f"Vector size mismatch: server has {collection_info.config.params.vectors.size}, data has {vector_size}"
            )
            return False

    # Import points in batches
    batch_size = 100
    total_imported = 0

    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]

        # Convert to PointStruct objects
        point_structs = []
        for point_data in batch:
            point_struct = PointStruct(id=point_data["id"], vector=point_data["vector"], payload=point_data["payload"])
            point_structs.append(point_struct)

        # Upload batch
        try:
            client.upsert(collection_name=collection_name, points=point_structs)
            total_imported += len(point_structs)
            logger.info(f"Imported {total_imported}/{len(points)} points")
        except Exception as e:
            logger.error(f"Failed to import batch: {e}")
            return False

    # Verify import
    collection_info = client.get_collection(collection_name)
    logger.info(f"Import complete. Server collection has {collection_info.points_count} points")

    if collection_info.points_count != len(points):
        logger.warning(f"Point count mismatch: expected {len(points)}, got {collection_info.points_count}")

    return True


def main():
    """Main migration function."""
    # Configuration
    EMBEDDED_PATH = "/data/qdrant"  # Path inside Docker container
    SERVER_URL = "http://localhost:6333"  # Qdrant server URL
    COLLECTION_NAME = "memories"

    # For running from host, use Docker volumes path
    import os

    if not os.path.exists(EMBEDDED_PATH):
        # Try Docker volume inspect to find actual path
        logger.info("Embedded path not found, trying to locate Docker volume...")
        import subprocess

        try:
            result = subprocess.run(
                ["docker", "volume", "inspect", "mcp-memory-service_mcp-memory-data"], capture_output=True, text=True, check=True
            )
            volume_info = json.loads(result.stdout)
            if volume_info:
                mount_point = volume_info[0].get("Mountpoint")
                if mount_point:
                    EMBEDDED_PATH = mount_point
                    logger.info(f"Found Docker volume at: {EMBEDDED_PATH}")
        except Exception as e:
            logger.warning(f"Could not locate Docker volume: {e}")

    # Check if embedded path exists
    if not Path(EMBEDDED_PATH).exists():
        logger.error(f"Embedded storage path not found: {EMBEDDED_PATH}")
        logger.info("Run this script from within the Docker container or adjust the path")
        return 1

    # Export from embedded
    logger.info("=== STEP 1: Export from embedded Qdrant ===")
    points = export_from_embedded(EMBEDDED_PATH, COLLECTION_NAME)

    if not points:
        logger.error("No points exported. Aborting migration.")
        return 1

    # Save backup
    backup_file = Path("qdrant_backup.json")
    logger.info(f"Saving backup to {backup_file}")
    with open(backup_file, "w") as f:
        json.dump(points, f, indent=2)
    logger.info(f"Backup saved: {len(points)} points")

    # Import to server
    logger.info("=== STEP 2: Import to Qdrant server ===")
    vector_size = len(points[0]["vector"]) if points else 384

    success = import_to_server(
        server_url=SERVER_URL,
        points=points,
        collection_name=COLLECTION_NAME,
        vector_size=vector_size,
        recreate=True,  # Recreate collection for clean migration
    )

    if success:
        logger.info("✅ Migration completed successfully!")
        logger.info(f"Migrated {len(points)} memories from embedded to server")
        logger.info("You can now use the production docker-compose with Qdrant server")
        return 0
    else:
        logger.error("❌ Migration failed!")
        logger.info(f"Backup is saved in {backup_file}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
