#!/usr/bin/env python3
"""
Migrate Qdrant embedded storage to Qdrant server mode.

This script:
1. Stops Docker services
2. Starts standalone Qdrant server
3. Exports memories from embedded storage
4. Imports memories to Qdrant server
5. Verifies migration
6. Provides cleanup instructions
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Migration-specific errors."""

    pass


class QdrantMigrator:
    """Handles migration from embedded to server mode."""

    def __init__(
        self,
        embedded_volume: str = "mcp-memory-service_mcp-memory-data",
        server_url: str = "http://localhost:6333",
        collection_name: str = "memories",
    ):
        self.embedded_volume = embedded_volume
        self.server_url = server_url
        self.collection_name = collection_name
        self.temp_container = "qdrant-migration-temp"

    def run_command(self, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command with logging."""
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.stdout:
            logger.debug(f"stdout: {result.stdout}")
        if result.stderr:
            logger.debug(f"stderr: {result.stderr}")

        if check and result.returncode != 0:
            raise MigrationError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")

        return result

    def stop_services(self):
        """Stop Docker Compose services."""
        logger.info("Stopping Docker Compose services...")
        self.run_command(["docker", "compose", "down"])

    def verify_volume_exists(self) -> bool:
        """Verify embedded volume exists and has data."""
        logger.info(f"Verifying volume {self.embedded_volume} exists...")
        result = self.run_command(["docker", "volume", "inspect", self.embedded_volume], check=False)

        if result.returncode != 0:
            logger.error(f"Volume {self.embedded_volume} not found")
            return False

        logger.info("Volume found")
        return True

    def start_qdrant_server(self):
        """Start Qdrant server in standalone mode."""
        logger.info("Starting Qdrant server...")

        # Check if already running
        result = self.run_command(["docker", "ps", "-q", "-f", "name=mcp-memory-qdrant"], check=False)

        if result.stdout.strip():
            logger.info("Qdrant server already running")
            return

        # Start via docker compose
        self.run_command(["docker", "compose", "up", "-d", "qdrant"])

        # Wait for health check
        logger.info("Waiting for Qdrant to be healthy...")
        for i in range(30):
            result = self.run_command(["docker", "ps", "-f", "name=mcp-memory-qdrant", "-f", "health=healthy"], check=False)
            if result.stdout.strip():
                logger.info("Qdrant server is healthy")
                return
            logger.info(f"Waiting... ({i + 1}/30)")
            import time

            time.sleep(2)

        raise MigrationError("Qdrant server failed to become healthy")

    def mount_and_read_embedded(self) -> list[dict[str, Any]]:
        """Mount embedded volume and read all memories."""
        logger.info("Reading memories from embedded storage...")

        # Start temporary container with embedded volume mounted
        logger.info("Starting temporary Qdrant container with embedded storage...")
        self.run_command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.temp_container,
                "-v",
                f"{self.embedded_volume}:/qdrant/storage",
                "-p",
                "6335:6333",  # Use port 6335 to avoid conflicts with server on 6333/6334
                "qdrant/qdrant:latest",
            ]
        )

        # Wait for startup
        import time

        logger.info("Waiting for temporary container to start...")
        time.sleep(10)

        try:
            # Connect to temporary instance (port 6335)
            temp_client = QdrantClient(url="http://localhost:6335")

            # Verify collection exists
            collections = temp_client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                raise MigrationError(
                    f"Collection '{self.collection_name}' not found in embedded storage. Available: {collection_names}"
                )

            logger.info(f"Found collection '{self.collection_name}'")

            # Get collection info
            collection_info = temp_client.get_collection(self.collection_name)
            total_points = collection_info.points_count
            logger.info(f"Total points in collection: {total_points}")

            # Scroll through all points
            memories = []
            offset = None
            batch_size = 100

            while True:
                points, next_offset = temp_client.scroll(
                    collection_name=self.collection_name, limit=batch_size, offset=offset, with_payload=True, with_vectors=True
                )

                if not points:
                    break

                # Filter out metadata point (ID = 1)
                for point in points:
                    if point.id != 1:  # Skip metadata point
                        memories.append({"id": point.id, "vector": point.vector, "payload": point.payload})

                logger.info(f"Read {len(memories)} memories so far...")

                if next_offset is None:
                    break
                offset = next_offset

            logger.info(f"Successfully read {len(memories)} memories from embedded storage")
            return memories

        finally:
            # Stop and remove temporary container
            logger.info("Cleaning up temporary container...")
            self.run_command(["docker", "stop", self.temp_container], check=False)
            self.run_command(["docker", "rm", self.temp_container], check=False)

    def import_to_server(self, memories: list[dict[str, Any]]):
        """Import memories to Qdrant server."""
        logger.info(f"Importing {len(memories)} memories to Qdrant server...")

        # Connect to server
        server_client = QdrantClient(url=self.server_url)

        # Check if collection exists
        collections = server_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self.collection_name in collection_names:
            logger.info(f"Collection '{self.collection_name}' exists on server")

            # Get count
            collection_info = server_client.get_collection(self.collection_name)
            existing_count = collection_info.points_count
            logger.info(f"Server already has {existing_count} points")

            # Ask for confirmation
            response = input("Collection exists. Merge or overwrite? [merge/overwrite/abort]: ").lower()
            if response == "abort":
                raise MigrationError("Migration aborted by user")
            elif response == "overwrite":
                logger.warning("Deleting existing collection...")
                server_client.delete_collection(self.collection_name)

        # Import in batches
        batch_size = 100
        imported = 0

        for i in range(0, len(memories), batch_size):
            batch = memories[i : i + batch_size]

            # Convert to PointStruct
            points = [PointStruct(id=mem["id"], vector=mem["vector"], payload=mem["payload"]) for mem in batch]

            # Upsert batch
            server_client.upsert(collection_name=self.collection_name, points=points)

            imported += len(points)
            logger.info(f"Imported {imported}/{len(memories)} memories...")

        logger.info(f"Successfully imported {imported} memories to server")

    def verify_migration(self, expected_count: int) -> bool:
        """Verify migration was successful."""
        logger.info("Verifying migration...")

        server_client = QdrantClient(url=self.server_url)

        # Get collection info
        collection_info = server_client.get_collection(self.collection_name)
        actual_count = collection_info.points_count

        # Subtract metadata point from actual count
        actual_memories = max(0, actual_count - 1)

        logger.info(f"Expected: {expected_count} memories")
        logger.info(f"Actual: {actual_memories} memories (server has {actual_count} total points)")

        if actual_memories == expected_count:
            logger.info("✅ Migration verification PASSED")
            return True
        else:
            logger.error("❌ Migration verification FAILED: count mismatch")
            return False

    def print_cleanup_instructions(self):
        """Print instructions for cleanup."""
        logger.info("\n" + "=" * 80)
        logger.info("MIGRATION COMPLETE")
        logger.info("=" * 80)
        logger.info("\nNext steps:")
        logger.info("\n1. Start services in server mode:")
        logger.info("   docker compose up -d")
        logger.info("\n2. Verify services are working:")
        logger.info("   docker compose ps")
        logger.info("   docker exec -i mcp-memory-mcp mcp-memory-server")
        logger.info("\n3. Test multi-client access (open 2+ terminals):")
        logger.info("   Terminal 1: docker exec -i mcp-memory-mcp mcp-memory-server")
        logger.info("   Terminal 2: docker exec -i mcp-memory-mcp mcp-memory-server")
        logger.info("\n4. After verification, clean up old volumes:")
        logger.info(f"   docker volume rm {self.embedded_volume}")
        logger.info("   docker volume rm mcp-memory-service_mcp-memory-data-mcp")
        logger.info("\n" + "=" * 80)


async def main():
    """Main migration flow."""
    logger.info("=" * 80)
    logger.info("Qdrant Embedded → Server Migration")
    logger.info("=" * 80)

    migrator = QdrantMigrator()

    try:
        # Step 1: Verify preconditions
        if not migrator.verify_volume_exists():
            raise MigrationError("Embedded volume not found. Nothing to migrate.")

        # Step 2: Stop services
        migrator.stop_services()

        # Step 3: Start Qdrant server
        migrator.start_qdrant_server()

        # Step 4: Read from embedded storage
        memories = migrator.mount_and_read_embedded()

        if not memories:
            logger.warning("No memories found in embedded storage")
            response = input("Continue anyway? [y/N]: ").lower()
            if response != "y":
                raise MigrationError("Migration aborted: no memories to migrate")

        # Step 5: Import to server
        migrator.import_to_server(memories)

        # Step 6: Verify
        if not migrator.verify_migration(len(memories)):
            raise MigrationError("Migration verification failed")

        # Step 7: Print next steps
        migrator.print_cleanup_instructions()

        return 0

    except MigrationError as e:
        logger.error(f"Migration failed: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("Migration interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error during migration: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
