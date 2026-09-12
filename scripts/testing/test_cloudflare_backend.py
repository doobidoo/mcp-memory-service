#!/usr/bin/env python3
"""
Test script for Cloudflare backend integration.
Run this after setting up your Cloudflare resources.
"""

import argparse
import asyncio
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add project root to sys.path so 'src' package is importable
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Cloudflare backend integration tests."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Run tests without confirmation",
    )
    return parser.parse_args(argv)


def load_project_env() -> None:
    """Load the repository .env after argparse has handled --help."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(PROJECT_ROOT / ".env", override=False)


def confirm_write(skip_confirmation: bool) -> bool:
    if skip_confirmation:
        return True

    try:
        answer = input(
            "This test will write to your Cloudflare resources. Continue? [y/N]: "
        )
    except EOFError:
        return False

    return answer.strip().lower() in {"y", "yes"}


def print_target() -> None:
    print("\nTarget configuration:")
    print("  Backend: Cloudflare")
    print(f"  Account ID: {os.getenv('CLOUDFLARE_ACCOUNT_ID')}")
    print(f"  Vectorize index: {os.getenv('CLOUDFLARE_VECTORIZE_INDEX')}")
    print(f"  D1 database ID: {os.getenv('CLOUDFLARE_D1_DATABASE_ID')}")
    print(f"  R2 bucket: {os.getenv('CLOUDFLARE_R2_BUCKET') or '(not configured)'}")
    print()


def missing_configuration() -> list[str]:
    placeholders = {
        "CLOUDFLARE_API_TOKEN": "your-cloudflare-api-token-here",
        "CLOUDFLARE_ACCOUNT_ID": "your-account-id-here",
        "CLOUDFLARE_VECTORIZE_INDEX": "your-vectorize-index-name",
        "CLOUDFLARE_D1_DATABASE_ID": "your-d1-database-id-here",
    }
    return [
        name
        for name, placeholder in placeholders.items()
        if not os.getenv(name) or os.getenv(name) == placeholder
    ]


async def test_cloudflare_backend() -> bool:
    """Test all Cloudflare backend functionality."""

    from src.mcp_memory_service.models.memory import Memory
    from src.mcp_memory_service.storage.cloudflare import CloudflareStorage

    # Check environment variables
    missing_vars = missing_configuration()
    if missing_vars:
        logger.error("Missing environment variables: %s", missing_vars)
        return False

    try:
        # Initialize storage
        logger.info("🔧 Initializing Cloudflare storage...")
        storage = CloudflareStorage(
            api_token=os.getenv("CLOUDFLARE_API_TOKEN"),
            account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
            vectorize_index=os.getenv("CLOUDFLARE_VECTORIZE_INDEX"),
            d1_database_id=os.getenv("CLOUDFLARE_D1_DATABASE_ID"),
            r2_bucket=os.getenv("CLOUDFLARE_R2_BUCKET"),  # Optional
        )

        # Test initialization
        logger.info("🚀 Testing storage initialization...")
        await storage.initialize()
        logger.info("✅ Storage initialized successfully")

        # Test storing a memory
        logger.info("💾 Testing memory storage...")
        test_content = "This is a test memory for Cloudflare backend integration."
        test_memory = Memory(
            content=test_content,
            content_hash=hashlib.sha256(test_content.encode()).hexdigest(),
            tags=["test", "cloudflare", "integration"],
            memory_type="test",
            metadata={"test_run": datetime.now(timezone.utc).isoformat()},
        )

        success, message = await storage.store(test_memory)
        if success:
            logger.info("✅ Memory stored: %s", message)
        else:
            logger.error("❌ Failed to store memory: %s", message)
            return False

        # Test retrieval
        logger.info("🔍 Testing memory retrieval...")
        results = await storage.retrieve("test memory cloudflare", n_results=5)
        if results:
            logger.info("✅ Retrieved %d memories", len(results))
            for i, result in enumerate(results):
                logger.info(
                    "  %d. Score: %.3f - %.50s...",
                    i + 1,
                    result.similarity_score,
                    result.memory.content,
                )
        else:
            logger.warning("⚠️  No memories retrieved")

        # Test tag search
        logger.info("🏷️  Testing tag search...")
        tag_results = await storage.search_by_tag(["test"])
        if tag_results:
            logger.info("✅ Found %d memories with 'test' tag", len(tag_results))
        else:
            logger.warning("⚠️  No memories found with 'test' tag")

        # Test statistics
        logger.info("📊 Testing statistics...")
        stats = await storage.get_stats()
        logger.info(
            "✅ Stats: %s memories, %s status",
            stats["total_memories"],
            stats["status"],
        )

        # Test cleanup (optional - uncomment to clean up test data)
        # logger.info("🧹 Cleaning up test data...")
        # deleted_count, delete_message = await storage.delete_by_tag("test")
        # logger.info("✅ Cleaned up: %s", delete_message)

        logger.info("🎉 All tests passed! Cloudflare backend is working correctly.")
        return True

    except (
        Exception  # noqa: BLE001 - integration test reports all failures cleanly
    ) as e:
        logger.error("❌ Test failed: %s", e)
        return False

    finally:
        if "storage" in locals():
            await storage.close()
            logger.info("🔒 Storage connection closed")


def print_setup_instructions(missing_vars: list[str] | None = None) -> None:
    """Print setup instructions if environment is not configured."""
    print("\n" + "=" * 60)
    print("🔧 CLOUDFLARE BACKEND SETUP REQUIRED")
    print("=" * 60)
    if missing_vars:
        print(f"Missing or placeholder settings: {', '.join(missing_vars)}")
        print()
    print()
    print("Please complete these steps:")
    print()
    print("1. Create API token with these permissions:")
    print("   - Vectorize:Edit")
    print("   - D1:Edit")
    print("   - Workers AI:Edit")
    print("   - R2:Edit (optional)")
    print()
    print("2. Create Cloudflare resources:")
    print(
        "   wrangler vectorize create mcp-memory-index --dimensions=768 --metric=cosine"
    )
    print("   wrangler d1 create mcp-memory-db")
    print("   wrangler r2 bucket create mcp-memory-content  # optional")
    print()
    print("3. Set environment variables:")
    print("   export CLOUDFLARE_API_TOKEN='your-token'")
    print("   export CLOUDFLARE_ACCOUNT_ID='be0e35a26715043ef8df90253268c33f'")
    print("   export CLOUDFLARE_VECTORIZE_INDEX='mcp-memory-index'")
    print("   export CLOUDFLARE_D1_DATABASE_ID='your-d1-id'")
    print("   export CLOUDFLARE_R2_BUCKET='mcp-memory-content'  # optional")
    print()
    print("4. Run this test again:")
    print("   python test_cloudflare_backend.py")
    print()
    print("See docs/cloudflare-setup.md for detailed instructions.")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_project_env()

    missing_vars = missing_configuration()
    if missing_vars:
        print_setup_instructions(missing_vars)
        return 1

    print_target()

    if not confirm_write(args.yes):
        print("Tests cancelled by user.")
        return 0

    success = asyncio.run(test_cloudflare_backend())
    if success:
        print("\n🎉 Cloudflare backend is ready for production use!")
        return 0

    print("\n❌ Tests failed. Check the logs above for details.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
