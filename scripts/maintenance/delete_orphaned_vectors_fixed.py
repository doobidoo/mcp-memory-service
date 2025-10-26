#!/usr/bin/env python3
"""
Delete orphaned vectors from Cloudflare Vectorize using correct endpoint.

Uses /delete_by_ids (underscores, not hyphens) with proper JSON payload format.
"""

import asyncio
import os
import sys
from pathlib import Path


async def main():
    # Set OAuth to false to avoid validation issues
    os.environ['MCP_OAUTH_ENABLED'] = 'false'

    # Import after setting environment
    from mcp_memory_service.storage.cloudflare import CloudflareStorage
    from mcp_memory_service.config import (
        CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID,
        CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID,
        EMBEDDING_MODEL_NAME
    )

    # Read vector IDs from the completed hash file
    hash_file = Path.home() / "cloudflare_d1_cleanup_completed.txt"

    if not hash_file.exists():
        print(f"❌ Error: Completed hash file not found: {hash_file}")
        print(f"   The D1 cleanup must be run first")
        sys.exit(1)

    print(f"📄 Reading vector IDs from: {hash_file}")

    with open(hash_file) as f:
        vector_ids = [line.strip() for line in f if line.strip()]

    if not vector_ids:
        print(f"✅ No vector IDs to delete (file is empty)")
        sys.exit(0)

    print(f"📋 Found {len(vector_ids)} orphaned vectors to delete")
    print(f"🔗 Connecting to Cloudflare...\n")

    # Initialize Cloudflare storage
    cloudflare = CloudflareStorage(
        api_token=CLOUDFLARE_API_TOKEN,
        account_id=CLOUDFLARE_ACCOUNT_ID,
        vectorize_index=CLOUDFLARE_VECTORIZE_INDEX,
        d1_database_id=CLOUDFLARE_D1_DATABASE_ID,
        embedding_model=EMBEDDING_MODEL_NAME
    )

    await cloudflare.initialize()

    print(f"✅ Connected to Cloudflare")
    print(f"🗑️  Deleting {len(vector_ids)} vectors using correct /delete_by_ids endpoint...\n")

    deleted = 0
    failed = []

    # Batch delete in groups of 100 (API recommended batch size)
    batch_size = 100
    total_batches = (len(vector_ids) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(vector_ids), batch_size), 1):
        batch = vector_ids[i:i+batch_size]

        try:
            # Use correct endpoint: /delete_by_ids with JSON payload {"ids": [...]}
            payload = {"ids": batch}
            response = await cloudflare._retry_request(
                "POST",
                f"{cloudflare.vectorize_url}/delete_by_ids",
                json=payload
            )
            result = response.json()

            if result.get("success"):
                deleted += len(batch)
                mutation_id = result.get("result", {}).get("mutationId", "N/A")
                print(f"Batch {batch_num}/{total_batches}: ✓ Deleted {len(batch)} vectors (mutation: {mutation_id[:16]}...)")
            else:
                failed.extend(batch)
                print(f"Batch {batch_num}/{total_batches}: ✗ Failed - {result.get('errors', 'Unknown error')}")

        except Exception as e:
            failed.extend(batch)
            print(f"Batch {batch_num}/{total_batches}: ✗ Exception - {str(e)[:100]}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 Vector Cleanup Summary")
    print(f"{'='*60}")
    print(f"✅ Successfully deleted: {deleted}/{len(vector_ids)}")
    print(f"✗  Failed: {len(failed)}/{len(vector_ids)}")
    print(f"{'='*60}\n")

    if deleted > 0:
        print(f"🎉 Vector cleanup complete!")
        print(f"📋 {deleted} orphaned vectors removed from Vectorize")
        print(f"⏱️  Note: Deletions are asynchronous and may take a few seconds to propagate\n")

    if failed:
        print(f"⚠️  {len(failed)} vectors failed to delete")
        print(f"   You may need to retry these manually\n")

    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
