#!/usr/bin/env python3
"""Reset all ONNX quality scores to implicit.

This script resets all memories with ONNX quality scores back to implicit
defaults (0.5). This is necessary after discovering that the bulk ONNX
evaluation was using self-matching queries, resulting in artificially
inflated scores (~1.0 for all memories).

The script:
1. Pauses hybrid backend sync during reset
2. Finds all memories with quality_provider='onnx_local'
3. Resets them to quality_score=0.5, quality_provider='implicit'
4. Preserves timestamps (doesn't change created_at/updated_at)
5. Resumes sync to propagate changes to Cloudflare

Usage:
    uv run python scripts/quality/reset_onnx_scores.py
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp_memory_service.storage.factory import create_storage_instance
from mcp_memory_service.config import STORAGE_BACKEND, SQLITE_VEC_PATH


async def reset_quality_scores():
    """Reset all ONNX scores to implicit defaults."""

    print("=" * 80)
    print("Reset ONNX Quality Scores to Implicit")
    print("=" * 80)
    print()
    print("This will reset all ONNX quality scores (quality_provider='onnx_local')")
    print("back to implicit defaults (quality_score=0.5).")
    print()
    print(f"📂 Storage backend: {STORAGE_BACKEND}")
    print(f"📁 SQLite path: {SQLITE_VEC_PATH}")
    print()

    storage = await create_storage_instance(SQLITE_VEC_PATH, server_type="script")

    try:
        # Pause sync if hybrid backend
        if STORAGE_BACKEND == 'hybrid' and hasattr(storage, 'pause_sync'):
            print("⏸️  Pausing hybrid backend sync...")
            await storage.pause_sync()
            print("✓ Sync paused")
            print()

        # Get all memories
        print("📊 Fetching all memories from local database...")
        all_memories = await storage.get_all_memories()
        total_count = len(all_memories)
        print(f"✓ Found {total_count} memories")
        print()

        # Find memories with ONNX scores
        onnx_memories = []
        for memory in all_memories:
            metadata = memory.metadata or {}
            provider = metadata.get('quality_provider')
            if provider == 'onnx_local':
                onnx_memories.append(memory)

        reset_count = len(onnx_memories)
        print(f"🎯 Memories with ONNX scores: {reset_count}/{total_count}")
        print()

        if reset_count == 0:
            print("✅ No ONNX scores to reset!")
            return

        print(f"🔄 Resetting {reset_count} quality scores...")
        print()

        success_count = 0
        error_count = 0

        for i, memory in enumerate(onnx_memories, 1):
            try:
                await storage.update_memory_metadata(
                    content_hash=memory.content_hash,
                    updates={
                        'quality_score': 0.5,
                        'quality_provider': 'implicit'
                    },
                    preserve_timestamps=True
                )
                success_count += 1

                # Progress every 500 memories
                if i % 500 == 0 or i == reset_count:
                    print(f"  [{i:5d}/{reset_count}] Reset, Errors: {error_count}")

            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Show first 5 errors
                    print(f"  ERROR [{i:5d}]: {memory.content_hash[:16]}... - {e}")

        print()
        print("=" * 80)
        print("✅ ONNX Quality Score Reset Complete!")
        print("=" * 80)
        print()

        # Summary statistics
        print("📈 Reset Summary:")
        print(f"   Total memories: {total_count}")
        print(f"   Reset successfully: {success_count}")
        print(f"   Errors: {error_count}")
        print()

        print("💡 Next Steps:")
        print("   1. Run bulk_evaluate_onnx.py with corrected query logic")
        print("   2. Verify quality distribution: curl -ks https://127.0.0.1:8000/api/quality/distribution")

    finally:
        # Resume sync if hybrid backend
        if STORAGE_BACKEND == 'hybrid' and hasattr(storage, 'resume_sync'):
            print()
            print("▶️  Resuming hybrid backend sync...")
            await storage.resume_sync()
            print("✓ Sync resumed (background service will push updates to Cloudflare)")

        # Close storage
        await storage.close()


if __name__ == '__main__':
    asyncio.run(reset_quality_scores())
