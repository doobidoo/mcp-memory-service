#!/usr/bin/env python3
"""Bulk ONNX quality evaluation using local-first pattern."""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp_memory_service.storage.factory import create_storage_instance
from mcp_memory_service.config import STORAGE_BACKEND
from mcp_memory_service.quality.scorer import QualityScorer


async def bulk_evaluate_all_memories():
    """Evaluate ONNX quality for all memories in local database."""

    print("=" * 80)
    print("Bulk ONNX Quality Evaluation (Local-First)")
    print("=" * 80)
    print()

    # Determine SQLite path based on backend
    from mcp_memory_service.config import SQLITE_VEC_PATH

    if STORAGE_BACKEND in ['hybrid', 'sqlite_vec']:
        # Both backends use SQLITE_VEC_PATH for primary storage
        sqlite_path = SQLITE_VEC_PATH
    else:
        print(f"⚠️  Backend '{STORAGE_BACKEND}' not supported for local-first evaluation")
        print(f"   Supported backends: hybrid, sqlite_vec")
        return

    print(f"📂 Storage backend: {STORAGE_BACKEND}")
    print(f"📁 SQLite path: {sqlite_path}")
    print()

    # Create storage instance
    storage = await create_storage_instance(sqlite_path, server_type="script")

    try:
        # Pause sync if hybrid backend
        if STORAGE_BACKEND == 'hybrid' and hasattr(storage, 'pause_sync'):
            print("⏸️  Pausing hybrid backend sync...")
            await storage.pause_sync()
            print("✓ Sync paused (will resume after evaluation)")
            print()

        # Get all memories from local database
        print("📊 Fetching all memories from local database...")
        all_memories = await storage.get_all_memories()
        total_count = len(all_memories)
        print(f"✓ Found {total_count} memories")
        print()

        # Filter memories that need ONNX evaluation
        needs_evaluation = []
        for memory in all_memories:
            metadata = memory.metadata or {}

            # Skip system-generated memories (associations, compressed clusters)
            memory_type = memory.memory_type or 'standard'
            if memory_type in ['association', 'compressed_cluster']:
                continue

            # Skip if metadata indicates system-generated (belt-and-suspenders)
            if 'source_memory_hashes' in metadata:
                continue

            provider = metadata.get('quality_provider', 'implicit')

            # Evaluate if not already ONNX-scored
            if provider != 'onnx_local':
                needs_evaluation.append(memory)

        eval_count = len(needs_evaluation)
        print(f"🎯 Memories needing ONNX evaluation: {eval_count}/{total_count}")
        print(f"   Estimated time: ~{eval_count * 0.175:.1f} seconds")
        print()

        if eval_count == 0:
            print("✅ All memories already have ONNX scores!")
            return

        # Initialize quality scorer
        quality_scorer = QualityScorer()

        # Batch evaluate all memories
        print("🔄 Starting bulk ONNX evaluation...")
        print()

        success_count = 0
        error_count = 0
        scores = []

        for i, memory in enumerate(needs_evaluation, 1):
            try:
                # Generate meaningful query from memory metadata and tags
                # This represents what the memory is *about*, not the memory itself
                query_parts = []

                # Use tags as primary query source (specific topics/categories)
                if memory.tags:
                    # Tags are often already a list, but handle string case too
                    if isinstance(memory.tags, list):
                        query_parts.append(' '.join(memory.tags[:5]))  # Up to 5 tags
                    else:
                        query_parts.append(str(memory.tags))

                # Add memory type as context
                memory_type = memory.memory_type or 'note'
                query_parts.append(memory_type)

                # Add summary if available in metadata
                if metadata and 'summary' in metadata:
                    query_parts.append(str(metadata['summary'])[:100])

                # Combine parts into query
                query = ' '.join(query_parts).strip()

                # Fallback to content if no metadata/tags available
                if not query:
                    query = memory.content[:200] if memory.content else "general knowledge"

                # Evaluate quality using ONNX model
                quality_score = await quality_scorer.calculate_quality_score(memory, query)

                # Extract provider from memory metadata (set by calculate_quality_score)
                quality_provider = memory.metadata.get('quality_provider', 'implicit')

                # Update memory metadata directly in local database
                await storage.update_memory_metadata(
                    content_hash=memory.content_hash,
                    updates={
                        'quality_score': quality_score,
                        'quality_provider': quality_provider
                    },
                    preserve_timestamps=True
                )

                scores.append(quality_score)
                success_count += 1

                # Progress every 100 memories
                if i % 100 == 0 or i == eval_count:
                    avg = sum(scores) / len(scores) if scores else 0
                    print(f"  [{i:5d}/{eval_count}] Avg score: {avg:.3f}, Errors: {error_count}")

            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Show first 5 errors
                    print(f"  ERROR [{i:5d}]: {memory.content_hash[:16]}... - {e}")

        print()
        print("=" * 80)
        print("✅ Bulk ONNX Evaluation Complete!")
        print("=" * 80)
        print()

        # Summary statistics
        print("📈 Evaluation Summary:")
        print(f"   Total memories: {total_count}")
        print(f"   Evaluated: {success_count}")
        print(f"   Errors: {error_count}")
        print()

        if scores:
            avg_score = sum(scores) / len(scores)
            high_quality = sum(1 for s in scores if s >= 0.7)
            medium_quality = sum(1 for s in scores if 0.5 <= s < 0.7)
            low_quality = sum(1 for s in scores if s < 0.5)

            print("📊 ONNX Quality Distribution:")
            print(f"   Average score: {avg_score:.3f}")
            print(f"   High quality (≥0.7): {high_quality} ({high_quality/len(scores)*100:.1f}%)")
            print(f"   Medium quality (0.5-0.7): {medium_quality} ({medium_quality/len(scores)*100:.1f}%)")
            print(f"   Low quality (<0.5): {low_quality} ({low_quality/len(scores)*100:.1f}%)")

    finally:
        # Resume sync if hybrid backend
        if STORAGE_BACKEND == 'hybrid' and hasattr(storage, 'resume_sync'):
            print()
            print("▶️  Resuming hybrid backend sync...")
            await storage.resume_sync()
            print("✓ Sync resumed")

            # Wait for sync queue to drain
            if hasattr(storage, 'wait_for_sync_completion'):
                print()
                print("⏳ Waiting for sync queue to drain...")
                try:
                    stats = await storage.wait_for_sync_completion(timeout=600)  # 10 min timeout
                    print("✓ Sync completed successfully")
                    print(f"   Synced: {stats['success_count']} operations")
                    if stats['failure_count'] > 0:
                        print(f"   ⚠️  Failed: {stats['failure_count']} operations (will retry)")
                except TimeoutError as e:
                    print(f"⚠️  {e}")
                    print("   Background sync will continue (check status manually)")

        # Close storage
        await storage.close()

        print()
        print("💡 Verify Results:")
        print("   curl -ks https://127.0.0.1:8000/api/quality/distribution | python3 -m json.tool")


if __name__ == '__main__':
    asyncio.run(bulk_evaluate_all_memories())
