#!/usr/bin/env python3
"""
Migration Validation Utility

Validates migration integrity between source and target storage backends.
Performs count validation, embedding similarity checks, metadata preservation,
and timestamp accuracy validation using sample-based approach for large datasets.

Usage:
    python scripts/migration/validate_migration.py \
        --source ~/Library/Application\\ Support/mcp-memory/sqlite_vec.db \
        --target ~/Library/Application\\ Support/mcp-memory/qdrant \
        --sample-size 100

Author: Claude Code
Date: 2025-01-16
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ValidationReport:
    """Validation report with detailed results."""

    count_match: bool
    count_source: int
    count_target: int
    embedding_similarity_pass: bool
    embedding_similarity_avg: float
    metadata_pass: bool
    timestamp_pass: bool
    errors: list[str]

    def is_valid(self) -> bool:
        """Check if validation passed all checks."""
        return (
            self.count_match
            and self.embedding_similarity_pass
            and self.metadata_pass
            and self.timestamp_pass
            and len(self.errors) == 0
        )

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "MIGRATION VALIDATION REPORT",
            "=" * 70,
            f"Count Validation: {'✅ PASS' if self.count_match else '❌ FAIL'}",
            f"  Source: {self.count_source:,} memories",
            f"  Target: {self.count_target:,} memories",
            "",
            f"Embedding Similarity: {'✅ PASS' if self.embedding_similarity_pass else '❌ FAIL'}",
            f"  Average Cosine Similarity: {self.embedding_similarity_avg:.4f}",
            "  Threshold: >0.99",
            "",
            f"Metadata Preservation: {'✅ PASS' if self.metadata_pass else '❌ FAIL'}",
            "",
            f"Timestamp Preservation: {'✅ PASS' if self.timestamp_pass else '❌ FAIL'}",
            "  Tolerance: 1ms",
            "",
        ]

        if self.errors:
            lines.extend(
                [
                    f"Errors Found: {len(self.errors)}",
                    "-" * 70,
                ]
            )
            for error in self.errors[:20]:  # Show first 20 errors
                lines.append(f"  - {error}")
            if len(self.errors) > 20:
                lines.append(f"  ... and {len(self.errors) - 20} more errors")
        else:
            lines.append("No errors found ✅")

        lines.extend(
            [
                "",
                "=" * 70,
                f"OVERALL: {'✅ VALIDATION PASSED' if self.is_valid() else '❌ VALIDATION FAILED'}",
                "=" * 70,
            ]
        )

        return "\n".join(lines)


def count_matches(source_storage, target_storage) -> tuple[bool, int, int]:
    """
    Verify total memory counts are equal.

    Args:
        source_storage: Source storage backend
        target_storage: Target storage backend

    Returns:
        Tuple of (counts_match, source_count, target_count)
    """
    # Get counts from both backends
    # Note: Assumes storage backends have a method to get count
    # For SqliteVecMemoryStorage, this might be a query
    # For QdrantStorage, use client.count()

    # Placeholder - implement based on storage interface
    # source_count = await source_storage.count()
    # target_count = await target_storage.count()

    # For now, return placeholder values
    # This will be implemented when integrated with actual storage
    source_count = 0
    target_count = 0
    return source_count == target_count, source_count, target_count


async def embedding_similarity_check(source_storage, target_storage, sample_size: int = 100) -> tuple[bool, float, list[str]]:
    """
    Sample-based cosine similarity check for embeddings.

    Args:
        source_storage: Source storage backend
        target_storage: Target storage backend
        sample_size: Number of memories to sample (default: 100)

    Returns:
        Tuple of (pass_check, avg_similarity, errors)
    """
    errors = []

    # Get random sample of memory hashes
    # Placeholder - implement based on storage interface
    # all_hashes = await source_storage.get_all_hashes()
    # sample_hashes = random.sample(all_hashes, min(sample_size, len(all_hashes)))

    sample_hashes = []  # Placeholder

    similarities = []

    for content_hash in sample_hashes:
        try:
            # Get memories from both storages
            # source_memory = await source_storage.get_by_hash(content_hash)
            # target_memory = await target_storage.get_by_hash(content_hash)

            # Placeholder - implement when integrated
            source_embedding = None  # source_memory.embedding
            target_embedding = None  # target_memory.embedding

            if source_embedding is None or target_embedding is None:
                errors.append(f"{content_hash}: Missing embedding")
                continue

            # Calculate cosine similarity
            similarity = cosine_similarity(source_embedding, target_embedding)
            similarities.append(similarity)

            if similarity < 0.99:
                errors.append(f"{content_hash}: Low similarity ({similarity:.4f} < 0.99)")

        except Exception as e:
            errors.append(f"{content_hash}: Error checking similarity - {e}")

    # Calculate average similarity
    avg_similarity = np.mean(similarities) if similarities else 0.0

    # Pass if average similarity > 0.99
    pass_check = avg_similarity > 0.99

    return pass_check, avg_similarity, errors


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (0-1)
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)

    # Normalize vectors
    vec1_norm = vec1_np / np.linalg.norm(vec1_np)
    vec2_norm = vec2_np / np.linalg.norm(vec2_np)

    # Dot product
    return float(np.dot(vec1_norm, vec2_norm))


async def metadata_preserved(source_storage, target_storage, sample_size: int = 100) -> tuple[bool, list[str]]:
    """
    Verify tags and metadata are preserved exactly.

    Args:
        source_storage: Source storage backend
        target_storage: Target storage backend
        sample_size: Number of memories to sample (default: 100)

    Returns:
        Tuple of (pass_check, errors)
    """
    errors = []

    # Get random sample of memory hashes
    # Placeholder - implement based on storage interface
    # all_hashes = await source_storage.get_all_hashes()
    # sample_hashes = random.sample(all_hashes, min(sample_size, len(all_hashes)))

    sample_hashes = []  # Placeholder

    for content_hash in sample_hashes:
        try:
            # Get memories from both storages
            # source_memory = await source_storage.get_by_hash(content_hash)
            # target_memory = await target_storage.get_by_hash(content_hash)

            # Placeholder - implement when integrated
            source_memory = None
            target_memory = None

            if source_memory is None or target_memory is None:
                errors.append(f"{content_hash}: Memory not found in target")
                continue

            # Check tags (order-independent)
            source_tags = set(source_memory.tags)
            target_tags = set(target_memory.tags)

            if source_tags != target_tags:
                errors.append(f"{content_hash}: Tag mismatch - source: {source_tags}, target: {target_tags}")

            # Check memory_type
            if source_memory.memory_type != target_memory.memory_type:
                errors.append(
                    f"{content_hash}: Memory type mismatch - "
                    f"source: {source_memory.memory_type}, "
                    f"target: {target_memory.memory_type}"
                )

            # Check metadata (if present)
            if source_memory.metadata != target_memory.metadata:
                errors.append(f"{content_hash}: Metadata mismatch")

        except Exception as e:
            errors.append(f"{content_hash}: Error checking metadata - {e}")

    # Pass if no errors
    pass_check = len(errors) == 0

    return pass_check, errors


async def timestamp_preserved(source_storage, target_storage, sample_size: int = 100) -> tuple[bool, list[str]]:
    """
    Verify timestamps are preserved within 1ms tolerance.

    Args:
        source_storage: Source storage backend
        target_storage: Target storage backend
        sample_size: Number of memories to sample (default: 100)

    Returns:
        Tuple of (pass_check, errors)
    """
    errors = []

    # Get random sample of memory hashes
    # Placeholder - implement based on storage interface
    # all_hashes = await source_storage.get_all_hashes()
    # sample_hashes = random.sample(all_hashes, min(sample_size, len(all_hashes)))

    sample_hashes = []  # Placeholder

    for content_hash in sample_hashes:
        try:
            # Get memories from both storages
            # source_memory = await source_storage.get_by_hash(content_hash)
            # target_memory = await target_storage.get_by_hash(content_hash)

            # Placeholder - implement when integrated
            source_memory = None
            target_memory = None

            if source_memory is None or target_memory is None:
                errors.append(f"{content_hash}: Memory not found in target")
                continue

            # Check created_at timestamp (within 1ms tolerance)
            created_diff = abs(source_memory.created_at - target_memory.created_at)
            if created_diff > 0.001:  # 1ms in seconds
                errors.append(f"{content_hash}: created_at mismatch - diff: {created_diff * 1000:.3f}ms")

            # Check updated_at timestamp (within 1ms tolerance)
            updated_diff = abs(source_memory.updated_at - target_memory.updated_at)
            if updated_diff > 0.001:
                errors.append(f"{content_hash}: updated_at mismatch - diff: {updated_diff * 1000:.3f}ms")

        except Exception as e:
            errors.append(f"{content_hash}: Error checking timestamps - {e}")

    # Pass if no errors
    pass_check = len(errors) == 0

    return pass_check, errors


async def validate_migration(source_storage, target_storage, sample_size: int = 100) -> ValidationReport:
    """
    Run all validation checks and return comprehensive report.

    Args:
        source_storage: Source storage backend
        target_storage: Target storage backend
        sample_size: Number of memories to sample for validation (default: 100)

    Returns:
        ValidationReport with detailed results
    """
    print("Starting migration validation...")
    print(f"Sample size: {sample_size}")
    print()

    # 1. Count validation
    print("1. Validating memory counts...")
    count_match, count_source, count_target = count_matches(source_storage, target_storage)
    print(f"   Source: {count_source:,} memories")
    print(f"   Target: {count_target:,} memories")
    print(f"   Result: {'✅ PASS' if count_match else '❌ FAIL'}")
    print()

    # 2. Embedding similarity
    print("2. Validating embedding similarity...")
    (embedding_pass, embedding_avg, embedding_errors) = await embedding_similarity_check(
        source_storage, target_storage, sample_size
    )
    print(f"   Average similarity: {embedding_avg:.4f}")
    print("   Threshold: >0.99")
    print(f"   Result: {'✅ PASS' if embedding_pass else '❌ FAIL'}")
    print()

    # 3. Metadata preservation
    print("3. Validating metadata preservation...")
    metadata_pass, metadata_errors = await metadata_preserved(source_storage, target_storage, sample_size)
    print(f"   Result: {'✅ PASS' if metadata_pass else '❌ FAIL'}")
    print()

    # 4. Timestamp preservation
    print("4. Validating timestamp preservation...")
    timestamp_pass, timestamp_errors = await timestamp_preserved(source_storage, target_storage, sample_size)
    print(f"   Result: {'✅ PASS' if timestamp_pass else '❌ FAIL'}")
    print()

    # Collect all errors
    all_errors = embedding_errors + metadata_errors + timestamp_errors

    # Create report
    report = ValidationReport(
        count_match=count_match,
        count_source=count_source,
        count_target=count_target,
        embedding_similarity_pass=embedding_pass,
        embedding_similarity_avg=embedding_avg,
        metadata_pass=metadata_pass,
        timestamp_pass=timestamp_pass,
        errors=all_errors,
    )

    return report


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate migration integrity between storage backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate SQLite to Qdrant migration
  python scripts/migration/validate_migration.py \\
      --source ~/Library/Application\\ Support/mcp-memory/sqlite_vec.db \\
      --target ~/Library/Application\\ Support/mcp-memory/qdrant \\
      --sample-size 100

  # Validate with larger sample for more thorough check
  python scripts/migration/validate_migration.py \\
      --source ~/Library/Application\\ Support/mcp-memory/sqlite_vec.db \\
      --target ~/Library/Application\\ Support/mcp-memory/qdrant \\
      --sample-size 500
        """,
    )

    parser.add_argument("--source", required=True, help="Path to source storage (SQLite database or Qdrant directory)")

    parser.add_argument("--target", required=True, help="Path to target storage (SQLite database or Qdrant directory)")

    parser.add_argument(
        "--sample-size", type=int, default=100, help="Number of memories to sample for validation (default: 100)"
    )

    args = parser.parse_args()

    # Validate paths exist
    source_path = Path(args.source)
    target_path = Path(args.target)

    if not source_path.exists():
        print(f"Error: Source path does not exist: {source_path}", file=sys.stderr)
        sys.exit(1)

    if not target_path.exists():
        print(f"Error: Target path does not exist: {target_path}", file=sys.stderr)
        sys.exit(1)

    print("Migration Validation Utility")
    print("=" * 70)
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print(f"Sample Size: {args.sample_size}")
    print()

    # TODO: Initialize storage backends based on paths
    # For now, this is a placeholder
    print("ERROR: Storage backend initialization not yet implemented")
    print()
    print("Next steps:")
    print("1. Implement storage backend detection from paths")
    print("2. Initialize SqliteVecMemoryStorage for SQLite databases")
    print("3. Initialize QdrantStorage for Qdrant directories")
    print("4. Connect validation functions to real storage interfaces")
    print()
    print("This script provides the validation framework.")
    print("Integration with storage backends will be completed in Phase 2.")

    # Placeholder validation report
    report = ValidationReport(
        count_match=False,
        count_source=0,
        count_target=0,
        embedding_similarity_pass=False,
        embedding_similarity_avg=0.0,
        metadata_pass=False,
        timestamp_pass=False,
        errors=["Storage backend initialization not implemented"],
    )

    print()
    print(report.summary())

    sys.exit(0 if report.is_valid() else 1)


if __name__ == "__main__":
    main()
