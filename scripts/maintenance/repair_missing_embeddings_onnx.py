#!/usr/bin/env python3
"""Fast repair: generate missing embeddings using the project's ONNX pipeline."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BATCH = 64  # ONNX batch size


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate missing embeddings for the configured database."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Run the repair without confirmation.",
    )
    parser.add_argument(
        "--allow-non-local",
        action="store_true",
        help="Allow the local SQLite repair while the hybrid backend is configured.",
    )
    return parser.parse_args(argv)


def confirm_write(skip_confirmation: bool) -> bool:
    if skip_confirmation:
        return True

    try:
        answer = input("This operation will modify the database. Continue? [y/N]: ")
    except EOFError:
        return False

    return answer.strip().lower() in {"y", "yes"}


async def repair_missing_embeddings(database_path: str) -> None:
    from sqlite_vec import serialize_float32

    from src.mcp_memory_service.storage.factory import create_storage_instance

    storage = await create_storage_instance(database_path)
    try:
        s = storage.primary if hasattr(storage, "primary") else storage
        s.conn.text_factory = lambda b: b.decode("utf-8", errors="replace")

        cur = s.conn.execute("""
            SELECT id, content FROM memories
            WHERE id NOT IN (SELECT rowid FROM memory_embeddings)
            """)
        rows = cur.fetchall()
        log.info("Missing embeddings: %d", len(rows))
        if not rows:
            log.info("Nothing to do.")
            return

        fixed = 0
        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            contents = [r[1] for r in batch]
            # _generate_embedding accepts one string; call the model directly for a batch.
            embeddings = s.embedding_model.encode(contents, convert_to_numpy=True)
            for (mem_id, _), emb in zip(batch, embeddings):
                s.conn.execute(
                    """
                    INSERT OR IGNORE INTO memory_embeddings(rowid, content_embedding)
                    VALUES (?, ?)
                    """,
                    (mem_id, serialize_float32(emb)),
                )
            s.conn.commit()
            fixed += len(batch)
            log.info(
                "Progress: %d/%d (%d%%)", fixed, len(rows), fixed * 100 // len(rows)
            )

        deleted = s.conn.execute("""
            DELETE FROM memory_embeddings
            WHERE rowid NOT IN (SELECT id FROM memories)
            """).rowcount
        s.conn.commit()
        log.info("Orphaned embeddings deleted: %d", deleted)
        log.info("Done. Fixed %d missing embeddings.", fixed)
    finally:
        await storage.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from src.mcp_memory_service.config import SQLITE_VEC_PATH, STORAGE_BACKEND

    database_path = SQLITE_VEC_PATH
    backend = STORAGE_BACKEND

    print("\nTarget configuration:")
    print(f"  Backend: {backend}")
    print(f"  Database: {database_path}")
    print()

    if database_path is None:
        print(f"Backend {backend!r} has no local SQLite database to repair.")
        return 2

    if backend != "sqlite_vec" and not args.allow_non_local:
        print(
            "Refusing to run while a non-local backend is configured. "
            "Use --allow-non-local to explicitly repair the hybrid backend's "
            "local SQLite database."
        )
        return 2

    if not confirm_write(args.yes):
        print("Operation cancelled by user.")
        return 0

    asyncio.run(repair_missing_embeddings(database_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
