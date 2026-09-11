#!/usr/bin/env python3
"""Fast repair: generate missing embeddings using the project's ONNX pipeline."""
import asyncio, sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp_memory_service.storage.factory import create_storage_instance
from mcp_memory_service.config import SQLITE_VEC_PATH
from sqlite_vec import serialize_float32

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

BATCH = 64  # ONNX batch size

async def main():
    storage = await create_storage_instance(SQLITE_VEC_PATH)
    s = storage.primary if hasattr(storage, 'primary') else storage
    s.conn.text_factory = lambda b: b.decode('utf-8', errors='replace')

    cur = s.conn.execute('''
        SELECT id, content FROM memories
        WHERE id NOT IN (SELECT rowid FROM memory_embeddings)
    ''')
    rows = cur.fetchall()
    log.info(f"Missing embeddings: {len(rows)}")
    if not rows:
        log.info("Nothing to do.")
        return

    fixed = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i+BATCH]
        contents = [r[1] for r in batch]
        # _generate_embedding accepts a single string; call model directly for batch
        embeddings = s.embedding_model.encode(contents, convert_to_numpy=True)
        for (mem_id, _), emb in zip(batch, embeddings):
            s.conn.execute(
                'INSERT OR IGNORE INTO memory_embeddings(rowid, content_embedding) VALUES (?, ?)',
                (mem_id, serialize_float32(emb))
            )
        s.conn.commit()
        fixed += len(batch)
        log.info(f"Progress: {fixed}/{len(rows)} ({fixed*100//len(rows)}%)")

    # Orphaned cleanup
    deleted = s.conn.execute('''
        DELETE FROM memory_embeddings
        WHERE rowid NOT IN (SELECT id FROM memories)
    ''').rowcount
    s.conn.commit()
    log.info(f"Orphaned embeddings deleted: {deleted}")
    log.info(f"Done. Fixed {fixed} missing embeddings.")
    await storage.close()

asyncio.run(main())
