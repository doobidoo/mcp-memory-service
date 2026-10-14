- **`regenerate_embeddings.py` wrote vectors without their store partition (#1521).**
  Since multi-store, `memory_embeddings` is a vec0 table with `store` as a partition key; the maintenance script inserted `(rowid, content_embedding)` with no `store`, so every regenerated vector landed outside the partition that store-scoped KNN queries, invisible to search while the script reported success. It now reads each memory's `store` and writes the vector into that partition. Also fixes a broken `serialize_float32` import that prevented the script from running at all.
- **Soft-deleted memories are no longer regenerated.** `_load_memories` now filters
  `WHERE deleted_at IS NULL` (count and select), so a tombstoned memory is not reloaded
  and does not get a fresh vector that would resurrect it in the index — matching the other
  maintenance scripts (`migrate_embeddings.py`, `repair_missing_embeddings_onnx.py`).
