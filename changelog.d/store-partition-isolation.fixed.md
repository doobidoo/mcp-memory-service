- **Store partition is now honored in delete and non-semantic search (#1465, #1466).**
  `delete_memories(store=...)` no longer crosses partitions: the `store` filter is
  forwarded to delete-by-hash, delete-by-tag (any/all), and time-range deletion, and
  the `store` kwarg was added to `delete`/`delete_by_tags`/`get_by_exact_content`/
  `get_memories_by_time_range` across every backend to keep the interface uniform.
  `search_memories(store=...)` now scopes `quality_boost`, `exact`, `tag-only`,
  `time-only` and `hybrid` modes (previously only `semantic`/`ranked` honored it);
  the hybrid path also applies a final store filter so the store-unaware BM25 half
  cannot leak other partitions. Hybrid `delete_by_tags` queues cloud deletes only for
  the hashes actually removed from the scoped primary, so deleting from one store no
  longer removes another store's cloud copies.
