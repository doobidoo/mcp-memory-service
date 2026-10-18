The embedding migrator (`scripts/maintenance/migrate_embeddings.py`) re-created the
`memory_embeddings` vec0 table with a fixed DDL, dropping the `store` partition key
used by current schemas. After a migration, every store-scoped query failed with
`no such column: store`, and `semantic`/`ranked`/`exact` searches returned zero
results while `hybrid` kept working and the script still reported success. The
migrator now preserves the existing DDL (substituting only the vector dimension) and
re-populates the partition column from `memories.store`.
