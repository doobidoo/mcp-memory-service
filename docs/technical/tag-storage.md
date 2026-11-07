# Tag Storage (Historical)

> **Historical document.** This page described the one-off tag migration that ran
> against a ChromaDB collection. ChromaDB was removed in v8.0.0, and the
> `validate_memories.py` and `migrate_tags.py` scripts it referenced are gone.

## Current tag storage

SQLite-vec stores tags alongside the memory in the `memories.tags` column;
Cloudflare keeps them in the relational `tags` / `memory_tags` tables. No
migration step is required on current releases: the schema is created and
populated when memories are written.

For backend setup and maintenance commands, see
[`docs/guides/STORAGE_BACKENDS.md`](../guides/STORAGE_BACKENDS.md) and
[`scripts/README.md`](../../scripts/README.md).

## Legacy ChromaDB data

If you still have a ChromaDB database, convert it to SQLite-vec with the v7.6.0
release before upgrading — see
[`docs/guides/chromadb-migration.md`](../guides/chromadb-migration.md).
