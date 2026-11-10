- **Milvus dropped the store a memory was written to and served it from any store (#1879, massimiliano1991).**
  The collection had no `store` field, so `store()` and `store_batch()` discarded the
  value, and `get_by_hash`, `update_memory_metadata` and `delete` found a memory whatever
  store the caller named. New collections get a `store` field (default `"default"`), the
  three hash-addressed calls are confined to it, and every upsert writes the row's own
  store back, so an update or a versioned update never moves a memory to `"default"`.
  A collection created before this has no field to confine on: there a named store is
  refused with one warning per instance (`None`, `"all"` and `"default"` are served as
  before), and the collection has to be recreated to pick up confinement, as for
  `content_lower`. `get_by_hash_is_store_scoped` stays `False`: search and listing still
  ignore `store` on this backend.
