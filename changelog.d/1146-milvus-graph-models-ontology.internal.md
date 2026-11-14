- **`storage/milvus_graph.py` and `models/ontology.py` log without bare outside values (#1146).**
  The `exc` of the `except Exception` blocks around the pymilvus calls, and the
  caller-supplied relationship type logged when it fails validation, are wrapped
  in `_sanitize_log_value`, as the entity-link blocks of the same module already
  did; the hop list in `find_shortest_path` is renamed so the guard no longer
  reads it as a file path. The f-strings in `load_custom_memory_types` become
  lazy `%s`/`%d` arguments with the same text. Both modules are added to
  `GUARDED_MODULES`.
