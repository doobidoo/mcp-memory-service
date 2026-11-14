- **`consolidation/quarantine.py` and `consolidation/relationship_inference.py` log without bare outside values (#1146).**
  The `content_hash[:8]` prefix in the quarantine, unquarantine and background
  check messages goes through `_sanitize_log_value`, as `storage/hybrid.py`
  already does for the same slice; the belief-challenged message and the four
  confidence/threshold messages of the relationship inferrer become lazy
  `%s`/`%.2f` arguments with the same text. Both modules are added to
  `GUARDED_MODULES`.
