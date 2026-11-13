- **`utils/content_splitter.py` logs with lazy `%`-style arguments (#1146).**
  The two logger calls in `split_content()` and `validate_chunk_lengths()` now
  log integer character counts and chunk indices via `%d` instead of f-strings,
  and the module is added to `GUARDED_MODULES`.
