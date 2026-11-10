- **`quality/metadata_codec.py` logs with lazy `%`-style arguments (#1146).**
  Its two f-string logger calls now pass their values as arguments: the part
  count of a malformed quality-metadata CSV string with `%d`, and the decode
  error through `_sanitize_log_value`. The module is added to
  `GUARDED_MODULES`, so the log-injection ratchet keeps it clean.
