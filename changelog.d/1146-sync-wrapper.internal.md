- **`api/sync_wrapper.py` logs with lazy `%`-style arguments (#1146).**
  When a wrapped call from the code-execution API fails, the exception is now
  passed through `_sanitize_log_value` instead of an f-string, so error text
  that echoes a caller's query or content cannot forge a log line. The module
  is added to `GUARDED_MODULES`.
