- **`server/client_detection.py` logs with lazy `%`-style arguments (#1146).**
  When reading the parent process's command line fails, the error is now
  passed through `_sanitize_log_value` instead of an f-string, so text from
  another process cannot forge a log line. The module is added to
  `GUARDED_MODULES`.
