- **`utils/debug.py` logs with lazy `%`-style arguments (#1146).**
  `exact_match_retrieve()`, which backs the `exact_match_retrieve` MCP tool,
  logged the backend's exception through an f-string. A backend error that
  echoed the caller's content could forge a second log line. The error now
  goes through `_sanitize_log_value`, and the backend class name is a `%s`
  argument. The module is added to `GUARDED_MODULES`.
