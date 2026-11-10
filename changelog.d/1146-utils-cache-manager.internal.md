- **`utils/cache_manager.py` logs with lazy `%`-style arguments (#1146).**
  Its f-string logger calls now pass their values as arguments. The backend
  name, the `backend:path` cache key and the caller's context label go through
  `_sanitize_log_value`, as the same cache lines in `mcp_server.py` already do;
  call counts and storage IDs are logged with `%d`, init times with `%.1f`.
  The module is added to `GUARDED_MODULES`, so the log-injection ratchet keeps
  it clean.
