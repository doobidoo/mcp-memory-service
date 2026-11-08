- **`web/oauth/storage/sqlite.py` logs with lazy `%`-style arguments (#1146).**
  Its eight f-string logger calls now pass their values as arguments. The client
  ids, the database path and the consumed authorization code stay wrapped in
  `_sanitize_log_value`; the expired-row counts from the cleanup job are
  internal integers and are logged with `%d`. The module is added to
  `GUARDED_MODULES`, so the log-injection ratchet keeps it clean.
