- **`server/cache_manager.py` logs with lazy `%`-style arguments (#1146).**
  Its f-string logger calls now pass their values as arguments. Storage IDs,
  hit/miss counts, and instance totals are logged with `%d`, and percentages
  and elapsed times with `%.1f`. The module is added to `GUARDED_MODULES`, so
  the log-injection ratchet keeps it clean.
