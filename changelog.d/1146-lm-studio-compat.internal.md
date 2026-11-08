- **`lm_studio_compat.py` logs with lazy `%`-style arguments (#1146).**
  Its six f-string logger calls now pass their values as arguments; the values
  that come from outside (the exception, the incoming notification and its
  `reason`) stay wrapped in `_sanitize_log_value`. The module is added to
  `GUARDED_MODULES`, so the log-injection ratchet keeps it clean.
