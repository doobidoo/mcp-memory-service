- **`storage/sync/orchestrator.py` joins `GUARDED_MODULES` (#1146).**
  The apply reason in the "Applied event" and "not applied" log lines, which can carry the peer's envelope data or an exception text from `apply_remote_event()`, is wrapped with `_sanitize_log_value`. A sentinel test pins the wrapper.
