- **`web/api/health.py` sanitizes its logged exception (#1146).**
  When the embedding integrity check behind `/api/health/detailed` fails, the
  exception is now passed through `_sanitize_log_value`, like the same message
  in `utils/health_check.py`, so a newline in it cannot forge a log line. The
  module is added to `GUARDED_MODULES`.
