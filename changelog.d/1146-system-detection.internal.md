- **`utils/system_detection.py` logs with sanitised lazy `%`-style arguments (#1146).**
  The three exception logger calls in `_get_system_memory()`, `_check_cuda_available()`,
  and `_check_mps_available()` now pass `_sanitize_log_value(e)` to `%s` instead of
  f-strings, and the module is added to `GUARDED_MODULES`.
