- **`ingestion/chunker.py` logs with lazy `%`-style arguments (#1146).**
  Its one f-string logger call now passes the chunk and character counts as
  `%d` arguments. `chunker.py` and the already clean `registry.py` join the
  five loaders in `GUARDED_MODULES`, so every module in `ingestion/` that logs
  is now held by the log-injection ratchet. `resolved_path`, the registry's
  canonical form of the caller's path, is added to `EXTERNAL_NAMES`.
