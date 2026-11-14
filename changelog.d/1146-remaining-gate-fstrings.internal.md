- **Clear the remaining five `Check 6.5` f-string logger modules and guard `self.logger` calls (#1146).**
  `config/documents.py`, `consolidation/associations.py`, `storage/migration_runner.py`,
  `sync/litestream_config.py`, and `utils/directory_ingestion.py` now use lazy `%`-style
  logger arguments with `_sanitize_log_value()` around migration names, error messages,
  config paths, and ingested filenames. All five join `GUARDED_MODULES`, and
  `test_log_injection_guard.py` now inspects `self.logger.<level>()` calls alongside
  module-level `logger` and `logging`.
