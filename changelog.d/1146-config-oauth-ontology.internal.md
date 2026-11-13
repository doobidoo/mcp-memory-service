- **`config/oauth.py` and `config/ontology.py` log with lazy `%`-style arguments (#1146).**
  The `ValueError` that `validate_oauth_configuration()` raises at import time
  carries `MCP_OAUTH_ISSUER` and a key path verbatim, and the base-type names
  parsed from `MCP_CUSTOM_MEMORY_TYPES` are logged as given: both are wrapped in
  `_sanitize_log_value`. The JWT algorithm and the three counts go as `%s`/`%d`
  arguments. Both modules are added to `GUARDED_MODULES`.
