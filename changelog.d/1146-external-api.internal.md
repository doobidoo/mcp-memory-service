- **`embeddings/external_api.py` logs with lazy `%`-style arguments (#1146).**
  Its two f-string logger calls now pass their values as arguments. The API
  URL and model name, read from `MCP_EXTERNAL_EMBEDDING_URL` and
  `MCP_EXTERNAL_EMBEDDING_MODEL`, go through `_sanitize_log_value`; the
  embedding dimension and batch size are logged with `%d`. The module is added
  to `GUARDED_MODULES`, with `api_url` in `EXTERNAL_NAMES` and `model_name`
  listed for this module only, so dropping either wrapper fails the test.
