- **`server_impl.py` logs with lazy `%`-style arguments (#1146).**
  Its four f-string logger calls each spanned lines, so the line gate never
  saw them and only the AST scan did. The Milvus URI and collection name read
  from the environment are wrapped in `_sanitize_log_value` on both the eager
  and the lazy init path; the caller-provided storage class name and the
  unresolved-conflict count go as `%s` arguments. The module is added to
  `GUARDED_MODULES`.
