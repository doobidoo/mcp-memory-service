- **`web/oauth/discovery.py` and `web/oauth/storage/factory.py` log with sanitised lazy `%`-style arguments (#1146).**
  `oauth_authorization_server_metadata()` and `create_oauth_storage()` now pass
  `_sanitize_log_value()` to `%s` instead of f-strings, and both modules are added
  to `GUARDED_MODULES`.
