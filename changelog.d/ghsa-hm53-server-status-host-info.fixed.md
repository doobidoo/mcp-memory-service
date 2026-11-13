- **`/api/server/status` no longer discloses host details below admin scope
  (GHSA-hm53-3f6c-cm5h).** Process ID, OS, kernel and Python version are now
  returned to admin callers only; read-scoped and anonymous callers get
  version, uptime and timestamp. The dashboard shows `-` for those fields
  without admin scope.
