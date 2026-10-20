- **One-command install for Hermes as a first-class client (`scripts/hermes/instalar.py`).**
  Registers the server via `hermes mcp add`, enables the skill and hooks, seeds and reads back a
  probe memory, and refuses to report success unless `hermes mcp list` shows the server on the
  installed port. Idempotent (second run is a no-op) and reversible (`--uninstall` undoes only what
  it created). Windows path uses `schtasks`; no POSIX calls. Ships with a `mms-doctor`-style
  post-install verification and 22 tests.
