- **`mms-doctor`: six checks with evidence, safe restart, and a new console entry point.**
  Verifies health (restarting only a `failed`/`inactive` unit, never one still `activating`),
  declared vs. listening port, installed vs. `pyproject` version, sync cursor age, author stamping,
  and live-vs-tombstone embedding coverage. Every check prints the evidence it read; a missing DB
  schema degrades to a message instead of raising. Port comes from `--port`, then env, then the
  service env-file. 49 tests.
