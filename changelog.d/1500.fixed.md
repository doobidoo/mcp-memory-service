- **Consolidation no longer stores pattern memories with nothing in them (#1492).**
  Compression summarized memories the service wrote itself, so discovered
  associations, auto-captured session summaries and earlier compression output
  turned into patterns about `0.682, 0.695` or `Session, User, copies, Hash`.
  Those memories are now left out of compression input; they are recognized by
  tag (`association`, `session-consolidation`, `session-summary`, or `cluster`
  plus `compressed`) or by `memory_type` `session`, since associations and most
  session summaries are stored as `observation`. The layout words `session`,
  `summary`, `user`, `copies` and `hash` are dropped from key concepts, and a
  cluster in which no sentence carries a key concept is skipped instead of being
  stored as the generated overview line alone.
