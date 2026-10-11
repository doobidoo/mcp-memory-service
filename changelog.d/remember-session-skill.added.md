- **Claude Code plugin: `remember-session` skill.** `/mcp-memory-service:remember-session`
  has the model save what a session actually decided and learned: decisions with their
  reasons, root causes, reusable lessons, and one open-items memory with owners and
  absolute dates. It searches before writing to skip duplicates, tags every memory with
  the session's project, and reads the stored hashes back. It complements the automatic
  keyword digest of the session-end hook rather than replacing it.
