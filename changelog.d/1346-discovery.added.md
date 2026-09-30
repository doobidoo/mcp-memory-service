- **Harvest discovers Kiro workspace-layout sessions, not just the flat CLI mirror (#1346).**
  `TranscriptParser.find_sessions()` now globs nested
  `{workspace_hash}/{session_uuid}/messages.jsonl` (Kiro v4 payload-wrapped
  workspace sessions, including sessions migrated from the old IDE) and the
  `cli/*.jsonl` mirror, in addition to flat `*.jsonl` at the root. The v4 parser
  already reads that content (#1366) — only discovery was missing:
  `find_sessions(~/.kiro/sessions)` returned 0 while `find_sessions(.../cli)`
  returned 1104. Now the root sees both layouts. Additive and backward-compatible:
  pointing directly at `cli/` behaves exactly as before, and the `cli/` glob is
  scoped rather than a broad `*/*.jsonl` wildcard.
