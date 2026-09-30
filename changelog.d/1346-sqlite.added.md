- **Harvest can parse Kiro CLI v3 sessions straight from the SQLite source (#1346).**
  `TranscriptParser.parse_sqlite()` reads `conversations_v2` from a Kiro
  `data.sqlite3` — the CLI v3 source of truth — extracting user/assistant text
  from each conversation's structured `history[]` (user `content.Prompt.prompt`;
  assistant `.content` of a `Response` or `ToolUse` variant). The store only read
  `.jsonl` before, so those conversations were invisible to harvest. Opened
  strictly read-only (`file:...?mode=ro`) and streamed row-by-row so the live
  Kiro db is never written and large values are not all loaded at once. Feeds the
  same Phase 0 coverage instrument (with language), so SQLite content enters the
  coverage matrix. On a real 195-conversation db this recovers 6553 messages
  (0 before), ~88% pt-BR — evidence that an English-only extractor would drop
  most of the value. Alvo B of RFC harvest-kiro-sessions v2.0.
