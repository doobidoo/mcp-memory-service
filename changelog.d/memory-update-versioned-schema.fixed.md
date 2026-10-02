- **`memory_update` schema declares the fields its versioned path requires, and the in-place path no longer reacts to them.**
  The handler hard-requires `updates.content` (and reads `updates.reason`) when
  `versioned=true`, but the tool schema listed only `tags`, `memory_type` and
  `metadata` under `updates` — clients that build arguments from the declared
  schema could never satisfy the requirement and always received "versioned update
  requires 'content' field in updates." The schema now declares `content` and
  `reason` and states the per-mode behavior of `metadata` and
  `preserve_timestamps`, and the handler strips `content`/`reason` from in-place
  updates so they cannot refresh `updated_at` or leak into custom metadata.
