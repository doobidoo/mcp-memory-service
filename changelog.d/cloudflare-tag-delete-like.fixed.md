- **Cloudflare tag deletes treated `%` and `_` in a tag as wildcards and deleted memories that did not have the tag (#[PR]).**
  `delete_by_tags`, `delete_by_timeframe` and `delete_before_date` built their D1
  `tags LIKE ?` patterns from the raw tag, so deleting the tag `%` deleted every memory
  with two or more tags, and `snake_case` also deleted a memory tagged `snakeXcase`. The
  patterns are escaped now and carry `ESCAPE '\'`, as the sqlite-vec deletes and the
  Cloudflare exact-content search already do.
