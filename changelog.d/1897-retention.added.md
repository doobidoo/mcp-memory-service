- **Retention periods for custom memory types, stored in the memory database (#1897, phase 4).**
  A base type added through the ontology overlay had no key in `retention_periods`, so its
  memories decayed on the 30-day fallback. `SqliteVecMemoryStorage.set_retention()` and
  `clear_retention()` store a period (1..3650 days) as a `retention` row in `ontology_entries`,
  and the decay calculator uses it for any type the consolidation config does not name.
  Subtypes without their own entry inherit their parent's period. The overlay only fills gaps:
  built-in periods and `MCP_RETENTION_*` overrides still win, and `set_retention()` refuses
  those types. Like the rest of the overlay, the rows are local to sqlite_vec for now.
