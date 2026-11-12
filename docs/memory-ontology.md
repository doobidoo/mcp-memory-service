# Memory Ontology

This service validates `memory_type` against a built-in taxonomy. Unknown
types are silently coerced to the base type `observation` and the MCP /
HTTP response includes a warning. This page documents the taxonomy and how
to extend it.

## Why an ontology?

A free-form `memory_type` field looks flexible but breaks two things at
scale:

1. **Filter queries become unreliable** — `memory_list?memory_type=foo`
   returns 0 results because storage holds `Foo`, `foo `, `Foos`, etc.
2. **Relationship inference loses signal** — the consolidation system
   uses `(source_type, target_type)` patterns to classify relationships
   (`causes`, `fixes`, `contradicts`). Unknown types collapse to
   `observation`, which weakens those signals.

The ontology trades a small amount of flexibility for predictable
filtering and meaningful graph relationships. Custom types are still
supported — they just have to be declared up front.

## Built-in taxonomy

Defined in [`models/ontology.py`](../src/mcp_memory_service/models/ontology.py).

| Base type       | Common subtypes                                                                                       |
|-----------------|-------------------------------------------------------------------------------------------------------|
| `observation`   | `code_edit`, `file_access`, `search`, `command`, `conversation`, `conversation_turn`, `session`, `document`, `note`, `reference` |
| `decision`      | `architecture`, `tool_choice`, `approach`, `configuration`                                            |
| `learning`      | `insight`, `best_practice`, `anti_pattern`, `gotcha`                                                  |
| `error`         | `bug`, `failure`, `exception`, `timeout`, `mistake`                                                   |
| `pattern`       | `recurring_issue`, `code_smell`, `design_pattern`, `workflow`                                         |
| `planning`      | `sprint_goal`, `backlog_item`, `story_point_estimate`, `velocity`, `retrospective`, `standup_note`, `acceptance_criteria` |
| `ceremony`      | `sprint_review`, `sprint_planning`, `daily_standup`, `retrospective_action`, `demo_feedback`          |
| `milestone`     | `deliverable`, `dependency`, `risk`, `constraint`, `assumption`, `deadline`                           |
| `stakeholder`   | `requirement`, `feedback`, `escalation`, `approval`, `change_request`, `status_update`                |
| `meeting`       | `action_item`, `attendee_note`, `agenda_item`, `follow_up`, `minutes`                                 |
| `research`      | `finding`, `comparison`, `recommendation`, `source`, `hypothesis`                                     |
| `communication` | `email_summary`, `chat_summary`, `announcement`, `request`, `response`                                |

Both base types and subtypes are valid `memory_type` values. Subtypes map
to their parent base type for relationship inference.

## What "coercion" looks like

```jsonc
// Request
{
  "content": "Test memory",
  "metadata": { "type": "foo" }
}

// MCP response (text)
"Memory stored successfully (hash: 75c13...)
Warning: requested memory_type 'foo' is not in the ontology — stored as 'observation'.
Register custom types via the MCP_CUSTOM_MEMORY_TYPES env var, e.g. '{\"foo\": []}'."

// HTTP response (JSON)
{
  "success": true,
  "message": "Memory stored successfully Warning: requested memory_type 'foo' is not in the ontology — stored as 'observation'. Register custom types via MCP_CUSTOM_MEMORY_TYPES, e.g. '{\"foo\": []}'.",
  "memory": { "memory_type": "observation", ... }
}
```

The memory is still stored — under `observation`, not `foo`. Subsequent
filter queries like `memory_list?memory_type=foo` return zero results.
The warning is your signal to either pick an existing type or register
the custom one.

## Registering custom types

Set `MCP_CUSTOM_MEMORY_TYPES` to a JSON object mapping base type → list
of subtypes. Empty list is allowed if you don't need subtypes.

```bash
# Single new base type, no subtypes
export MCP_CUSTOM_MEMORY_TYPES='{"foo": []}'

# Extend an existing base type with new subtypes
export MCP_CUSTOM_MEMORY_TYPES='{"planning": ["okr", "north_star"]}'

# Multiple new base types
export MCP_CUSTOM_MEMORY_TYPES='{"foo": ["bar"], "experiment": ["hypothesis", "result"]}'
```

### Docker / Kubernetes

```yaml
- name: MCP_CUSTOM_MEMORY_TYPES
  value: '{"foo": []}'
```

### `.env` file

```dotenv
MCP_CUSTOM_MEMORY_TYPES={"foo": []}
```

After updating the env var, **restart the server** — the merged taxonomy
is cached in-process. With `memory restart` (CLI) or
`./scripts/update_and_restart.sh` (legacy).

### Validation rules

- Base type names must be valid Python identifiers (`[A-Za-z_][A-Za-z0-9_]*`).
- Subtype names must be alphanumeric + underscore.
- Invalid entries are skipped with a warning in the server log; the rest
  of the configuration still loads.
- Built-in types take precedence — extending an existing base type adds
  subtypes, it doesn't replace them.

## Types stored in the database (sqlite_vec)

On the `sqlite_vec` backend, custom types can also live in the memory
database itself, in the `ontology_entries` table (#1897). They are merged
after the built-ins and `MCP_CUSTOM_MEMORY_TYPES`, and they need no restart:

```python
await storage.add_memory_type("legal")                      # new base type
await storage.add_memory_type("contract", parent="legal")   # subtype
await storage.remove_memory_type("contract")                # tombstone
```

- The overlay only adds. A name that already exists is refused, so it can
  neither replace nor re-parent a built-in or env type.
- Every write bumps `ontology_version` in the `metadata` table. Other
  processes on the same database file (the HTTP server and the MCP server)
  check that counter at most once per second and reload their caches when
  it moved.
- Removing a type does not touch stored memories. They keep the type; only
  new writes with it fall back to `observation`.
- A new base type has no retention period in the consolidation config, so
  its memories decay on the 30-day fallback. Store one next to the type:

  ```python
  await storage.set_retention("legal", 400)     # days, 1..3650
  await storage.clear_retention("legal")        # back to the fallback
  ```

  Subtypes without their own entry use their parent's period. The overlay
  only fills gaps: a type that has a built-in period or an
  `MCP_RETENTION_*` override keeps it, and `set_retention` refuses it.
- **Sync:** with delta sync on (`MCP_SYNC_EVENTLOG`), each overlay write is
  also a sync event, so the rows reach the other `sqlite_vec` nodes through
  the hub. Concurrent writes to the same entry resolve by HLC: the later one
  wins on every node. A peer on an older version is not sent these events
  and keeps syncing memories. A row written while the event log was off has
  no event and does not sync; write it again with the event log on.
- **Limit:** Cloudflare (D1) does not carry the table. On the `cloudflare`
  backend, and between `hybrid` nodes that only share Cloudflare, keep using
  `MCP_CUSTOM_MEMORY_TYPES`.

### Relationship types and inference patterns

Relationship types (`RELATIONSHIPS` in `models/ontology.py`) and the base-type
pairs that consolidation uses to infer them (`TYPE_PATTERNS`) can be extended
the same way:

```python
await storage.add_relationship_type(
    "blocks", ["decision → error", "pattern → decision"],  # "->" works too
    description="A blocks B", symmetric=False,
)
await storage.set_type_pattern("decision", "error", "blocks", 0.8)
await storage.clear_type_pattern("decision", "error")
await storage.remove_relationship_type("blocks")
```

- Pattern ends are base types or `any`. A name that already exists is
  refused, so built-in relationships keep their patterns and symmetry.
- A type pattern maps a `(source, target)` base-type pair to a relationship
  and a confidence in (0, 1]. Built-in pairs cannot be replaced, and a pair
  the relationship's `valid_patterns` do not allow is refused (inference would
  drop it). Below 0.75 the typed-label gate falls back to `related`.
- Removing a relationship type keeps existing edges; new edges with it stop
  validating, and its type patterns are ignored until it is added again.
- `has_entity` (memory → entity key) is a built-in relationship. Its target is
  an entity, not a memory, so a type pattern cannot name it.

### REST API and dashboard

The HTTP server exposes the overlay under `/api/ontology`. Reads need the
`read` scope. Writes need the global `write` scope: the ontology is shared by
every store, so a `write:<store>` token is refused.

| Method | Path | Body |
|---|---|---|
| `GET` | `/api/ontology` | none |
| `POST` | `/api/ontology/memory-types` | `{"name", "parent"}` |
| `DELETE` | `/api/ontology/memory-types/{name}` | none |
| `PUT` | `/api/ontology/retention/{memory_type}` | `{"days"}` |
| `DELETE` | `/api/ontology/retention/{memory_type}` | none |
| `POST` | `/api/ontology/relationships` | `{"name", "valid_patterns", "description", "symmetric"}` |
| `DELETE` | `/api/ontology/relationships/{name}` | none |
| `PUT` | `/api/ontology/type-patterns/{source}/{target}` | `{"relationship", "confidence"}` |
| `DELETE` | `/api/ontology/type-patterns/{source}/{target}` | none |

`GET` returns the merged memory types, relationships, type patterns and
retention periods, plus `stored`: the live overlay rows, which are the only
entries the `DELETE` routes can remove. A write the storage refuses is a
`400` with the reason; a `DELETE` with nothing stored is a `404`. On a
backend without the overlay table the writes return `501`.

The dashboard's **Manage** tab has an **Ontology** section that adds memory
types, sets retention periods and lists the stored rows with a remove button.
Relationship types and type patterns are added through the API.

## When in doubt

- **Prototyping?** Use the closest base type (`observation`, `note`).
  Don't register custom types until you actually filter on them.
- **Building a domain workflow?** Pick a meaningful base + subtype pair,
  e.g. `meeting` → `action_item`. The graph layer benefits from this.
- **Importing from another tool?** Register the foreign vocabulary
  explicitly via `MCP_CUSTOM_MEMORY_TYPES` so downstream filters work.

## Related

- [`models/ontology.py`](../src/mcp_memory_service/models/ontology.py) — taxonomy + custom-type loader
- [`models/memory.py`](../src/mcp_memory_service/models/memory.py) — `Memory.__post_init__` validation
- Issue #842 — original report that motivated the warning
- Issue #843 — UX fix tracking issue
