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
