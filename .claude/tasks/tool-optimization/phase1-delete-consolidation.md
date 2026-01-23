# Phase 1: Delete Tool Consolidation (6 → 1)

## Ziel

Konsolidiere diese 6 Delete-Tools zu einem `memory_delete` Tool:
- `delete_memory` (by hash)
- `delete_by_tag` (single tag)
- `delete_by_tags` (any tag match)
- `delete_by_all_tags` (all tags match)
- `delete_by_timeframe` (date range)
- `delete_before_date` (before date)

## Neue Tool-Signatur

```python
async def delete_memories(
    self,
    content_hash: Optional[str] = None,
    tags: Optional[List[str]] = None,
    tag_match: Literal["any", "all"] = "any",
    before: Optional[str] = None,  # ISO date
    after: Optional[str] = None,   # ISO date
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Unified memory deletion with flexible filtering.
    
    Filter Logic:
    - If content_hash: Delete single memory (ignores other filters)
    - If tags: Filter by tags using tag_match mode
    - If before/after: Filter by time range
    - Multiple filters combine with AND logic
    - dry_run: Preview deletions without executing
    
    Returns:
        {
            "success": bool,
            "deleted_count": int,
            "deleted_hashes": List[str],
            "dry_run": bool
        }
    """
```

---

## Task 1.1: Backend Implementation

**Datei:** `src/mcp_memory_service/services/memory_service.py`

### Amp Prompt

```
Refactor delete operations in src/mcp_memory_service/services/memory_service.py:

1. Use finder to locate all delete methods:
   - delete_memory
   - delete_by_tag
   - delete_by_tags
   - delete_by_all_tags
   - delete_memories_by_timeframe
   - delete_memories_before_date

2. Create ONE unified method delete_memories with this signature:
   
   async def delete_memories(
       self,
       content_hash: Optional[str] = None,
       tags: Optional[List[str]] = None,
       tag_match: Literal["any", "all"] = "any",
       before: Optional[str] = None,
       after: Optional[str] = None,
       dry_run: bool = False
   ) -> Dict[str, Any]

3. Implementation logic:
   
   a) If content_hash is provided:
      - Delete single memory by hash
      - Ignore other filters
      - Return immediately
   
   b) Build filter criteria:
      - If tags provided: Add tag filter with tag_match mode
      - If before provided: Add created_at < before filter
      - If after provided: Add created_at > after filter
      - Combine all filters with AND
   
   c) If no filters provided:
      - Return error: "At least one filter required"
   
   d) If dry_run:
      - Query matching memories
      - Return count and hashes without deleting
   
   e) Execute deletion:
      - Delete all matching memories
      - Return count and hashes

4. Mark old methods as deprecated:
   
   import warnings
   
   async def delete_by_tag(self, tag: str) -> Dict[str, Any]:
       warnings.warn(
           "delete_by_tag is deprecated, use delete_memories(tags=[tag])",
           DeprecationWarning,
           stacklevel=2
       )
       return await self.delete_memories(tags=[tag], tag_match="any")
   
   # Same pattern for other deprecated methods

5. Add comprehensive type hints and docstring.

6. Verify existing tests still pass.
```

---

## Task 1.2: MCP Tool Definition

**Datei:** `src/mcp_memory_service/server_impl.py`

### Amp Prompt

```
Update tool definitions in src/mcp_memory_service/server_impl.py:

1. Find the handle_list_tools method and locate these tool definitions:
   - delete_memory
   - delete_by_tag
   - delete_by_tags
   - delete_by_all_tags
   - delete_by_timeframe
   - delete_before_date

2. Replace ALL 6 tools with ONE unified tool definition:

types.Tool(
    name="memory_delete",
    description="""Delete memories with flexible filtering. Combine filters for precise targeting.

USE THIS WHEN:
- User says "delete", "remove", "forget" specific memories
- Need to clean up by tag, time range, or specific hash
- Bulk deletion with safety preview (dry_run=true)

FILTER COMBINATIONS (AND logic when multiple specified):
- content_hash only: Delete single memory by hash
- tags only: Delete memories with matching tags
- before/after: Delete memories in time range
- tags + time: Delete tagged memories within time range

SAFETY FEATURES:
- dry_run=true: Preview what will be deleted without deleting
- Returns deleted_hashes for audit trail
- No filters = error (prevents accidental mass deletion)

Examples:
{"content_hash": "abc123def456"}
{"tags": ["temporary", "draft"], "tag_match": "any"}
{"tags": ["archived", "old"], "tag_match": "all"}
{"before": "2024-01-01"}
{"after": "2024-06-01", "before": "2024-12-31"}
{"tags": ["cleanup"], "before": "2024-01-01", "dry_run": true}
""",
    inputSchema={
        "type": "object",
        "properties": {
            "content_hash": {
                "type": "string",
                "description": "Specific memory hash to delete (ignores other filters if provided)"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by these tags"
            },
            "tag_match": {
                "type": "string",
                "enum": ["any", "all"],
                "default": "any",
                "description": "Match ANY tag or ALL tags"
            },
            "before": {
                "type": "string",
                "description": "Delete memories created before this date (ISO format: YYYY-MM-DD)"
            },
            "after": {
                "type": "string",
                "description": "Delete memories created after this date (ISO format: YYYY-MM-DD)"
            },
            "dry_run": {
                "type": "boolean",
                "default": false,
                "description": "Preview deletions without executing"
            }
        }
    }
)

3. Add handler method:

async def handle_memory_delete(self, arguments: dict) -> List[types.TextContent]:
    result = await self.memory_service.delete_memories(
        content_hash=arguments.get("content_hash"),
        tags=arguments.get("tags"),
        tag_match=arguments.get("tag_match", "any"),
        before=arguments.get("before"),
        after=arguments.get("after"),
        dry_run=arguments.get("dry_run", False)
    )
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

4. Update handle_call_tool to route "memory_delete":

elif name == "memory_delete":
    return await self.handle_memory_delete(arguments)

5. Keep old tool names temporarily with deprecation routing:

# In handle_call_tool, before the main switch:
DEPRECATED_TOOLS = {
    "delete_memory": ("memory_delete", lambda a: {"content_hash": a["content_hash"]}),
    "delete_by_tag": ("memory_delete", lambda a: {"tags": [a["tag"]], "tag_match": "any"}),
    "delete_by_tags": ("memory_delete", lambda a: {"tags": a["tags"], "tag_match": "any"}),
    "delete_by_all_tags": ("memory_delete", lambda a: {"tags": a["tags"], "tag_match": "all"}),
    "delete_by_timeframe": ("memory_delete", lambda a: {"after": a.get("start_date"), "before": a.get("end_date")}),
    "delete_before_date": ("memory_delete", lambda a: {"before": a["before_date"]}),
}

if name in DEPRECATED_TOOLS:
    new_name, transformer = DEPRECATED_TOOLS[name]
    logger.warning(f"Tool '{name}' is deprecated. Use '{new_name}' instead.")
    arguments = transformer(arguments)
    name = new_name
```

---

## Validation

Nach Abschluss dieser Phase:

```bash
# Test unified delete
uv run python -c "
import asyncio
from mcp_memory_service.services.memory_service import MemoryService

async def test():
    # Test dry_run
    result = await service.delete_memories(tags=['test'], dry_run=True)
    print('Dry run:', result)

asyncio.run(test())
"

# Check deprecation warnings
uv run python -W default::DeprecationWarning -c "
import asyncio
from mcp_memory_service.services.memory_service import MemoryService

async def test():
    await service.delete_by_tag('test')  # Should warn

asyncio.run(test())
"
```

---

## Checkliste

- [ ] `delete_memories` Methode in memory_service.py implementiert
- [ ] Alte Methoden deprecated und routen zu `delete_memories`
- [ ] `memory_delete` Tool in server_impl.py definiert
- [ ] Handler-Methode implementiert
- [ ] Deprecation-Routing für alte Tool-Namen
- [ ] Tests grün
- [ ] Deprecation-Warnings funktionieren
