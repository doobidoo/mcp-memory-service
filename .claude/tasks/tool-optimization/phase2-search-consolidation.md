# Phase 2: Search Tool Consolidation (5 → 1)

## Ziel

Konsolidiere diese 5 Search/Retrieve-Tools zu einem `memory_search` Tool:
- `retrieve_memory` (semantic search)
- `recall_memory` (natural language time expressions)
- `recall_by_timeframe` (explicit date range)
- `retrieve_with_quality_boost` (semantic + quality weighting)
- `exact_match_retrieve` (exact string match)

Bonus: `debug_retrieve` wird als `include_debug` Parameter integriert.

## Neue Tool-Signatur

```python
async def search_memories(
    self,
    query: Optional[str] = None,
    mode: Literal["semantic", "exact", "hybrid"] = "semantic",
    time_expr: Optional[str] = None,      # "last week", "yesterday", "2 days ago"
    after: Optional[str] = None,          # ISO date
    before: Optional[str] = None,         # ISO date
    tags: Optional[List[str]] = None,     # Filter by tags
    quality_boost: float = 0.0,           # 0.0-1.0, weight for quality scoring
    limit: int = 10,
    include_debug: bool = False
) -> Dict[str, Any]:
    """
    Unified memory search with flexible modes and filters.
    
    Modes:
    - semantic: Vector similarity search (default)
    - exact: Exact string match in content
    - hybrid: Semantic + quality-based reranking
    
    Time Filters:
    - time_expr: Natural language ("last week", "yesterday")
    - after/before: Explicit ISO dates
    
    Quality Boost:
    - 0.0 = pure semantic ranking
    - 0.3 = 30% quality, 70% semantic (recommended)
    - 1.0 = pure quality ranking
    
    Returns:
        {
            "memories": [...],
            "total": int,
            "query": str,
            "mode": str,
            "debug": {...}  # Only if include_debug=True
        }
    """
```

---

## Task 2.1: Backend Implementation

**Datei:** `src/mcp_memory_service/services/memory_service.py`

### Amp Prompt

```
Refactor search operations in src/mcp_memory_service/services/memory_service.py:

1. Use finder to locate all retrieve/recall methods:
   - retrieve_memories
   - recall_memory  
   - recall_by_timeframe
   - retrieve_with_quality_boost
   - exact_match_retrieve
   - debug_retrieve

2. Analyze how each method works:
   - retrieve_memories: Uses embedding similarity search
   - recall_memory: Parses time expressions, then searches
   - recall_by_timeframe: Filters by date range
   - retrieve_with_quality_boost: Over-fetches, reranks by quality
   - exact_match_retrieve: String matching
   - debug_retrieve: Adds debug info to results

3. Create ONE unified method:

from typing import Literal, Optional, List, Dict, Any

async def search_memories(
    self,
    query: Optional[str] = None,
    mode: Literal["semantic", "exact", "hybrid"] = "semantic",
    time_expr: Optional[str] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
    tags: Optional[List[str]] = None,
    quality_boost: float = 0.0,
    limit: int = 10,
    include_debug: bool = False
) -> Dict[str, Any]:

4. Implementation logic:

   a) Parse time expressions if provided:
      if time_expr:
          # Reuse existing time parsing logic from recall_memory
          after, before = self._parse_time_expression(time_expr)
   
   b) Build base query based on mode:
      if mode == "semantic":
          if not query:
              return {"error": "query required for semantic mode"}
          results = await self._semantic_search(query, limit * 3 if quality_boost > 0 else limit)
      
      elif mode == "exact":
          if not query:
              return {"error": "query required for exact mode"}
          results = await self._exact_match_search(query, limit)
      
      elif mode == "hybrid":
          # Combine semantic + quality
          results = await self._semantic_search(query, limit * 3)
   
   c) Apply time filters:
      if after or before:
          results = self._filter_by_time(results, after, before)
   
   d) Apply tag filters:
      if tags:
          results = [r for r in results if any(t in r.get("tags", []) for t in tags)]
   
   e) Apply quality reranking if requested:
      if quality_boost > 0 and mode in ["semantic", "hybrid"]:
          results = self._rerank_by_quality(results, quality_boost)
   
   f) Limit results:
      results = results[:limit]
   
   g) Build response:
      response = {
          "memories": results,
          "total": len(results),
          "query": query,
          "mode": mode
      }
      
      if include_debug:
          response["debug"] = {
              "time_filter": {"after": after, "before": before},
              "tag_filter": tags,
              "quality_boost": quality_boost,
              "pre_filter_count": pre_filter_count,
              "embedding_model": self.embedding_model
          }
      
      return response

5. Extract helper methods from existing code:
   - _parse_time_expression() from recall_memory
   - _semantic_search() from retrieve_memories  
   - _exact_match_search() from exact_match_retrieve
   - _rerank_by_quality() from retrieve_with_quality_boost
   - _filter_by_time() from recall_by_timeframe

6. Mark old methods as deprecated:

async def retrieve_memories(self, query: str, n_results: int = 5) -> Dict[str, Any]:
    warnings.warn(
        "retrieve_memories is deprecated, use search_memories(query=query, limit=n_results)",
        DeprecationWarning,
        stacklevel=2
    )
    return await self.search_memories(query=query, limit=n_results)

async def recall_memory(self, query: str, time_expr: str = None, n_results: int = 5) -> Dict[str, Any]:
    warnings.warn(
        "recall_memory is deprecated, use search_memories(query=query, time_expr=time_expr)",
        DeprecationWarning,
        stacklevel=2
    )
    return await self.search_memories(query=query, time_expr=time_expr, limit=n_results)

# ... same pattern for others

7. Ensure all type hints are complete.

8. Test that existing functionality still works.
```

---

## Task 2.2: MCP Tool Definition

**Datei:** `src/mcp_memory_service/server_impl.py`

### Amp Prompt

```
Update tool definitions in src/mcp_memory_service/server_impl.py:

1. Find and identify these tool definitions in handle_list_tools:
   - retrieve_memory
   - recall_memory
   - recall_by_timeframe
   - retrieve_with_quality_boost
   - exact_match_retrieve
   - debug_retrieve

2. Replace ALL 6 tools with ONE unified tool:

types.Tool(
    name="memory_search",
    description="""Search memories with flexible modes and filters. Primary tool for finding stored information.

USE THIS WHEN:
- User asks "what do you remember about X", "recall", "find memories"
- Looking for past decisions, preferences, context from previous sessions
- Need semantic, exact, or time-based search
- User references "last time we discussed", "you should know"

MODES:
- semantic (default): Finds conceptually similar content even if exact words differ
- exact: Finds memories containing the exact query string
- hybrid: Combines semantic similarity with quality scoring

TIME FILTERS (can combine with other filters):
- time_expr: Natural language like "yesterday", "last week", "2 days ago", "last month"
- after/before: Explicit ISO dates (YYYY-MM-DD)

QUALITY BOOST (for semantic/hybrid modes):
- 0.0 = pure semantic ranking (default)
- 0.3 = 30% quality weight, 70% semantic (recommended for important lookups)
- 1.0 = pure quality ranking

TAG FILTER:
- Filter results to only memories with specific tags
- Useful for categorical searches ("find all 'reference' memories about databases")

DEBUG:
- include_debug=true adds timing, embedding info, filter details

Examples:
{"query": "python async patterns"}
{"query": "API endpoint", "mode": "exact"}
{"time_expr": "last week", "limit": 20}
{"query": "database config", "time_expr": "yesterday"}
{"query": "architecture decisions", "tags": ["important"], "quality_boost": 0.3}
{"after": "2024-01-01", "before": "2024-06-30", "limit": 50}
{"query": "error handling", "include_debug": true}
""",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (required for semantic/exact modes, optional for time-only searches)"
            },
            "mode": {
                "type": "string",
                "enum": ["semantic", "exact", "hybrid"],
                "default": "semantic",
                "description": "Search mode"
            },
            "time_expr": {
                "type": "string",
                "description": "Natural language time filter (e.g., 'last week', 'yesterday', '3 days ago')"
            },
            "after": {
                "type": "string",
                "description": "Return memories created after this date (ISO format: YYYY-MM-DD)"
            },
            "before": {
                "type": "string",
                "description": "Return memories created before this date (ISO format: YYYY-MM-DD)"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter to memories with any of these tags"
            },
            "quality_boost": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0,
                "description": "Quality weight for reranking (0.0-1.0)"
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 100,
                "description": "Maximum results to return"
            },
            "include_debug": {
                "type": "boolean",
                "default": false,
                "description": "Include debug information in response"
            }
        }
    }
)

3. Add handler method:

async def handle_memory_search(self, arguments: dict) -> List[types.TextContent]:
    result = await self.memory_service.search_memories(
        query=arguments.get("query"),
        mode=arguments.get("mode", "semantic"),
        time_expr=arguments.get("time_expr"),
        after=arguments.get("after"),
        before=arguments.get("before"),
        tags=arguments.get("tags"),
        quality_boost=arguments.get("quality_boost", 0.0),
        limit=arguments.get("limit", 10),
        include_debug=arguments.get("include_debug", False)
    )
    return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

4. Update handle_call_tool routing:

elif name == "memory_search":
    return await self.handle_memory_search(arguments)

5. Add deprecation mapping for old tool names:

# Add to DEPRECATED_TOOLS dict:
"retrieve_memory": ("memory_search", lambda a: {
    "query": a["query"], 
    "limit": a.get("n_results", 5)
}),
"recall_memory": ("memory_search", lambda a: {
    "query": a.get("query"),
    "time_expr": a.get("time_expr"),
    "limit": a.get("n_results", 5)
}),
"recall_by_timeframe": ("memory_search", lambda a: {
    "after": a.get("start_date"),
    "before": a.get("end_date"),
    "limit": a.get("n_results", 5)
}),
"retrieve_with_quality_boost": ("memory_search", lambda a: {
    "query": a["query"],
    "quality_boost": a.get("quality_weight", 0.3),
    "limit": a.get("n_results", 10),
    "mode": "hybrid"
}),
"exact_match_retrieve": ("memory_search", lambda a: {
    "query": a["content"],
    "mode": "exact"
}),
"debug_retrieve": ("memory_search", lambda a: {
    "query": a["query"],
    "limit": a.get("n_results", 5),
    "include_debug": True
}),
```

---

## Validation

```bash
# Test unified search modes
uv run python -c "
import asyncio
from mcp_memory_service.services.memory_service import MemoryService

async def test():
    # Semantic search
    r1 = await service.search_memories(query='python patterns', limit=5)
    print('Semantic:', len(r1['memories']))
    
    # Exact match
    r2 = await service.search_memories(query='async def', mode='exact')
    print('Exact:', len(r2['memories']))
    
    # Time-based
    r3 = await service.search_memories(time_expr='last week', limit=20)
    print('Time-based:', len(r3['memories']))
    
    # Combined
    r4 = await service.search_memories(
        query='database',
        time_expr='last month',
        quality_boost=0.3
    )
    print('Combined:', len(r4['memories']))

asyncio.run(test())
"
```

---

## Checkliste

- [ ] `search_memories` Methode implementiert
- [ ] Time expression parsing funktioniert
- [ ] Alle drei Modi (semantic/exact/hybrid) funktionieren
- [ ] Quality boost reranking funktioniert
- [ ] Tag filtering funktioniert
- [ ] Debug output funktioniert
- [ ] Alte Methoden deprecated
- [ ] `memory_search` Tool definiert
- [ ] Deprecation routing für 6 alte Tools
- [ ] Tests grün
