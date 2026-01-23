# Phase 4: Naming Convention Migration

## Ziel

Alle Tools auf konsistentes `memory_{action}` Pattern migrieren.

## Naming Map

| Alt | Neu | Status |
|-----|-----|--------|
| `store_memory` | `memory_store` | Rename |
| `retrieve_memory` | `memory_search` | Phase 2 erledigt |
| `recall_memory` | `memory_search` | Phase 2 erledigt |
| `search_by_tag` | `memory_list` | Merge mit list_memories |
| `delete_memory` | `memory_delete` | Phase 1 erledigt |
| `list_memories` | `memory_list` | Rename + merge |
| `check_database_health` | `memory_health` | Rename |
| `get_cache_stats` | `memory_stats` | Rename |
| `cleanup_duplicates` | `memory_cleanup` | Rename |
| `update_memory_metadata` | `memory_update` | Rename |
| `ingest_document` | `memory_ingest` | Merge |
| `ingest_directory` | `memory_ingest` | Merge |
| `rate_memory` | `memory_rate` | Rename |
| `get_memory_quality` | `memory_quality` | Merge |
| `analyze_quality_distribution` | `memory_quality` | Merge |
| `find_connected_memories` | `memory_graph` | Merge |
| `find_shortest_path` | `memory_graph` | Merge |
| `get_memory_subgraph` | `memory_graph` | Merge |

---

## Task 4.1: Simple Renames

### Amp Prompt

```
Rename tools in src/mcp_memory_service/server_impl.py following memory_{action} pattern:

1. In handle_list_tools, rename these tools:

OLD NAME                  → NEW NAME
store_memory              → memory_store
check_database_health     → memory_health
get_cache_stats           → memory_stats
cleanup_duplicates        → memory_cleanup
update_memory_metadata    → memory_update
rate_memory               → memory_rate

2. For each renamed tool, update:
   - The name= field in types.Tool()
   - The handler routing in handle_call_tool
   - Add deprecation mapping

3. Example for store_memory → memory_store:

# In handle_list_tools:
types.Tool(
    name="memory_store",  # Changed from "store_memory"
    description="""...""",
    inputSchema={...}
)

# In handle_call_tool:
elif name == "memory_store":
    return await self.handle_store_memory(arguments)

# In DEPRECATED_TOOLS dict:
"store_memory": ("memory_store", lambda a: a),  # Pass args unchanged

4. Update all 6 tools following this pattern.
```

---

## Task 4.2: Merge list_memories + search_by_tag → memory_list

### Amp Prompt

```
Merge list_memories and search_by_tag into memory_list in server_impl.py:

1. Create unified tool definition:

types.Tool(
    name="memory_list",
    description="""List and browse memories with pagination and optional filters.

USE THIS WHEN:
- User wants to browse all memories ("show me my memories", "list everything")
- Need to paginate through large result sets
- Filter by tag OR memory type for categorical browsing
- User asks "what do I have stored", "browse my memories"

Unlike memory_search (semantic search), this does categorical listing/filtering.

PAGINATION:
- page: 1-based page number (default: 1)
- page_size: Results per page (default: 20, max: 100)

FILTERS (combine with AND logic):
- tags: Filter to memories with any of these tags
- memory_type: Filter by type (note, reference, decision, etc.)

Examples:
{}  // List first 20 memories
{"page": 2, "page_size": 50}
{"tags": ["python", "reference"]}
{"memory_type": "decision", "page_size": 10}
{"tags": ["important"], "memory_type": "note"}
""",
    inputSchema={
        "type": "object",
        "properties": {
            "page": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Page number (1-based)"
            },
            "page_size": {
                "type": "integer",
                "default": 20,
                "minimum": 1,
                "maximum": 100,
                "description": "Results per page"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by tags (returns memories with ANY of these tags)"
            },
            "memory_type": {
                "type": "string",
                "description": "Filter by memory type"
            }
        }
    }
)

2. Update handler to support both listing and tag filtering:

async def handle_memory_list(self, arguments: dict) -> List[types.TextContent]:
    tags = arguments.get("tags")
    
    if tags:
        # Use tag search
        result = await self.memory_service.search_by_tag(tags=tags)
        # Apply pagination to results
        page = arguments.get("page", 1)
        page_size = arguments.get("page_size", 20)
        start = (page - 1) * page_size
        end = start + page_size
        
        memories = result.get("memories", [])
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "memories": memories[start:end],
                "total": len(memories),
                "page": page,
                "page_size": page_size,
                "total_pages": (len(memories) + page_size - 1) // page_size
            }, indent=2, default=str)
        )]
    else:
        # Use standard listing
        result = await self.memory_service.list_memories(
            page=arguments.get("page", 1),
            page_size=arguments.get("page_size", 20),
            memory_type=arguments.get("memory_type")
        )
        return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

3. Add deprecation mappings:

"list_memories": ("memory_list", lambda a: a),
"search_by_tag": ("memory_list", lambda a: {"tags": a.get("tags", [a.get("tag")])}),
```

---

## Task 4.3: Merge ingest_document + ingest_directory → memory_ingest

### Amp Prompt

```
Merge ingest tools into memory_ingest in server_impl.py:

1. Create unified tool:

types.Tool(
    name="memory_ingest",
    description="""Ingest documents or directories into memory database.

USE THIS WHEN:
- User wants to import a document (PDF, TXT, MD, JSON)
- Need to batch import a directory of documents
- Building knowledge base from existing files

SUPPORTED FORMATS:
- PDF files (.pdf)
- Text files (.txt, .md, .markdown, .rst)
- JSON files (.json)

MODE:
- file: Ingest single document (requires file_path)
- directory: Batch ingest all documents in directory (requires directory_path)

CHUNKING:
- Documents are split into chunks for better retrieval
- chunk_size: Target characters per chunk (default: 1000)
- chunk_overlap: Overlap between chunks (default: 200)

Examples:
{"file_path": "/path/to/document.pdf"}
{"file_path": "/path/to/notes.md", "tags": ["documentation"]}
{"directory_path": "/path/to/docs", "recursive": true}
{"directory_path": "/path/to/project", "file_extensions": ["md", "txt"], "tags": ["project-docs"]}
""",
    inputSchema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to single document (for file mode)"
            },
            "directory_path": {
                "type": "string",
                "description": "Path to directory (for directory mode)"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": "Tags to apply to all ingested memories"
            },
            "chunk_size": {
                "type": "integer",
                "default": 1000,
                "description": "Target chunk size in characters"
            },
            "chunk_overlap": {
                "type": "integer",
                "default": 200,
                "description": "Overlap between chunks"
            },
            "memory_type": {
                "type": "string",
                "default": "document",
                "description": "Type label for created memories"
            },
            "recursive": {
                "type": "boolean",
                "default": true,
                "description": "For directory mode: process subdirectories"
            },
            "file_extensions": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["pdf", "txt", "md", "json"],
                "description": "For directory mode: file types to process"
            },
            "max_files": {
                "type": "integer",
                "default": 100,
                "description": "For directory mode: maximum files to process"
            }
        }
    }
)

2. Handler routes based on which path is provided:

async def handle_memory_ingest(self, arguments: dict) -> List[types.TextContent]:
    file_path = arguments.get("file_path")
    directory_path = arguments.get("directory_path")
    
    if file_path and directory_path:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": "Provide either file_path OR directory_path, not both"})
        )]
    
    if file_path:
        result = await self.memory_service.ingest_document(
            file_path=file_path,
            tags=arguments.get("tags", []),
            chunk_size=arguments.get("chunk_size", 1000),
            chunk_overlap=arguments.get("chunk_overlap", 200),
            memory_type=arguments.get("memory_type", "document")
        )
    elif directory_path:
        result = await self.memory_service.ingest_directory(
            directory_path=directory_path,
            tags=arguments.get("tags", []),
            recursive=arguments.get("recursive", True),
            file_extensions=arguments.get("file_extensions", ["pdf", "txt", "md", "json"]),
            chunk_size=arguments.get("chunk_size", 1000),
            max_files=arguments.get("max_files", 100)
        )
    else:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": "Either file_path or directory_path required"})
        )]
    
    return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

3. Deprecation mappings:

"ingest_document": ("memory_ingest", lambda a: {"file_path": a["file_path"], **{k:v for k,v in a.items() if k != "file_path"}}),
"ingest_directory": ("memory_ingest", lambda a: {"directory_path": a["directory_path"], **{k:v for k,v in a.items() if k != "directory_path"}}),
```

---

## Task 4.4: Merge Quality Tools → memory_quality

### Amp Prompt

```
Merge quality tools into memory_quality:

types.Tool(
    name="memory_quality",
    description="""Memory quality management - rate, inspect, and analyze.

ACTIONS:
- rate: Manually rate a memory's quality (thumbs up/down)
- get: Get quality metrics for a specific memory
- analyze: Analyze quality distribution across all memories

Examples:
{"action": "rate", "content_hash": "abc123", "rating": 1, "feedback": "Very useful"}
{"action": "get", "content_hash": "abc123"}
{"action": "analyze"}
{"action": "analyze", "min_quality": 0.5, "max_quality": 1.0}
""",
    inputSchema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["rate", "get", "analyze"],
                "description": "Quality action to perform"
            },
            "content_hash": {
                "type": "string",
                "description": "Memory hash (required for rate/get)"
            },
            "rating": {
                "type": "integer",
                "enum": [-1, 0, 1],
                "description": "For 'rate': -1 (thumbs down), 0 (neutral), 1 (thumbs up)"
            },
            "feedback": {
                "type": "string",
                "description": "For 'rate': Optional feedback text"
            },
            "min_quality": {
                "type": "number",
                "default": 0.0,
                "description": "For 'analyze': minimum quality threshold"
            },
            "max_quality": {
                "type": "number",
                "default": 1.0,
                "description": "For 'analyze': maximum quality threshold"
            }
        },
        "required": ["action"]
    }
)
```

---

## Task 4.5: Merge Graph Tools → memory_graph

### Amp Prompt

```
Merge graph tools into memory_graph:

types.Tool(
    name="memory_graph",
    description="""Memory association graph operations - explore connections between memories.

ACTIONS:
- connected: Find memories connected via associations (BFS traversal)
- path: Find shortest path between two memories
- subgraph: Get graph structure around a memory for visualization

Examples:
{"action": "connected", "hash": "abc123", "max_hops": 2}
{"action": "path", "hash1": "abc123", "hash2": "def456", "max_depth": 5}
{"action": "subgraph", "hash": "abc123", "radius": 2}
""",
    inputSchema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["connected", "path", "subgraph"],
                "description": "Graph operation to perform"
            },
            "hash": {
                "type": "string",
                "description": "Memory hash (for connected/subgraph)"
            },
            "hash1": {
                "type": "string",
                "description": "Start memory hash (for path)"
            },
            "hash2": {
                "type": "string",
                "description": "End memory hash (for path)"
            },
            "max_hops": {
                "type": "integer",
                "default": 2,
                "description": "For 'connected': max traversal depth"
            },
            "max_depth": {
                "type": "integer",
                "default": 5,
                "description": "For 'path': max path length"
            },
            "radius": {
                "type": "integer",
                "default": 2,
                "description": "For 'subgraph': nodes to include"
            }
        },
        "required": ["action"]
    }
)
```

---

## Finale Tool-Liste nach Phase 4

1. `memory_store` (renamed from store_memory)
2. `memory_search` (Phase 2)
3. `memory_list` (merged list_memories + search_by_tag)
4. `memory_delete` (Phase 1)
5. `memory_update` (renamed)
6. `memory_health` (renamed)
7. `memory_stats` (renamed)
8. `memory_consolidate` (Phase 3)
9. `memory_cleanup` (renamed)
10. `memory_ingest` (merged)
11. `memory_quality` (merged)
12. `memory_graph` (merged)

**Total: 12 Tools** ✓

---

## Checkliste

- [ ] Simple renames durchgeführt (6 Tools)
- [ ] list_memories + search_by_tag → memory_list
- [ ] ingest_document + ingest_directory → memory_ingest
- [ ] Quality tools → memory_quality
- [ ] Graph tools → memory_graph
- [ ] Alle Handler aktualisiert
- [ ] Alle Deprecation mappings eingetragen
- [ ] Tests grün
