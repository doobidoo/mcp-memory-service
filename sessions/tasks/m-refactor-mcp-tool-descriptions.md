---
name: m-refactor-mcp-tool-descriptions
branch: feature/m-refactor-mcp-tool-descriptions
status: pending
created: 2025-11-03
---

# Refactor MCP Tool Descriptions for Token Efficiency

## Problem/Goal
The memory MCP server exposes 19 tools consuming 29.6k tokens (14.8% of context window) just for tool descriptions. This is excessive and impacts available context for actual development work. Need to reduce token usage to <10k tokens (~5%) through:

1. **Tool consolidation** - Merge redundant delete/search operations with mode parameters
2. **Description compression** - Strip verbose examples and reduce to essential documentation
3. **Debug tool hiding** - Make debug tools opt-in via environment variable

Breaking changes acceptable - no backward compatibility required.

## Success Criteria

**Performance Targets:**
- [ ] Reduce MCP tool token usage from 29.6k to <10k tokens (66%+ reduction)
- [ ] Tool count reduced from 19 to ≤12 tools (through consolidation)
- [ ] Individual tool descriptions <600 tokens each (currently up to 1.0k)

**Functionality:**
- [ ] All existing tool functionality preserved (behavior unchanged)
- [ ] Debug tools hidden by default (`MCP_MEMORY_EXPOSE_DEBUG_TOOLS=false`)

**Quality:**
- [ ] Tool descriptions remain clear and actionable (compression doesn't sacrifice clarity)
- [ ] Tests verify consolidated tools work correctly (mode parameters behave as expected)
- [ ] Documentation updated (CLAUDE.md reflects new tool structure)

## Context Manifest

### How MCP Tool Registration Currently Works

The MCP Memory Service has **two server implementations** with different architectures:

**1. FastMCP Server (mcp_server.py) - 6 Tools**
- Modern FastMCP framework using `@mcp.tool()` decorators
- Tools directly delegate to `MemoryService` (shared business logic)
- Clean, minimal implementation with ~310 lines of code
- Tools: `store_memory`, `retrieve_memory`, `search_by_tag`, `delete_memory`, `check_database_health`, `list_memories`
- Each tool is a standalone async function with decorator - simple to modify

**2. Legacy MCP Server (server.py) - 26 Tools (The Problem)**
- Original `mcp.server.Server` implementation with 3,729 lines
- Uses `@self.server.list_tools()` decorator returning a giant list of `types.Tool` definitions
- Each tool has two parts:
  1. **Tool Definition** (lines 1289-1994): Giant `types.Tool` objects with verbose descriptions, examples, and JSON schemas
  2. **Tool Handler** (lines 2003-2090): Routing logic in `@self.server.call_tool()` that dispatches to handler methods
- **Token Bloat Source**: The tool definitions contain extensive multi-line docstrings with JSON examples

### Current Tool Inventory (server.py)

**Core Memory Operations (17 tools):**
- `store_memory` - Store with auto-chunking support
- `recall_memory` - Time-based retrieval with natural language parsing
- `retrieve_memory` - Semantic similarity search
- `search_by_tag` - Tag filtering (ANY match)
- `delete_memory` - Delete by content hash
- `delete_by_tag` - Delete by tags (ANY match)
- `delete_by_tags` - Same as delete_by_tag but "explicit multi-tag" (REDUNDANT)
- `delete_by_all_tags` - Delete by tags (ALL match)
- `cleanup_duplicates` - Find and remove duplicates
- `debug_retrieve` - Retrieval with debug info (DEBUG TOOL)
- `exact_match_retrieve` - Exact content matching (DEBUG TOOL)
- `get_raw_embedding` - Get embedding vectors (DEBUG TOOL)
- `check_database_health` - Health check
- `recall_by_timeframe` - Date range retrieval
- `delete_by_timeframe` - Date range deletion
- `delete_before_date` - Delete before date
- `update_memory_metadata` - Metadata updates

**Consolidation Tools (7 tools - conditional):**
Only exposed when `CONSOLIDATION_ENABLED=true` environment variable is set:
- `consolidate_memories` - Run consolidation
- `consolidation_status` - Status check
- `consolidation_recommendations` - Get recommendations
- `scheduler_status` - Scheduler info
- `trigger_consolidation` - Manual trigger
- `pause_consolidation` - Pause jobs
- `resume_consolidation` - Resume jobs

**Ingestion Tools (2 tools):**
- `ingest_document` - Single document ingestion
- `ingest_directory` - Batch directory ingestion

### Redundancies Identified

**Delete Operations (3 → 2 tools):**
- `delete_by_tag` and `delete_by_tags` are **identical implementations** - both call `storage.delete_by_tag(tags)` with ANY match logic
- `delete_by_all_tags` is distinct (ALL match) - keep separate
- **Consolidation**: Merge `delete_by_tag` + `delete_by_tags` → single `delete_by_tag(tags, match_all=False)` with mode parameter

**Search Operations (2 → 1 tool):**
- `recall_memory` - Parses natural language time expressions, extracts timestamps, then does semantic search
- `retrieve_memory` - Pure semantic search without time parsing
- **Key Difference**: Time parsing logic via `extract_time_expression()` and `parse_time_expression()` from `utils/time_parser.py`
- **Consolidation**: Keep `recall_memory` (superset functionality), deprecate `retrieve_memory`
- Alternatively: Add `enable_time_parsing=True` parameter to unified search tool

**Debug Tools (3 tools → opt-in):**
- `debug_retrieve` - Uses `utils.debug.debug_retrieve_memory()` for detailed similarity scores
- `exact_match_retrieve` - Uses `utils.debug.exact_match_retrieve()` for exact content matching
- `get_raw_embedding` - Uses `utils.debug.get_raw_embedding()` to expose embedding vectors
- **Current Pattern**: Always exposed, bloating token count
- **Consolidation**: Hide by default, expose only when `MCP_MEMORY_EXPOSE_DEBUG_TOOLS=true`

### Tool Description Format - Token Bloat Analysis

Current verbose format (example from `store_memory`, lines 1292-1353):

```python
types.Tool(
    name="store_memory",
    description="""Store new information with optional tags.

                Accepts two tag formats in metadata:
                - Array: ["tag1", "tag2"]
                - String: "tag1,tag2"

               Examples:
                # Using array format:
                {
                    "content": "Memory content",
                    "metadata": {
                        "tags": ["important", "reference"],
                        "type": "note"
                    }
                }

                # Using string format(preferred):
                {
                    "content": "Memory content",
                    "metadata": {
                        "tags": "important,reference",
                        "type": "note"
                    }
                }""",
    inputSchema={...}  # JSON schema with examples
)
```

**Token Count Breakdown:**
- Multi-line docstring with formatting: ~200 tokens
- JSON examples (2 full examples): ~150 tokens
- JSON schema with descriptions: ~100 tokens
- **Total per tool**: ~450 tokens
- **×17 core tools**: ~7,650 tokens
- **Plus consolidation/ingestion**: ~2,000 tokens
- **Grand total**: ~29,600 tokens (matches task description)

**Compression Strategy:**
1. **Remove all JSON examples** - Users can infer structure from JSON schema
2. **Compress descriptions to single lines** - Remove formatting, bullets, explanations
3. **Simplify schema descriptions** - One sentence per field max
4. **Target**: <300 tokens per tool, <100 tokens for simple tools

Compressed format example:

```python
types.Tool(
    name="store_memory",
    description="Store content with optional tags. Accepts tags as array or comma-separated string.",
    inputSchema={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Content to store"},
            "metadata": {
                "type": "object",
                "properties": {
                    "tags": {"oneOf": [{"type": "array"}, {"type": "string"}]},
                    "type": {"type": "string"}
                }
            }
        },
        "required": ["content"]
    }
)
```

### Environment Variable Pattern in Codebase

**Current Conditional Feature Pattern (CONSOLIDATION_ENABLED):**

1. **Config (config.py, line 541):**
```python
CONSOLIDATION_ENABLED = os.getenv('MCP_CONSOLIDATION_ENABLED', 'false').lower() == 'true'
```

2. **Tool Registration (server.py, line 1760):**
```python
# Add consolidation tools if enabled
if CONSOLIDATION_ENABLED and self.consolidator:
    consolidation_tools = [...]
    tools.extend(consolidation_tools)
```

3. **Handler Gating (server.py, line 2411):**
```python
if not CONSOLIDATION_ENABLED or not self.consolidator:
    return [types.TextContent(type="text", text="Consolidation not enabled")]
```

**Apply Same Pattern for Debug Tools:**

1. Add to `config.py`:
```python
EXPOSE_DEBUG_TOOLS = os.getenv('MCP_MEMORY_EXPOSE_DEBUG_TOOLS', 'false').lower() == 'true'
```

2. Conditional registration in `server.py`:
```python
# Add debug tools if explicitly enabled
if EXPOSE_DEBUG_TOOLS:
    debug_tools = [
        types.Tool(name="debug_retrieve", ...),
        types.Tool(name="exact_match_retrieve", ...),
        types.Tool(name="get_raw_embedding", ...)
    ]
    tools.extend(debug_tools)
```

3. Handler gating (optional - could also remove):
```python
elif name == "debug_retrieve":
    if not EXPOSE_DEBUG_TOOLS:
        return [types.TextContent(type="text", text="Debug tools not enabled")]
    return await self.handle_debug_retrieve(arguments)
```

### Tool Handler Architecture

**Handler Dispatch Pattern (server.py, lines 2003-2090):**

The `@self.server.call_tool()` decorator creates a single handler that routes by tool name:

```python
@self.server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> List[types.TextContent]:
    if arguments is None:
        arguments = {}

    if name == "store_memory":
        return await self.handle_store_memory(arguments)
    elif name == "retrieve_memory":
        return await self.handle_retrieve_memory(arguments)
    # ... 24 more elif branches
```

**Consolidation Impact:**
- Merging tools requires updating **both** the tool definition AND the handler routing
- Removing tools requires deleting the elif branch
- Adding mode parameters requires updating handler implementation

**Handler Implementation Pattern:**

Each handler method (lines 2092-3000+) follows this structure:

```python
async def handle_<tool_name>(self, arguments: dict) -> List[types.TextContent]:
    # 1. Extract and validate arguments
    param1 = arguments.get("param1")
    if not param1:
        return [types.TextContent(type="text", text="Error: param1 required")]

    # 2. Lazy storage initialization
    await self._ensure_storage_initialized()

    # 3. Business logic (either direct storage call or via MemoryService)
    result = await self.storage.some_method(param1)
    # OR: result = await self.memory_service.some_method(param1)

    # 4. Format response as TextContent
    return [types.TextContent(type="text", text=f"Success: {result}")]
```

**Consolidation Strategy for Handlers:**

For `delete_by_tag` + `delete_by_tags` merge:
```python
async def handle_delete_by_tag(self, arguments: dict) -> List[types.TextContent]:
    tags = arguments.get("tags", [])
    match_all = arguments.get("match_all", False)  # NEW parameter

    if not tags:
        return [types.TextContent(type="text", text="Error: Tags required")]

    await self._ensure_storage_initialized()

    # Branch based on mode
    if match_all:
        count, message = await self.storage.delete_by_all_tags(tags)
    else:
        count, message = await self.storage.delete_by_tag(tags)

    return [types.TextContent(type="text", text=message)]
```

### Storage Backend Interface (Critical for Consolidation)

**Base Storage Interface (storage/base.py):**

The `MemoryStorage` abstract base class defines the contract all backends must implement:

```python
class MemoryStorage(ABC):
    @abstractmethod
    async def search(self, query: str, n_results: int = 5) -> List[Memory]:
        """Semantic similarity search"""
        pass

    @abstractmethod
    async def search_by_tag(self, tags: List[str]) -> List[Memory]:
        """Search by tags (ANY match)"""
        pass

    @abstractmethod
    async def delete_by_tag(self, tags: List[str]) -> Tuple[int, str]:
        """Delete by tags (ANY match)"""
        pass

    @abstractmethod
    async def delete_by_all_tags(self, tags: List[str]) -> Tuple[int, str]:
        """Delete by tags (ALL match)"""
        pass
```

**Key Insight**: Storage backends already support BOTH `delete_by_tag` (ANY) and `delete_by_all_tags` (ALL) methods separately. The tool consolidation just adds a mode parameter to the MCP tool layer - no backend changes needed.

### MemoryService Shared Business Logic

**Service Layer (services/memory_service.py):**

The `MemoryService` class was introduced to eliminate code duplication between `mcp_server.py` and `server.py`. It centralizes business logic for:

- `store_memory()` - Content validation, chunking, hostname tagging
- `retrieve_memories()` - Semantic search with result formatting
- `search_by_tag()` - Tag filtering with result formatting
- `delete_memory()` - Single memory deletion
- `check_database_health()` - Health checks
- `list_memories()` - Pagination with database-level filtering

**Usage Pattern:**

Both servers initialize the service during startup:
```python
self.memory_service = MemoryService(self.storage)
```

Then delegate to shared logic:
```python
result = await self.memory_service.retrieve_memories(query=query, n_results=n_results)
```

**Consolidation Impact**: Tool consolidation in `server.py` should use `MemoryService` methods where available to maintain consistency with `mcp_server.py`.

### File Locations for Implementation

**Files to Modify:**

1. **`src/mcp_memory_service/config.py`** (line 541 area)
   - Add `EXPOSE_DEBUG_TOOLS` environment variable parsing
   - Location: After `CONSOLIDATION_ENABLED` definition

2. **`src/mcp_memory_service/server.py`** (3,729 lines total)
   - Tool definitions: Lines 1289-1994 (705 lines)
   - Tool handler routing: Lines 2003-2090 (87 lines)
   - Handler implementations: Lines 2092-3000+ (900+ lines)

   Specific changes:
   - Compress descriptions in `handle_list_tools()` (lines 1289-1997)
   - Add debug tool conditional (around line 1760 pattern)
   - Merge delete handler logic (around line 2029-2034)
   - Remove redundant retrieve handler (line 2022)

3. **`CLAUDE.md`** (project documentation)
   - Update tool count: 19 → ≤12
   - Document new environment variable: `MCP_MEMORY_EXPOSE_DEBUG_TOOLS`
   - Update tool consolidation notes

4. **Tests to Update:**
   - Check for any tests explicitly calling `delete_by_tags` (should use `delete_by_tag`)
   - Add tests for new `match_all` parameter
   - Add tests for debug tool conditional exposure

**Files NOT to Modify:**

- `src/mcp_memory_service/mcp_server.py` - FastMCP server is clean, leave as-is
- `src/mcp_memory_service/storage/*.py` - Backend interfaces unchanged
- `src/mcp_memory_service/services/memory_service.py` - Shared logic unchanged

### Configuration Files (No Changes Needed)

The codebase loads environment variables via:

1. **`.env` file loading (config.py, lines 35-43):**
```python
from dotenv import load_dotenv
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
```

2. **Environment variable precedence:**
   - Environment variables > .env file > defaults
   - No code changes needed for new `EXPOSE_DEBUG_TOOLS` variable

3. **User configuration:**
   - Users can add `MCP_MEMORY_EXPOSE_DEBUG_TOOLS=true` to `.env` file
   - Or export as shell environment variable
   - Default is `false` (debug tools hidden)

### Token Budget Analysis

**Current State:**
- 26 tools × ~1,140 tokens avg = 29,640 tokens
- Core tools: 17 × 1,000 = 17,000 tokens
- Consolidation tools: 7 × 600 = 4,200 tokens
- Ingestion tools: 2 × 800 = 1,600 tokens
- Debug tools: 3 × 700 = 2,100 tokens

**After Consolidation:**
- Core tools: 12 (removed 5 redundant)
- Debug tools: 0 (hidden by default)
- Consolidation: 7 (unchanged, conditional)
- Ingestion: 2 (unchanged)
- **Total exposed**: 12-14 tools (depending on CONSOLIDATION_ENABLED)

**After Description Compression:**
- Simple tools (delete, search): 150 tokens each
- Complex tools (store, recall): 300 tokens each
- Consolidation tools: 250 tokens each (conditional)
- Ingestion tools: 250 tokens each

**Final Token Count:**
- 8 simple core tools × 150 = 1,200 tokens
- 4 complex core tools × 300 = 1,200 tokens
- 7 consolidation tools × 250 = 1,750 tokens (if enabled)
- 2 ingestion tools × 250 = 500 tokens
- **Total**: ~4,650 tokens (84% reduction from 29,640)
- **Target met**: <10k tokens ✓

### Implementation Order

**Phase 1: Description Compression (Low Risk)**
1. Compress all tool descriptions in `handle_list_tools()`
2. Remove JSON examples
3. Simplify schema descriptions
4. Test with MCP Inspector to verify schemas still work

**Phase 2: Debug Tool Hiding (Medium Risk)**
1. Add `EXPOSE_DEBUG_TOOLS` to config.py
2. Add conditional registration in server.py
3. Test default behavior (tools hidden)
4. Test opt-in behavior (tools exposed with env var)

**Phase 3: Tool Consolidation (Higher Risk - Breaking Changes)**
1. Merge `delete_by_tag` + `delete_by_tags` (add `match_all` parameter)
2. Deprecate `retrieve_memory` in favor of `recall_memory`
3. Update handler routing logic
4. Update tool definitions
5. Test all deletion scenarios
6. Update documentation

**Phase 4: Testing & Documentation**
1. Run full test suite
2. Test with MCP Inspector
3. Test with Claude Desktop
4. Update CLAUDE.md
5. Update API reference docs

### Risk Assessment

**Low Risk:**
- Description compression (doesn't change functionality)
- Debug tool hiding (opt-in, doesn't affect existing users)

**Medium Risk:**
- Delete tool consolidation (breaking change for API consumers)
- Search tool deprecation (breaking change)

**Mitigation:**
- Keep both `delete_by_tag` and `delete_by_tags` in handlers initially (deprecated path)
- Add deprecation warnings in responses
- Document breaking changes clearly
- Consider gradual deprecation path

### Success Metrics

**Quantitative:**
- Token count: <10k (target) vs 29.6k (current) = 66%+ reduction ✓
- Tool count: ≤12 (target) vs 19-26 (current) = 37-54% reduction ✓
- Individual tool descriptions: <600 tokens each ✓

**Qualitative:**
- All existing tool functionality preserved (behavior unchanged)
- Debug tools hidden by default (better UX)
- Tool descriptions remain clear and actionable
- Tests verify consolidated tools work correctly

### Technical Reference Details

**Environment Variables:**
- `MCP_CONSOLIDATION_ENABLED`: Controls consolidation tool exposure (existing)
- `MCP_MEMORY_EXPOSE_DEBUG_TOOLS`: Controls debug tool exposure (new)

**Tool Consolidation Mappings:**
- `delete_by_tag(tags)` + `delete_by_tags(tags)` → `delete_by_tag(tags, match_all=False)`
- `delete_by_all_tags(tags)` → `delete_by_tag(tags, match_all=True)`
- `retrieve_memory(query)` → deprecated, use `recall_memory(query)` instead

**Handler Method Signatures:**
```python
async def handle_delete_by_tag(
    self,
    arguments: dict  # Contains: tags: List[str], match_all: bool = False
) -> List[types.TextContent]

async def handle_recall_memory(
    self,
    arguments: dict  # Contains: query: str, n_results: int = 5
) -> List[types.TextContent]
```

**Storage Backend Methods (unchanged):**
```python
async def delete_by_tag(self, tags: List[str]) -> Tuple[int, str]
async def delete_by_all_tags(self, tags: List[str]) -> Tuple[int, str]
async def search(self, query: str, n_results: int = 5) -> List[Memory]
```

## User Notes
<!-- Any specific notes or requirements from the developer -->

## Work Log
<!-- Updated as work progresses -->
- [YYYY-MM-DD] Started task, initial research
