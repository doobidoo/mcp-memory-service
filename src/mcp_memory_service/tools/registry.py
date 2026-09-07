"""Declarative tool registry for MCP Memory Service.

Each ToolDef maps 1:1 to a types.Tool(...) that was previously inline in
server_impl.py list_tools(). Annotations drive OAuth scope and MUST be
preserved exactly (GHSA-2r68-g678-7qr3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDef:
    """Declarative definition of an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] = field(default_factory=dict)


TOOL_REGISTRY: list[ToolDef] = [
    ToolDef(
        name="memory_store",
        description="""Store new information with optional tags.

                    Accepts two tag formats in metadata:
                    - Array: ["tag1", "tag2"]
                    - String: "tag1,tag2"

                    Use `conversation_id` to bypass semantic deduplication when saving
                    incremental memories from the same conversation.

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
                    }

                    # Using conversation_id to save incremental notes:
                    {
                        "content": "User prefers dark mode",
                        "conversation_id": "conv_abc123",
                        "metadata": {
                            "tags": "preference,ui"
                        }
                    }""",
        input_schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to store, such as a fact, note, or piece of information.",
                },
                "conversation_id": {
                    "type": "string",
                    "description": "Optional conversation identifier. When provided, semantic deduplication is skipped, allowing multiple incremental memories from the same conversation to be stored even if their content is topically similar. Exact duplicate hashes are still rejected.",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata about the memory, including tags and type.",
                    "properties": {
                        "tags": {
                            "oneOf": [
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Tags as an array of strings",
                                },
                                {
                                    "type": "string",
                                    "description": "Tags as comma-separated string",
                                },
                            ],
                            "description": "Tags to categorize the memory. Accepts either an array of strings or a comma-separated string.",
                            "examples": ["tag1,tag2,tag3", ["tag1", "tag2", "tag3"]],
                        },
                        "type": {
                            "type": "string",
                            "description": 'Optional memory type. Validated against the built-in ontology. Common base types: observation, decision, learning, error, pattern, planning, ceremony, milestone, stakeholder, meeting, research, communication. Common subtypes: note, reference, code_edit, command, document, insight, gotcha, bug, action_item, finding. Unknown types are silently coerced to \'observation\' and the response includes a warning. Register additional types via the MCP_CUSTOM_MEMORY_TYPES env var, e.g. \'{"foo": ["sub_a", "sub_b"]}\'. See docs/memory-ontology.md for the full taxonomy.',
                        },
                    },
                },
                "store": {
                    "type": "string",
                    "description": "Target store partition (default: 'default'). Use 'docs' for documents, 'all' for cross-store search.",
                },
            },
            "required": ["content"],
        },
        annotations={"destructiveHint": False},
    ),
    ToolDef(
        name="memory_store_session",
        description="""Store a full conversation session as a single memory unit.

Use this instead of memory_store when you want to preserve the full context
of a multi-turn conversation. All turns are stored together, making session-level
retrieval more reliable than storing individual turns separately.

Example:
{
"turns": [
    {"role": "user", "content": "How do I configure Redis?"},
    {"role": "assistant", "content": "Set REDIS_URL in your .env file..."}
],
"session_id": "optional-stable-id",
"tags": "redis,configuration"
}""",
        input_schema={
            "type": "object",
            "properties": {
                "turns": {
                    "type": "array",
                    "description": "Ordered list of conversation turns.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "description": "Speaker role, e.g. 'user' or 'assistant'",
                            },
                            "content": {
                                "type": "string",
                                "description": "Turn content",
                            },
                        },
                        "required": ["role", "content"],
                    },
                    "minItems": 1,
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional stable identifier for this session. Auto-generated UUID if omitted.",
                },
                "tags": {
                    "oneOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "string"},
                    ],
                    "description": "Additional tags (comma-separated string or array). 'session:<id>' is always added automatically.",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional extra metadata.",
                },
                "store": {
                    "type": "string",
                    "description": "Target store partition (default: 'default'). Use 'docs' for documents, 'all' for cross-store search.",
                },
            },
            "required": ["turns"],
        },
        annotations={"destructiveHint": False},
    ),
    ToolDef(
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
{"query": "error handling", "include_debug": true}""",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (required for semantic/exact modes, optional for time-only searches)",
                },
                "mode": {
                    "type": "string",
                    "enum": ["semantic", "exact", "hybrid", "ranked"],
                    "default": "semantic",
                    "description": "Search mode",
                },
                "time_expr": {
                    "type": "string",
                    "description": "Natural language time filter (e.g., 'last week', 'yesterday', '3 days ago')",
                },
                "after": {
                    "type": "string",
                    "description": "Return memories created after this date (ISO format: YYYY-MM-DD)",
                },
                "before": {
                    "type": "string",
                    "description": "Return memories created before this date (ISO format: YYYY-MM-DD)",
                },
                "tags": {
                    "oneOf": [
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags as an array of strings",
                        },
                        {
                            "type": "string",
                            "description": "Tags as comma-separated string",
                        },
                    ],
                    "description": "Filter to memories with any of these tags",
                },
                "tag_match": {
                    "type": "string",
                    "enum": ["any", "all"],
                    "default": "any",
                    "description": "Match ANY tag (OR, default) or ALL tags (AND)",
                },
                "quality_boost": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0,
                    "description": "Quality weight for reranking (0.0-1.0)",
                },
                "ranking_weights": {
                    "type": "object",
                    "description": "Custom weights for ranked mode (keys: semantic, time_decay, access_frequency, quality). Values are normalized to sum=1.0.",
                    "properties": {
                        "semantic": {"type": "number"},
                        "time_decay": {"type": "number"},
                        "access_frequency": {"type": "number"},
                        "quality": {"type": "number"},
                    },
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum results to return",
                },
                "include_debug": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include debug information in response",
                },
                "max_response_chars": {
                    "type": "number",
                    "description": "Maximum response size in characters. Truncates at memory boundaries to prevent context overflow. Recommended: 30000-50000. Default: unlimited.",
                },
                "include_superseded": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include memories that have been superseded by newer contradicting memories. Default: false (superseded memories are hidden).",
                },
                "entity": {
                    "type": "string",
                    "description": "Filter by linked entity name. Returns only memories linked to this entity via entity extraction. Requires a populated entity graph: run memory_quality {\"action\": \"maintain\", \"dry_run\": false} first (the dry_run default of true stores nothing). Unavailable on the Cloudflare backend.",
                },
                "fallback": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable cascading fallback when semantic results are sparse. When true and fewer than 3 results are found with scores below 0.4, automatically attempts BM25 keyword match and tag intersection. Each result includes match_method field. Default: false.",
                },
                "include_beliefs": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include derived beliefs alongside memories. Beliefs are confidence-scored knowledge derived from observations. Each belief result includes result_type='belief' and confidence score.",
                },
                "store": {
                    "type": "string",
                    "description": "Target store partition (default: 'default'). Use 'docs' for documents, 'all' for cross-store search.",
                },
            },
        },
        annotations={"readOnlyHint": True},
    ),
    ToolDef(
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
- tags: Filter to memories with matching tags
- tag_match: "any" (OR, default) or "all" (AND) — controls how multiple tags are matched
- memory_type: Filter by type (note, reference, decision, etc.)
- stale_days: Filter to memories not accessed in the last N days

Examples:
{}  // List first 20 memories
{"page": 2, "page_size": 50}
{"tags": ["python", "reference"]}
{"tags": ["python", "reference"], "tag_match": "all"}
{"memory_type": "decision", "page_size": 10}
{"tags": ["important"], "memory_type": "note"}
{"stale_days": 30}  // Memories not accessed in 30 days
{"stale_days": 7, "tags": ["archived"]}  // Stale archived memories
""",
        input_schema={
            "type": "object",
            "properties": {
                "page": {
                    "type": "integer",
                    "default": 1,
                    "minimum": 1,
                    "description": "Page number (1-based)",
                },
                "page_size": {
                    "type": "integer",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Results per page",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags (returns memories with ANY of these tags by default, use tag_match to control)",
                },
                "tag_match": {
                    "type": "string",
                    "default": "any",
                    "enum": ["any", "all"],
                    "description": "Match ANY tag (OR, default) or ALL tags (AND)",
                },
                "memory_type": {
                    "type": "string",
                    "description": "Filter by memory type",
                },
                "stale_days": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Filter to memories not accessed in the last N days. Uses COALESCE(last_accessed, created_at) for memories never read. Currently supported on sqlite-vec backend only.",
                },
                "store": {
                    "type": "string",
                    "description": "Target store partition (default: 'default'). Use 'docs' for documents, 'all' for cross-store search.",
                },
            },
        },
        annotations={"readOnlyHint": True},
    ),
    ToolDef(
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
{"tags": ["cleanup"], "before": "2024-01-01", "dry_run": true}""",
        input_schema={
            "type": "object",
            "properties": {
                "content_hash": {
                    "type": "string",
                    "description": "Specific memory hash to delete (ignores other filters if provided)",
                },
                "tags": {
                    "oneOf": [
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags as an array of strings",
                        },
                        {
                            "type": "string",
                            "description": "Tags as comma-separated string",
                        },
                    ],
                    "description": "Filter by these tags",
                },
                "tag_match": {
                    "type": "string",
                    "enum": ["any", "all"],
                    "default": "any",
                    "description": "Match ANY tag or ALL tags",
                },
                "before": {
                    "type": "string",
                    "description": "Delete memories created before this date (ISO format: YYYY-MM-DD)",
                },
                "after": {
                    "type": "string",
                    "description": "Delete memories created after this date (ISO format: YYYY-MM-DD)",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": False,
                    "description": "Preview deletions without executing",
                },
                "store": {
                    "type": "string",
                    "description": "Target store partition (default: 'default'). Use 'docs' for documents, 'all' for cross-store search.",
                },
            },
        },
        annotations={"destructiveHint": True},
    ),
    ToolDef(
        name="memory_cleanup",
        description="""Find and remove duplicate entries""",
        input_schema={"type": "object", "properties": {}},
        annotations={"destructiveHint": True},
    ),
    ToolDef(
        name="memory_health",
        description="""Check database health and get statistics""",
        input_schema={"type": "object", "properties": {}},
        annotations={"readOnlyHint": True},
    ),
    ToolDef(
        name="memory_stats",
        description="""Get MCP server global cache statistics for performance monitoring.

                    Returns detailed metrics about storage and memory service caching,
                    including hit rates, initialization times, and cache sizes.

                    This tool is useful for:
                    - Monitoring cache effectiveness
                    - Debugging performance issues
                    - Verifying cache persistence across MCP tool calls

                    Returns cache statistics including total calls, hit rate percentage,
                    storage/service cache metrics, performance metrics, and backend info.""",
        input_schema={"type": "object", "properties": {}},
        annotations={"readOnlyHint": True},
    ),
    ToolDef(
        name="memory_update",
        description="""Update memory metadata without recreating the entire memory entry.
                    
                    This provides efficient metadata updates while preserving the original
                    memory content, embeddings, and optionally timestamps.
                    
                    Examples:
                    # Add tags to a memory
                    {
                        "content_hash": "abc123...",
                        "updates": {
                            "tags": ["important", "reference", "new-tag"]
                        }
                    }
                    
                    # Update memory type and custom metadata
                    {
                        "content_hash": "abc123...",
                        "updates": {
                            "memory_type": "reminder",
                            "metadata": {
                                "priority": "high",
                                "due_date": "2024-01-15"
                            }
                        }
                    }
                    
                    # Update custom fields directly
                    {
                        "content_hash": "abc123...",
                        "updates": {
                            "priority": "urgent",
                            "status": "active"
                        }
                    }""",
        input_schema={
            "type": "object",
            "properties": {
                "content_hash": {
                    "type": "string",
                    "description": "The content hash of the memory to update.",
                },
                "updates": {
                    "type": "object",
                    "description": "Dictionary of metadata fields to update.",
                    "properties": {
                        "tags": {
                            "oneOf": [
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Tags as an array of strings",
                                },
                                {
                                    "type": "string",
                                    "description": "Tags as comma-separated string",
                                },
                            ],
                            "description": "Replace existing tags with this list. Accepts either an array of strings or a comma-separated string.",
                        },
                        "memory_type": {
                            "type": "string",
                            "description": "Update the memory type (e.g., 'note', 'reminder', 'fact').",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Custom metadata fields to merge with existing metadata.",
                        },
                    },
                },
                "preserve_timestamps": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to preserve the original created_at timestamp (default: true).",
                },
                "versioned": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, creates a new version instead of overwriting. The old memory is marked as superseded. Requires content in updates to create the new version. Creates a new memory version and marks the old one as superseded. Supported backends: sqlite_vec. Unsupported backends return an error.",
                },
            },
            "required": ["content_hash", "updates"],
        },
        annotations={"destructiveHint": True},
    ),
    ToolDef(
        name="memory_consolidate",
        description="""Memory consolidation management - run, monitor, and control memory optimization.

USE THIS WHEN:
- User asks about memory optimization, cleanup, or consolidation
- Need to check consolidation system health
- Want to manually trigger or schedule consolidation
- Need to pause/resume consolidation jobs

ACTIONS:
- run: Execute consolidation for a time horizon (requires time_horizon)
  Performs dream-inspired memory consolidation:
  • Exponential decay scoring
  • Creative association discovery
  • Semantic clustering and compression
  • Controlled forgetting with archival

- status: Get consolidation system health and statistics

- recommend: Get optimization recommendations for a time horizon (requires time_horizon)
  Analyzes memory state and suggests whether consolidation is needed

- scheduler: View all scheduled consolidation jobs
  Shows next run times and execution statistics

- pause: Pause consolidation (all or specific horizon)
  Temporarily stops automatic consolidation jobs

- resume: Resume paused consolidation
  Re-enables previously paused consolidation jobs

- merge: Combine multiple existing memories into one consolidated entry (requires content_hashes and merged_content)
  Verifies all source memories exist, stores the merged content, deletes originals

TIME HORIZONS:
- daily: Consolidate last 24 hours
- weekly: Consolidate last 7 days
- monthly: Consolidate last 30 days
- quarterly: Consolidate last 90 days
- yearly: Consolidate last 365 days

Examples:
{"action": "status"}
{"action": "run", "time_horizon": "weekly"}
{"action": "run", "time_horizon": "daily", "immediate": true}
{"action": "recommend", "time_horizon": "monthly"}
{"action": "scheduler"}
{"action": "pause"}
{"action": "pause", "time_horizon": "daily"}
{"action": "resume", "time_horizon": "weekly"}
{"action": "merge", "content_hashes": ["hash1", "hash2"], "merged_content": "Consolidated text"}
{"action": "merge", "content_hashes": ["hash1", "hash2"], "merged_content": "text", "tags": ["important"], "memory_type": "note"}""",
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "run",
                        "status",
                        "recommend",
                        "scheduler",
                        "pause",
                        "resume",
                        "merge",
                    ],
                    "description": "Consolidation action to perform",
                },
                "time_horizon": {
                    "type": "string",
                    "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"],
                    "description": "Time horizon (required for run/recommend, optional for pause/resume)",
                },
                "immediate": {
                    "type": "boolean",
                    "default": True,
                    "description": "For 'run' action: execute immediately vs schedule for later",
                },
                "content_hashes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Content hashes of the memories to merge (minimum 2)",
                },
                "merged_content": {
                    "type": "string",
                    "description": "The consolidated content that replaces the source memories",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for the merged memory (default: union of source tags)",
                },
                "memory_type": {
                    "type": "string",
                    "description": "Memory type for the merged memory (default: inherited from first source)",
                },
            },
            "required": ["action"],
        },
        annotations={"destructiveHint": False},
    ),
    ToolDef(
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
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to single document (for file mode)",
                },
                "directory_path": {
                    "type": "string",
                    "description": "Path to directory (for directory mode)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Tags to apply to all ingested memories",
                },
                "chunk_size": {
                    "type": "integer",
                    "default": 1000,
                    "description": "Target chunk size in characters",
                },
                "chunk_overlap": {
                    "type": "integer",
                    "default": 200,
                    "description": "Overlap between chunks",
                },
                "memory_type": {
                    "type": "string",
                    "default": "document",
                    "description": "Type label for created memories",
                },
                "recursive": {
                    "type": "boolean",
                    "default": True,
                    "description": "For directory mode: process subdirectories",
                },
                "file_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["pdf", "txt", "md", "json"],
                    "description": "For directory mode: file types to process",
                },
                "max_files": {
                    "type": "integer",
                    "default": 100,
                    "description": "For directory mode: maximum files to process",
                },
                "store": {
                    "type": "string",
                    "description": "Target store partition (default: 'default'). Use 'docs' for documents, 'all' for cross-store search.",
                },
            },
        },
        annotations={"destructiveHint": False},
    ),
    ToolDef(
        name="memory_harvest",
        description="""Extract learnings from Claude Code session transcripts.

USE THIS WHEN:
- End of session — auto-capture decisions, bugs, conventions, learnings
- User asks to "harvest" or "extract learnings" from sessions
- Building knowledge base from past sessions

MODES:
- dry_run=true (DEFAULT): Preview candidates without storing
- dry_run=false: Store candidates as tagged memories

MEMORY TYPES EXTRACTED:
- decision: Architectural choices, tool selections
- bug: Issues encountered and root causes
- convention: Patterns/rules established
- learning: Insights, mistakes learned from
- context: Session state for continuity

LLM CLASSIFICATION (Phase 2):
- use_llm=true: Validate candidates with Groq LLM (higher precision, ~$0.001/candidate)
- Filters conversation fragments, deduplicates, refines content
- Requires GROQ_API_KEY environment variable

Examples:
{"sessions": 1, "dry_run": true}
{"sessions": 3, "types": ["decision", "bug"]}
{"session_ids": ["abc123"], "dry_run": false}
{"min_confidence": 0.7, "sessions": 1}
{"sessions": 1, "use_llm": true, "dry_run": true}
""",
        input_schema={
            "type": "object",
            "properties": {
                "sessions": {
                    "type": "integer",
                    "default": 1,
                    "description": "Number of recent sessions to harvest",
                },
                "session_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific session IDs to harvest",
                },
                "types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "decision",
                            "bug",
                            "convention",
                            "learning",
                            "context",
                        ],
                    },
                    "description": "Filter by memory types (default: all)",
                },
                "min_confidence": {
                    "type": "number",
                    "default": 0.6,
                    "description": "Minimum confidence threshold (0.0-1.0)",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview candidates without storing (default: true)",
                },
                "project_path": {
                    "type": "string",
                    "description": "Override Claude Code project directory path",
                },
                "use_llm": {
                    "type": "boolean",
                    "default": False,
                    "description": "Use LLM to validate and refine candidates (requires GROQ_API_KEY)",
                },
                "auto_commit": {
                    "type": "boolean",
                    "default": False,
                    "description": "Bridge harvested candidates to commit_session_legacy (feeds bootstrap profile)",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Agent ID for auto_commit attribution (default: MCP_AGENT_ID env or 'unknown')",
                },
            },
        },
        annotations={"destructiveHint": False},
    ),
    ToolDef(
        name="memory_quality",
        description="""Memory quality management - rate, inspect, and analyze.

ACTIONS:
- rate: Manually rate a memory's quality (thumbs up/down)
- get: Get quality metrics for a specific memory
- analyze: Analyze quality distribution across all memories
- maintain: Run maintenance cycle (cleanup + conflicts + stale + quality)
- maintain_status: Get stats from last maintenance run

Examples:
{"action": "rate", "content_hash": "abc123", "rating": 1, "feedback": "Very useful"}
{"action": "get", "content_hash": "abc123"}
{"action": "analyze"}
{"action": "analyze", "min_quality": 0.5, "max_quality": 1.0}
{"action": "maintain"}
{"action": "maintain", "dry_run": false}
{"action": "maintain_status"}
""",
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["rate", "get", "analyze", "maintain", "maintain_status"],
                    "description": "Quality action to perform",
                },
                "content_hash": {
                    "type": "string",
                    "description": "Memory hash (required for rate/get)",
                },
                "rating": {
                    "type": "string",
                    "enum": ["-1", "0", "1"],
                    "description": "For 'rate': '-1' (thumbs down), '0' (neutral), '1' (thumbs up)",
                },
                "feedback": {
                    "type": "string",
                    "description": "For 'rate': Optional feedback text",
                },
                "min_quality": {
                    "type": "number",
                    "default": 0.0,
                    "description": "For 'analyze': minimum quality threshold",
                },
                "max_quality": {
                    "type": "number",
                    "default": 1.0,
                    "description": "For 'analyze': maximum quality threshold",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "For 'maintain': preview mode — no modifications (default: true)",
                },
            },
            "required": ["action"],
        },
        annotations={"destructiveHint": False},
    ),
    ToolDef(
        name="memory_graph",
        description="""Memory association graph operations - explore connections between memories.

ACTIONS:
- connected: Find memories connected via associations (BFS traversal)
- path: Find shortest path between two memories
- subgraph: Get graph structure around a memory for visualization
- infer: Find transitive relationships (A→B→C implies A→C). Params: rel_type (string), max_hops (int, default 2)
- suggest: Suggest potential relationships for a memory based on shared neighbors. Params: hash (string)

Examples:
{"action": "connected", "hash": "abc123", "max_hops": 2}
{"action": "path", "hash1": "abc123", "hash2": "def456", "max_depth": 5}
{"action": "subgraph", "hash": "abc123", "radius": 2}
{"action": "infer", "rel_type": "causes", "max_hops": 2}
{"action": "suggest", "hash": "abc123"}
""",
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["connected", "path", "subgraph", "infer", "suggest"],
                    "description": "Graph operation to perform",
                },
                "hash": {
                    "type": "string",
                    "description": "Memory hash (for connected/subgraph)",
                },
                "hash1": {
                    "type": "string",
                    "description": "Start memory hash (for path)",
                },
                "hash2": {
                    "type": "string",
                    "description": "End memory hash (for path)",
                },
                "max_hops": {
                    "type": "integer",
                    "default": 2,
                    "description": "For 'connected': max traversal depth",
                },
                "max_depth": {
                    "type": "integer",
                    "default": 5,
                    "description": "For 'path': max path length",
                },
                "radius": {
                    "type": "integer",
                    "default": 2,
                    "description": "For 'subgraph': nodes to include",
                },
                "rel_type": {
                    "type": "string",
                    "description": "For 'infer': relationship type to traverse",
                },
            },
            "required": ["action"],
        },
        annotations={"readOnlyHint": True},
    ),
    ToolDef(
        name="memory_conflicts",
        description="""List unresolved memory conflicts (contradictory memories detected by similarity analysis)""",
        input_schema={"type": "object", "properties": {}, "required": []},
        annotations={"readOnlyHint": True, "destructiveHint": False},
    ),
    ToolDef(
        name="memory_resolve",
        description="""Resolve a memory conflict by choosing a winner""",
        input_schema={
            "type": "object",
            "properties": {
                "winner_hash": {
                    "type": "string",
                    "description": "Content hash of the correct memory",
                },
                "loser_hash": {
                    "type": "string",
                    "description": "Content hash of the incorrect memory",
                },
            },
            "required": ["winner_hash", "loser_hash"],
        },
        annotations={"destructiveHint": True},
    ),
    ToolDef(
        name="mistake_note_add",
        description="""Record a mistake pattern for error replay. Tracks what went wrong and the correct action. Auto-increments failure_count for repeated patterns.""",
        input_schema={
            "type": "object",
            "properties": {
                "error_pattern": {
                    "type": "string",
                    "description": "The error pattern or message",
                },
                "context_signature": {
                    "type": "string",
                    "description": "Context where the error occurred (file, function, task type)",
                },
                "incorrect_action": {
                    "type": "string",
                    "description": "What was done incorrectly",
                },
                "correct_action": {
                    "type": "string",
                    "description": "What should have been done instead",
                },
            },
            "required": [
                "error_pattern",
                "context_signature",
                "incorrect_action",
                "correct_action",
            ],
        },
        annotations={"destructiveHint": False},
    ),
    ToolDef(
        name="mistake_note_search",
        description="""Search mistake notes by semantic similarity. Use before starting a task to check for known pitfalls and past errors.""",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (error message, context, or task description)",
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Max results (default: 5)",
                },
            },
            "required": ["query"],
        },
        annotations={"readOnlyHint": True},
    ),
    ToolDef(
        name="mistake_note_update",
        description="""Update fields of an existing mistake note. Can update failure_count, error_pattern, context_signature, incorrect_action, or correct_action. Content changes create a new hash (delete + re-store).""",
        input_schema={
            "type": "object",
            "properties": {
                "content_hash": {
                    "type": "string",
                    "description": "Content hash of the mistake note to update",
                },
                "failure_count": {
                    "type": "integer",
                    "description": "New failure count",
                },
                "error_pattern": {
                    "type": "string",
                    "description": "Updated error pattern",
                },
                "context_signature": {
                    "type": "string",
                    "description": "Updated context",
                },
                "incorrect_action": {
                    "type": "string",
                    "description": "Updated incorrect action",
                },
                "correct_action": {
                    "type": "string",
                    "description": "Updated correct action",
                },
            },
            "required": ["content_hash"],
        },
        annotations={"destructiveHint": False},
    ),
    ToolDef(
        name="mistake_note_delete",
        description="""Delete a mistake note by content hash. Only deletes memories with memory_type='mistake'.""",
        input_schema={
            "type": "object",
            "properties": {
                "content_hash": {
                    "type": "string",
                    "description": "Content hash of the mistake note to delete",
                },
            },
            "required": ["content_hash"],
        },
        annotations={"destructiveHint": True},
    ),
    ToolDef(
        name="get_quarantined_memories",
        description="""List memories that have been quarantined due to contradicting active beliefs. Use to review flagged memories.""",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "description": "Maximum results to return",
                },
            },
        },
        annotations={"readOnlyHint": True},
    ),
    ToolDef(
        name="unquarantine_memory",
        description="""Release a memory from quarantine. Use when a quarantined memory is valid (e.g., the contradicted belief was wrong).""",
        input_schema={
            "type": "object",
            "properties": {
                "content_hash": {
                    "type": "string",
                    "description": "Content hash of the quarantined memory",
                },
            },
            "required": ["content_hash"],
        },
        annotations={"readOnlyHint": False},
    ),
    ToolDef(
        name="commit_session_legacy",
        description="""Record end-of-session learnings from an ephemeral agent. Stores decisions, errors, corrections as structured observations.""",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Unique session identifier",
                },
                "agent_id": {"type": "string", "description": "Agent identifier"},
                "task_summary": {
                    "type": "string",
                    "description": "Brief description of what was attempted",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["success", "partial", "failure"],
                },
                "decisions": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Decisions made",
                },
                "errors": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Errors encountered",
                },
                "user_corrections": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "User corrections",
                },
                "belief_updates": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Belief updates",
                },
            },
            "required": ["session_id", "agent_id", "task_summary", "outcome"],
        },
        annotations={"destructiveHint": False},
    ),
    ToolDef(
        name="get_bootstrap_profile",
        description="""Generate a behavioral bootstrap profile for an ephemeral agent.""",
        input_schema={
            "type": "object",
            "properties": {
                "agent_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Agent identifiers",
                },
                "project_id": {
                    "type": "string",
                    "description": "Optional project scope",
                },
                "task_summary": {
                    "type": "string",
                    "description": "Optional task context",
                },
                "max_tokens": {
                    "type": "integer",
                    "default": 2048,
                    "description": "Token budget",
                },
                "budget_tokens": {
                    "type": "integer",
                    "description": "Override token budget for this call (takes precedence over max_tokens)",
                },
            },
            "required": ["agent_ids"],
        },
        annotations={"readOnlyHint": True},
    ),
    ToolDef(
        name="get_onboarding_guide",
        description="""Get integration guide for a specific client type. """,
        input_schema={
            "type": "object",
            "properties": {
                "client_type": {
                    "type": "string",
                    "default": "generic",
                    "description": "Client type (generic, kiro, claude-code)",
                },
            },
        },
        annotations={"readOnlyHint": True},
    ),
    ToolDef(
        name="memory_distill",
        description="""Extract insights from existing memories via LLM rewriter (batch mode). """,
        input_schema={
            "type": "object",
            "properties": {
                "batch_size": {
                    "type": "integer",
                    "default": 20,
                    "description": "Max memories to process per run",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview candidates without storing (default: true)",
                },
                "memory_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory types to process (default: observation, decision, reference, learning)",
                },
            },
        },
        annotations={"destructiveHint": False},
    ),
    ToolDef(
        name="memory_explore",
        description="""Explore a knowledge map of entities, with query-scoped chunks per entity (LLM-free).

REQUIRES A POPULATED ENTITY GRAPH — returns {"entities": [], "count": 0} otherwise, for
every query. Check with memory_graph {"action": "list_entities"}; populate with
memory_quality {"action": "maintain", "dry_run": false} (dry_run defaults to true and
stores nothing). Unavailable on the Cloudflare backend.
Guide: docs/guides/token-efficient-retrieval.md

USE THIS WHEN:
- User wants a high-level overview ("what do I know about X", "map out", "explore")
- Need entities + their relationships rather than raw memory chunks
- Building a knowledge graph view or entity dashboard

Returns a list of entities. Per entity:
- entity_id: canonical slug (NFKD-normalized, hash fallback for non-Latin scripts)
- name, entity_type
- summary: extractive (LLM-free) summary from top chunks
- relation_count: number of graph relationships
- top_chunks: most relevant memory chunks for the entity

Read-only. Composes with `store` for multi-store scoping.

Examples:
{"query": "authentication design"}
{"query": "python", "max_entities": 5, "chunks_per_entity": 2}
{"query": "database", "tags": ["reference"], "min_score": 0.3}
""",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query used to discover candidate entities",
                },
                "max_entities": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum number of entities to return",
                },
                "chunks_per_entity": {
                    "type": "integer",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum top chunks to include per entity",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict candidate retrieval to memories with any of these tags",
                },
                "min_score": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum relevance score for candidate chunks",
                },
                "scoring": {
                    "type": "string",
                    "enum": ["composite"],
                    "description": "Opt-in ranking mode (#55). 'composite' re-ranks chunks by semantic relevance + graph proximity + entity centrality, adding composite_score and score_components fields. Omit for default relevance-only ranking.",
                },
                "store": {
                    "type": "string",
                    "description": "Optional store/namespace to scope retrieval (composes with multi-store support)",
                },
            },
            "required": ["query"],
        },
        annotations={"readOnlyHint": True},
    ),
    ToolDef(
        name="memory_detail",
        description="""Get the full ranked detail for a single entity.

USE THIS WHEN:
- User drills into one entity from a knowledge map ("tell me more about X")
- Need the full ranked list of chunks for an entity plus its related entities

Returns:
- entity_id, name
- chunks: ranked memory chunks (relevance now; composite_score + score_components once #55 lands)
- related_entities: neighbouring entities (gated by include_related / max_hops)

Read-only.

Examples:
{"entity_id": "authentication-design"}
{"entity_id": "python", "limit": 20, "include_related": false}
{"entity_id": "database", "max_hops": 2}
""",
        input_schema={
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "Canonical entity id (slug) returned by memory_explore",
                },
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Maximum ranked chunks to return",
                },
                "include_related": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include related entities discovered via graph traversal",
                },
                "max_hops": {
                    "type": "integer",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 4,
                    "description": "Maximum hops for related-entity discovery",
                },
                "scoring": {
                    "type": "string",
                    "enum": ["composite"],
                    "description": "Opt-in ranking mode (#55). 'composite' re-ranks chunks by relevance + graph proximity + entity centrality, adding composite_score and score_components fields. Omit for default relevance-only ranking.",
                },
                "store": {
                    "type": "string",
                    "description": "Optional store/namespace to scope retrieval (composes with multi-store support)",
                },
            },
            "required": ["entity_id"],
        },
        annotations={"readOnlyHint": True},
    ),
]
