# Recall Memories by Time and Context

I'll help you retrieve memories from your MCP Memory Service using natural language time expressions and contextual queries. This command excels at finding past conversations, decisions, and notes based on when they occurred.

## What I'll do:

1. **Parse Time Expressions**: I'll interpret natural language time queries like:
   - "yesterday", "last week", "two months ago"
   - "last Tuesday", "this morning", "last summer"
   - "before the database migration", "since we started using SQLite"

2. **Context-Aware Search**: I'll consider the current project context to find relevant memories related to your current work.

3. **Smart Filtering**: I'll automatically filter results to show the most relevant memories first, considering:
   - Temporal relevance to your query
   - Project and directory context matching
   - Semantic similarity to current work

4. **Present Results**: I'll format the retrieved memories with clear context about when they were created and why they're relevant.

## Usage Examples:

```bash
claude /memory-recall "what did we decide about the database last week?"
claude /memory-recall "yesterday's architectural decisions"
claude /memory-recall "memories from when we were working on the mDNS feature"
claude /memory-recall --project "mcp-memory-service" "last month's progress"
```

## Implementation:

I'll connect to your MCP Memory Service at `https://memory.local:8443/` and use its API endpoints. The recall process involves:

1. **Query Processing**: Parse the natural language time expression and extract context clues
2. **Memory Retrieval**: Use the appropriate API endpoints:
   - `POST /api/search/by-time` - Natural language time-based queries
   - `POST /api/search` - Semantic search for context-based recall
   - `GET /api/memories` - List memories with pagination and filtering
   - `GET /api/memories/{hash}` - Retrieve specific memory by hash
3. **Context Matching**: Filter results based on current project and directory context
4. **Relevance Scoring**: Use similarity scores from the API responses
5. **Result Presentation**: Format memories with timestamps, tags, and relevance context

All requests use curl with `-k` flag for HTTPS and proper JSON formatting.

For each recalled memory, I'll show:
- **Content**: The actual memory content
- **Created**: When the memory was stored
- **Tags**: Associated tags and categories
- **Context**: Project and session context when stored
- **Relevance**: Why this memory matches your query

## Time Expression Examples:

- **Relative**: "yesterday", "last week", "two days ago", "this month"
- **Seasonal**: "last summer", "this winter", "spring 2024"

**Note**: Some expressions like "last hour" may not be supported by the time parser. Standard expressions like "today", "yesterday", "last week" work reliably.
- **Event-based**: "before the refactor", "since we switched to SQLite", "during the testing phase"
- **Specific**: "January 15th", "last Tuesday morning", "end of last month"

## Arguments:

- `$ARGUMENTS` - The time-based query, with optional flags:
  - `--limit N` - Maximum number of memories to retrieve (default: 10)
  - `--project "name"` - Filter by specific project
  - `--tags "tag1,tag2"` - Additional tag filtering
  - `--type "note|decision|task"` - Filter by memory type
  - `--include-context` - Show full session context for each memory

If no memories are found for the specified time period, I'll suggest broadening the search or checking if the MCP Memory Service contains data for that timeframe.
