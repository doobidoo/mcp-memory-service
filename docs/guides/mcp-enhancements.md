# MCP Protocol Enhancements Guide

This guide covers the enhanced MCP (Model Context Protocol) features introduced in v4.1.0, including Resources, Prompts, and Progress Tracking.

## Table of Contents
- [Enhanced Resources](#enhanced-resources)
- [Guided Prompts](#guided-prompts)
- [Progress Tracking](#progress-tracking)
- [Integration Examples](#integration-examples)

## Enhanced Resources

The MCP Memory Service now exposes memory collections through URI-based resources, allowing clients to access structured data directly.

### Available Resources

#### 1. Memory Statistics
```
URI: memory://stats
Returns: JSON object with database statistics
```

Example response:
```json
{
  "total_memories": 1234,
  "storage_backend": "SqliteVecStorage",
  "status": "operational",
  "total_tags": 45,
  "storage_size": "12.3 MB"
}
```

#### 2. Available Tags
```
URI: memory://tags
Returns: List of all unique tags in the database
```

Example response:
```json
{
  "tags": ["work", "personal", "learning", "project-x", "meeting-notes"],
  "count": 5
}
```

#### 3. Recent Memories
```
URI: memory://recent/{n}
Parameters: n = number of memories to retrieve
Returns: N most recent memories
```

Example: `memory://recent/10` returns the 10 most recent memories.

#### 4. Memories by Tag
```
URI: memory://tag/{tagname}
Parameters: tagname = specific tag to filter by
Returns: All memories with the specified tag
```

Example: `memory://tag/learning` returns all memories tagged with "learning".

#### 5. Dynamic Search
```
URI: memory://search/{query}
Parameters: query = search query
Returns: Search results matching the query
```

Example: `memory://search/python%20programming` searches for memories about Python programming.

### Resource Templates

The service provides templates for dynamic resource access:

```json
[
  {
    "uriTemplate": "memory://recent/{n}",
    "name": "Recent Memories",
    "description": "Get N most recent memories"
  },
  {
    "uriTemplate": "memory://tag/{tag}",
    "name": "Memories by Tag",
    "description": "Get all memories with a specific tag"
  },
  {
    "uriTemplate": "memory://search/{query}",
    "name": "Search Memories",
    "description": "Search memories by query"
  }
]
```

## Guided Prompts

Interactive workflows guide users through common memory operations with structured inputs and outputs.

### Available Prompts

#### 1. Memory Review
Review and organize memories from a specific time period.

**Arguments:**
- `time_period` (required): Time period to review (e.g., "last week", "yesterday")
- `focus_area` (optional): Area to focus on (e.g., "work", "personal")

**Example:**
```json
{
  "name": "memory_review",
  "arguments": {
    "time_period": "last week",
    "focus_area": "work"
  }
}
```

#### 2. Memory Analysis
Analyze patterns and themes in stored memories.

**Arguments:**
- `tags` (optional): Comma-separated tags to analyze
- `time_range` (optional): Time range for analysis (e.g., "last month")

**Example:**
```json
{
  "name": "memory_analysis",
  "arguments": {
    "tags": "learning,python",
    "time_range": "last month"
  }
}
```

#### 3. Knowledge Export
Export memories in various formats.

**Arguments:**
- `format` (required): Export format ("json", "markdown", "text")
- `filter` (optional): Filter criteria (tags or search query)

**Example:**
```json
{
  "name": "knowledge_export",
  "arguments": {
    "format": "markdown",
    "filter": "project-x"
  }
}
```

#### 4. Memory Cleanup
Identify and remove duplicate or outdated memories.

**Arguments:**
- `older_than` (optional): Remove memories older than specified period
- `similarity_threshold` (optional): Threshold for duplicate detection (0.0-1.0)

**Example:**
```json
{
  "name": "memory_cleanup",
  "arguments": {
    "older_than": "6 months",
    "similarity_threshold": "0.95"
  }
}
```

#### 5. Learning Session
Store structured learning notes with automatic categorization.

**Arguments:**
- `topic` (required): Learning topic or subject
- `key_points` (required): Comma-separated key points learned
- `questions` (optional): Questions for further study

**Example:**
```json
{
  "name": "learning_session",
  "arguments": {
    "topic": "Machine Learning Basics",
    "key_points": "supervised learning, neural networks, backpropagation",
    "questions": "How does gradient descent work?, What is overfitting?"
  }
}
```

## Progress Tracking

Long-running operations now provide real-time progress updates through the MCP notification system.

### Operations with Progress Tracking

#### 1. Bulk Deletion (`delete_by_tags`)
Provides step-by-step progress when deleting memories by tags:

```
0% - Starting deletion of memories with tags: [tag1, tag2]
25% - Searching for memories to delete...
50% - Deleting memories...
90% - Deleted 45 memories
100% - Deletion completed: Successfully deleted 45 memories
```

### Operation IDs

Each long-running operation receives a unique ID for tracking:

```
Operation ID: delete_by_tags_a1b2c3d4
```

### Progress Notification Structure

Progress notifications follow the MCP protocol:

```json
{
  "progress": 50,
  "progress_token": "operation_id_12345",
  "message": "Processing memories..."
}
```

## Integration Examples

### Accessing Resources in Claude Code

```python
# List available resources
resources = await mcp_client.list_resources()

# Read specific resource
stats = await mcp_client.read_resource("memory://stats")
recent = await mcp_client.read_resource("memory://recent/20")
```

### Using Prompts

```python
# Execute a memory review prompt
result = await mcp_client.get_prompt(
    name="memory_review",
    arguments={
        "time_period": "yesterday",
        "focus_area": "meetings"
    }
)
```

### Tracking Progress

```python
# Start operation and track progress
operation = await mcp_client.call_tool(
    name="delete_by_tags",
    arguments={"tags": ["temporary", "test"]}
)

# Progress notifications will be sent automatically
# Monitor via operation_id in the response
```

## Best Practices

1. **Resources**: Use resources for read-only access to memory data
2. **Prompts**: Use prompts for interactive, guided workflows
3. **Progress Tracking**: Monitor operation IDs for long-running tasks
4. **Error Handling**: All operations return structured error messages
5. **Performance**: Resources are optimized for quick access

## Compatibility

These enhancements maintain full backward compatibility with existing MCP clients while providing richer functionality for clients that support the extended features.

## Further Reading

- [MCP Specification](https://modelcontextprotocol.info/specification/2024-11-05/)
- [Memory Service API Documentation](../api/README.md)
- [Claude Code Integration Guide](./claude-code-integration.md)
