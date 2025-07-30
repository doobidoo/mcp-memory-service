# Search Memory via Direct API

I'll help you search through stored memories using direct HTTP API calls to the remote memory server. This bypasses all MCP complexity and uses reliable tag-based search for maximum effectiveness.

## What I'll do:

1. **Direct API Search**: I'll query the memory server directly via HTTP POST to `/api/search/by-tag` (recommended) or `/api/search` (semantic)

2. **Smart Search Strategy**: I'll:
   - Use tag-based search as the primary method (most reliable)
   - Fall back to semantic search if specifically requested
   - Combine multiple search approaches when needed

3. **Intelligent Query Processing**: I'll analyze your search request to:
   - Extract relevant tags from your query
   - Identify project-specific context
   - Add temporal filters if time references are mentioned
   - Suggest alternative search terms

4. **Result Processing**: I'll format and present results with:
   - Relevance scoring and match explanations
   - Content excerpts with highlighted matches
   - Metadata like creation dates and tags
   - Content hashes for reference

## Usage Examples:

```bash
claude /memory-api-search "remote memory bridge"
claude /memory-api-search --tags "implementation,bridge" 
claude /memory-api-search --semantic "database performance optimization"
claude /memory-api-search --tags "July 30 2025" --limit 10
```

## Search Methods:

### Tag-Based Search (Recommended)
- **Endpoint**: `/api/search/by-tag`
- **Reliability**: 100% - always works
- **Speed**: Very fast (1-2ms typical)
- **Use Case**: Finding memories by known categories

### Semantic Search
- **Endpoint**: `/api/search`
- **Reliability**: Currently limited due to embedding indexing issues
- **Speed**: Moderate (300-500ms typical)
- **Use Case**: Natural language content matching

## API Implementation:

I'll use these curl commands:

**Tag Search:**
```bash
curl -k -H "Authorization: Bearer API_KEY" \\
  -H "Content-Type: application/json" \\
  -X POST https://memory.local/api/search/by-tag \\
  -d '{"tags":["tag1","tag2"],"n_results":10}'
```

**Semantic Search:**
```bash
curl -k -H "Authorization: Bearer API_KEY" \\
  -H "Content-Type: application/json" \\
  -X POST https://memory.local/api/search \\
  -d '{"query":"search terms","n_results":5}'
```

## Configuration:

- **Server Endpoint**: `https://memory.local/api` (default)
- **API Key**: `mcp-0b1ccbde2197a08dcb12d41af4044be6` (default)  
- **SSL Verification**: Disabled for self-signed certificates
- **Default Results**: 10 for tag search, 5 for semantic search

## Arguments:

- `$ARGUMENTS` - Search query or options:
  - `--tags "tag1,tag2"` - Search by specific tags (recommended)
  - `--semantic` - Force semantic search instead of tag search
  - `--limit N` - Number of results to return
  - `--server "https://server/api"` - Override server endpoint
  - `--key "api-key"` - Override API key
  - `--after "date"` - Results after specific date
  - `--before "date"` - Results before specific date

## Smart Query Processing:

I'll automatically detect and transform queries:

- **"remote memory"** → tags: ["remote-memory", "remote", "memory"]
- **"July 30 2025"** → tags: ["July 30 2025", "2025-07-30"]
- **"database decision"** → tags: ["database", "decision"]
- **"last week's work"** → tags: [date-based-tags]

## Result Presentation:

For each result, I'll show:
- **Content Preview**: First 200 characters with key terms highlighted
- **Relevance**: Match explanation and similarity score
- **Metadata**: Creation date, tags, content hash
- **Context**: Project and session information
- **Actions**: Quick commands to view full content or related memories

## Known Limitations:

- **Semantic Search**: Currently has embedding indexing issues on the server
- **Workaround**: Tag-based search is 100% reliable and recommended
- **Performance**: Tag searches are significantly faster than semantic searches

## Error Handling:

If searches fail, I'll:
- Show the exact API call that was attempted
- Suggest alternative search strategies
- Provide diagnostic information about server connectivity
- Offer manual curl commands for direct testing

This command provides the most reliable way to search memories using direct API access without MCP dependencies.