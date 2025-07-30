# Store Memory via Direct API

I'll help you store information directly to a remote MCP Memory Service using HTTP API calls, bypassing the MCP protocol entirely. This command uses direct REST API access for maximum reliability and simplicity.

## What I'll do:

1. **Direct API Storage**: I'll send your memory content directly to the memory server via HTTP POST to `/api/memories`

2. **Automatic Context Detection**: I'll analyze the current session to add relevant context:
   - Current project directory and git repository
   - Programming languages and file types detected
   - Session timestamp and working context

3. **Smart Tagging**: I'll generate intelligent tags based on:
   - Project name and directory structure
   - Code languages and frameworks detected
   - Any explicit tags you provide
   - Current date and session context

4. **Authentication Handling**: I'll automatically use the configured API key for the remote memory server

## Usage Examples:

```bash
claude /memory-api-store "We implemented remote memory bridge with retry logic and error handling"
claude /memory-api-store --tags "implementation,bridge,api" "Remote memory connection working successfully"
claude /memory-api-store --server "https://memory.local/api" "Store to specific server"
```

## API Configuration:

The command uses these environment variables or defaults:
- **Server Endpoint**: `https://memory.local/api` (default)
- **API Key**: `mcp-0b1ccbde2197a08dcb12d41af4044be6` (default)
- **SSL Verification**: Disabled for self-signed certificates

## Implementation:

I'll use curl commands directly for maximum reliability:

```bash
curl -k -H "Authorization: Bearer API_KEY" \\
  -H "Content-Type: application/json" \\
  -X POST https://memory.local/api/memories \\
  -d '{"content":"MEMORY_CONTENT","tags":["tag1","tag2"]}'
```

## Response Handling:

After storing, I'll:
- Confirm successful storage with content hash
- Display the generated tags and metadata
- Show the exact API endpoint used
- Provide retrieval hints for future access

## Arguments:

- `$ARGUMENTS` - The content to store, or additional options:
  - `--tags "tag1,tag2"` - Explicit comma-separated tags
  - `--server "https://server/api"` - Override server endpoint
  - `--key "api-key"` - Override API key
  - `--project "name"` - Override project name detection
  - `--type "note|decision|task|reference"` - Memory type classification

## Advantages over MCP:

- **No Bridge Required**: Direct HTTP calls, no Node.js bridge needed
- **Maximum Reliability**: Fewer layers, direct connection
- **Easy Debugging**: Standard HTTP requests, easy to troubleshoot
- **Cross-Platform**: Works anywhere curl is available
- **No Dependencies**: No MCP client libraries required

## Error Handling:

If the API call fails, I'll:
- Show the exact curl command that failed
- Provide troubleshooting steps for common issues
- Suggest alternative servers or configurations
- Offer MCP fallback options if available

This command provides the most direct and reliable way to store memories without any MCP complexity.