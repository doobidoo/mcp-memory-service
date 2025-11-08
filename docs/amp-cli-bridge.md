# Amp CLI Bridge (Semi-Automated Workflow)

**Purpose**: Leverage Amp CLI capabilities (research, code analysis, web search) from Claude Code without consuming Claude Code credits, using a semi-automated file-based workflow.

## Quick Start

**1. Claude Code creates prompt**:
```
You: "Use @agent-amp-bridge to research TypeScript 5.0 features"
Claude: [Creates prompt file and shows command]
```

**2. Run the command shown**:
```bash
amp @.claude/amp/prompts/pending/research-xyz.json
```

**3. Amp processes and writes response automatically**

**4. Claude Code continues automatically**

## Architecture

```
Claude Code (@agent-amp-bridge) → .claude/amp/prompts/pending/{uuid}.json
                                            ↓
                          You run: amp @prompts/pending/{uuid}.json
                                            ↓
                          Amp writes: responses/ready/{uuid}.json
                                            ↓
                   Claude Code reads response ← Workflow continues
```

## File Structure

```
.claude/amp/
├── prompts/
│   └── pending/        # Prompts waiting for you to process
├── responses/
│   ├── ready/          # Responses written by Amp
│   └── consumed/       # Archive of processed responses
└── README.md           # Documentation
```

## Message Format

**Prompt** (`.claude/amp/prompts/pending/{uuid}.json`):
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-11-04T20:00:00.000Z",
  "prompt": "Research async/await best practices in Python",
  "context": {
    "project": "mcp-memory-service",
    "cwd": "/path/to/project"
  },
  "options": {
    "timeout": 300000,
    "format": "markdown"
  }
}
```

**Response** (`.claude/amp/responses/ready/{uuid}.json`):
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-11-04T20:05:00.000Z",
  "success": true,
  "output": "## Async/Await Best Practices\n\n...",
  "error": null,
  "duration": 300000
}
```

## Configuration

**File**: `.claude/amp/config.json`

```json
{
  "pollInterval": 1000,      // Check for new prompts every 1s
  "timeout": 300000,          // 5 minute timeout per prompt
  "debug": false,             // Enable debug logging
  "ampCommand": "amp"         // Amp CLI command
}
```

## Use Cases

- Web Research: "Research latest React 18 features"
- Code Analysis: "Analyze our storage backend architecture"
- Documentation: "Generate API docs for MCP tools"
- Code Generation: "Create TypeScript type definitions"
- Best Practices: "Find OAuth 2.1 security recommendations"

## Manual Inspection (Optional)

```bash
# List pending prompts
ls -lt .claude/amp/prompts/pending/

# View prompt content
cat .claude/amp/prompts/pending/{uuid}.json | jq -r '.prompt'
```

## Troubleshooting

**Amp CLI credit errors:**
```bash
# Test if Amp is authenticated
amp

# If credits exhausted, check status
# https://ampcode.com/settings
```

**Response not appearing:**
```bash
# Verify Amp wrote the file
ls -lt .claude/amp/responses/ready/
```

**Permission issues:**
```bash
# Ensure directories exist
ls -la .claude/amp/

# Check write permissions
touch .claude/amp/responses/ready/test.json && rm .claude/amp/responses/ready/test.json
```

## Benefits

- Zero Claude Code Credits: Uses your separate Amp session
- Uses Free Tier: Works with Amp's free tier (when credits available)
- Simple Workflow: No background processes
- Full Control: You decide when/what to process
- Fault Tolerant: File-based queue survives crashes
- Audit Trail: All prompts/responses saved
- Reusable: Can replay prompts or review past responses

## Limitations

- Manual Step Required: You must run the `amp @` command
- Amp Credits: Still consumes Amp API credits
- Semi-Async: Claude Code waits for you to process
- Best for Research: Optimized for async research tasks, not real-time chat
