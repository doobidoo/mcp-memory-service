# Claude Code Memory Awareness Hooks

This directory contains Claude Code hooks that implement automatic memory awareness and intelligent context injection for the MCP Memory Service.

## Architecture

The memory awareness system consists of three main components:

### Core Hooks
- `session-start.js` - Automatic memory injection at session initialization
- `session-end.js` - Memory consolidation and outcome storage
- `topic-change.js` - Dynamic memory loading based on conversation evolution

### Utilities
- `project-detector.js` - Project context detection and analysis
- `memory-scorer.js` - Relevance scoring algorithms for memory selection
- `context-formatter.js` - Memory formatting for Claude Code injection

### Configuration
- `config.json` - Hook configuration and memory service endpoints
- `memory-filters.json` - Memory filtering rules and preferences

## Features

### Automatic Memory Injection
- **Session Startup**: Automatically loads relevant project memories
- **Dynamic Updates**: Real-time memory injection based on conversation topics
- **Cross-Session Continuity**: Links conversations across different sessions

### Intelligent Memory Selection  
- **Project Awareness**: Detects current project and loads relevant context
- **Relevance Scoring**: AI-powered memory selection based on conversation topics
- **Time Decay**: Prioritizes recent memories while maintaining historical context

### Memory Consolidation
- **Session Outcomes**: Automatically stores conversation insights
- **Auto-Tagging**: Intelligent categorization of memory content
- **Knowledge Building**: Progressive memory organization and linking

## Installation

1. Copy hooks to Claude Code hooks directory:
```bash
cp claude-hooks/* ~/.claude-code/hooks/
```

2. Configure memory service endpoint:
```bash
cd ~/.claude-code/hooks
cp config.template.json config.json
# Edit config.json with your memory service details
```

3. Test hook installation:
```bash
claude-hooks test memory-awareness
```

## Configuration

### Memory Service Setup
```json
{
  "memoryService": {
    "endpoint": "https://your-server:8443",
    "apiKey": "your-api-key",
    "defaultTags": ["claude-code", "auto-generated"],
    "maxMemoriesPerSession": 10
  }
}
```

### Project Detection Rules
```json
{
  "projectDetection": {
    "gitRepository": true,
    "packageFiles": ["package.json", "pyproject.toml", "Cargo.toml"],
    "frameworkDetection": true,
    "languageDetection": true
  }
}
```

## Usage

Once installed, the hooks work automatically:

1. **Starting a Claude Code session** triggers automatic memory loading
2. **Conversation topic changes** dynamically inject additional relevant memories  
3. **Ending a session** consolidates outcomes and stores new insights

No manual intervention required - the system learns and adapts to your development patterns.

## Development

### Testing Hooks
```bash
# Test individual hooks
npm test claude-hooks/tests/session-start.test.js

# Test full workflow
npm test claude-hooks/tests/integration.test.js
```

### Debugging
Set environment variable for verbose logging:
```bash
export CLAUDE_HOOKS_DEBUG=true
claude
```

## Architecture Diagrams

### Memory Injection Flow
```
Session Start → Project Detection → Memory Query → Relevance Scoring → Context Injection
```

### Dynamic Updates Flow  
```
Topic Change → Semantic Analysis → Additional Memory Query → Context Update
```

### Consolidation Flow
```
Session End → Conversation Analysis → Auto-Tagging → Memory Storage → Cross-Linking
```

This system transforms Claude Code into a memory-aware development assistant that maintains perfect context across all interactions.