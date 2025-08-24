# Claude Code Memory Awareness Hooks

Automatic memory awareness and intelligent context injection for Claude Code using the MCP Memory Service.

## Quick Start

```bash
cd claude-hooks
./install.sh
```

This installs hooks that automatically:
- Load relevant project memories when Claude Code starts
- Inject contextual information based on your current project
- Store session insights and decisions for future reference

## Components

- **Core Hooks**: `session-start.js`, `session-end.js` - Automatic memory injection and consolidation
- **Utilities**: Project detection, memory scoring, context formatting
- **Tests**: Comprehensive integration test suite (14 tests)

## Features

- **Automatic Memory Injection**: Load relevant memories at session start
- **Project Awareness**: Detect current project context and frameworks  
- **Memory Consolidation**: Store session outcomes and insights
- **Intelligent Selection**: AI-powered relevance scoring and time decay

## Installation

### Automated (Recommended)
```bash
cd claude-hooks
./install.sh
```

### Manual
```bash
cp -r claude-hooks/* ~/.claude/hooks/
# Edit ~/.claude/settings.json and ~/.claude/hooks/config.json
```

## Verification

After installation:
```bash
claude --debug hooks  # Should show "Found 1 hook matchers in settings"
cd ~/.claude/hooks && node tests/integration-test.js  # Run 14 integration tests
```

## Configuration

Edit `~/.claude/hooks/config.json`:
```json
{
  "memoryService": {
    "endpoint": "https://your-server:8443",
    "apiKey": "your-api-key",
    "maxMemoriesPerSession": 10
  }
}
```

## Usage

Once installed, hooks work automatically:
- **Session start**: Load relevant project memories
- **Session end**: Store insights and decisions
- No manual intervention required

## Troubleshooting

### Quick Fixes
- **Hooks not detected**: `ls ~/.claude/settings.json` → Reinstall if missing
- **JSON parse errors**: Update to latest version (includes Python dict conversion)
- **Connection failed**: Check `curl -k https://your-endpoint:8443/api/health`
- **Wrong directory**: Move `~/.claude-code/hooks/*` to `~/.claude/hooks/`

### Debug Mode
```bash
claude --debug hooks  # Shows hook execution details
node ~/.claude/hooks/core/session-start.js  # Test individual hooks
```

## Documentation

For comprehensive documentation including detailed troubleshooting, advanced configuration, and development guides, see:

**[📖 Memory Awareness Hooks - Detailed Guide](https://github.com/doobidoo/mcp-memory-service/wiki/Memory-Awareness-Hooks-Detailed-Guide)**

This guide covers:
- Advanced installation methods and configuration options
- Comprehensive troubleshooting with solutions
- Custom hook development and architecture diagrams
- Memory service integration and testing frameworks