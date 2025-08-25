# Claude Code Memory Awareness Hooks

Automatic memory awareness and intelligent context injection for Claude Code using the MCP Memory Service.

## Quick Start

```bash
cd claude-hooks
./install.sh
```

This installs hooks that automatically:
- Load relevant project memories when Claude Code starts
- Inject meaningful contextual information (no more generic fluff!)
- Store session insights and decisions for future reference
- Provide on-demand memory retrieval when you need it

## Components

- **Core Hooks**: `session-start.js` (v2.0), `session-end.js`, `memory-retrieval.js` - Smart memory management
- **Utilities**: Project detection, quality-aware scoring, intelligent formatting, context shift detection
- **Tests**: Comprehensive integration test suite (14 tests)

## Features

### ✨ **NEW in v6.7.0**: Smart Memory Context
- **Quality Content Extraction**: Extracts actual decisions/insights from session summaries instead of "implementation..." fluff
- **Duplicate Filtering**: Automatically removes repetitive session summaries
- **Smart Timing**: Only injects memories when contextually appropriate (no more mid-session disruptions)
- **On-Demand Retrieval**: Manual memory refresh with `memory-retrieval.js` hook

### 🧠 **Core Features**
- **Automatic Memory Injection**: Load relevant memories at session start with quality filtering
- **Project Awareness**: Detect current project context and frameworks  
- **Memory Consolidation**: Store session outcomes and insights
- **Intelligent Selection**: Quality-aware scoring that prioritizes meaningful content over just recency

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
    "maxMemoriesPerSession": 8,
    "injectAfterCompacting": false
  },
  "memoryScoring": {
    "weights": {
      "timeDecay": 0.25,
      "tagRelevance": 0.35,
      "contentRelevance": 0.15,
      "contentQuality": 0.25
    }
  }
}
```

### ⚙️ **New Configuration Options (v6.7.0)**
- `injectAfterCompacting`: Controls whether to inject memories after compacting events (default: `false`)
- `contentQuality`: New scoring weight for content quality assessment (filters generic summaries)
- Enhanced memory filtering automatically removes "implementation..." fluff

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