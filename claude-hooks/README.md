# Claude Code Memory Awareness Hooks

Automatic memory awareness and intelligent context injection for Claude Code using the MCP Memory Service.

## Quick Start

```bash
cd claude-hooks

# Install Natural Memory Triggers v7.1.3 (recommended)
python install_hooks.py --natural-triggers

# OR install basic memory awareness hooks
python install_hooks.py --basic
```

> **Note**: The unified Python installer replaces all previous installers and provides cross-platform compatibility with enhanced features. See [MIGRATION.md](MIGRATION.md) for details.

This installs hooks that automatically:
- Load relevant project memories when Claude Code starts
- Inject meaningful contextual information (no more generic fluff!)
- Store session insights and decisions for future reference
- Provide on-demand memory retrieval when you need it

## Components

- **Core Hooks**: `session-start.js` (Hook v2.2), `session-end.js`, `memory-retrieval.js`, `permission-request.js` (v1.0) - Smart memory management and permission automation
- **Utilities**: Project detection, quality-aware scoring, intelligent formatting, context shift detection
- **Tests**: Comprehensive integration test suite (14 tests)

## Features

### 🎯 **NEW in v8.5.7**: SessionStart Hook Visibility Features
Three complementary ways to view session memory context:

1. **Visible Summary Output** - Clean bordered console display at session start
   - Shows: project name, storage backend, memory count (with recent indicator), git context
   - Respects `cleanMode` configuration for minimal output

2. **Detailed Log File** - `~/.claude/last-session-context.txt`
   - Auto-generated on each session start
   - Contains: project details, storage backend, memory statistics, git analysis, top loaded memories

3. **Status Line Display** ⭐ - Always-visible status bar at bottom of terminal
   - Format: `🧠 8 (8 recent) memories | 📊 10 commits`
   - Displays static session memory context (set once at session start)
   - **Requires**: `jq` (JSON parser) and Claude Code statusLine configuration
   - **Platform**: Linux/macOS (Windows SessionStart hook broken - issue #160)
   - **Windows Workaround**: Use `/session-start` slash command for manual session initialization

### ✨ **Hook v2.2.0**: Enhanced Output Control
- **Clean Output Mode**: Configurable verbosity levels for minimal or detailed output
- **Smart Filtering**: Hide memory scoring details while keeping essential information
- **Professional UX**: Removed noisy wrapper tags and improved ANSI formatting
- **Granular Control**: Fine-tune what information is displayed during hook execution

### 🧠 **Previous Features (Project v6.7.0)**: Smart Memory Context  
- **Quality Content Extraction**: Extracts actual decisions/insights from session summaries instead of "implementation..." fluff
- **Duplicate Filtering**: Automatically removes repetitive session summaries
- **Smart Timing**: Only injects memories when contextually appropriate (no more mid-session disruptions)
- **On-Demand Retrieval**: Manual memory refresh with `memory-retrieval.js` hook

### 🧠 **Core Features**
- **Automatic Memory Injection**: Load relevant memories at session start with quality filtering
- **Project Awareness**: Detect current project context and frameworks
- **Memory Consolidation**: Store session outcomes and insights
- **Intelligent Selection**: Quality-aware scoring that prioritizes meaningful content over just recency
- **🆕 Permission Automation** (v1.0): Auto-approve safe MCP tool operations, block destructive actions - See [README-PERMISSION-REQUEST.md](README-PERMISSION-REQUEST.md)

### 🎮 **User Overrides** (`#skip` / `#remember`)

Control memory hook behavior directly from your messages:

| Trigger | Effect |
|---------|--------|
| `#skip` | Skip all memory operations for this message |
| `#remember` | Force memory operations (bypass cooldowns, thresholds) |

**Examples:**
```
User: "Quick test, ignore this #skip"
→ ⏭️ Memory Hook → Skipped by user override (#skip)

User: "Important architecture decision #remember"
→ 💾 Memory Hook → Force triggered by user override (#remember)
```

**Supported Hooks:**

| Hook | `#skip` | `#remember` |
|------|---------|-------------|
| session-start | Skips memory retrieval | Forces retrieval |
| mid-conversation | Skips analysis | Bypasses cooldown, forces trigger |
| session-end | Skips consolidation | Bypasses length/confidence checks |
| auto-capture | Skips storage | Forces storage |

Configuration in `config.json`:
```json
{
  "autoCapture": {
    "userOverrides": {
      "forceRemember": "#remember",
      "forceSkip": "#skip",
      "applyTo": ["auto-capture", "session-start", "mid-conversation", "session-end"]
    }
  }
}
```

## Installation

### Unified Installer (Recommended)
```bash
cd claude-hooks
python install_hooks.py  # Installs all features

# OR install specific features
python install_hooks.py --basic             # Basic memory hooks only
python install_hooks.py --natural-triggers  # Natural Memory Triggers only
```

**Prerequisites:**
- Python 3.10+
- Node.js (for hooks execution)
- **`jq`** (required for statusLine feature - displays memory context in Claude Code status bar)
  - macOS: `brew install jq`
  - Linux: `sudo apt install jq` (Ubuntu/Debian) or equivalent
  - Windows: `choco install jq` or download from https://jqlang.github.io/jq/

### Manual
```bash
cp -r claude-hooks/* ~/.claude/hooks/
chmod +x ~/.claude/hooks/statusline.sh  # Make statusline executable (Unix only)
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

### ⚙️ **Output Verbosity Control (Hook v2.2.0)**
```json
{
  "output": {
    "verbose": true,           // Show hook activity messages
    "showMemoryDetails": false, // Hide detailed memory scoring
    "showProjectDetails": true, // Show project detection info
    "showScoringDetails": false,// Hide scoring breakdowns
    "cleanMode": false         // Ultra-minimal output mode
  }
}
```

**Verbosity Levels**:
- **Normal** (`verbose: true`, others `false`): Shows essential information only
- **Detailed** (`showMemoryDetails: true`): Include memory scoring details  
- **Clean** (`cleanMode: true`): Minimal output, only success/error messages
- **Silent** (`verbose: false`): Hook works silently in background

### ⚙️ **Previous Configuration Options (Project v6.7.0)**
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

### Windows-Specific Issues

#### Path Configuration
- **Directory Structure**: Hooks should be installed to `%USERPROFILE%\.claude\hooks\`
- **JSON Path Format**: Use forward slashes in settings.json: `"command": "node C:/Users/username/.claude/hooks/core/session-start.js"`
- **Avoid Backslashes**: Windows backslashes in JSON need escaping: `"C:\\\\Users\\\\..."` (prefer forward slashes instead)

#### Settings Configuration Example
```json
{
  "hooks": [
    {
      "pattern": "session-start",
      "command": "node C:/Users/your-username/.claude/hooks/core/session-start.js"
    }
  ]
}
```

#### Common Fixes
- **Wrong Path Format**: If you see `session-start-wrapper.bat` errors, update your settings.json to use the Node.js script directly
- **Legacy Directory**: If using old `.claude-code` directory, move contents to `.claude` directory
- **Permission Issues**: Run installation scripts as Administrator if needed

## Changelog

### Hook v2.2.0 (2025-01-25) - Enhanced Output Control
**🎯 Focus**: Professional UX and configurable verbosity

**New Features**:
- **Output Verbosity Control**: Granular configuration for hook output levels
- **Clean Mode**: Ultra-minimal output option for distraction-free usage
- **Smart Filtering**: Hide memory scoring details while preserving essential information

**Improvements**:
- **Removed Noise**: Eliminated `<session-start-hook>` wrapper tags and verbose logging
- **Enhanced ANSI**: Improved color consistency and formatting throughout
- **Better Defaults**: Less verbose output by default while maintaining functionality

**Configuration**:
- Added `output` section with `verbose`, `showMemoryDetails`, `showProjectDetails`, `cleanMode` options
- Backwards compatible - existing configurations work without changes
- Self-documenting configuration with clear field names

### Hook v2.1.0 - Smart Memory Integration
- Advanced memory scoring and quality assessment
- Enhanced context injection with deduplication
- Improved project detection and context awareness

### Project v6.7.0 - Smart Memory Context
- Quality content extraction and duplicate filtering
- Smart timing and context shift detection
- On-demand memory retrieval capabilities

## Documentation

For comprehensive documentation including detailed troubleshooting, advanced configuration, and development guides, see:

**[📖 Memory Awareness Hooks - Detailed Guide](https://github.com/doobidoo/mcp-memory-service/wiki/Memory-Awareness-Hooks-Detailed-Guide)**

This guide covers:
- Advanced installation methods and configuration options
- Comprehensive troubleshooting with solutions
- Custom hook development and architecture diagrams
- Memory service integration and testing frameworks