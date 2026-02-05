# Natural Memory Triggers v7.1.0 - Installation Guide

This guide provides detailed installation instructions for Natural Memory Triggers, the intelligent automatic memory awareness system for Claude Code.

## Prerequisites

Before installing Natural Memory Triggers, ensure you have:

- ✅ **Claude Code CLI** installed and working (`claude --version`)
- ✅ **Node.js 14+** for hook execution (`node --version`)
- ✅ **MCP Memory Service** running (`curl -k https://localhost:8443/api/health`)
- ✅ **Valid configuration** at `~/.claude/hooks/config.json`

## Installation Methods

### Method 1: Automated Installation (Recommended)

The automated installer handles the complete setup with comprehensive testing:

```bash
# Navigate to the claude-hooks directory
cd mcp-memory-service/claude-hooks

# Install with unified Python installer
python install_hooks.py --natural-triggers
```

**What the installer does:**

1. **System Verification**
   - Checks Claude Code CLI availability
   - Validates Node.js version compatibility
   - Tests MCP Memory Service connectivity
   - Verifies directory permissions

2. **Backup Operations**
   - Creates backup of existing `~/.claude/hooks/` directory
   - Preserves current configuration files
   - Backs up existing hook implementations

3. **Component Installation**
   - Copies Natural Memory Triggers core components
   - Installs multi-tier conversation monitor
   - Sets up performance manager and git analyzer
   - Installs CLI management controller

4. **Configuration Setup**
   - Merges new configuration sections with existing settings
   - Preserves user customizations
   - Adds Natural Memory Triggers specific settings
   - Configures performance profiles

5. **Testing and Validation**
   - Runs 18 comprehensive tests
   - Tests semantic analysis functionality
   - Validates CLI controller operations
   - Checks memory service integration

6. **Installation Report**
   - Provides detailed installation summary
   - Lists installed components and their versions
   - Shows configuration status and recommendations
   - Provides next steps and usage instructions

### Method 2: Manual Installation

For users who prefer manual control or need custom configurations:

#### Step 1: Directory Setup

```bash
# Create required directory structure
mkdir -p ~/.claude/hooks/{core,utilities,tests}

# Verify directory creation
ls -la ~/.claude/hooks/
```

#### Step 2: Copy Core Components

```bash
# Copy main hook implementation
cp claude-hooks/core/mid-conversation.js ~/.claude/hooks/core/

# Copy utility modules
cp claude-hooks/utilities/tiered-conversation-monitor.js ~/.claude/hooks/utilities/
cp claude-hooks/utilities/performance-manager.js ~/.claude/hooks/utilities/
cp claude-hooks/utilities/git-analyzer.js ~/.claude/hooks/utilities/
cp claude-hooks/utilities/mcp-client.js ~/.claude/hooks/utilities/

# Copy CLI management system
cp claude-hooks/memory-mode-controller.js ~/.claude/hooks/

# Copy test suite
cp claude-hooks/test-natural-triggers.js ~/.claude/hooks/
```

#### Step 3: Configuration Setup

```bash
# Copy base configuration if it doesn't exist
if [ ! -f ~/.claude/hooks/config.json ]; then
    cp claude-hooks/config.template.json ~/.claude/hooks/config.json
fi

# Edit configuration file
nano ~/.claude/hooks/config.json
```

Add the following sections to your configuration:

```json
{
  "naturalTriggers": {
    "enabled": true,
    "triggerThreshold": 0.6,
    "cooldownPeriod": 30000,
    "maxMemoriesPerTrigger": 5
  },
  "performance": {
    "defaultProfile": "balanced",
    "enableMonitoring": true,
    "autoAdjust": true,
    "profiles": {
      "speed_focused": {
        "maxLatency": 100,
        "enabledTiers": ["instant"],
        "backgroundProcessing": false,
        "degradeThreshold": 200,
        "description": "Fastest response, minimal memory awareness"
      },
      "balanced": {
        "maxLatency": 200,
        "enabledTiers": ["instant", "fast"],
        "backgroundProcessing": true,
        "degradeThreshold": 400,
        "description": "Moderate latency, smart memory triggers"
      },
      "memory_aware": {
        "maxLatency": 500,
        "enabledTiers": ["instant", "fast", "intensive"],
        "backgroundProcessing": true,
        "degradeThreshold": 1000,
        "description": "Full memory awareness, accept higher latency"
      },
      "adaptive": {
        "autoAdjust": true,
        "degradeThreshold": 800,
        "backgroundProcessing": true,
        "description": "Auto-adjust based on performance and user preferences"
      }
    }
  }
}
```

#### Step 4: Set File Permissions

```bash
# Make hook files executable
chmod +x ~/.claude/hooks/core/*.js
chmod +x ~/.claude/hooks/memory-mode-controller.js
chmod +x ~/.claude/hooks/test-natural-triggers.js

# Set appropriate directory permissions
chmod 755 ~/.claude/hooks
chmod -R 644 ~/.claude/hooks/*.json
```

#### Step 5: Test Installation

```bash
# Run comprehensive test suite
cd ~/.claude/hooks
node test-natural-triggers.js

# Test CLI controller
node memory-mode-controller.js status

# Test specific components
node -e "
const { TieredConversationMonitor } = require('./utilities/tiered-conversation-monitor');
const monitor = new TieredConversationMonitor();
console.log('✅ TieredConversationMonitor loaded successfully');
"
```

## Installation Verification

### Test 1: System Components

```bash
# Verify all components are in place
ls ~/.claude/hooks/core/mid-conversation.js
ls ~/.claude/hooks/utilities/tiered-conversation-monitor.js
ls ~/.claude/hooks/utilities/performance-manager.js
ls ~/.claude/hooks/utilities/git-analyzer.js
ls ~/.claude/hooks/memory-mode-controller.js
```

### Test 2: Configuration Validation

```bash
# Check configuration syntax
cat ~/.claude/hooks/config.json | node -e "
try {
  const config = JSON.parse(require('fs').readFileSync(0, 'utf8'));
  console.log('✅ Configuration JSON is valid');
  console.log('Natural Triggers enabled:', config.naturalTriggers?.enabled);
  console.log('Default profile:', config.performance?.defaultProfile);
} catch (error) {
  console.error('❌ Configuration error:', error.message);
}
"
```

### Test 3: CLI Controller

```bash
# Test CLI management system
node ~/.claude/hooks/memory-mode-controller.js status
node ~/.claude/hooks/memory-mode-controller.js profiles
```

Expected output:
```
📊 Memory Hook Status
Current Profile: balanced
Description: Moderate latency, smart memory triggers
Natural Triggers: enabled
Sensitivity: 0.6
Performance: 0ms avg latency, 0 degradation events
```

### Test 4: Memory Service Integration

```bash
# Test memory service connectivity
node ~/.claude/hooks/memory-mode-controller.js test "What did we decide about authentication?"
```

Expected behavior:
- Should attempt to analyze the test query
- Should show tier processing (instant → fast → intensive)
- Should either retrieve relevant memories or show "no relevant memories found"
- Should complete without errors

## Post-Installation Configuration

### Performance Profile Selection

Choose the appropriate profile for your workflow:

```bash
# For quick coding sessions (minimal interruption)
node memory-mode-controller.js profile speed_focused

# For general development work (recommended)
node memory-mode-controller.js profile balanced

# For architecture and research work (maximum context)
node memory-mode-controller.js profile memory_aware

# For adaptive learning (system learns your preferences)
node memory-mode-controller.js profile adaptive
```

### Sensitivity Tuning

Adjust trigger sensitivity based on your preferences:

```bash
# More triggers (lower threshold)
node memory-mode-controller.js sensitivity 0.4

# Balanced triggers (recommended)
node memory-mode-controller.js sensitivity 0.6

# Fewer triggers (higher threshold)
node memory-mode-controller.js sensitivity 0.8
```

### Git Integration Setup

For enhanced Git-aware context, ensure your repository has:

- **Recent commit history** (Natural Memory Triggers analyzes last 14 days)
- **Readable CHANGELOG.md** (parsed for version context)
- **Proper git configuration** (for commit author and timestamps)

## Troubleshooting Installation Issues

### Issue 1: Node.js Not Found

**Error**: `node: command not found`

**Solution**:
```bash
# Install Node.js (version 14 or higher)
# macOS with Homebrew:
brew install node

# Ubuntu/Debian:
sudo apt update && sudo apt install nodejs npm

# Windows:
# Download from https://nodejs.org/

# Verify installation
node --version
npm --version
```

### Issue 2: Permission Errors

**Error**: `Permission denied` when running hooks

**Solution**:
```bash
# Fix file permissions
chmod +x ~/.claude/hooks/core/*.js
chmod +x ~/.claude/hooks/memory-mode-controller.js

# Fix directory permissions
chmod 755 ~/.claude/hooks
chmod -R 644 ~/.claude/hooks/*.json
```

### Issue 3: Memory Service Connection Failed

**Error**: `Network error` or `ENOTFOUND`

**Diagnosis**:
```bash
# Test memory service directly
curl -k https://localhost:8443/api/health

# Check configuration
cat ~/.claude/hooks/config.json | grep -A 5 "memoryService"
```

**Solutions**:
1. **Start Memory Service**: `uv run memory server`
2. **Check API Key**: Ensure valid API key in configuration
3. **Firewall Settings**: Verify port 8443 is accessible
4. **SSL Issues**: Self-signed certificates may need special handling

### Issue 4: Configuration Conflicts

**Error**: `Parse error: Expected property name or '}' in JSON`

**Solution**:
```bash
# Validate JSON syntax
cat ~/.claude/hooks/config.json | python -m json.tool

# If corrupted, restore from backup
cp ~/.claude/hooks/config.json.backup ~/.claude/hooks/config.json

# Or reset to defaults
node memory-mode-controller.js reset
```

### Issue 5: Claude Code Integration Issues

**Error**: Hooks not detected by Claude Code

**Diagnosis**:
```bash
# Check Claude Code settings
cat ~/.claude/settings.json | grep -A 10 "hooks"

# Verify hook files location
ls -la ~/.claude/hooks/core/
```

**Solutions**:
1. **Correct Location**: Ensure hooks are in `~/.claude/hooks/` not `~/.claude-code/hooks/`
2. **Settings Update**: Update `~/.claude/settings.json` with correct paths
3. **Restart Claude Code**: Some changes require restart
4. **Debug Mode**: Run `claude --debug hooks` to see hook loading messages

## Installation Verification Checklist

- [ ] All core components copied to `~/.claude/hooks/`
- [ ] Configuration file includes `naturalTriggers` and `performance` sections
- [ ] File permissions set correctly (executable hooks, readable configs)
- [ ] CLI controller responds to `status` command
- [ ] Test suite passes all 18 tests
- [ ] Memory service connectivity verified
- [ ] Performance profile selected and applied
- [ ] Git integration working (if applicable)
- [ ] Claude Code detects and loads hooks

## Next Steps

After successful installation:

1. **Read the User Guide**: Comprehensive usage instructions at [Natural Memory Triggers v7.1.0 Guide](https://github.com/doobidoo/mcp-memory-service/wiki/Natural-Memory-Triggers-v7.1.0)

2. **Try the System**: Ask Claude Code questions like:
   - "What approach did we use for authentication?"
   - "How did we handle error handling in this project?"
   - "What were the main architectural decisions we made?"

3. **Monitor Performance**: Check system metrics periodically:
   ```bash
   node memory-mode-controller.js metrics
   ```

4. **Customize Settings**: Adjust profiles and sensitivity based on your workflow:
   ```bash
   node memory-mode-controller.js profile memory_aware
   node memory-mode-controller.js sensitivity 0.7
   ```

5. **Provide Feedback**: The adaptive profile learns from your usage patterns, so use the system regularly for best results.

---

**Natural Memory Triggers v7.1.0** transforms Claude Code into an intelligent development assistant that automatically understands when you need context from your project history! 🚀
