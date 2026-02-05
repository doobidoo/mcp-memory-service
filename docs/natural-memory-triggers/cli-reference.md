# Natural Memory Triggers v7.1.0 - CLI Reference

Complete reference for the CLI management system that provides real-time configuration and monitoring of Natural Memory Triggers without requiring file edits or Claude Code restarts.

## Overview

The CLI controller (`memory-mode-controller.js`) is the primary interface for managing Natural Memory Triggers. It provides:

- ✅ **Real-time configuration** changes without restart
- ✅ **Performance monitoring** and metrics
- ✅ **Profile management** for different workflows
- ✅ **Sensitivity tuning** for trigger frequency
- ✅ **System diagnostics** and health checks

## Command Syntax

```bash
node ~/.claude/hooks/memory-mode-controller.js <command> [options] [arguments]
```

## Core Commands

### `status` - System Status and Information

Display current system status, configuration, and performance metrics.

```bash
node memory-mode-controller.js status
```

**Output Example:**
```
📊 Memory Hook Status
Current Profile: balanced
Description: Moderate latency, smart memory triggers
Natural Triggers: enabled
Sensitivity: 0.6
Cooldown Period: 30000ms
Max Memories per Trigger: 5
Performance: 145ms avg latency, 2 degradation events
Cache Size: 12 entries
Conversation History: 8 messages
```

**Options:**
- `--verbose` - Show detailed performance metrics
- `--json` - Output in JSON format for scripting

```bash
node memory-mode-controller.js status --verbose
node memory-mode-controller.js status --json
```

### `profiles` - List Available Performance Profiles

Display all available performance profiles with descriptions and configurations.

```bash
node memory-mode-controller.js profiles
```

**Output:**
```
📋 Available Performance Profiles

🏃 speed_focused
  Max Latency: 100ms
  Enabled Tiers: instant
  Description: Fastest response, minimal memory awareness

⚖️ balanced (current)
  Max Latency: 200ms
  Enabled Tiers: instant, fast
  Description: Moderate latency, smart memory triggers

🧠 memory_aware
  Max Latency: 500ms
  Enabled Tiers: instant, fast, intensive
  Description: Full memory awareness, accept higher latency

🤖 adaptive
  Max Latency: auto-adjusting
  Enabled Tiers: dynamic
  Description: Auto-adjust based on performance and user preferences
```

### `profile` - Switch Performance Profile

Change the active performance profile for different workflow requirements.

```bash
node memory-mode-controller.js profile <profile_name>
```

**Available Profiles:**

#### `speed_focused` - Maximum Speed
```bash
node memory-mode-controller.js profile speed_focused
```
- **Latency**: < 100ms
- **Tiers**: Instant only (pattern matching, cache checks)
- **Use Case**: Quick coding sessions, pair programming
- **Trade-off**: Minimal memory awareness for maximum speed

#### `balanced` - Recommended Default
```bash
node memory-mode-controller.js profile balanced
```
- **Latency**: < 200ms
- **Tiers**: Instant + Fast (semantic analysis)
- **Use Case**: General development work, most productive for daily use
- **Trade-off**: Good balance of speed and context awareness

#### `memory_aware` - Maximum Context
```bash
node memory-mode-controller.js profile memory_aware
```
- **Latency**: < 500ms
- **Tiers**: All tiers (deep semantic understanding)
- **Use Case**: Complex projects, architectural decisions, research
- **Trade-off**: Maximum context awareness, higher latency acceptable

#### `adaptive` - Machine Learning
```bash
node memory-mode-controller.js profile adaptive
```
- **Latency**: Auto-adjusting based on usage patterns
- **Tiers**: Dynamic selection based on user feedback
- **Use Case**: Users who want the system to learn automatically
- **Trade-off**: Requires learning period but becomes highly personalized

### `sensitivity` - Adjust Trigger Sensitivity

Control how often Natural Memory Triggers activate by adjusting the confidence threshold.

```bash
node memory-mode-controller.js sensitivity <value>
```

**Sensitivity Values:**
- `0.0` - Maximum triggers (activates on any potential memory-seeking pattern)
- `0.4` - High sensitivity (more triggers, useful for research/architecture work)
- `0.6` - Balanced (recommended default)
- `0.8` - Low sensitivity (fewer triggers, high-confidence only)
- `1.0` - Minimum triggers (only explicit memory requests)

**Examples:**
```bash
# More triggers for architecture work
node memory-mode-controller.js sensitivity 0.4

# Balanced triggers (recommended)
node memory-mode-controller.js sensitivity 0.6

# Fewer triggers for focused coding
node memory-mode-controller.js sensitivity 0.8
```

## System Management Commands

### `enable` - Enable Natural Memory Triggers

Activate the Natural Memory Triggers system.

```bash
node memory-mode-controller.js enable
```

**Output:**
```
✅ Natural Memory Triggers enabled
Current sensitivity: 0.6
Active profile: balanced
Ready to detect memory-seeking patterns
```

### `disable` - Disable Natural Memory Triggers

Temporarily disable the Natural Memory Triggers system without uninstalling.

```bash
node memory-mode-controller.js disable
```

**Output:**
```
⏸️ Natural Memory Triggers disabled
Manual memory commands still available
Use 'enable' to reactivate triggers
```

### `reset` - Reset to Default Settings

Reset all configuration to default values.

```bash
node memory-mode-controller.js reset
```

**What gets reset:**
- Performance profile → `balanced`
- Sensitivity → `0.6`
- Natural triggers → `enabled`
- Cooldown period → `30000ms`
- Max memories per trigger → `5`

**Confirmation prompt:**
```
⚠️ This will reset all Natural Memory Triggers settings to defaults.
Are you sure? (y/N): y
✅ Settings reset to defaults
```

**Options:**
- `--force` - Skip confirmation prompt

```bash
node memory-mode-controller.js reset --force
```

## Testing and Diagnostics

### `test` - Test Trigger Detection

Test the trigger detection system with a specific query to see how it would be processed.

```bash
node memory-mode-controller.js test "your test query"
```

**Example:**
```bash
node memory-mode-controller.js test "What did we decide about authentication?"
```

**Output:**
```
🧪 Testing Natural Memory Triggers

Query: "What did we decide about authentication?"
Processing tiers: instant → fast → intensive

Tier 1 (Instant): 42ms
  - Pattern match: ✅ "what...decide" detected
  - Cache check: ❌ No cached result
  - Confidence: 0.85

Tier 2 (Fast): 127ms
  - Key phrases: ["decide", "authentication"]
  - Topic shift: 0.2 (moderate)
  - Question pattern: ✅ Detected
  - Confidence: 0.78

Memory Query Generated:
  - Type: recent-development
  - Query: "authentication decision approach implementation"
  - Weight: 1.0

Result: Would trigger memory retrieval (confidence 0.85 > threshold 0.6)
```

### `metrics` - Performance Metrics

Display detailed performance metrics and system health information.

```bash
node memory-mode-controller.js metrics
```

**Output:**
```
📊 Natural Memory Triggers Performance Metrics

System Performance:
  - Active Profile: balanced
  - Average Latency: 145ms
  - Degradation Events: 2
  - User Tolerance: 0.7

Tier Performance:
  - Instant Tier: 47ms avg (120 calls)
  - Fast Tier: 142ms avg (89 calls)
  - Intensive Tier: 387ms avg (23 calls)

Trigger Statistics:
  - Total Triggers: 45
  - Success Rate: 89%
  - False Positives: 5%
  - User Satisfaction: 87%

Cache Performance:
  - Cache Size: 15 entries
  - Hit Rate: 34%
  - Average Hit Time: 3ms

Memory Service:
  - Connection Status: ✅ Connected
  - Average Response: 89ms
  - Error Rate: 0%
```

### `health` - System Health Check

Perform comprehensive health check of all system components.

```bash
node memory-mode-controller.js health
```

**Output:**
```
🏥 Natural Memory Triggers Health Check

Core Components:
  ✅ TieredConversationMonitor loaded
  ✅ PerformanceManager initialized
  ✅ GitAnalyzer functional
  ✅ MCP Client connected

Configuration:
  ✅ config.json syntax valid
  ✅ naturalTriggers section present
  ✅ performance profiles configured
  ✅ memory service endpoint accessible

Dependencies:
  ✅ Node.js version compatible (v18.17.0)
  ✅ Required packages available
  ✅ File permissions correct

Memory Service Integration:
  ✅ Connection established
  ✅ Authentication valid
  ✅ API responses normal
  ⚠️ High response latency (245ms)

Git Integration:
  ✅ Repository detected
  ✅ Recent commits available
  ✅ Changelog found
  ❌ Branch name unavailable

Recommendations:
  - Consider optimizing memory service for faster responses
  - Check git configuration for branch detection
```

## Advanced Commands

### `config` - Configuration Management

View and modify configuration settings directly through CLI.

```bash
# View current configuration
node memory-mode-controller.js config show

# Get specific setting
node memory-mode-controller.js config get naturalTriggers.triggerThreshold

# Set specific setting
node memory-mode-controller.js config set naturalTriggers.cooldownPeriod 45000
```

### `cache` - Cache Management

Manage the semantic analysis cache.

```bash
# View cache statistics
node memory-mode-controller.js cache stats

# Clear cache
node memory-mode-controller.js cache clear

# Show cache contents (debug)
node memory-mode-controller.js cache show
```

**Cache Stats Output:**
```
💾 Semantic Cache Statistics

Size: 18/50 entries
Memory Usage: 2.4KB
Hit Rate: 34% (89/260 requests)
Average Hit Time: 2.8ms
Last Cleanup: 15 minutes ago

Most Accessed Patterns:
  1. "what did we decide" (12 hits)
  2. "how did we implement" (8 hits)
  3. "similar to what we" (6 hits)
```

### `export` - Export Configuration and Metrics

Export system configuration and performance data for backup or analysis.

```bash
# Export configuration
node memory-mode-controller.js export config > my-config-backup.json

# Export metrics
node memory-mode-controller.js export metrics > performance-report.json

# Export full system state
node memory-mode-controller.js export all > system-state.json
```

### `import` - Import Configuration

Import previously exported configuration.

```bash
node memory-mode-controller.js import config my-config-backup.json
```

## Scripting and Automation

### JSON Output Mode

Most commands support `--json` flag for machine-readable output:

```bash
# Get status in JSON format
node memory-mode-controller.js status --json

# Example output:
{
  "profile": "balanced",
  "enabled": true,
  "sensitivity": 0.6,
  "performance": {
    "avgLatency": 145,
    "degradationEvents": 2
  },
  "cache": {
    "size": 12,
    "hitRate": 0.34
  }
}
```

### Batch Operations

Run multiple commands in sequence:

```bash
# Setup for architecture work
node memory-mode-controller.js profile memory_aware
node memory-mode-controller.js sensitivity 0.4

# Daily development setup
node memory-mode-controller.js profile balanced
node memory-mode-controller.js sensitivity 0.6

# Quick coding setup
node memory-mode-controller.js profile speed_focused
node memory-mode-controller.js sensitivity 0.8
```

### Environment Variables

Control CLI behavior with environment variables:

```bash
# Enable debug output
export CLAUDE_HOOKS_DEBUG=true
node memory-mode-controller.js status

# Disable colored output
export NO_COLOR=1
node memory-mode-controller.js status

# Set alternative config path
export CLAUDE_HOOKS_CONFIG=/path/to/config.json
node memory-mode-controller.js status
```

## Error Handling and Debugging

### Common Error Messages

#### `Configuration Error: Cannot read config file`
**Cause**: Missing or corrupted configuration file
**Solution**:
```bash
# Check if config exists
ls ~/.claude/hooks/config.json

# Validate JSON syntax
cat ~/.claude/hooks/config.json | node -e "console.log(JSON.parse(require('fs').readFileSync(0, 'utf8')))"

# Reset to defaults if corrupted
node memory-mode-controller.js reset --force
```

#### `Memory Service Connection Failed`
**Cause**: MCP Memory Service not running or unreachable
**Solution**:
```bash
# Check memory service status
curl -k https://localhost:8443/api/health

# Start memory service
uv run memory server

# Check configuration
node memory-mode-controller.js config get memoryService.endpoint
```

#### `Permission Denied`
**Cause**: Incorrect file permissions
**Solution**:
```bash
# Fix permissions
chmod +x ~/.claude/hooks/memory-mode-controller.js
chmod 644 ~/.claude/hooks/config.json
```

### Debug Mode

Enable verbose debugging:

```bash
export CLAUDE_HOOKS_DEBUG=true
node memory-mode-controller.js status
```

**Debug Output Example:**
```
[DEBUG] Loading configuration from ~/.claude/hooks/config.json
[DEBUG] Configuration loaded successfully
[DEBUG] Initializing TieredConversationMonitor
[DEBUG] PerformanceManager initialized with profile: balanced
[DEBUG] GitAnalyzer detecting repository context
[DEBUG] MCP Client connecting to https://localhost:8443
[DEBUG] Status command executed successfully
```

## Integration Examples

### Shell Aliases

Add to your `.bashrc` or `.zshrc`:

```bash
# Quick aliases for common operations
alias nmt-status='node ~/.claude/hooks/memory-mode-controller.js status'
alias nmt-balanced='node ~/.claude/hooks/memory-mode-controller.js profile balanced'
alias nmt-speed='node ~/.claude/hooks/memory-mode-controller.js profile speed_focused'
alias nmt-memory='node ~/.claude/hooks/memory-mode-controller.js profile memory_aware'
alias nmt-metrics='node ~/.claude/hooks/memory-mode-controller.js metrics'
```

### VS Code Integration

Create VS Code tasks (`.vscode/tasks.json`):

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "NMT: Check Status",
      "type": "shell",
      "command": "node ~/.claude/hooks/memory-mode-controller.js status",
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "NMT: Switch to Memory Aware",
      "type": "shell",
      "command": "node ~/.claude/hooks/memory-mode-controller.js profile memory_aware",
      "group": "build"
    }
  ]
}
```

### Automated Performance Monitoring

Monitor system performance with cron job:

```bash
# Add to crontab (crontab -e)
# Check metrics every hour and log to file
0 * * * * node ~/.claude/hooks/memory-mode-controller.js metrics --json >> ~/nmt-metrics.log 2>&1
```

---

The CLI controller provides complete control over Natural Memory Triggers v7.1.0, enabling real-time optimization of your intelligent memory awareness system! 🚀
