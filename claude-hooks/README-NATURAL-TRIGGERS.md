# Natural Memory Triggers for Claude Code

🧠 **Intelligent mid-conversation memory awareness with performance optimization**

## Overview

The Natural Memory Triggers system provides seamless, intelligent memory awareness during conversations that feels like Claude naturally "remembers" rather than executing explicit system hooks. It uses multi-tiered performance architecture to balance memory intelligence with responsiveness.

## Key Features

### 🎯 **Natural Language Pattern Detection**
- **Explicit Memory Requests**: "What did we decide about...?", "Remind me how we..."
- **Past Work References**: "Similar to what we did...", "Like we discussed before..."
- **Technical Discussions**: Architecture, security, database topics that benefit from context
- **Project Continuity**: "Continue with...", "Next step...", problem-solving patterns

### ⚡ **Performance-Aware Architecture**
- **Tiered Processing**: Instant (< 50ms), Fast (< 150ms), Intensive (< 500ms)
- **Smart Performance Profiles**: Speed-focused, Balanced, Memory-aware, Adaptive
- **Automatic Degradation**: Gracefully reduces complexity when performance budgets are exceeded
- **User-Configurable Trade-offs**: Full control over speed vs intelligence balance

### 🔄 **Adaptive Learning**
- **User Preference Learning**: Adapts to user tolerance for latency vs memory awareness
- **Pattern Confidence Adjustment**: Learns which patterns are most valuable to the user
- **Context-Aware Triggering**: Considers project context, conversation history, and topic shifts

## Quick Start

### Installation

The system is integrated into the existing Claude Code hooks. No additional installation required.

### Basic Usage

```bash
# Check current status
node claude-hooks/memory-mode-controller.js status

# Switch to balanced mode (recommended)
node claude-hooks/memory-mode-controller.js profile balanced

# Enable natural triggers
node claude-hooks/memory-mode-controller.js enable
```

### Performance Profiles

Choose the profile that best matches your preferences:

```bash
# Fastest response, minimal memory awareness (< 100ms)
node claude-hooks/memory-mode-controller.js profile speed_focused

# Moderate latency, smart triggers (< 200ms) - RECOMMENDED
node claude-hooks/memory-mode-controller.js profile balanced

# Full memory awareness, accept higher latency (< 500ms)
node claude-hooks/memory-mode-controller.js profile memory_aware

# Auto-adjust based on usage patterns
node claude-hooks/memory-mode-controller.js profile adaptive
```

## How It Works

### Trigger Detection

The system uses a three-tiered approach to detect when memory context would be helpful:

#### **Tier 1: Instant Detection (< 50ms)**
- Regex-based pattern matching for explicit memory requests
- Cache lookups for previously analyzed messages
- Simple keyword extraction for technical terms

#### **Tier 2: Fast Analysis (< 150ms)**
- Contextual analysis with project information
- Topic shift detection from conversation history
- Enhanced pattern matching with semantic context

#### **Tier 3: Intensive Analysis (< 500ms)**
- Deep semantic understanding (when available)
- Full conversation context analysis
- Complex pattern relationships

### Example Triggers

**Explicit Memory Requests** (High Confidence):
```
"What did we decide about the authentication approach?"
"Remind me how we handled user sessions"
"Remember when we discussed the database schema?"
```

**Past Work References** (Medium Confidence):
```
"Similar to what we implemented last time"
"Like we discussed in the previous meeting"
"The same approach we used for the API"
```

**Technical Discussions** (Context-Dependent):
```
"Let's design the authentication architecture"
"How should we handle database migrations?"
"What's our security strategy?"
```

## Configuration

### Basic Configuration

Edit `claude-hooks/config.json`:

```json
{
  "naturalTriggers": {
    "enabled": true,
    "sensitivity": 0.7,           // 0-1, higher = more sensitive
    "triggerThreshold": 0.6,      // 0-1, confidence needed to trigger
    "cooldownPeriod": 30000,      // Milliseconds between triggers
    "maxMemoriesPerTrigger": 5,   // Max memories to inject per trigger
    "adaptiveLearning": true      // Learn from user feedback
  }
}
```

### Performance Profiles

Customize performance profiles in the configuration:

```json
{
  "performance": {
    "defaultProfile": "balanced",
    "profiles": {
      "speed_focused": {
        "maxLatency": 100,
        "enabledTiers": ["instant"],
        "backgroundProcessing": false
      },
      "balanced": {
        "maxLatency": 200,
        "enabledTiers": ["instant", "fast"],
        "backgroundProcessing": true
      },
      "memory_aware": {
        "maxLatency": 500,
        "enabledTiers": ["instant", "fast", "intensive"],
        "backgroundProcessing": true
      }
    }
  }
}
```

## Command Line Interface

### Memory Mode Controller

```bash
# Get current status and configuration
node claude-hooks/memory-mode-controller.js status

# Switch performance profiles
node claude-hooks/memory-mode-controller.js profile <speed_focused|balanced|memory_aware|adaptive>

# Adjust sensitivity (0-1, higher = more triggers)
node claude-hooks/memory-mode-controller.js sensitivity 0.8

# Adjust trigger threshold (0-1, higher = need more confidence)
node claude-hooks/memory-mode-controller.js threshold 0.7

# Enable/disable natural triggers
node claude-hooks/memory-mode-controller.js enable
node claude-hooks/memory-mode-controller.js disable
node claude-hooks/memory-mode-controller.js toggle

# List all available profiles
node claude-hooks/memory-mode-controller.js list

# Reset to defaults
node claude-hooks/memory-mode-controller.js reset
```

## Testing

Run the comprehensive test suite:

```bash
# Full test suite
node claude-hooks/test-natural-triggers.js

# Test dual protocol functionality
node claude-hooks/test-dual-protocol-hook.js
```

The test suite covers:
- Performance management and timing
- Pattern detection accuracy
- Conversation monitoring
- Integration testing
- Performance profile behavior

## Performance Optimization

### Latency Targets

| Profile | Target Latency | Use Case |
|---------|---------------|----------|
| Speed Focused | < 100ms | Priority on responsiveness |
| Balanced | < 200ms | Good balance (recommended) |
| Memory Aware | < 500ms | Maximum memory intelligence |
| Adaptive | Variable | Learns user preferences |

### Performance Monitoring

The system automatically tracks:
- Hook execution latency
- Pattern detection accuracy
- User acceptance rates
- Memory query performance

### Optimization Tips

1. **Start with Balanced Mode**: Good default for most users
2. **Monitor Performance**: Check status regularly to see average latencies
3. **Adjust Sensitivity**: Lower sensitivity = fewer false positives
4. **Use Cooldown Period**: Prevents excessive triggering
5. **Enable Learning**: Let the system adapt to your preferences

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Mid-Conversation Hook                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │ Performance     │  │ Conversation     │  │ Pattern     │ │
│  │ Manager         │  │ Monitor          │  │ Detector    │ │
│  │                 │  │                  │  │             │ │
│  │ • Timing        │  │ • Topic Extract  │  │ • Regex     │ │
│  │ • Profiles      │  │ • Semantic Shift │  │ • Context   │ │
│  │ • Learning      │  │ • Caching        │  │ • Learning  │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Memory Client                          │
│            (Dual Protocol: HTTP + MCP)                     │
└─────────────────────────────────────────────────────────────┘
```

### Key Classes

- **`MidConversationHook`**: Main orchestrator for trigger detection and execution
- **`TieredConversationMonitor`**: Multi-tier conversation analysis with performance awareness
- **`AdaptivePatternDetector`**: Natural language pattern detection with learning
- **`PerformanceManager`**: Performance monitoring, budgeting, and profile management
- **`MemoryClient`**: Unified interface for HTTP and MCP memory operations

## Troubleshooting

### Common Issues

**Q: Triggers aren't firing when expected**
```bash
# Check if natural triggers are enabled
node claude-hooks/memory-mode-controller.js status

# Lower the trigger threshold
node claude-hooks/memory-mode-controller.js threshold 0.5

# Increase sensitivity
node claude-hooks/memory-mode-controller.js sensitivity 0.8
```

**Q: Performance is slower than expected**
```bash
# Switch to speed-focused mode
node claude-hooks/memory-mode-controller.js profile speed_focused

# Check current latency
node claude-hooks/memory-mode-controller.js status
```

**Q: Too many false positive triggers**
```bash
# Lower sensitivity
node claude-hooks/memory-mode-controller.js sensitivity 0.5

# Increase threshold
node claude-hooks/memory-mode-controller.js threshold 0.8

# Increase cooldown period (edit config.json)
```

### Debug Mode

Enable detailed logging:

```json
{
  "logging": {
    "level": "debug",
    "enableDebug": true,
    "logToFile": true
  }
}
```

### Performance Analysis

Monitor hook performance:

```bash
# Check status for performance metrics
node claude-hooks/memory-mode-controller.js status

# Run performance tests
node claude-hooks/test-natural-triggers.js
```

## Integration with Claude Code

### Session Start Integration

The natural triggers work alongside the existing session start hooks:

1. **Session Start**: Loads initial memory context (existing functionality)
2. **Mid-Conversation**: Intelligently refreshes context when patterns suggest it's needed
3. **Adaptive Learning**: Learns from user interactions to improve trigger accuracy

### Memory Storage Integration

Uses the existing dual-protocol memory service:
- **HTTP Protocol**: Web-based memory service (https://localhost:8443)
- **MCP Protocol**: Direct server process communication
- **Smart Fallback**: Automatically switches protocols if one fails

## Roadmap

### Planned Enhancements

1. **Advanced Semantic Analysis**: Integration with more sophisticated NLP models
2. **Cross-Session Learning**: Remember user preferences across Claude Code sessions
3. **Project-Specific Patterns**: Learn patterns specific to different projects
4. **Real-time Performance Tuning**: Dynamic adjustment based on system resources
5. **Visual Performance Dashboard**: Web-based interface for monitoring and configuration

### Contributing

The natural triggers system is designed to be extensible:

1. **Custom Pattern Categories**: Add new pattern types in `AdaptivePatternDetector`
2. **Performance Profiles**: Define custom profiles in the configuration
3. **Integration Points**: Hook into additional Claude Code events
4. **Learning Algorithms**: Enhance the adaptive learning mechanisms

## License

This system is part of the MCP Memory Service project and follows the same licensing terms.

---

🧠 **The goal is to make memory awareness feel natural and seamless, like Claude simply "remembers" your conversations and project context.**