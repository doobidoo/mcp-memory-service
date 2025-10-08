# Memory Hooks Configuration Guide

## Overview

This guide documents all configuration properties for the Claude Code memory awareness hooks, with detailed explanations of their behavior and impact on memory retrieval.

## Configuration Structure

The hooks are configured via `config.json` in the hooks directory. Configuration follows this hierarchy:

1. **Memory Service** - Connection and protocol settings
2. **Project Detection** - How projects are identified
3. **Memory Scoring** - How memories are ranked for relevance
4. **Git Analysis** - Repository context integration
5. **Time Windows** - Temporal scoping for queries
6. **Output** - Display and logging options

---

## Memory Service Connection Configuration

### `memoryService` Object

Controls how the hooks connect to the MCP Memory Service.

```json
"memoryService": {
  "protocol": "auto",
  "preferredProtocol": "http",
  "fallbackEnabled": true,
  "http": {
    "endpoint": "http://127.0.0.1:8889",
    "apiKey": "YOUR_API_KEY_HERE",
    "healthCheckTimeout": 3000,
    "useDetailedHealthCheck": true
  },
  "mcp": {
    "serverCommand": ["uv", "run", "memory", "server", "-s", "hybrid"],
    "serverWorkingDir": "../",
    "connectionTimeout": 2000,
    "toolCallTimeout": 3000
  }
}
```

#### HTTP Configuration

**`endpoint`** (String): URL of the HTTP memory service.

**Security Considerations:**
- **HTTP (`http://`)**: Default for local development. Traffic is **unencrypted** - only use for localhost connections.
- **HTTPS (`https://`)**: Recommended if connecting to remote servers or when encryption-in-transit is required.
  - For self-signed certificates, your system must trust the certificate authority.
  - The hooks enforce certificate validation - `rejectUnauthorized` is always enabled for security.

**`apiKey`** (String): API key for authenticating with the memory service.
- **Default**: Empty string `""` - the application will validate and prompt for a valid key on startup
- **Best practice**: Set via environment variable or secure configuration file
- **Security**: Never commit actual API keys to version control

#### MCP Configuration

**`serverCommand`** (Array): Command to launch the MCP memory service locally.
- Example: `["uv", "run", "memory", "server", "-s", "hybrid"]`
- Adjust storage backend flag (`-s`) as needed: `hybrid`, `cloudflare`, `sqlite_vec`, `chromadb`

**`serverWorkingDir`** (String): Working directory for the MCP server process.
- **Relative paths**: `"../"` assumes hooks are in a subdirectory (e.g., `project/claude-hooks/`)
- **Absolute paths**: Use full path for explicit configuration
- **Environment variables**: Consider using `process.env.MCP_MEMORY_PROJECT_ROOT` for flexibility

**Directory Structure Assumption (for `../` relative path):**
```
project-root/
├── src/                    # MCP Memory Service code
├── claude-hooks/           # This hooks directory
│   ├── config.json
│   └── utilities/
└── pyproject.toml
```

If your structure differs, update `serverWorkingDir` accordingly or use an absolute path.

**`connectionTimeout`** (Number): Milliseconds to wait for MCP server connection (default: 2000).

**`toolCallTimeout`** (Number): Milliseconds to wait for MCP tool call responses (default: 3000).

---

## Memory Scoring Configuration

### `memoryScoring` Object

Controls how memories are scored and ranked for relevance to the current session.

#### `weights` (Object)

Relative importance of different scoring factors. These weights are applied to individual component scores (0.0-1.0 each), then summed together with additive bonuses (typeBonus, recencyBonus). The final score is clamped to [0, 1].

**Note**: Weights don't need to sum to exactly 1.0 since additional bonuses are added separately and the final score is normalized by clamping. The weights shown below sum to 1.00 for the base scoring (without conversation context) or 1.25 when conversation context is enabled.

```json
"weights": {
  "timeDecay": 0.40,           // Recency weight (default: 0.40)
  "tagRelevance": 0.25,        // Tag matching weight (default: 0.25)
  "contentRelevance": 0.15,    // Content keyword weight (default: 0.15)
  "contentQuality": 0.20,      // Quality assessment weight (default: 0.20)
  "conversationRelevance": 0.25 // Conversation context weight (default: 0.25, only when enabled)
}
```

**Property Details:**

- **`timeDecay`** (0.0-1.0, recommended: 0.35-0.45)
  - Weight given to memory age in scoring
  - Higher values prioritize recent memories
  - Lower values allow older, high-quality memories to rank higher
  - **Impact**: At 0.40, a 7-day-old memory with perfect tags can outscore a 60-day-old memory with perfect tags and high quality
  - **Recommendation**: Set to 0.40-0.45 for active development, 0.25-0.35 for research/reference work

- **`tagRelevance`** (0.0-1.0, recommended: 0.20-0.30)
  - Weight given to tag matching with project context
  - Higher values favor well-tagged memories
  - **Impact**: Tags like `projectName`, `language`, `framework` significantly boost scores
  - **Trade-off**: High tag weight can cause old, well-documented memories to dominate over recent work
  - **Recommendation**: Set to 0.25 for balanced tag importance, 0.20 if recency is critical

- **`contentRelevance`** (0.0-1.0, recommended: 0.10-0.20)
  - Weight for keyword matching in memory content
  - Matches against project name, language, frameworks, technical terms
  - **Impact**: Memories mentioning project-specific terms rank higher
  - **Recommendation**: Keep at 0.15 unless doing very keyword-focused work

- **`contentQuality`** (0.0-1.0, recommended: 0.15-0.25)
  - Weight for assessed content quality (length, diversity, meaningful indicators)
  - Penalizes generic session summaries
  - **Impact**: Filters out low-quality auto-generated content
  - **Quality Indicators**: "decided", "implemented", "fixed", "because", "approach", "solution"
  - **Recommendation**: Set to 0.20 to balance quality with other factors

- **`conversationRelevance`** (0.0-1.0, recommended: 0.20-0.30)
  - Weight for matching current conversation topics and intent
  - Only active when conversation context is available
  - **Impact**: Dynamically adjusts based on what user is discussing
  - **Recommendation**: Keep at 0.25 for adaptive context awareness

#### `minRelevanceScore` (Number)

Minimum score threshold for a memory to be included in context.

```json
"minRelevanceScore": 0.4  // Default: 0.4
```

**Details:**
- Range: 0.0 to 1.0
- Memories below this threshold are filtered out entirely
- **Impact on Quality**:
  - `0.3`: Permissive, may include generic old content
  - `0.4`: Balanced, filters most low-quality memories (recommended)
  - `0.5`: Strict, only high-relevance memories
- **Trade-off**: Higher threshold = fewer but higher quality memories

#### `timeDecayRate` (Number)

Rate of exponential decay for time-based scoring.

```json
"timeDecayRate": 0.05  // Default: 0.05
```

**Formula**: `score = e^(-rate * days)`

**Details:**
- Range: 0.01 to 0.2 (practical range)
- Lower rate = gentler decay (memories age slower)
- Higher rate = aggressive decay (memories age faster)

**Decay Examples**:

| Days Old | Rate 0.05 | Rate 0.10 | Rate 0.15 |
|----------|-----------|-----------|-----------|
| 7 days   | 0.70      | 0.50      | 0.35      |
| 14 days  | 0.50      | 0.25      | 0.12      |
| 30 days  | 0.22      | 0.05      | 0.01      |
| 60 days  | 0.05      | 0.002     | ~0        |

**Recommendation**:
- `0.05`: Balanced, keeps 2-4 week memories relevant (recommended)
- `0.10`: Aggressive, prioritizes last 1-2 weeks only
- `0.03`: Gentle, treats 1-2 month memories as still valuable

#### `enableConversationContext` (Boolean)

Whether to use conversation analysis for dynamic memory scoring.

```json
"enableConversationContext": true  // Default: true
```

---

## Git Analysis Configuration

### `gitAnalysis` Object

Controls how git repository context influences memory retrieval.

```json
"gitAnalysis": {
  "enabled": true,
  "commitLookback": 14,
  "maxCommits": 20,
  "includeChangelog": true,
  "maxGitMemories": 3,
  "gitContextWeight": 1.8
}
```

#### `gitContextWeight` (Number)

Multiplier applied to memories derived from git context queries.

**Details:**
- Range: 1.0 to 2.5 (practical range)
- Applied multiplicatively to base memory score
- **Impact Examples**:
  - Base score 0.5 × weight 1.2 = final 0.6
  - Base score 0.5 × weight 1.8 = final 0.9

**Behavior by Value**:
- `1.0`: No boost (git context treated equally)
- `1.2`: Small boost (git-aware memories slightly favored)
- `1.8`: Strong boost (git-aware memories highly prioritized) ✅ **Recommended**
- `2.0+`: Very strong boost (git context dominates)

**Use Cases**:
- **Active development** (`1.8`): Prioritize memories matching recent commits/keywords
- **Maintenance work** (`1.2-1.5`): Balance git context with other signals
- **Research/planning** (`1.0`): Disable git preference

#### Other Git Properties

- **`commitLookback`** (Number, default: 14): Days of git history to analyze
- **`maxCommits`** (Number, default: 20): Maximum commits to process
- **`includeChangelog`** (Boolean, default: true): Parse CHANGELOG.md for context
- **`maxGitMemories`** (Number, default: 3): Max memories from git-context phase

---

## Time Windows Configuration

### Memory Service Time Windows

Controls temporal scoping for memory queries.

```json
"memoryService": {
  "recentTimeWindow": "last-month",      // Default: "last-month"
  "fallbackTimeWindow": "last-3-months"  // Default: "last-3-months"
}
```

#### `recentTimeWindow` (String)

Time window for Phase 1 recent memory queries.

**Supported Values:**
- `"last-day"`: Last 24 hours
- `"last-week"`: Last 7 days
- `"last-2-weeks"`: Last 14 days
- `"last-month"`: Last 30 days ✅ **Recommended**
- `"last-3-months"`: Last 90 days

**Impact:**
- **Narrow window** (`last-week`): Only very recent memories, may miss context during development gaps
- **Balanced window** (`last-month`): Captures recent sprint/iteration cycle
- **Wide window** (`last-3-months`): Includes seasonal patterns, may dilute recency focus

**Recommendation**:
- Active development: `"last-month"`
- Periodic/seasonal work: `"last-3-months"`

#### `fallbackTimeWindow` (String)

Time window for fallback queries when recent memories are insufficient.

**Supported Values:** Same as `recentTimeWindow`

**Purpose:** Ensures minimum context when recent work is sparse.

**Recommendation**: Set 2-3× wider than recent window (e.g., `last-month` → `last-3-months`)

---

## Recency Bonus System

### Automatic Recency Bonuses

The memory scorer applies explicit additive bonuses based on memory age (implemented in `memory-scorer.js`):

```javascript
// Automatic bonuses (no configuration needed)
< 7 days:  +0.15 bonus  // Strong boost for last week
< 14 days: +0.10 bonus  // Moderate boost for last 2 weeks
< 30 days: +0.05 bonus  // Small boost for last month
> 30 days: 0 bonus      // No bonus for older memories
```

**How It Works:**
- Applied **additively** (not multiplicatively) to final score
- Ensures very recent memories get absolute advantage
- Creates clear tier separation (weekly/biweekly/monthly)

**Example Impact:**
```
Memory A (5 days old):
  Base score: 0.50
  Recency bonus: +0.15
  Final score: 0.65

Memory B (60 days old):
  Base score: 0.60 (higher quality/tags)
  Recency bonus: 0
  Final score: 0.60

Result: Recent memory wins despite lower base score
```

**Design Rationale:**
- Compensates for aggressive time decay
- Prevents old, well-tagged memories from dominating
- Aligns with user expectation that recent work is most relevant

---

## Complete Configuration Example

### Optimized for Active Development (Recommended)

```json
{
  "memoryService": {
    "maxMemoriesPerSession": 8,
    "recentFirstMode": true,
    "recentMemoryRatio": 0.6,
    "recentTimeWindow": "last-month",
    "fallbackTimeWindow": "last-3-months"
  },
  "memoryScoring": {
    "weights": {
      "timeDecay": 0.40,
      "tagRelevance": 0.25,
      "contentRelevance": 0.15,
      "contentQuality": 0.20,
      "conversationRelevance": 0.25
    },
    "minRelevanceScore": 0.4,
    "timeDecayRate": 0.05,
    "enableConversationContext": true
  },
  "gitAnalysis": {
    "enabled": true,
    "commitLookback": 14,
    "maxCommits": 20,
    "includeChangelog": true,
    "maxGitMemories": 3,
    "gitContextWeight": 1.8
  }
}
```

### Optimized for Research/Reference Work

```json
{
  "memoryService": {
    "recentTimeWindow": "last-month",
    "fallbackTimeWindow": "last-3-months"
  },
  "memoryScoring": {
    "weights": {
      "timeDecay": 0.25,
      "tagRelevance": 0.35,
      "contentRelevance": 0.20,
      "contentQuality": 0.30,
      "conversationRelevance": 0.20
    },
    "minRelevanceScore": 0.3,
    "timeDecayRate": 0.03
  },
  "gitAnalysis": {
    "gitContextWeight": 1.0
  }
}
```

---

## Tuning Guide

### Problem: Recent work not appearing in context

**Symptoms:**
- Old documentation/decisions dominate
- Recent bug fixes/features missing
- Context feels outdated

**Solutions:**
1. Increase `timeDecay` weight: `0.40` → `0.45`
2. Increase `gitContextWeight`: `1.8` → `2.0`
3. Widen `recentTimeWindow`: `"last-week"` → `"last-month"`
4. Reduce `tagRelevance` weight: `0.25` → `0.20`

### Problem: Too many low-quality memories

**Symptoms:**
- Generic session summaries in context
- Duplicate or trivial information
- Context feels noisy

**Solutions:**
1. Increase `minRelevanceScore`: `0.4` → `0.5`
2. Increase `contentQuality` weight: `0.20` → `0.25`
3. Reduce `maxMemoriesPerSession`: `8` → `5`

### Problem: Missing important old architectural decisions

**Symptoms:**
- Lose context of foundational decisions
- Architectural rationale missing
- Only seeing recent tactical work

**Solutions:**
1. Reduce `timeDecay` weight: `0.40` → `0.30`
2. Increase `tagRelevance` weight: `0.25` → `0.30`
3. Gentler `timeDecayRate`: `0.05` → `0.03`
4. Tag important decisions with `"architecture"`, `"decision"` tags

### Problem: Git context overwhelming other signals

**Symptoms:**
- Only git-keyword memories showing up
- Missing memories that don't match commit messages
- Over-focused on recent commits

**Solutions:**
1. Reduce `gitContextWeight`: `1.8` → `1.4`
2. Reduce `maxGitMemories`: `3` → `2`
3. Disable git analysis temporarily: `"enabled": false`

---

## Migration from Previous Versions

### v1.0 → v2.0 (Recency Optimization)

**Breaking Changes:**
- `timeDecay` weight increased from `0.25` to `0.40`
- `tagRelevance` weight decreased from `0.35` to `0.25`
- `timeDecayRate` decreased from `0.10` to `0.05`
- `minRelevanceScore` increased from `0.3` to `0.4`
- `gitContextWeight` increased from `1.2` to `1.8`

**Impact:** Recent memories (< 30 days) will rank significantly higher. Adjust weights if you need more historical context.

**Migration Steps:**
1. Backup current `config.json`
2. Update weights to new defaults
3. Test with `test-recency-scoring.js`
4. Fine-tune based on your workflow

---

## Advanced: Scoring Algorithm Details

### Final Score Calculation

```javascript
// Step 1: Calculate base score (weighted sum of components + bonuses)
let baseScore =
  (timeDecayScore * timeDecayWeight) +
  (tagRelevanceScore * tagRelevanceWeight) +
  (contentRelevanceScore * contentRelevanceWeight) +
  (contentQualityScore * contentQualityWeight) +
  typeBonus +
  recencyBonus

// Step 2: Add conversation context if enabled (additive)
if (conversationContextEnabled) {
  baseScore += (conversationRelevanceScore * conversationRelevanceWeight)
}

// Step 3: Apply git context boost (multiplicative - boosts ALL components)
// Note: This multiplies the entire score including conversation relevance
// Implementation: Applied in session-start.js after scoring, not in memory-scorer.js
if (isGitContextMemory) {
  baseScore *= gitContextWeight
}

// Step 4: Apply quality penalty for very low quality (multiplicative)
if (contentQualityScore < 0.2) {
  baseScore *= 0.5
}

// Step 5: Normalize to [0, 1]
finalScore = clamp(baseScore, 0, 1)
```

### Score Component Ranges

- **Time Decay**: 0.01 - 1.0 (exponential decay based on age)
- **Tag Relevance**: 0.1 - 1.0 (0.3 default if no tags)
- **Content Relevance**: 0.1 - 1.0 (0.3 default if no keywords)
- **Content Quality**: 0.05 - 1.0 (0.3 default for normal content)
- **Type Bonus**: -0.1 - 0.3 (based on memory type)
- **Recency Bonus**: 0 - 0.15 (tiered based on age)

### Type Bonuses

```javascript
{
  'decision': 0.3,      // Architectural decisions
  'architecture': 0.3,  // Architecture docs
  'reference': 0.2,     // Reference materials
  'session': 0.15,      // Session summaries
  'insight': 0.2,       // Insights
  'bug-fix': 0.15,      // Bug fixes
  'feature': 0.1,       // Feature descriptions
  'note': 0.05,         // General notes
  'temporary': -0.1     // Temporary notes (penalized)
}
```

---

## Testing Configuration Changes

Use the included test script to validate your configuration:

```bash
cd /path/to/claude-hooks
node test-recency-scoring.js
```

This will show:
- Time decay calculations for different ages
- Recency bonus application
- Final scoring with your config weights
- Ranking of test memories

Expected output should show recent memories (< 7 days) in top 3 positions.

---

## See Also

- [README.md](./README.md) - General hooks documentation
- [MIGRATION.md](./MIGRATION.md) - Migration guides
- [README-NATURAL-TRIGGERS.md](./README-NATURAL-TRIGGERS.md) - Natural triggers documentation
