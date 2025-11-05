# TOON Format PoC Results

**Branch**: `feature/toon-format-poc`
**Date**: November 5, 2025
**Status**: ❌ **FAILED - Do Not Merge**
**Conclusion**: TOON format provides **WORSE token efficiency** than existing markdown format

---

## Executive Summary

We implemented a proof of concept to evaluate TOON (Token-Oriented Object Notation) format for session hook memory injection. The hypothesis was that TOON would reduce token usage by 30-60% compared to our current markdown format.

**Result**: TOON format uses **38% MORE tokens** than markdown (602 vs 436 tokens for 8 memories).

**Recommendation**: **Abandon PoC** and **keep existing markdown format**. Do not merge to main branch.

---

## Implementation Details

### What Was Built

1. **TOON Formatter Utility** (`~/.claude/hooks/utilities/toon-formatter.js`)
   - Implemented `formatMemoriesAsTOON()` function
   - Uses `@toon-format/toon` NPM package (v0.7.3)
   - Includes error handling and fallback support

2. **Session Hook Integration** (`~/.claude/hooks/core/session-start.js`)
   - Added format detection logic
   - Implemented try-catch with automatic fallback to markdown
   - Added console logging for format selection

3. **Configuration** (`~/.claude/hooks/config.json`)
   - Added `format: "toon"` option
   - Added `toonConfig` section with customization options

4. **Dependencies**
   - NPM: `@toon-format/toon@0.7.3`
   - Python: `toon-format>=0.1.0` (added to pyproject.toml)

### Testing Performed

- ✅ Unit tests for TOON formatter (all passed)
- ✅ Integration test with session-start hook (successful)
- ✅ Token measurement comparison (revealed the problem)
- ✅ Live hook execution test (worked correctly, but verbose)

---

## Token Measurement Results

### Test Data
- **8 memories** with realistic content from mcp-memory-service
- **Git context**: 20 commits, 3 changelog entries, 5 keywords
- **Storage info**: sqlite-vec backend, 2224 memories, 8.78MB

### Measurements

| Format | Characters | Tokens (est) | vs Baseline |
|--------|-----------|--------------|-------------|
| **Markdown (Current)** | 1,741 | 436 | Baseline |
| **TOON (PoC)** | 2,406 | 602 | +38% ❌ |

**Token Estimation**: ~4 characters per token (standard approximation)

### Why TOON Failed

The `@toon-format/toon` library outputs **YAML-style format** with indentation and repeated field names:

```toon
memories[8]:
  - content: "Critical analytics fix..."
    tags[4]: dashboard,fix,v8.17.1,analytics
    created_at_iso: "2025-11-01T08:45:00Z"
    relevance: 0.95
    type: implementation
    age_days: 4
  - content: "Memory hook retrieves git context..."
    ...
```

**Problems:**
1. ❌ **Field names repeated** for EVERY memory (content, tags, created_at_iso, relevance, type, age_days)
2. ❌ **Indentation overhead** (2-4 spaces per line)
3. ❌ **YAML-style bullets** (`-`) for each memory
4. ❌ **Tag array notation** (`tags[4]:`) adds extra characters

### Expected TOON Format (Spec)

According to TOON specification, the format should be **tabular**:

```toon
memories[8]{content,tags,created_at_iso,relevance,type,age_days}:
  "Critical...",["dashboard","fix","v8.17.1"],2025-11-01T08:45:00Z,0.95,"implementation",4
  "Memory...",["hooks","git","context"],2025-10-31T20:15:00Z,0.87,"feature",4
  ...
```

**This format would be more compact**, but the `@toon-format/toon@0.7.3` library does NOT implement this tabular format. It uses YAML-style instead.

---

## Markdown Format Advantages

Our current markdown format is **already highly optimized**:

```markdown
**Loaded 8 relevant memories from your project history:**

1. Memory hook retrieves git context... (Oct 31)
   Tags: hooks, git, context

2. Hybrid backend provides SQLite speed... (Oct 30)
   Tags: hybrid, backend, architecture
```

**Optimization Features:**
- ✅ **Adaptive truncation**: 300-800 chars based on memory count
- ✅ **Standardized dates**: "Oct 31" instead of full ISO timestamps
- ✅ **Categorization**: Groups memories by type (Recent Work, Problems, Decisions)
- ✅ **Minimal formatting**: Only essential structure
- ✅ **Human-readable**: Easy to debug and understand

**Token Efficiency:**
- Content-first approach (important text not buried in metadata)
- Field names NOT repeated for every memory
- No indentation overhead
- Compact date representation

---

## Root Cause Analysis

### Why the PoC Failed

1. **Library Implementation Gap**: `@toon-format/toon` v0.7.3 implements YAML-style format, not the compact tabular format described in TOON specification

2. **Specification vs Implementation**: The TOON spec promises 30-60% token reduction, but the library implementation doesn't deliver this

3. **Use Case Mismatch**: TOON is designed for **uniform arrays of objects**, but our memory data has **variable-length content** which breaks the tabular format

4. **Already Optimized Baseline**: Our markdown format is highly optimized (adaptive truncation, date standardization), making it a tough baseline to beat

### Potential Alternative Approaches

If we wanted to pursue token optimization further (NOT RECOMMENDED), we could:

1. **Manual TOON Implementation**: Write custom formatter that outputs true tabular format
   - Complexity: High
   - Risk: Parsing errors, maintenance burden
   - Benefit: Unclear (may still not beat markdown)

2. **CSV-like Format**: Ultra-compact representation
   ```csv
   content,tags,date,score,type
   "...",["..."],"Oct 31",0.87,"feature"
   ```
   - Complexity: Medium
   - Risk: Less readable for debugging
   - Benefit: Potentially 20-30% reduction

3. **Compressed JSON**: Use JSON with minimal field names
   ```json
   {"m":[{"c":"...","t":["..."],"d":"2025-10-31","s":0.87,"y":"feature"}]}
   ```
   - Complexity: Low
   - Risk: Claude parsing may struggle
   - Benefit: Moderate reduction (15-25%)

**However**: All alternatives sacrifice readability and debugging ease for minimal token savings. **NOT WORTH IT.**

---

## Lessons Learned

### ✅ What Went Well

1. **Clean Implementation**: Code is well-structured with proper error handling
2. **Testing Approach**: Comprehensive tests caught the problem before production
3. **Branch Strategy**: Feature branch isolated PoC from production
4. **Amp Bridge Usage**: Delegating implementation to Amp was efficient

### ❌ What Went Wrong

1. **Insufficient Research**: Should have verified library implementation matches spec
2. **Assumption Validation**: Assumed TOON library would deliver promised token savings
3. **Benchmark Baseline**: Didn't account for how optimized our markdown format already is

### 📚 Takeaways

1. **Library != Specification**: Always verify implementation matches spec claims
2. **Measure Before Implementing**: Should have tested library output before full integration
3. **Optimization Has Limits**: Our markdown format is already near-optimal for this use case
4. **Readability Matters**: Token savings aren't worth sacrificing debugability

---

## Files Created/Modified

### Created Files
- `~/.claude/hooks/utilities/toon-formatter.js` (274 lines)
- `/Users/hkr/Documents/GitHub/mcp-memory-service/scripts/testing/test-toon-formatter.js` (185 lines)
- `/Users/hkr/Documents/GitHub/mcp-memory-service/scripts/testing/compare-formats.js` (158 lines)
- `/Users/hkr/Documents/GitHub/mcp-memory-service/docs/analysis/toon-format-evaluation.md` (745 lines)
- `/Users/hkr/Documents/GitHub/mcp-memory-service/docs/poc/toon-format-results.md` (this file)

### Modified Files
- `~/.claude/hooks/config.json` (added format and toonConfig)
- `~/.claude/hooks/core/session-start.js` (added TOON integration with fallback)
- `/Users/hkr/Documents/GitHub/mcp-memory-service/pyproject.toml` (added toon-format dependency)

### Dependencies Added
- NPM: `@toon-format/toon@0.7.3`
- Python: `toon-format>=0.1.0`

---

## Rollback Plan

### Option 1: Keep Branch for Reference (RECOMMENDED)

```bash
# Revert config to markdown format
# In ~/.claude/hooks/config.json, change:
"format": "markdown"  # was "toon"

# Branch remains available for future reference
git checkout main
# Do NOT merge feature/toon-format-poc
```

**Benefits:**
- PoC code available for reference
- Documentation preserved
- Zero impact on production

### Option 2: Delete Branch Completely

```bash
git checkout main
git branch -D feature/toon-format-poc
git push origin --delete feature/toon-format-poc
```

**Benefits:**
- Clean repository history
- No confusion about PoC status

### Option 3: Cherry-pick Documentation Only

```bash
git checkout main
git cherry-pick <commit-hash-for-docs-only>
# Keep analysis and results docs, discard implementation
```

**Benefits:**
- Preserve lessons learned
- No implementation code in main

---

## Recommendation

### ❌ **DO NOT MERGE** to main branch

**Reasons:**
1. TOON format uses **38% MORE tokens** than markdown
2. No performance benefit whatsoever
3. Adds unnecessary complexity
4. Reduces debugability

### ✅ **KEEP** existing markdown format

**Reasons:**
1. Already highly optimized (436 tokens for 8 memories)
2. Human-readable for debugging
3. Adaptive truncation works well
4. Proven stable in production

### 📚 **ARCHIVE** PoC branch for reference

Keep `feature/toon-format-poc` branch available but don't merge:
- Documentation shows what was tried and why it failed
- Code available if future TOON library improvements make it viable
- Lessons learned documented for future optimization attempts

---

## Alternative: Focus on Other Optimizations

Instead of format changes, consider these proven optimizations:

1. **Reduce Memory Count**: Drop from 8 to 6 memories (save ~100 tokens)
2. **Shorter Content Truncation**: 300 → 250 chars when 5+ memories (save ~50 tokens)
3. **Remove Git Context**: Optional section, saves ~30 tokens when disabled
4. **Remove Storage Info**: Optional section, saves ~20 tokens when disabled

**Total Potential Savings**: ~200 tokens (45% reduction) with **zero readability loss**.

---

## Conclusion

TOON format PoC **failed to deliver promised token savings** and actually **increased** token usage by 38%. The `@toon-format/toon` library implementation does not match the specification's compact tabular format, instead using verbose YAML-style output.

**Final Decision**: **Abandon PoC**, **do not merge**, **keep existing markdown format**.

Our current markdown format is already well-optimized for this use case. Future optimization efforts should focus on:
- Reducing memory count
- Shorter truncation lengths
- Optional sections (git context, storage info)

Rather than changing the fundamental format representation.

---

**Branch Status**: Keep for reference, do not merge
**Configuration**: Reverted to `"format": "markdown"` in config.json
**Impact**: Zero (PoC isolated in feature branch)
**Next Steps**: Document lessons learned, move on to other priorities

---

**Generated**: November 5, 2025
**Reviewed By**: Claude Code
**Decision**: ❌ **REJECT - Do Not Merge**
