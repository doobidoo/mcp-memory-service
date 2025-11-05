# GitHub Discussion Draft - TOON Format Repository

**Repository**: https://github.com/toon-format/toon
**Discussion Type**: Questions & Feedback
**Status**: Ready to post

---

## Suggested Title

**Library vs Spec Implementation: 38% Token Discrepancy & Real-world Performance Comparison**

---

## Discussion Content

```markdown
# TOON Format Library vs Specification: Token Efficiency Analysis

Hi TOON team! 👋

I recently conducted a comprehensive evaluation of TOON format for optimizing memory context injection in Claude Code session hooks. I wanted to share my findings as they revealed some interesting discrepancies between the library implementation and the specification, and might provide valuable feedback for the project.

## Use Case Context

**Project**: MCP Memory Service - Claude Code integration
**Goal**: Optimize token usage when injecting 8 recent memories into Claude Code sessions
**Baseline**: Already-optimized markdown format (436 tokens)

Our session hooks inject project-relevant memories at the start of each Claude Code session. With context windows being precious, we're always looking for ways to reduce token consumption without sacrificing readability.

## Test Results Summary

I tested three formats with identical data (8 memories with realistic content):

| Format | Characters | Tokens (est.) | vs Baseline |
|--------|------------|---------------|-------------|
| **Markdown** (our baseline) | 1,741 | **436** | - |
| **Manual TOON** (spec-compliant tabular) | 1,721 | **431** | **-1.1%** ✅ |
| **Library TOON** (`@toon-format/toon` v0.7.3) | 2,406 | **602** | **+38%** ❌ |

## Key Findings

### 1. Library Output vs Specification

The `@toon-format/toon` library produces **YAML-style output** rather than the **tabular format** described in the specification:

**Library Output (YAML-style)**:
```yaml
memories:
  - content: "Critical analytics fix: Dashboard now shows accurate memory count..."
    tags:
      - dashboard
      - fix
      - v8.17.1
      - analytics
    created_at_iso: "2025-11-01T08:45:00Z"
    relevance: 0.95
    type: implementation
    age_days: 4
  - content: "Memory hook retrieves git context..."
    tags:
      - hooks
      - git
    ...
```

**Expected Output (per spec)**:
```toon
memories[8]{content,tags,created_at_iso,relevance,type,age_days}:
  "Critical analytics fix...",["dashboard","fix","v8.17.1","analytics"],"2025-11-01T08:45:00Z",0.95,"implementation",4
  "Memory hook retrieves...",["hooks","git"],"2025-10-31T20:15:00Z",0.87,"feature",5
  ...
```

### 2. Manual Implementation Results

To verify whether the spec's claims about token efficiency were accurate, I manually implemented the tabular format described in the specification:

**Implementation**: ~180 lines with proper CSV escaping, compact date formatting, and field name deduplication

**Results**:
- Manual tabular TOON: **431 tokens** (1.1% better than markdown)
- Library YAML-style TOON: **602 tokens** (38% worse than markdown)
- **Difference**: Manual implementation is **28.4% more efficient** than library output

This confirms the TOON spec's approach has merit, but the library implementation doesn't deliver the promised token savings.

### 3. Real-world Comparison vs Optimized Markdown

Our markdown format is already highly optimized (v8.4.0+):
- Adaptive truncation (300-800 chars based on memory count)
- Compact date formatting ("Nov 1" vs ISO timestamps)
- Content-first structure (no repeated field names)
- Visual grouping by category

**Finding**: Manual TOON saves only **5 tokens** (1.1%) vs our optimized markdown.

This suggests that for well-optimized baseline formats, TOON's tabular approach provides marginal gains - though every token counts!

## Questions for the Maintainers

I'm genuinely curious about the design decisions here:

1. **Why YAML-style instead of tabular?**
   - Is this intentional for readability/debugging?
   - Does the library plan to support spec-compliant tabular output?
   - Are there use cases where YAML-style is preferred?

2. **Specification vs Implementation**
   - Is the spec outdated, or is the library diverging from it?
   - Are there plans to align the library with the tabular format shown in docs?

3. **Token Efficiency Claims**
   - The spec mentions 30-60% token savings - what baseline is this compared to?
   - Have there been benchmarks against optimized JSON/YAML/Markdown?

4. **Use Case Fit**
   - Is TOON optimized for specific data patterns (uniform arrays, short values)?
   - Our memories have variable-length content (50-800 chars) - is this a mismatch?

## What I Learned

Despite the token efficiency not meeting our needs, this was a valuable exercise:

1. **TOON's tabular approach has merit** - The spec-compliant format does reduce redundancy
2. **Library implementations matter** - There's a 38% difference between library and spec
3. **Baseline optimization matters** - Hard to beat already-optimized formats (diminishing returns)
4. **Readability trade-offs** - Dense tabular formats sacrifice debugability

## Documentation

I've documented the full evaluation process with:
- Complete token measurements and methodology
- Code examples of both library and manual implementations
- Detailed analysis of why improvements were minimal

**PoC Results**:
- [Library TOON Evaluation](https://github.com/doobidoo/mcp-memory-service/blob/feature/toon-format-poc/docs/poc/toon-format-results.md)
- [Manual Tabular Implementation](https://github.com/doobidoo/mcp-memory-service/blob/feature/toon-manual-tabular/docs/poc/toon-manual-tabular-results.md)
- [Manual Implementation Code](https://github.com/doobidoo/mcp-memory-service/blob/feature/toon-manual-tabular/.claude/hooks/utilities/toon-formatter.js#L360-L444)

## Constructive Feedback

I think TOON has interesting potential for specific use cases, especially:
- Uniform data structures (database rows, API responses)
- Short, predictable field values
- Systems where every token truly matters
- Non-human-readable contexts (pure machine consumption)

For our use case (human+AI collaboration contexts), the 1.1% savings didn't justify the readability trade-off, but I can see TOON shining in other scenarios.

## Thank You!

Thanks for creating an interesting approach to token optimization! I hope this real-world evaluation provides useful data points for the project's development. Happy to discuss further or provide additional testing data if helpful.

---

**Test Environment**:
- Library: `@toon-format/toon@0.7.3`
- Data: 8 memories with realistic technical content (50-300 chars each)
- Baseline: Already-optimized markdown with adaptive truncation
- Token Estimation: ~4 chars/token (standard approximation)
- Use Case: Claude Code session hook memory injection
```

---

## Discussion Metadata

**Target Repository**: https://github.com/toon-format/toon
**Discussion Category**: General / Q&A
**Tags (if available)**: `question`, `enhancement`, `documentation`
**Word Count**: ~650 words
**Tone**: Constructive, data-driven, respectful

## Before Posting - Checklist

- [ ] Review the content for tone and accuracy
- [ ] Ensure GitHub links work (may need to push branches first)
- [ ] Consider adding code snippet attachments
- [ ] Choose appropriate discussion category on GitHub
- [ ] Add relevant tags/labels if available

## Follow-up Strategy

**If maintainers respond positively**:
- Offer to help test spec-compliant implementation
- Share additional benchmark data if needed
- Consider contributing PR for tabular format option

**If they explain it's intentional**:
- Ask about roadmap for spec alignment
- Request documentation clarification
- Understand use cases where YAML-style is preferred

**If spec is outdated**:
- Suggest updating documentation to reflect current approach
- Ask about reasoning for YAML-style choice
- Request benchmark comparisons

## Additional Context

This discussion is based on two comprehensive PoCs:
1. **Library TOON PoC** (feature/toon-format-poc) - 38% worse than markdown
2. **Manual TOON PoC** (feature/toon-manual-tabular) - 1.1% better than markdown

Both branches contain full implementation, tests, and documentation.
