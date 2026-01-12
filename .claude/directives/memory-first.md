# Information Retrieval Directive

## RULE: Layered Information Lookup

When you need information about this project, follow this order:

### 1. Files First (Source of Truth)

**Always check files before anything else:**
- **Code**: `src/`, `tests/`, `scripts/`
- **Docs**: `docs/`, `CHANGELOG.md`, `README.md`
- **Config**: `.env`, `pyproject.toml`, `.claude/directives/`

**Why**: Files are always current. Code doesn't lie. Documentation may lag, but code is truth.

**Examples:**
- How does X work? → Read the implementation
- What's the current API? → Check the code
- What's configured? → Read config files

### 2. Memory Second (Historical Context)

**Search memory when files don't explain WHY:**

Use `mcp__memory__retrieve_memory(query)` for:
- **Design decisions**: Why was X chosen over Y?
- **Known issues**: Has this problem been solved before?
- **Performance baselines**: What metrics did we measure?
- **Release history**: What changed and why?
- **Learnings**: What mistakes did we make?

**Common search patterns:**
```bash
# Design decisions
mcp__memory__retrieve_memory("consolidation design decision")

# Known issues
mcp__memory__search_by_tag(["bug-fix", "troubleshooting"])

# Performance context
mcp__memory__retrieve_memory("performance baseline measurements")
```

**Tags to search**: `mcp-memory-service`, `architecture`, `decision`, `bug-fix`, `performance`, `release`

### 3. User Last (Genuine Unknowns)

**Only ask the user when:**
- Files don't contain the information
- Memory has no relevant context
- User preference or decision is genuinely needed
- Ambiguity requires clarification

**Don't ask when:**
- The answer is in the code (read it first!)
- It's documented in files
- Memory contains the context

## Decision Tree

```
Need info?
   ↓
Files have it? → YES → Use file info
   ↓ NO
Memory has it? → YES → Use memory context
   ↓ NO
Ask user
```

## Examples

### Example 1: How does consolidation work?

❌ **Wrong**: "I don't know, let me ask the user"
✅ **Right**:
1. Read `src/mcp_memory_service/consolidation/compression.py`
2. Search memory: `"consolidation system design"`
3. Only ask if still unclear (rare)

### Example 2: Why did we choose ONNX over Groq?

❌ **Wrong**: Read all quality scoring code
✅ **Right**:
1. Check if code comments explain (quick scan)
2. **Search memory**: `"quality scoring design decision ONNX Groq"`
3. Ask user if no memory found

### Example 3: Should I use approach A or B?

✅ **Right**:
1. Check if similar pattern exists in codebase
2. Search memory for architectural decisions
3. **Ask user** (design choice, user decides)

### Example 4: What's the correct API call format?

❌ **Wrong**: Ask user
✅ **Right**:
1. Check code examples in `src/mcp_memory_service/web/api/`
2. Search memory: `"api reference quick-reference"`
3. Only ask if genuinely unclear

## Memory Search Best Practices

- **Be specific**: `"consolidation cluster deduplication"` not just `"consolidation"`
- **Use project tag**: All relevant memories tagged with `mcp-memory-service`
- **Search by tags**: `retrieve_by_tags(["architecture", "decision"])`
- **Timeframe matters**: Recent decisions may override old ones (check created_at)
- **Check quality**: High-quality memories (score ≥0.7) are more reliable

## When Memory Conflicts with Files

**Files always win.** If memory says X but code does Y:
- Code is current truth
- Memory is historical context
- Note the discrepancy (code evolved since memory was stored)
