---
name: Forge MCP Task Pattern
description: Create Forge tasks with @ references to load full context from files
---

# Forge MCP Task Pattern

**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`
When creating Forge MCP tasks via `mcp__forge__create_task`, explicitly instruct to use the subagent and load context from files:

```
Use the <persona> subagent to [action verb] this task.

@agent-<persona>
@.genie/wishes/<slug>/task-<group>.md
@.genie/wishes/<slug>/<slug>-wish.md

Load all context from the referenced files above. Do not duplicate content here.
```

**Example:**
```
Use the implementor subagent to implement this task.

@agent-implementor
@.genie/wishes/claude-executor/task-a.md
@.genie/wishes/claude-executor/claude-executor-wish.md

Load all context from the referenced files above. Do not duplicate content here.
```

**Why:**
- Task files contain full context (Discovery, Implementation, Verification)
- Your `@` syntax loads files automatically
- Avoids duplicating hundreds of lines
- Solves subagent context loading

**Critical Distinction:**

**Task files** (`.genie/wishes/<slug>/task-*.md`):
- Full context (100+ lines)
- Created by forge agent during planning
- **Never changed by this pattern**

**Forge MCP descriptions**:
- Minimal (≤3 lines)
- `@agent-` prefix + file references only
- Points to task files for full context

**Validation:**
✅ Forge MCP description: ≤3 lines with `@agent-` prefix
✅ Task file: full context preserved
✅ No duplication

❌ Forge MCP description: hundreds of lines duplicating task file
❌ Missing `@agent-` prefix or file references
