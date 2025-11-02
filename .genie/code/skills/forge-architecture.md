---
name: Forge Architecture Understanding
description: How Forge creates tasks, worktrees, branches, and metadata encoding
---

# Forge Architecture Understanding

**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`
**Purpose:** Know how Forge works to enable pre-commit automation and state linking
**Source:** Direct experimentation with Forge MCP (2025-10-18)

---

## Forge Task Lifecycle

### 1. Task Creation
- **API:** `mcp__automagik_forge__create_task`
- **Returns:** task_id (UUID format, e.g., `e84ff7e9-db49-4cdb-8f5b-3c1afd2df94f`)
- **Status:** starts as "todo"

### 2. Task Attempt Start
- **API:** `mcp__automagik_forge__start_task_attempt`
- **Parameters:** task_id, executor (CLAUDE_CODE, etc.)
- **Returns:** attempt_id (UUID format, e.g., `35a403e3-fe62-4545-bffe-0285dbfa472d`)

### 3. Worktree Creation (Automatic)
Forge automatically creates a worktree with the pattern:

```
<attempt-id-prefix>-<abbreviated-task-title>
```

**Example:**
- Attempt ID: `35a403e3-fe62-4545-bffe-0285dbfa472d`
- Prefix (first 4 chars): `35a4`
- Task title: "Forge Metadata Investigation - Extract task_id structure"
- Abbreviation: "test-forge-metad"
- **Worktree dir:** `35a4-test-forge-metad`
- **Location:** `/var/tmp/automagik-forge/worktrees/35a4-test-forge-metad/`

### 4. Branch Creation (Automatic)
Forge creates a forge branch with the pattern:

```
forge/<attempt-id-prefix>-<abbreviated-task-title>
```

**Example:**
```
forge/35a4-test-forge-metad
```

This becomes the current branch when genie starts work.

---

## Metadata Encoding

### Attempt ID Recovery Pattern

From any point in the worktree, the attempt ID prefix can be recovered:

**Option 1: From directory name**
```bash
pwd  # /var/tmp/automagik-forge/worktrees/35a4-test-forge-metad
# Extract prefix: 35a4
```

**Option 2: From current branch**
```bash
git rev-parse --abbrev-ref HEAD
# Output: forge/35a4-test-forge-metad
# Extract prefix: 35a4
```

**Option 3: From branch config**
```bash
git config --list | grep "branch.forge"
# Output: branch.forge/35a4-test-forge-metad.remote=origin
# Extract prefix: 35a4
```

### Data Structure

```
Task Layer (Forge API):
  ├─ task_id: e84ff7e9-db49-4cdb-8f5b-3c1afd2df94f (full UUID, persistent)
  └─ task metadata: title, description, status (todo/in-progress/complete)

Attempt Layer (Forge API):
  ├─ attempt_id: 35a403e3-fe62-4545-bffe-0285dbfa472d (full UUID)
  └─ Created when: start_task_attempt() called

Worktree Layer (File System):
  ├─ directory: /var/tmp/automagik-forge/worktrees/35a4-test-forge-metad/
  ├─ prefix: 35a4 (first 4 chars of attempt_id)
  └─ branch: forge/35a4-test-forge-metad

Wish Layer (Genie):
  ├─ wish slug: extracted from abbreviated task title
  ├─ wish file: .genie/wishes/<slug>/<slug>-wish.md
  └─ must be linked in SESSION-STATE.md
```

---

## Pre-Commit Hook Integration Point

When genie commits in the worktree:

**Flow:**
```
Pre-commit hook fires
  ↓
Extract worktree directory name: 35a4-test-forge-metad
  ↓
Extract attempt_id prefix: 35a4
  ↓
Extract wish slug from abbreviation: test-forge-metad → forge-metadata-investigation (lookup in .genie/wishes/)
  ↓
Link everything:
  - Forge task_id: e84ff7e9-db49-4cdb-8f5b-3c1afd2df94f
  - Attempt ID: 35a403e3-fe62-4545-bffe-0285dbfa472d
  - Wish: forge-metadata-investigation
  - GitHub issue: (if present in wish metadata)
  ↓
Update SESSION-STATE.md with full linkage
```

---

## Reverse-Mapping Algorithm

**Given:** Current directory in worktree
**Goal:** Find wish slug and link to Forge task

### Step 1: Extract Worktree Info
```bash
# Get current directory name
WTDIR=$(basename "$(pwd)")
# Example: 35a4-test-forge-metad

# Extract prefix
PREFIX=$(echo "$WTDIR" | cut -d- -f1)
# Result: 35a4

# Extract abbreviation (remainder after prefix and hyphen)
ABBREV=$(echo "$WTDIR" | cut -d- -f2-)
# Result: test-forge-metad
```

### Step 2: Find Matching Wish
```bash
# Search .genie/wishes for directory containing this abbreviation
for wish_dir in .genie/wishes/*/; do
  wish_slug=$(basename "$wish_dir")
  if [[ "$wish_slug" == *"${ABBREV}"* ]]; then
    # Found match
    WISH_SLUG="$wish_slug"
    break
  fi
done
```

### Step 3: Query Forge (Optional - if we need full attempt_id)
```bash
# If prefix is insufficient, could query Forge for attempts matching criteria
# (This may not be exposed via current MCP)
```

### Step 4: Update STATE
```bash
# Update SESSION-STATE.md with:
# - Forge task_id
# - Attempt ID prefix
# - Wish slug
# - GitHub issue (from wish metadata if present)
```

---

## Key Insights for Automation

1. **Worktree directory name is the primary signal** - it contains both:
   - Attempt ID prefix (first 4 chars) - identifies the Forge task attempt
   - Abbreviated task title - helps identify wish

2. **Git branch also encodes this** - `forge/35a4-...` is always available and reliable

3. **No additional metadata files needed** - Forge doesn't leave .forge-context.json or similar
   - All metadata is in file system paths and git branch names

4. **Pre-commit hook has enough info** to:
   - Extract attempt prefix (identify task)
   - Find wish slug (identify work)
   - Link them in SESSION-STATE.md
   - No external API calls needed (except optional Forge MCP query)

5. **Limitation:** Can't reliably get full attempt_id UUID from worktree alone
   - Only have first 4 chars (prefix)
   - May be sufficient for most purposes (session identification)
   - Full UUID stored in Forge database but not exposed in worktree

---

## Testing

Created test task to verify these patterns:

**Task:**
- ID: `e84ff7e9-db49-4cdb-8f5b-3c1afd2df94f`
- Title: "[TEST] Forge Metadata Investigation - Extract task_id structure"

**Attempt:**
- ID: `35a403e3-fe62-4545-bffe-0285dbfa472d`
- Prefix: `35a4`

**Worktree:**
- Location: `/var/tmp/automagik-forge/worktrees/35a4-test-forge-metad/`
- Current branch: `forge/35a4-test-forge-metad`

All patterns confirmed ✅

---

## Next Steps

- Use this knowledge to implement pre-commit hook reverse-extraction
- Create forge-task-link workflow that runs on first commit
- Update SESSION-STATE.md automatically without manual intervention
