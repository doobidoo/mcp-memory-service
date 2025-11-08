# PR Review Thread Management with GraphQL

**Status:** ✅ Implemented in v8.20.0
**Motivation:** Eliminate manual "mark as resolved" clicks, reduce PR review friction
**Key Benefit:** Automatic thread resolution when code is fixed

---

## Table of Contents

1. [Overview](#overview)
2. [Why GraphQL?](#why-graphql)
3. [Components](#components)
4. [Usage Guide](#usage-guide)
5. [Integration with Automation](#integration-with-automation)
6. [Troubleshooting](#troubleshooting)
7. [API Reference](#api-reference)

---

## Overview

This system provides **automated PR review thread management** using GitHub's GraphQL API. It eliminates the manual work of resolving review threads by:

1. **Detecting** which code changes address review comments
2. **Automatically resolving** threads for fixed code
3. **Adding explanatory comments** with commit references
4. **Tracking thread status** across review iterations

### Problem Solved

**Before:**
```bash
# Manual workflow (time-consuming, error-prone)
1. Gemini reviews code → creates 30 inline comments
2. You fix all issues → push commit
3. Manually click "Resolve" 30 times on GitHub web UI
4. Trigger new review: gh pr comment $PR --body "/gemini review"
5. Repeat...
```

**After:**
```bash
# Automated workflow (zero manual clicks)
1. Gemini reviews code → creates 30 inline comments
2. You fix all issues → push commit
3. Auto-resolve: bash scripts/pr/resolve_threads.sh $PR HEAD --auto
4. Trigger new review: gh pr comment $PR --body "/gemini review"
5. Repeat...
```

Even better with `auto_review.sh` - it auto-resolves threads after each fix iteration!

---

## Why GraphQL?

### GitHub API Limitation

**Critical discovery:** GitHub's REST API **cannot** resolve PR review threads.

```bash
# ❌ REST API - No thread resolution endpoint
gh api repos/OWNER/REPO/pulls/PR_NUMBER/comments
# Can list comments, but cannot resolve threads

# ✅ GraphQL API - Full thread management
gh api graphql -f query='mutation { resolveReviewThread(...) }'
# Can query threads, resolve them, add replies
```

### GraphQL Advantages

| Feature | REST API | GraphQL API |
|---------|----------|-------------|
| **List review comments** | ✅ Yes | ✅ Yes |
| **Get thread status** | ❌ No | ✅ Yes (`isResolved`, `isOutdated`) |
| **Resolve threads** | ❌ No | ✅ Yes (`resolveReviewThread` mutation) |
| **Add thread replies** | ❌ Limited | ✅ Yes (`addPullRequestReviewThreadReply`) |
| **Thread metadata** | ❌ No | ✅ Yes (line, path, diffSide) |

---

## Components

### 1. GraphQL Helpers Library

**File:** `scripts/pr/lib/graphql_helpers.sh`
**Purpose:** Reusable GraphQL operations for PR review threads

**Key Functions:**

```bash
# Get all review threads for a PR
get_review_threads <PR_NUMBER>

# Resolve a thread (with optional comment)
resolve_review_thread <THREAD_ID> [COMMENT]

# Add a reply to a thread
add_thread_reply <THREAD_ID> <COMMENT>

# Check if a line was modified in a commit
was_line_modified <FILE_PATH> <LINE_NUMBER> <COMMIT_SHA>

# Get thread statistics
get_thread_stats <PR_NUMBER>
count_unresolved_threads <PR_NUMBER>

# Verify gh CLI supports GraphQL
check_graphql_support
```

**GraphQL Queries Used:**

1. **Query review threads:**
   ```graphql
   query($pr: Int!, $owner: String!, $repo: String!) {
     repository(owner: $owner, name: $repo) {
       pullRequest(number: $pr) {
         reviewThreads(first: 100) {
           nodes {
             id
             isResolved
             isOutdated
             path
             line
             comments(first: 10) {
               nodes {
                 id
                 author { login }
                 body
                 createdAt
               }
             }
           }
         }
       }
     }
   }
   ```

2. **Resolve a thread:**
   ```graphql
   mutation($threadId: ID!) {
     resolveReviewThread(input: {threadId: $threadId}) {
       thread {
         id
         isResolved
       }
     }
   }
   ```

3. **Add thread reply:**
   ```graphql
   mutation($threadId: ID!, $body: String!) {
     addPullRequestReviewThreadReply(input: {
       pullRequestReviewThreadId: $threadId
       body: $body
     }) {
       comment { id }
     }
   }
   ```

### 2. Smart Thread Resolution Tool

**File:** `scripts/pr/resolve_threads.sh`
**Purpose:** Automatically resolve threads when code is fixed

**Usage:**

```bash
# Interactive mode (prompts for each thread)
bash scripts/pr/resolve_threads.sh <PR_NUMBER> [COMMIT_SHA]

# Automatic mode (no prompts)
bash scripts/pr/resolve_threads.sh <PR_NUMBER> HEAD --auto

# Example
bash scripts/pr/resolve_threads.sh 212 HEAD --auto
```

**Decision Logic:**

```bash
For each unresolved thread:
  1. Is the file modified in this commit?
     → Yes: Check if the specific line was changed
        → Yes: Resolve with "Line X modified in commit ABC"
        → No: Skip
     → No: Check if thread is marked "outdated" by GitHub
        → Yes: Resolve with "Thread outdated by subsequent commits"
        → No: Skip
```

**Resolution Comment Format:**

```markdown
✅ Resolved: Line 123 in file.py was modified in commit abc1234

Verified by automated thread resolution script.
```

### 3. Thread Status Display

**File:** `scripts/pr/thread_status.sh`
**Purpose:** Display comprehensive thread status with filtering

**Usage:**

```bash
# Show all threads with summary
bash scripts/pr/thread_status.sh <PR_NUMBER>

# Show only unresolved threads
bash scripts/pr/thread_status.sh <PR_NUMBER> --unresolved

# Show only resolved threads
bash scripts/pr/thread_status.sh <PR_NUMBER> --resolved

# Show only outdated threads
bash scripts/pr/thread_status.sh <PR_NUMBER> --outdated

# Example
bash scripts/pr/thread_status.sh 212 --unresolved
```

**Output Format:**

```
========================================
  PR Review Thread Status
========================================
PR Number: #212
Filter: unresolved

========================================
  Summary
========================================
Total Threads:      45
Resolved:           39
Unresolved:         6
Outdated:           12

========================================
  Thread Details
========================================

○ Thread #1
  Status: UNRESOLVED | OUTDATED
  File: scripts/pr/auto_review.sh:89
  Side: RIGHT
  Author: gemini-code-assist[bot]
  Created: 2025-11-08T12:30:45Z
  Comments: 1
  "Variable $review_comments is undefined. Define it before use..."
  Thread ID: MDEyOlB1bGxSZXF1ZXN...

...
```

### 4. Integration with Auto-Review

**File:** `scripts/pr/auto_review.sh` (enhanced)
**Added functionality:**

1. **Startup:** Check GraphQL availability
   ```bash
   GraphQL Thread Resolution: Enabled
   ```

2. **Per-iteration:** Display thread stats
   ```bash
   Review Threads: 45 total, 30 resolved, 15 unresolved
   ```

3. **After pushing fixes:** Auto-resolve threads
   ```bash
   Resolving review threads for fixed code...
   ✅ Review threads auto-resolved (8 threads)
   ```

### 5. Integration with Watch Mode

**File:** `scripts/pr/watch_reviews.sh` (enhanced)
**Added functionality:**

1. **Startup:** Check GraphQL availability
   ```bash
   GraphQL Thread Tracking: Enabled
   ```

2. **Per-check:** Display thread stats
   ```bash
   Review Threads: 45 total, 30 resolved, 15 unresolved
   ```

3. **On new review:** Show unresolved thread details
   ```bash
   Thread Status:
   [Displays thread_status.sh --unresolved output]

   Options:
     1. View detailed thread status:
        bash scripts/pr/thread_status.sh 212
     ...
   ```

---

## Usage Guide

### Basic Workflow

**1. Check thread status:**

```bash
bash scripts/pr/thread_status.sh 212
```

**2. Fix issues and push:**

```bash
# Fix code based on review comments
git add .
git commit -m "fix: address review feedback"
git push
```

**3. Resolve threads for fixed code:**

```bash
# Automatic resolution
bash scripts/pr/resolve_threads.sh 212 HEAD --auto

# Interactive resolution (with prompts)
bash scripts/pr/resolve_threads.sh 212 HEAD
```

**4. Trigger new review:**

```bash
gh pr comment 212 --body "/gemini review"
```

### Integrated Workflow (Recommended)

**Use auto_review.sh - it handles everything:**

```bash
bash scripts/pr/auto_review.sh 212 5 true
```

This will:
- Fetch review feedback
- Categorize issues
- Generate fixes
- Apply and push fixes
- **Auto-resolve threads** ← New!
- Wait for next review
- Repeat

**Use watch_reviews.sh for monitoring:**

```bash
bash scripts/pr/watch_reviews.sh 212 120
```

This will:
- Check for new reviews every 120s
- **Display thread status** ← New!
- Show unresolved threads when reviews arrive
- Optionally trigger auto_review.sh

### Advanced Usage

**Manual thread resolution with custom comment:**

```bash
# Interactive mode allows custom comments
bash scripts/pr/resolve_threads.sh 212 HEAD

# When prompted:
Resolve this thread? (y/N): y
Add custom comment? (leave empty for auto): Fixed by refactoring storage backend

# Result:
✅ Fixed by refactoring storage backend
```

**Query thread info programmatically:**

```bash
# Source the helpers
source scripts/pr/lib/graphql_helpers.sh

# Get all threads as JSON
threads=$(get_review_threads 212)

# Extract specific data
echo "$threads" | jq '.data.repository.pullRequest.reviewThreads.nodes[] |
  select(.isResolved == false) |
  {file: .path, line: .line, comment: .comments.nodes[0].body}'
```

**Check specific file's threads:**

```bash
source scripts/pr/lib/graphql_helpers.sh

# Get threads for specific file
get_unresolved_threads_for_file 212 "scripts/pr/auto_review.sh"
```

---

## Integration with Automation

### Gemini PR Automator Agent

The gemini-pr-automator agent (`.claude/agents/gemini-pr-automator.md`) now includes GraphQL thread management:

**Phase 1: Initial PR Creation**
```bash
# After creating PR, start watch mode with GraphQL tracking
bash scripts/pr/watch_reviews.sh $PR_NUMBER 180 &
```

**Phase 2: Review Iteration**
```bash
# Auto-review now auto-resolves threads
bash scripts/pr/auto_review.sh $PR_NUMBER 5 true
# Includes:
# - Fix issues
# - Push commits
# - Resolve threads  ← Automatic!
# - Trigger new review
```

**Phase 3: Manual Fixes**
```bash
# After manual fixes
git push
bash scripts/pr/resolve_threads.sh $PR_NUMBER HEAD --auto
gh pr comment $PR_NUMBER --body "/gemini review"
```

### Pre-commit Integration (Future)

**Potential enhancement:** Warn about unresolved threads before allowing new commits

```bash
# In .git/hooks/pre-commit
if [ -n "$PR_BRANCH" ]; then
  unresolved=$(bash scripts/pr/thread_status.sh $PR_NUMBER --unresolved | grep "Unresolved:" | awk '{print $2}')

  if [ "$unresolved" -gt 0 ]; then
    echo "⚠️  Warning: $unresolved unresolved review threads"
    echo "Consider resolving before committing new changes"
  fi
fi
```

---

## Troubleshooting

### Issue 1: GraphQL helpers not found

**Symptom:**
```
Warning: GraphQL helpers not available, thread auto-resolution disabled
```

**Cause:** `scripts/pr/lib/graphql_helpers.sh` not found

**Fix:**
```bash
# Verify file exists
ls -la scripts/pr/lib/graphql_helpers.sh

# If missing, re-pull from main branch
git checkout main -- scripts/pr/lib/
```

### Issue 2: gh CLI doesn't support GraphQL

**Symptom:**
```
Error: GitHub CLI version X.Y.Z is too old
GraphQL support requires v2.20.0 or later
```

**Fix:**
```bash
# Update gh CLI
gh upgrade

# Or install latest from https://cli.github.com/
```

### Issue 3: Thread resolution fails

**Symptom:**
```
❌ Failed to resolve
```

**Causes and fixes:**

1. **Invalid thread ID:**
   ```bash
   # Verify thread exists
   bash scripts/pr/thread_status.sh $PR_NUMBER
   ```

2. **Network issues:**
   ```bash
   # Check GitHub connectivity
   gh auth status
   gh api graphql -f query='query { viewer { login } }'
   ```

3. **Permissions:**
   ```bash
   # Ensure you have write access to the repository
   gh repo view --json viewerPermission
   ```

### Issue 4: Threads not auto-resolving during auto_review

**Symptom:**
Auto-review runs but threads remain unresolved

**Debug steps:**

1. **Check GraphQL availability:**
   ```bash
   bash scripts/pr/auto_review.sh 212 1 true 2>&1 | grep "GraphQL"
   # Should show: GraphQL Thread Resolution: Enabled
   ```

2. **Verify thread resolution script works:**
   ```bash
   bash scripts/pr/resolve_threads.sh 212 HEAD --auto
   # Should resolve threads if any changes match
   ```

3. **Check commit SHA detection:**
   ```bash
   git rev-parse HEAD
   # Should return valid SHA
   ```

### Issue 5: "No threads needed resolution" when threads exist

**Symptom:**
```
ℹ️  No threads needed resolution
```

**Cause:** Threads reference lines that weren't modified in the commit

**Explanation:**

The tool only resolves threads for **code that was actually changed**:

```bash
# Thread on line 89 of file.py
# Your commit modified lines 100-120
# → Thread NOT resolved (line 89 unchanged)

# Thread on line 105 of file.py
# Your commit modified lines 100-120
# → Thread RESOLVED (line 105 changed)
```

**Fix:** Either:
1. Modify the code that the thread references
2. Manually resolve via GitHub web UI if thread is no longer relevant
3. Wait for thread to become "outdated" (GitHub marks it automatically after subsequent commits)

---

## API Reference

### GraphQL Helper Functions

#### `get_review_threads <PR_NUMBER>`

**Description:** Fetch all review threads for a PR

**Returns:** JSON with thread data

**Example:**
```bash
source scripts/pr/lib/graphql_helpers.sh
threads=$(get_review_threads 212)
echo "$threads" | jq '.data.repository.pullRequest.reviewThreads.nodes | length'
# Output: 45
```

#### `resolve_review_thread <THREAD_ID> [COMMENT]`

**Description:** Resolve a review thread with optional comment

**Parameters:**
- `THREAD_ID`: GraphQL node ID (e.g., `MDEyOlB1bGxSZXF1ZXN...`)
- `COMMENT`: Optional explanatory comment

**Returns:** 0 on success, 1 on failure

**Example:**
```bash
resolve_review_thread "MDEyOlB1bGxSZXF1ZXN..." "Fixed in commit abc1234"
```

#### `add_thread_reply <THREAD_ID> <COMMENT>`

**Description:** Add a reply to a thread without resolving

**Parameters:**
- `THREAD_ID`: GraphQL node ID
- `COMMENT`: Reply text (required)

**Returns:** 0 on success, 1 on failure

**Example:**
```bash
add_thread_reply "MDEyOlB1bGxSZXF1ZXN..." "Working on this now, will fix in next commit"
```

#### `was_line_modified <FILE_PATH> <LINE_NUMBER> <COMMIT_SHA>`

**Description:** Check if a specific line was modified in a commit

**Parameters:**
- `FILE_PATH`: Relative path to file
- `LINE_NUMBER`: Line number to check
- `COMMIT_SHA`: Commit to check (e.g., `HEAD`, `abc1234`)

**Returns:** 0 if modified, 1 if not

**Example:**
```bash
if was_line_modified "scripts/pr/auto_review.sh" 89 "HEAD"; then
  echo "Line 89 was modified"
fi
```

#### `get_thread_stats <PR_NUMBER>`

**Description:** Get summary statistics for PR review threads

**Returns:** JSON with counts

**Example:**
```bash
stats=$(get_thread_stats 212)
echo "$stats" | jq '.unresolved'
# Output: 6
```

#### `count_unresolved_threads <PR_NUMBER>`

**Description:** Get count of unresolved threads

**Returns:** Integer count

**Example:**
```bash
count=$(count_unresolved_threads 212)
echo "Unresolved threads: $count"
# Output: Unresolved threads: 6
```

---

## Best Practices

### 1. Use Auto-Resolution Conservatively

**Do auto-resolve when:**
- ✅ You fixed the exact code mentioned in the comment
- ✅ The commit directly addresses the review feedback
- ✅ Tests pass after the fix

**Don't auto-resolve when:**
- ❌ Unsure if the fix fully addresses the concern
- ❌ The review comment asks a question (not a fix request)
- ❌ Breaking changes involved (needs discussion)

### 2. Add Meaningful Comments

**Good resolution comments:**
```
✅ Fixed: Refactored using async/await pattern as suggested
✅ Resolved: Added type hints for all parameters
✅ Addressed: Extracted helper function to reduce complexity
```

**Bad resolution comments:**
```
❌ Done
❌ Fixed
❌ OK
```

### 3. Verify Before Auto-Resolving

```bash
# 1. Check what will be resolved
bash scripts/pr/resolve_threads.sh 212 HEAD

# Review the prompts, then run in auto mode
bash scripts/pr/resolve_threads.sh 212 HEAD --auto
```

### 4. Monitor Thread Status

```bash
# Regular check during review cycle
bash scripts/pr/thread_status.sh 212 --unresolved

# Track progress
bash scripts/pr/thread_status.sh 212
# Shows: 45 total, 39 resolved, 6 unresolved
```

---

## Performance Considerations

### API Rate Limits

GitHub GraphQL API has rate limits:
- **Authenticated:** 5,000 points per hour
- **Points per query:** ~1 point for simple queries, ~10 for complex

**Our usage:**
- `get_review_threads`: ~5 points (fetches 100 threads with comments)
- `resolve_review_thread`: ~1 point
- `get_thread_stats`: ~5 points

**Typical PR with 30 threads:**
- Initial status check: 5 points
- Resolve 30 threads: 30 points
- Final status check: 5 points
- **Total: ~40 points** (0.8% of hourly limit)

**Conclusion:** Rate limits not a concern for typical PR workflows.

### Network Latency

- GraphQL API calls: ~200-500ms each
- Auto-resolving 30 threads: ~1-2 seconds total
- Minimal impact on review cycle time

---

## Future Enhancements

### 1. Bulk Thread Operations

**Idea:** Resolve all threads for a file in one mutation

```bash
# Current: 30 API calls for 30 threads
for thread in $threads; do
  resolve_review_thread "$thread"
done

# Future: 1 API call for 30 threads
resolve_threads_batch "${thread_ids[@]}"
```

### 2. Smart Thread Filtering

**Idea:** Only show threads relevant to recent commits

```bash
bash scripts/pr/thread_status.sh 212 --since="2 hours ago"
bash scripts/pr/thread_status.sh 212 --author="gemini-code-assist[bot]"
```

### 3. Thread Diff View

**Idea:** Show what changed for each thread

```bash
bash scripts/pr/thread_diff.sh 212
# Shows:
# Thread #1: scripts/pr/auto_review.sh:89
#   Before: review_comments=$(undefined)
#   After:  review_comments=$(gh api "repos/$REPO/pulls/$PR_NUMBER/comments" | ...)
#   Status: Fixed ✅
```

### 4. Pre-Push Hook Integration

**Idea:** Warn before pushing if unresolved threads exist

```bash
# .git/hooks/pre-push
unresolved=$(count_unresolved_threads $PR_NUMBER)
if [ "$unresolved" -gt 0 ]; then
  echo "⚠️  $unresolved unresolved threads"
  read -p "Continue? (y/N): " response
fi
```

---

## Related Documentation

- **Gemini PR Automator:** `.claude/agents/gemini-pr-automator.md`
- **Code Quality Guard:** `.claude/agents/code-quality-guard.md`
- **Auto-Review Script:** `scripts/pr/auto_review.sh`
- **Watch Mode Script:** `scripts/pr/watch_reviews.sh`
- **GitHub GraphQL API:** https://docs.github.com/en/graphql

---

**Last Updated:** 2025-11-08
**Version:** 8.20.0
**Maintainer:** MCP Memory Service Team
