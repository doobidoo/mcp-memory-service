#!/bin/bash
# scripts/maintenance/scan_todos.sh - Scan codebase for TODOs and prioritize
#
# Usage: bash scripts/maintenance/scan_todos.sh [DIRECTORY]
# Example: bash scripts/maintenance/scan_todos.sh src/

set -e

SCAN_DIR=${1:-src}

if ! command -v gemini &> /dev/null; then
    echo "Error: Gemini CLI is not installed"
    exit 1
fi

if [ ! -d "$SCAN_DIR" ]; then
    echo "Error: Directory not found: $SCAN_DIR"
    exit 1
fi

echo "=== TODO Scanner ==="
echo "Scanning directory: $SCAN_DIR"
echo ""

# Extract all TODOs with file and line number
echo "Finding TODO comments..."
todos=$(find "$SCAN_DIR" -name '*.py' -exec grep -Hn "TODO\|FIXME\|HACK\|XXX" {} \; 2>/dev/null || echo "")

if [ -z "$todos" ]; then
    echo "✅ No TODOs found in $SCAN_DIR"
    exit 0
fi

todo_count=$(echo "$todos" | wc -l)
echo "Found $todo_count TODO comments"
echo ""

# Save to temp file
echo "$todos" > /tmp/todos_raw.txt

echo "Analyzing and prioritizing TODOs with Gemini..."
echo ""

# Use Gemini to prioritize
prioritized=$(gemini "Analyze these TODO/FIXME/HACK/XXX comments from a Python codebase and categorize by priority.

Priority Levels:
- **CRITICAL (P0)**: Security vulnerabilities, data corruption risks, blocking bugs, production-breaking issues
- **HIGH (P1)**: Performance bottlenecks (>100ms), user-facing bugs, incomplete core features, API breaking changes
- **MEDIUM (P2)**: Code quality improvements, minor optimizations, technical debt, convenience features
- **LOW (P3)**: Documentation, cosmetic changes, nice-to-haves, future enhancements

Consider:
- Security impact (SQL injection, XSS, etc.)
- Performance implications
- Feature completeness
- User impact
- Technical debt accumulation

TODO comments (format: file:line:comment):
$(cat /tmp/todos_raw.txt)

Output format (be concise):
## CRITICAL (P0)
- file.py:123 - Brief description of issue

## HIGH (P1)
- file.py:456 - Brief description

## MEDIUM (P2)
- file.py:789 - Brief description

## LOW (P3)
- file.py:012 - Brief description" 2>&1)

echo "$prioritized" > /tmp/todos_prioritized.txt

# Display results
cat /tmp/todos_prioritized.txt
echo ""

# Generate summary
critical_count=$(grep -c '## CRITICAL' /tmp/todos_prioritized.txt || echo "0")
high_count=$(grep -c '## HIGH' /tmp/todos_prioritized.txt || echo "0")
medium_count=$(grep -c '## MEDIUM' /tmp/todos_prioritized.txt || echo "0")
low_count=$(grep -c '## LOW' /tmp/todos_prioritized.txt || echo "0")

# Count actual items (lines starting with -)
critical_items=$(sed -n '/## CRITICAL/,/## HIGH/p' /tmp/todos_prioritized.txt | grep -c '^-' || echo "0")
high_items=$(sed -n '/## HIGH/,/## MEDIUM/p' /tmp/todos_prioritized.txt | grep -c '^-' || echo "0")
medium_items=$(sed -n '/## MEDIUM/,/## LOW/p' /tmp/todos_prioritized.txt | grep -c '^-' || echo "0")
low_items=$(sed -n '/## LOW/,/$/p' /tmp/todos_prioritized.txt | grep -c '^-' || echo "0")

echo "=== Summary ==="
echo "Total TODOs: $todo_count"
echo ""
echo "By Priority:"
echo "  CRITICAL (P0): $critical_items"
echo "  HIGH (P1):     $high_items"
echo "  MEDIUM (P2):   $medium_items"
echo "  LOW (P3):      $low_items"
echo ""

# Save to docs (optional)
if [ -d "docs/development" ]; then
    echo "Saving to docs/development/todo-tracker.md..."
    cat > docs/development/todo-tracker.md << EOF
# TODO Tracker

**Last Updated:** $(date '+%Y-%m-%d %H:%M:%S')
**Scan Directory:** $SCAN_DIR
**Total TODOs:** $todo_count

## Summary

| Priority | Count | Description |
|----------|-------|-------------|
| CRITICAL (P0) | $critical_items | Security, data corruption, blocking bugs |
| HIGH (P1) | $high_items | Performance, user-facing, incomplete features |
| MEDIUM (P2) | $medium_items | Code quality, optimizations, technical debt |
| LOW (P3) | $low_items | Documentation, cosmetic, nice-to-haves |

---

$(cat /tmp/todos_prioritized.txt)

---

## How to Address

1. **CRITICAL**: Address immediately, block releases if necessary
2. **HIGH**: Schedule for current/next sprint
3. **MEDIUM**: Add to backlog, address in refactoring sprints
4. **LOW**: Address opportunistically or when touching related code

## Updating This Tracker

Run: \`bash scripts/maintenance/scan_todos.sh\`
EOF
    echo "✅ Saved to docs/development/todo-tracker.md"
fi

# Exit with warning if critical TODOs found
if [ $critical_items -gt 0 ]; then
    echo ""
    echo "⚠️  WARNING: $critical_items CRITICAL TODOs found!"
    echo "These should be addressed immediately."
    exit 1
fi

exit 0
