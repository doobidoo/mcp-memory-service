#!/bin/bash
# scripts/pr/detect_breaking_changes.sh - Analyze API changes for breaking changes
#
# Usage: bash scripts/pr/detect_breaking_changes.sh <BASE_BRANCH> [HEAD_BRANCH]
# Example: bash scripts/pr/detect_breaking_changes.sh main feature/new-api

set -e

BASE_BRANCH=${1:-main}
HEAD_BRANCH=${2:-$(git branch --show-current)}

if [ -z "$BASE_BRANCH" ]; then
    echo "Usage: $0 <BASE_BRANCH> [HEAD_BRANCH]"
    echo "Example: $0 main feature/new-api"
    exit 1
fi

if ! command -v gemini &> /dev/null; then
    echo "Error: Gemini CLI is not installed"
    exit 1
fi

echo "=== Breaking Change Detection ==="
echo "Base branch: $BASE_BRANCH"
echo "Head branch: $HEAD_BRANCH"
echo ""

# Get API-related file changes
echo "Analyzing API changes..."
api_changes=$(git diff $BASE_BRANCH...$HEAD_BRANCH -- \
    src/mcp_memory_service/tools.py \
    src/mcp_memory_service/web/api/ \
    src/mcp_memory_service/storage/base.py \
    2>/dev/null || echo "")

if [ -z "$api_changes" ]; then
    echo "✅ No API changes detected"
    echo ""
    echo "Checked paths:"
    echo "- src/mcp_memory_service/tools.py (MCP tools)"
    echo "- src/mcp_memory_service/web/api/ (Web API endpoints)"
    echo "- src/mcp_memory_service/storage/base.py (Storage interface)"
    exit 0
fi

echo "API changes detected. Analyzing for breaking changes..."
echo ""

# Check diff size and warn if large
diff_lines=$(echo "$api_changes" | wc -l)
if [ $diff_lines -gt 200 ]; then
    echo "⚠️  Warning: Large diff ($diff_lines lines) - analysis may miss changes beyond model context window"
    echo "   Consider reviewing the full diff manually for breaking changes"
fi

# Analyze with Gemini (full diff, not truncated)
result=$(gemini "Analyze these API changes for BREAKING CHANGES ONLY.

A breaking change is:
1. **Removed** function, method, class, or HTTP endpoint
2. **Changed function signature**: parameters removed, reordered, or made required
3. **Changed return type**: incompatible return value structure
4. **Renamed public API**: function, class, endpoint renamed without alias
5. **Changed HTTP endpoint**: path or method changed
6. **Removed configuration option**: environment variable or config field removed

NON-BREAKING changes (ignore these):
- Added new functions/endpoints (backward compatible)
- Added optional parameters with defaults
- Improved documentation
- Internal implementation changes
- Refactoring that preserves public interface

For each breaking change, provide:
- Severity: CRITICAL (data loss/security) / HIGH (blocks upgrade) / MEDIUM (migration effort)
- Type: Removed / Signature Changed / Renamed / etc.
- Location: File and function/endpoint name
- Impact: What breaks for users
- Migration: How users should adapt

API Changes:
\`\`\`diff
$api_changes
\`\`\`

Output format:
If breaking changes found:
## BREAKING CHANGES DETECTED

### [SEVERITY] Type: Location
**Impact:** <description>
**Migration:** <instructions>

If no breaking changes:
No breaking changes detected.")

echo "$result"
echo ""

# Check severity
if echo "$result" | grep -qi "CRITICAL"; then
    echo "🔴 CRITICAL breaking changes detected!"
    exit 3
elif echo "$result" | grep -qi "HIGH"; then
    echo "🟠 HIGH severity breaking changes detected!"
    exit 2
elif echo "$result" | grep -qi "MEDIUM"; then
    echo "🟡 MEDIUM severity breaking changes detected"
    exit 1
elif echo "$result" | grep -qi "breaking"; then
    echo "⚠️  Breaking changes detected (unspecified severity)"
    exit 1
else
    echo "✅ No breaking changes detected"
    exit 0
fi
