#!/bin/bash
# scripts/pr/run_quality_checks.sh - Run quality checks on a PR
# Wrapper for quality_gate.sh to maintain consistent naming in workflows
#
# Usage: bash scripts/pr/run_quality_checks.sh <PR_NUMBER>

set -e

PR_NUMBER=$1

if [ -z "$PR_NUMBER" ]; then
    echo "Usage: $0 <PR_NUMBER>"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run quality gate checks
exec "$SCRIPT_DIR/quality_gate.sh" "$PR_NUMBER"
