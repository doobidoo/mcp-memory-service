#!/bin/bash
# scripts/pr/generate_tests.sh - Auto-generate tests for new code in PR
#
# Usage: bash scripts/pr/generate_tests.sh <PR_NUMBER>
# Example: bash scripts/pr/generate_tests.sh 123

set -e

PR_NUMBER=$1

if [ -z "$PR_NUMBER" ]; then
    echo "Usage: $0 <PR_NUMBER>"
    exit 1
fi

if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed"
    exit 1
fi

if ! command -v gemini &> /dev/null; then
    echo "Error: Gemini CLI is not installed"
    exit 1
fi

echo "=== Test Generation for PR #$PR_NUMBER ==="
echo ""

# Get changed Python files (exclude tests/)
changed_files=$(gh pr diff $PR_NUMBER --name-only | grep '\.py$' | grep -v '^tests/' || echo "")

if [ -z "$changed_files" ]; then
    echo "No Python source files changed (excluding tests/)"
    exit 0
fi

echo "Files to generate tests for:"
echo "$changed_files"
echo ""

tests_generated=0

# Process files safely (handle spaces in filenames)
echo "$changed_files" | while IFS= read -r file; do
    if [ -z "$file" ]; then
        continue
    fi

    if [ ! -f "$file" ]; then
        echo "Skipping $file (not found in working directory)"
        continue
    fi

    echo "=== Processing: $file ==="

    # Extract basename for temp files
    base_name=$(basename "$file" .py)

    # Determine test file path (mirror source structure)
    # e.g., src/api/utils.py -> tests/api/test_utils.py
    test_dir="tests/$(dirname "${file#src/}")"
    mkdir -p "$test_dir"
    test_file="$test_dir/test_$(basename "$file")"

    if [ -f "$test_file" ]; then
        echo "Test file exists: $test_file"
        echo "Suggesting additional test cases..."

        # Read existing tests
        existing_tests=$(cat "$test_file")

        # Read source code
        source_code=$(cat "$file")

        # Generate additional tests
        additional_tests=$(gemini "Existing pytest test file:
\`\`\`python
$existing_tests
\`\`\`

Source code with new/changed functionality:
\`\`\`python
$source_code
\`\`\`

Task: Suggest additional pytest test functions to cover new/changed code that isn't already tested.

Requirements:
- Use pytest framework
- Include async tests if source has async functions
- Test happy paths and edge cases
- Test error handling
- Follow existing test style
- Output ONLY the new test functions (no imports, no existing tests)

Format: Complete Python test functions ready to append.")

        # Use mktemp for output file
        output_file=$(mktemp -t test_additions_${base_name}.XXXXXX)
        echo "$additional_tests" > "$output_file"

        echo "Additional tests generated: $output_file"
        echo ""
        echo "--- Preview ---"
        head -20 "$output_file"
        echo "..."
        echo "--- End Preview ---"
        echo ""
        echo "To append: cat $output_file >> $test_file"

    else
        echo "Creating new test file: $test_file"

        # Read source code
        source_code=$(cat "$file")

        # Generate complete test file
        new_tests=$(gemini "Generate comprehensive pytest tests for this Python module:

\`\`\`python
$source_code
\`\`\`

Requirements:
- Complete pytest test file with imports
- Test all public functions/methods
- Include happy paths and edge cases
- Test error handling and validation
- Use pytest fixtures if appropriate
- Include async tests for async functions
- Follow pytest best practices
- Add docstrings to test functions

Format: Complete, ready-to-use Python test file.")

        # Use mktemp for output file
        output_file=$(mktemp -t test_new_${base_name}.XXXXXX)
        echo "$new_tests" > "$output_file"

        echo "New test file generated: $output_file"
        echo ""
        echo "--- Preview ---"
        head -30 "$output_file"
        echo "..."
        echo "--- End Preview ---"
        echo ""
        echo "To create: cp $output_file $test_file"
    fi

    tests_generated=$((tests_generated + 1))
    echo ""
done

echo "=== Test Generation Complete ==="
echo "Files processed: $tests_generated"
echo ""
echo "Generated test files are in /tmp/"
echo "Review and apply manually with the commands shown above."
echo ""
echo "After applying tests:"
echo "1. Run: pytest $test_file"
echo "2. Verify tests pass"
echo "3. Commit: git add $test_file && git commit -m 'test: add tests for <feature>'"
