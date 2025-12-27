#!/usr/bin/env python3
# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Check that all handler functions have corresponding integration tests.

This script validates test coverage by:
1. Parsing all handler function names from memory.py
2. Checking for corresponding test classes/functions in test files
3. Reporting any untested handlers

Exit codes:
0 - All handlers tested
1 - Untested handlers found or parsing error
"""

import re
import sys
from pathlib import Path


def parse_handlers_from_file(file_path: Path) -> list[str]:
    """
    Parse handler function names from memory.py.

    Returns:
        List of handler function names (e.g., ["handle_store_memory", ...])
    """
    handlers = []
    try:
        content = file_path.read_text()
        # Match: async def handle_<something>(
        pattern = r'^async def (handle_\w+)\('
        for match in re.finditer(pattern, content, re.MULTILINE):
            handlers.append(match.group(1))
    except Exception as e:
        print(f"❌ Error parsing {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    return handlers


def parse_tested_handlers_from_file(file_path: Path) -> set[str]:
    """
    Parse tested handler names from test file.

    Looks for:
    - Test class names: TestHandleDeleteMemory -> handle_delete_memory
    - Direct handler calls: server.handle_delete_memory(
    - Handler imports: from ... import handle_delete_memory

    Returns:
        Set of tested handler names
    """
    tested = set()
    try:
        content = file_path.read_text()

        # Pattern 1: Test class names (TestHandleDeleteMemory -> handle_delete_memory)
        class_pattern = r'class TestHandle(\w+):'
        for match in re.finditer(class_pattern, content):
            # Convert PascalCase to snake_case
            name = match.group(1)
            # Insert underscore before capitals
            snake_name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
            tested.add(f"handle_{snake_name}")

        # Pattern 2: Direct handler calls (server.handle_<name>()
        call_pattern = r'server\.(handle_\w+)\('
        for match in re.finditer(call_pattern, content):
            tested.add(match.group(1))

        # Pattern 3: Import statements
        import_pattern = r'from .+ import[^(]+\(([^)]+)\)'
        for match in re.finditer(import_pattern, content):
            # Extract all imported names
            imports = match.group(1).split(',')
            for imp in imports:
                imp = imp.strip()
                if imp.startswith('handle_'):
                    tested.add(imp)

        # Pattern 4: Direct import list matching (in test_all_memory_handlers.py)
        # Match: "handle_store_memory",
        quoted_pattern = r'"(handle_\w+)"'
        for match in re.finditer(quoted_pattern, content):
            tested.add(match.group(1))

    except Exception as e:
        print(f"❌ Error parsing {file_path}: {e}", file=sys.stderr)
        # Don't exit - this is best-effort

    return tested


def main():
    """Main validation logic."""
    print("🔍 Checking handler test coverage...\n")

    # Find project root (assumes script is in scripts/validation/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    # Paths
    handlers_file = project_root / "src" / "mcp_memory_service" / "server" / "handlers" / "memory.py"
    test_dir = project_root / "tests"

    # Validate handlers file exists
    if not handlers_file.exists():
        print(f"❌ Handlers file not found: {handlers_file}", file=sys.stderr)
        sys.exit(1)

    # Parse all handlers
    all_handlers = parse_handlers_from_file(handlers_file)
    if not all_handlers:
        print("❌ No handlers found in memory.py", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(all_handlers)} handlers in memory.py:")
    for handler in sorted(all_handlers):
        print(f"  - {handler}")
    print()

    # Find all test files
    test_files = list(test_dir.glob("**/test_*.py"))
    if not test_files:
        print(f"❌ No test files found in {test_dir}", file=sys.stderr)
        sys.exit(1)

    # Parse tested handlers from all test files
    tested_handlers = set()
    for test_file in test_files:
        file_tested = parse_tested_handlers_from_file(test_file)
        tested_handlers.update(file_tested)

    print(f"Found {len(tested_handlers)} tested handlers across {len(test_files)} test files:")
    for handler in sorted(tested_handlers):
        if handler in all_handlers:  # Only show handlers that exist
            print(f"  ✓ {handler}")
    print()

    # Find untested handlers
    untested = set(all_handlers) - tested_handlers

    if untested:
        print(f"❌ {len(untested)} handlers without test coverage:\n")
        for handler in sorted(untested):
            print(f"  ⚠️  {handler}")
        print()
        print("💡 Add tests to tests/integration/test_all_memory_handlers.py")
        print("   or create new test file with test class/function.")
        sys.exit(1)
    else:
        print("✅ All handlers have test coverage!")
        print(f"   Total handlers: {len(all_handlers)}")
        print(f"   Coverage: 100%")
        sys.exit(0)


if __name__ == "__main__":
    main()
