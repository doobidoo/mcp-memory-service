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
MCP-MEMORY-SERVICE Cleanup and Organization Script

This script implements the cleanup plan documented in CLEANUP_PLAN.md.
It creates the necessary directory structure, moves files to their new locations,
and prepares everything for committing to the new branch.
"""

import datetime
import os
import shutil
import subprocess


def run_command(command):
    """Run a shell command and print the result."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    print(f"SUCCESS: {result.stdout}")
    return True


def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        print(f"Creating directory: {path}")
        os.makedirs(path)
    else:
        print(f"Directory already exists: {path}")


def move_file(src, dest):
    """Move a file from src to dest, creating destination directory if needed."""
    dest_dir = os.path.dirname(dest)
    # Only try to create the directory if dest_dir is not empty
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if os.path.exists(src):
        print(f"Moving {src} to {dest}")
        shutil.move(src, dest)
    else:
        print(f"WARNING: Source file does not exist: {src}")


def copy_file(src, dest):
    """Copy a file from src to dest, creating destination directory if needed."""
    dest_dir = os.path.dirname(dest)
    # Only try to create the directory if dest_dir is not empty
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if os.path.exists(src):
        print(f"Copying {src} to {dest}")
        shutil.copy2(src, dest)
    else:
        print(f"WARNING: Source file does not exist: {src}")


def create_readme(path, content):
    """Create a README.md file with the given content."""
    with open(path, "w") as f:
        f.write(content)
    print(f"Created README: {path}")


def main():
    # Change to the repository root directory
    repo_root = os.getcwd()
    print(f"Working in repository root: {repo_root}")

    # 1. Create test directory structure
    test_dirs = ["tests/integration", "tests/unit", "tests/performance"]
    for directory in test_dirs:
        create_directory(directory)

    # 2. Create documentation directory structure
    doc_dirs = ["docs/guides", "docs/implementation", "docs/api", "docs/examples"]
    for directory in doc_dirs:
        create_directory(directory)

    # 3. Create archive directory
    today = datetime.date.today().strftime("%Y-%m-%d")
    archive_dir = f"archive/{today}"
    create_directory(archive_dir)

    # 4. Move test files
    test_files = [
        ("test_chromadb.py", "tests/unit/test_chroma.py"),
        ("test_health_check_fixes.py", "tests/integration/test_storage.py"),
        ("test_issue_5_fix.py", "tests/unit/test_tags.py"),
        ("test_performance_optimizations.py", "tests/performance/test_caching.py"),
    ]
    for src, dest in test_files:
        move_file(src, dest)

    # 5. Move documentation files
    doc_files = [
        ("HEALTH_CHECK_FIXES_SUMMARY.md", "docs/implementation/health_checks.md"),
        ("PERFORMANCE_OPTIMIZATION_SUMMARY.md", "docs/implementation/performance.md"),
        ("CLAUDE.md", "docs/guides/claude_integration.md"),
    ]
    for src, dest in doc_files:
        move_file(src, dest)

    # 6. Archive backup files
    archive_files = [("backup_performance_optimization", f"{archive_dir}/backup_performance_optimization")]
    for src, dest in archive_files:
        if os.path.exists(src):
            if os.path.exists(dest):
                print(f"Archive destination already exists: {dest}")
            else:
                shutil.copytree(src, dest)
                print(f"Archived {src} to {dest}")
        else:
            print(f"WARNING: Source directory does not exist: {src}")

    # 7. Update CHANGELOG.md
    if os.path.exists("CHANGELOG.md.new"):
        move_file("CHANGELOG.md.new", "CHANGELOG.md")

    # 8. Create test README files
    test_readme = """# MCP-MEMORY-SERVICE Tests

This directory contains tests for the MCP-MEMORY-SERVICE project.

## Directory Structure

- `integration/` - Integration tests between components
- `unit/` - Unit tests for individual components
- `performance/` - Performance benchmarks

## Running Tests

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```
"""
    create_readme("tests/README.md", test_readme)

    # 9. Create docs README file
    docs_readme = """# MCP-MEMORY-SERVICE Documentation

This directory contains documentation for the MCP-MEMORY-SERVICE project.

## Directory Structure

- `guides/` - User guides and tutorials
- `implementation/` - Implementation details and technical documentation
- `api/` - API reference documentation
- `examples/` - Example code and usage patterns
"""
    create_readme("docs/README.md", docs_readme)

    # 10. Create archive README file
    archive_readme = f"""# Archive Directory

This directory contains archived files from previous versions of MCP-MEMORY-SERVICE.

## {today}/

- `backup_performance_optimization/` - Backup files from performance optimization work
"""
    create_readme(f"{archive_dir}/README.md", archive_readme)

    # 11. Create pytest.ini
    pytest_ini = """[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: unit tests
    integration: integration tests
    performance: performance tests
"""
    with open("pytest.ini", "w") as f:
        f.write(pytest_ini)
    print("Created pytest.ini")

    # 12. Create conftest.py
    conftest = """import pytest
import os
import sys
import tempfile
import shutil

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

@pytest.fixture
def temp_db_path():
    '''Create a temporary directory for ChromaDB testing.'''
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    shutil.rmtree(temp_dir)
"""
    with open("tests/conftest.py", "w") as f:
        f.write(conftest)
    print("Created tests/conftest.py")

    print("\nCleanup and organization completed!")
    print("Next steps:")
    print("1. Verify all files are in their correct locations")
    print("2. Run tests to ensure everything still works")
    print("3. Create a new git branch and commit the changes")
    print("   git checkout -b feature/cleanup-and-organization")
    print("   git add .")
    print('   git commit -m "Organize tests, documentation, and archive old files"')
    print("4. Push the branch for hardware testing")
    print("   git push origin feature/cleanup-and-organization")


if __name__ == "__main__":
    main()
