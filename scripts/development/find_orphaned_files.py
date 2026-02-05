#!/usr/bin/env python3
"""
Orphaned File Detection Script

Finds files and directories that may be unused, redundant, or orphaned in the repository.
This helps maintain a lean and clean codebase by identifying cleanup candidates.

Usage:
    python scripts/find_orphaned_files.py
    python scripts/find_orphaned_files.py --include-safe-files
    python scripts/find_orphaned_files.py --verbose
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path


class OrphanDetector:
    def __init__(self, repo_root: Path, include_safe_files: bool = False, verbose: bool = False):
        self.repo_root = repo_root
        self.include_safe_files = include_safe_files
        self.verbose = verbose

        # Files/dirs to always ignore
        self.ignore_patterns = {
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".DS_Store",
            ".gitignore",
            ".gitattributes",
            "LICENSE",
            "CHANGELOG.md",
            "*.pyc",
            "*.pyo",
            "*.egg-info",
            "dist",
            "build",
        }

        # Safe files that are commonly unreferenced but important
        self.safe_files = {
            "README.md",
            "pyproject.toml",
            "uv.lock",
            "setup.py",
            "requirements.txt",
            "Dockerfile",
            "docker-compose.yml",
            ".dockerignore",
            "Makefile",
            "__init__.py",
            "main.py",
            "server.py",
            "config.py",
            "settings.py",
        }

        # Extensions that are likely to be referenced
        self.code_extensions = {".py", ".js", ".ts", ".sh", ".md", ".yml", ".yaml", ".json"}

    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored."""
        path_str = str(path)
        for pattern in self.ignore_patterns:
            if pattern in path_str or path.name == pattern:
                return True
        return False

    def is_safe_file(self, path: Path) -> bool:
        """Check if a file is considered 'safe' (commonly unreferenced but important)."""
        return path.name in self.safe_files

    def find_all_files(self) -> list[Path]:
        """Find all files in the repository."""
        all_files = []
        for root, dirs, files in os.walk(self.repo_root):
            # Remove ignored directories from dirs list to skip them
            dirs[:] = [d for d in dirs if not any(ignore in d for ignore in self.ignore_patterns)]

            for file in files:
                file_path = Path(root) / file
                if not self.should_ignore(file_path):
                    all_files.append(file_path)

        return all_files

    def extract_references(self, file_path: Path) -> set[str]:
        """Extract potential file references from a file."""
        references = set()

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Find various types of references
            patterns = [
                # Python imports: from module import, import module
                r"(?:from\s+|import\s+)([a-zA-Z_][a-zA-Z0-9_.]*)",
                # File paths in quotes
                r'["\']([^"\']*\.[a-zA-Z0-9]+)["\']',
                # Common file references
                r"([a-zA-Z_][a-zA-Z0-9_.-]*\.[a-zA-Z0-9]+)",
                # Directory references
                r"([a-zA-Z_][a-zA-Z0-9_-]*/)(?:[a-zA-Z0-9_.-]+)",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                references.update(matches)

        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not read {file_path}: {e}")

        return references

    def build_reference_map(self, files: list[Path]) -> dict[str, set[Path]]:
        """Build a map of what files reference what."""
        reference_map = defaultdict(set)

        for file_path in files:
            if file_path.suffix in self.code_extensions:
                references = self.extract_references(file_path)
                for ref in references:
                    reference_map[ref].add(file_path)

        return reference_map

    def find_orphaned_files(self) -> tuple[list[Path], list[Path], list[Path]]:
        """Find potentially orphaned files."""
        all_files = self.find_all_files()
        reference_map = self.build_reference_map(all_files)

        # Convert file paths to strings for easier matching
        {f.name for f in all_files}
        {f.stem for f in all_files}
        {str(f.relative_to(self.repo_root)) for f in all_files}

        potentially_orphaned = []
        safe_unreferenced = []
        directories_to_check = []

        for file_path in all_files:
            rel_path = file_path.relative_to(self.repo_root)
            file_name = file_path.name
            file_stem = file_path.stem

            # Check if file is referenced
            is_referenced = False

            # Check various forms of references
            reference_forms = [
                file_name,
                file_stem,
                str(rel_path),
                str(rel_path).replace("/", "."),  # Python module style
                file_stem.replace("_", "-"),  # kebab-case variants
                file_stem.replace("-", "_"),  # snake_case variants
            ]

            for form in reference_forms:
                if form in reference_map and reference_map[form]:
                    is_referenced = True
                    break

            # Special checks for Python files
            if file_path.suffix == ".py":
                # Check if it's imported as a module
                module_path = str(rel_path).replace("/", ".").replace(".py", "")
                if module_path in reference_map:
                    is_referenced = True

            # Categorize unreferenced files
            if not is_referenced:
                if self.is_safe_file(file_path) and not self.include_safe_files:
                    safe_unreferenced.append(file_path)
                else:
                    potentially_orphaned.append(file_path)

        # Check for empty directories
        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if not any(ignore in d for ignore in self.ignore_patterns)]

            if not dirs and not files:  # Empty directory
                empty_dir = Path(root)
                if not self.should_ignore(empty_dir):
                    directories_to_check.append(empty_dir)

        return potentially_orphaned, safe_unreferenced, directories_to_check

    def find_duplicate_files(self) -> dict[str, list[Path]]:
        """Find files with identical names that might be duplicates."""
        all_files = self.find_all_files()
        name_groups = defaultdict(list)

        for file_path in all_files:
            name_groups[file_path.name].append(file_path)

        # Only return groups with multiple files
        return {name: paths for name, paths in name_groups.items() if len(paths) > 1}

    def analyze_config_files(self) -> list[tuple[Path, str]]:
        """Find potentially redundant configuration files."""
        all_files = self.find_all_files()
        config_files = []

        config_patterns = [
            (r".*requirements.*\.txt$", "Requirements file"),
            (r".*requirements.*\.lock$", "Requirements lock"),
            (r".*package.*\.json$", "Package.json"),
            (r".*package.*lock.*\.json$", "Package lock"),
            (r".*\.lock$", "Lock file"),
            (r".*config.*\.(py|json|yaml|yml)$", "Config file"),
            (r".*settings.*\.(py|json|yaml|yml)$", "Settings file"),
            (r".*\.env.*", "Environment file"),
        ]

        for file_path in all_files:
            rel_path = str(file_path.relative_to(self.repo_root))
            for pattern, description in config_patterns:
                if re.match(pattern, rel_path, re.IGNORECASE):
                    config_files.append((file_path, description))
                    break

        return config_files

    def generate_report(self):
        """Generate a comprehensive orphan detection report."""
        print("🔍 ORPHANED FILE DETECTION REPORT")
        print("=" * 60)

        orphaned, safe_unreferenced, empty_dirs = self.find_orphaned_files()
        duplicates = self.find_duplicate_files()
        config_files = self.analyze_config_files()

        # Potentially orphaned files
        if orphaned:
            print(f"\n❌ POTENTIALLY ORPHANED FILES ({len(orphaned)}):")
            for file_path in sorted(orphaned):
                rel_path = file_path.relative_to(self.repo_root)
                print(f"  📄 {rel_path}")
        else:
            print("\n✅ No potentially orphaned files found!")

        # Safe unreferenced files (if requested)
        if self.include_safe_files and safe_unreferenced:
            print(f"\n🟡 SAFE UNREFERENCED FILES ({len(safe_unreferenced)}):")
            print("   (These are commonly unreferenced but usually important)")
            for file_path in sorted(safe_unreferenced):
                rel_path = file_path.relative_to(self.repo_root)
                print(f"  📄 {rel_path}")

        # Empty directories
        if empty_dirs:
            print(f"\n📁 EMPTY DIRECTORIES ({len(empty_dirs)}):")
            for dir_path in sorted(empty_dirs):
                rel_path = dir_path.relative_to(self.repo_root)
                print(f"  📁 {rel_path}")

        # Duplicate file names
        if duplicates:
            print(f"\n👥 DUPLICATE FILE NAMES ({len(duplicates)} groups):")
            for name, paths in sorted(duplicates.items()):
                print(f"  📄 {name}:")
                for path in sorted(paths):
                    rel_path = path.relative_to(self.repo_root)
                    print(f"    - {rel_path}")

        # Configuration files analysis
        if config_files:
            print(f"\n⚙️  CONFIGURATION FILES ({len(config_files)}):")
            print("   (Review for redundancy)")
            config_by_type = defaultdict(list)
            for path, desc in config_files:
                config_by_type[desc].append(path)

            for desc, paths in sorted(config_by_type.items()):
                print(f"  {desc}:")
                for path in sorted(paths):
                    rel_path = path.relative_to(self.repo_root)
                    print(f"    - {rel_path}")

        print("\n" + "=" * 60)
        print("📊 SUMMARY:")
        print(f"Potentially orphaned files: {len(orphaned)}")
        print(f"Empty directories: {len(empty_dirs)}")
        print(f"Duplicate name groups: {len(duplicates)}")
        print(f"Configuration files: {len(config_files)}")

        if orphaned or empty_dirs:
            print("\n⚠️  Review these files carefully before deletion!")
            print("Some may be important despite not being directly referenced.")
        else:
            print("\n✅ Repository appears clean with no obvious orphans!")


def main():
    parser = argparse.ArgumentParser(description="Find orphaned files in the repository")
    parser.add_argument(
        "--include-safe-files", "-s", action="store_true", help="Include commonly unreferenced but safe files in report"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output including warnings")

    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    detector = OrphanDetector(repo_root, args.include_safe_files, args.verbose)
    detector.generate_report()


if __name__ == "__main__":
    main()
