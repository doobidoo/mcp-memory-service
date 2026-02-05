#!/usr/bin/env python3
"""
Script to check for broken internal links in markdown files.
Checks relative links to files within the repository.

Usage:
    python scripts/check_documentation_links.py
    python scripts/check_documentation_links.py --verbose
    python scripts/check_documentation_links.py --fix-suggestions
"""

import argparse
import re
from pathlib import Path


def find_markdown_files(root_dir: str) -> list[Path]:
    """Find all markdown files in the repository."""
    root = Path(root_dir)
    md_files = []

    for path in root.rglob("*.md"):
        # Skip venv and node_modules
        if ".venv" in path.parts or "venv" in path.parts or "node_modules" in path.parts:
            continue
        md_files.append(path)

    return md_files


def extract_links(content: str) -> list[tuple[str, str]]:
    """Extract markdown links from content with their text."""
    # Pattern for markdown links: [text](url)
    link_pattern = r"\[([^\]]*)\]\(([^)]+)\)"
    links = re.findall(link_pattern, content)
    return links  # Return (text, url) tuples


def is_internal_link(link: str) -> bool:
    """Check if a link is internal (relative path)."""
    # Skip external URLs, anchors, and mailto links
    if link.startswith("http://") or link.startswith("https://") or link.startswith("mailto:") or link.startswith("#"):
        return False
    return True


def resolve_link_path(md_file_path: Path, link: str) -> Path:
    """Resolve relative link path from markdown file location."""
    # Remove any anchor fragments
    link_path = link.split("#")[0]

    # Resolve relative to the markdown file's directory
    return (md_file_path.parent / link_path).resolve()


def suggest_fixes(broken_link: str, repo_root: Path) -> list[str]:
    """Suggest possible fixes for broken links."""
    suggestions = []

    # Extract filename from the broken link
    filename = Path(broken_link).name

    # Search for files with similar names
    for md_file in find_markdown_files(str(repo_root)):
        if md_file.name.lower() == filename.lower():
            suggestions.append(str(md_file.relative_to(repo_root)))
        elif filename.lower() in md_file.name.lower():
            suggestions.append(str(md_file.relative_to(repo_root)))

    return suggestions[:3]  # Return top 3 suggestions


def check_links_in_file(md_file: Path, repo_root: Path) -> list[tuple[str, str, str, bool]]:
    """Check all internal links in a markdown file."""
    try:
        with open(md_file, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {md_file}: {e}")
        return []

    links = extract_links(content)
    internal_links = [(text, link) for text, link in links if is_internal_link(link)]

    results = []
    for link_text, link in internal_links:
        try:
            target_path = resolve_link_path(md_file, link)
            exists = target_path.exists()
            results.append((link_text, link, str(target_path), exists))
        except Exception as e:
            results.append((link_text, link, f"Error resolving: {e}", False))

    return results


def main():
    parser = argparse.ArgumentParser(description="Check for broken internal links in markdown documentation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all links, not just broken ones")
    parser.add_argument("--fix-suggestions", "-s", action="store_true", help="Suggest fixes for broken links")
    parser.add_argument("--format", choices=["text", "markdown", "json"], default="text", help="Output format")

    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    md_files = find_markdown_files(str(repo_root))

    print(f"Checking {len(md_files)} markdown files for broken links...\n")

    broken_links = []
    total_links = 0
    file_results = {}

    for md_file in sorted(md_files):
        rel_path = md_file.relative_to(repo_root)
        link_results = check_links_in_file(md_file, repo_root)

        if link_results:
            file_results[str(rel_path)] = link_results

            if args.verbose or any(not exists for _, _, _, exists in link_results):
                print(f"\n[FILE] {rel_path}")

            for link_text, link, target, exists in link_results:
                total_links += 1
                status = "[OK]" if exists else "[ERROR]"

                if args.verbose or not exists:
                    print(f"  {status} [{link_text}]({link})")
                    if not exists:
                        print(f"     -> Target: {target}")
                        broken_links.append((str(rel_path), link_text, link, target))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Total internal links checked: {total_links}")
    print(f"Broken links found: {len(broken_links)}")

    if broken_links:
        print("\n❌ BROKEN LINKS:")
        for file_path, link_text, link, target in broken_links:
            print(f"\n  📄 {file_path}")
            print(f"     Text: {link_text}")
            print(f"     Link: {link}")
            print(f"     Target: {target}")

            if args.fix_suggestions:
                suggestions = suggest_fixes(link, repo_root)
                if suggestions:
                    print("     💡 Suggestions:")
                    for suggestion in suggestions:
                        print(f"        - {suggestion}")

    # Exit with error code if broken links found
    exit_code = 1 if broken_links else 0

    if broken_links:
        print(f"\n⚠️  Found {len(broken_links)} broken links. Use --fix-suggestions for repair ideas.")
    else:
        print("\n✅ All documentation links are working correctly!")

    return exit_code


if __name__ == "__main__":
    exit(main())
