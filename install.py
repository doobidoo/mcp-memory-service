#!/usr/bin/env python3
"""
Root-level install.py for mcp-memory-service.

This script exists to help users who follow the wiki installation guide
(https://github.com/doobidoo/mcp-memory-service/wiki/01-Installation-Guide)
and run `python install.py` from the repository root.

It delegates to the appropriate installer based on what the user needs:
  - Python package installer: scripts/installation/install.py
  - Claude Code hooks installer: claude-hooks/install_hooks.py

Usage:
  python install.py              # Interactive menu
  python install.py --package    # Install Python package (delegates to scripts/installation/install.py)
  python install.py --hooks      # Install Claude Code hooks (delegates to claude-hooks/install_hooks.py)
  python install.py --help       # Show this help
"""

import sys
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parent.resolve()
PACKAGE_INSTALLER = REPO_ROOT / "scripts" / "installation" / "install.py"
HOOKS_INSTALLER = REPO_ROOT / "claude-hooks" / "install_hooks.py"


def print_banner():
    print()
    print("=" * 60)
    print("  mcp-memory-service installer")
    print("=" * 60)
    print()


def print_python313_warning():
    """Warn users about known safetensors / Python 3.13 issues."""
    if sys.version_info >= (3, 13):
        print("WARNING: You are running Python 3.13.")
        print("  Some dependencies (e.g. safetensors) may not yet support")
        print("  Python 3.13 and could cause installation failures.")
        print()
        print("  Recommended: Use Python 3.11 or 3.12 for best compatibility.")
        print("  See docs: https://github.com/doobidoo/mcp-memory-service/wiki")
        print("            or docs/first-time-setup.md")
        print()


def run_installer(script_path: Path, extra_args: list) -> int:
    """Run a sub-installer and return its exit code."""
    if not script_path.exists():
        print(f"ERROR: Installer not found: {script_path}")
        print("  This is likely a bug. Please report it at:")
        print("  https://github.com/doobidoo/mcp-memory-service/issues")
        return 1

    cmd = [sys.executable, str(script_path)] + extra_args
    print(f"Running: {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd, check=False)
    return result.returncode


def show_help():
    print(__doc__)


def interactive_menu() -> int:
    """Show an interactive menu and delegate to the chosen installer."""
    print("This project has two separate installation options:")
    print()
    print("  1. Python package installation")
    print("     Installs the mcp-memory-service Python package and its")
    print("     dependencies using pip/uv.")
    print("     Script: scripts/installation/install.py")
    print()
    print("  2. Claude Code hooks installation")
    print("     Installs memory awareness hooks for Claude Code (session-")
    print("     start/end hooks, natural memory triggers, etc.).")
    print("     Script: claude-hooks/install_hooks.py")
    print()
    print("  q. Quit")
    print()

    try:
        choice = input("Enter choice [1/2/q]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        print("Aborted.")
        return 0

    print()

    if choice == "1":
        print("Delegating to: scripts/installation/install.py")
        print()
        return run_installer(PACKAGE_INSTALLER, [])
    elif choice == "2":
        print("Delegating to: claude-hooks/install_hooks.py")
        print()
        return run_installer(HOOKS_INSTALLER, [])
    elif choice in ("q", "quit", "exit"):
        print("Exiting.")
        return 0
    else:
        print(f"Unknown choice: {choice!r}")
        print("Run `python install.py --help` for usage.")
        return 1


def main() -> int:
    print_banner()
    print_python313_warning()

    args = sys.argv[1:]

    # Pass-through: any unrecognised flags are forwarded to the package installer
    if "--help" in args or "-h" in args:
        show_help()
        return 0

    if "--package" in args:
        extra = [a for a in args if a != "--package"]
        print("Delegating to: scripts/installation/install.py")
        print()
        return run_installer(PACKAGE_INSTALLER, extra)

    if "--hooks" in args:
        extra = [a for a in args if a != "--hooks"]
        print("Delegating to: claude-hooks/install_hooks.py")
        print()
        return run_installer(HOOKS_INSTALLER, extra)

    # No recognised flags - show interactive menu
    return interactive_menu()


if __name__ == "__main__":
    sys.exit(main())
