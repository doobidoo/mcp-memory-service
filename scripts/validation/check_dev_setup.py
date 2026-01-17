#!/usr/bin/env python3
"""
Verify development environment setup for MCP Memory Service.
Detects common issues like stale venv packages vs updated source code.

Usage:
    python scripts/validation/check_dev_setup.py

Exit codes:
    0 - Development environment is correctly configured
    1 - Critical issues detected (editable install missing or version mismatch)
"""
import sys
import os
from pathlib import Path

def check_editable_install():
    """Check if package is installed in editable mode."""
    try:
        import mcp_memory_service
        package_location = Path(mcp_memory_service.__file__).parent

        # Check if location is in source directory (editable) or site-packages
        if 'site-packages' in str(package_location):
            return False, str(package_location)
        else:
            return True, str(package_location)
    except ImportError:
        return None, "Package not installed"

def check_version_match():
    """Check if installed version matches source code version."""
    # Read source version from _version.py (single source of truth)
    repo_root = Path(__file__).parent.parent.parent
    version_file = repo_root / "src" / "mcp_memory_service" / "_version.py"
    source_version = None

    if not version_file.exists():
        return None, "Unknown", "Source version file not found"

    # Import version directly from _version.py instead of text parsing
    import sys
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    try:
        from mcp_memory_service._version import __version__ as source_version
    except ImportError:
        # Fallback to text parsing if import fails
        with open(version_file) as f:
            for line in f:
                if line.startswith('__version__'):
                    source_version = line.split('=')[1].strip().strip('"\'')
                    break

    # Get installed version
    try:
        import mcp_memory_service
        installed_version = mcp_memory_service.__version__
    except ImportError:
        return None, source_version, "Not installed"

    if source_version is None:
        return None, "Unknown", installed_version

    return source_version == installed_version, source_version, installed_version

def main():
    print("=" * 70)
    print("MCP Memory Service - Development Environment Check")
    print("=" * 70)

    has_error = False

    # Check 1: Editable install
    print("\n[1/2] Checking installation mode...")
    is_editable, location = check_editable_install()

    if is_editable is None:
        print("  ❌ CRITICAL: Package not installed")
        print(f"     Location: {location}")
        print("\n  Fix: pip install -e .")
        has_error = True
    elif not is_editable:
        print("  ⚠️  WARNING: Package installed in site-packages (not editable)")
        print(f"     Location: {location}")
        print("\n  This means source code changes won't take effect!")
        print("  Fix: pip uninstall mcp-memory-service && pip install -e .")
        has_error = True
    else:
        print(f"  ✅ OK: Editable install detected")
        print(f"     Location: {location}")

    # Check 2: Version match
    print("\n[2/2] Checking version consistency...")
    versions_match, source_ver, installed_ver = check_version_match()

    if versions_match is None:
        print("  ⚠️  WARNING: Could not determine versions")
        print(f"     Source:    {source_ver}")
        print(f"     Installed: {installed_ver}")
    elif not versions_match:
        print(f"  ❌ CRITICAL: Version mismatch detected!")
        print(f"     Source code: v{source_ver}")
        print(f"     Installed:   v{installed_ver}")
        print("\n  This is the 'stale venv' issue!")
        print("  Fix: pip install -e . --force-reinstall")
        has_error = True
    else:
        print(f"  ✅ OK: Versions match (v{source_ver})")

    print("\n" + "=" * 70)
    if has_error:
        print("❌ Development environment has CRITICAL issues!")
        print("=" * 70)
        sys.exit(1)
    else:
        print("✅ Development environment is correctly configured!")
        print("=" * 70)
        sys.exit(0)

if __name__ == "__main__":
    main()
