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
Enhanced fix script for sitecustomize.py recursion issues.
This script replaces the problematic sitecustomize.py with a fixed version
that works on Linux WSL2 with CUDA 12.4 and other platforms.
"""

import os
import platform
import shutil
import site
import sys


def print_info(text):
    """Print formatted info text."""
    print(f"[INFO] {text}")


def print_error(text):
    """Print formatted error text."""
    print(f"[ERROR] {text}")


def print_success(text):
    """Print formatted success text."""
    print(f"[SUCCESS] {text}")


def print_warning(text):
    """Print formatted warning text."""
    print(f"[WARNING] {text}")


def fix_sitecustomize():
    """Fix the sitecustomize.py file to prevent recursion."""
    # Get site-packages directory
    site_packages = site.getsitepackages()[0]

    # Path to sitecustomize.py
    sitecustomize_path = os.path.join(site_packages, "sitecustomize.py")

    # Check if file exists
    if not os.path.exists(sitecustomize_path):
        print_error(f"sitecustomize.py not found at {sitecustomize_path}")
        return False

    # Create backup
    backup_path = sitecustomize_path + ".bak"
    if not os.path.exists(backup_path):
        print_info(f"Creating backup of sitecustomize.py at {backup_path}")
        shutil.copy2(sitecustomize_path, backup_path)
        print_success(f"Backup created at {backup_path}")
    else:
        print_warning(f"Backup already exists at {backup_path}")

    # Create fixed sitecustomize.py
    print_info(f"Creating fixed sitecustomize.py at {sitecustomize_path}")

    # Detect system for platform-specific fixes
    system = platform.system().lower()
    is_wsl = "microsoft" in platform.release().lower() if system == "linux" else False

    # Create content based on platform
    if is_wsl:
        # Special content for WSL with enhanced error handling
        content = """# Fixed sitecustomize.py to prevent recursion issues on WSL
# Import standard library modules first to avoid recursion
import sys
import os
import importlib.util
import importlib.machinery
import warnings

# Disable warnings to reduce noise
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

# Print debug info to stderr to avoid interfering with MCP protocol
print("sitecustomize.py loaded", file=sys.stderr)

# Set environment variables to prevent pip from installing dependencies
os.environ["PIP_NO_DEPENDENCIES"] = "1"
os.environ["PIP_NO_INSTALL"] = "1"

# Disable automatic torch installation
os.environ["PYTORCH_IGNORE_DUPLICATE_MODULE_REGISTRATION"] = "1"

# Create a custom import hook to prevent automatic installation
class PreventAutoInstallImportHook:
    def __init__(self):
        self.blocked_packages = ['torch', 'torchvision', 'torchaudio', 'torchao']
        # Keep track of packages we've already tried to find to prevent recursion
        self.checked_packages = set()

    def find_spec(self, fullname, path, target=None):
        # Prevent recursion by checking if we've already tried to find this package
        if fullname in self.checked_packages:
            return None

        # Check if this is a package we want to block
        if any(fullname.startswith(pkg) for pkg in self.blocked_packages):
            # Add to checked packages to prevent recursion
            self.checked_packages.add(fullname)

            # Try to find the package directly using the loader
            try:
                # Try to find the module directly
                loader = importlib.machinery.PathFinder.find_spec(fullname, path)
                if loader is not None:
                    return loader
            except Exception:
                pass

            # If not found, print a warning and return None
            print(f"WARNING: Blocked automatic installation of {fullname}", file=sys.stderr)
            return None

        # Return None to let the normal import system handle it
        return None

# Register the import hook
sys.meta_path.insert(0, PreventAutoInstallImportHook())

# Disable distutils setup hooks that can cause recursion
try:
    import setuptools
    setuptools._distutils_hack = None
except Exception:
    pass

# Disable _distutils_hack completely
sys.modules['_distutils_hack'] = None
"""
    else:
        # Standard content for other platforms
        content = """# Fixed sitecustomize.py to prevent recursion issues
import sys
import os
import importlib.util
import importlib.machinery

# Print debug info
print("sitecustomize.py loaded", file=sys.stderr)

# Set environment variables to prevent pip from installing dependencies
os.environ["PIP_NO_DEPENDENCIES"] = "1"
os.environ["PIP_NO_INSTALL"] = "1"

# Create a custom import hook to prevent automatic installation
class PreventAutoInstallImportHook:
    def __init__(self):
        self.blocked_packages = ['torch', 'torchvision', 'torchaudio']
        # Keep track of packages we've already tried to find to prevent recursion
        self.checked_packages = set()

    def find_spec(self, fullname, path, target=None):
        # Prevent recursion by checking if we've already tried to find this package
        if fullname in self.checked_packages:
            return None

        # Check if this is a package we want to block
        if any(fullname.startswith(pkg) for pkg in self.blocked_packages):
            # Add to checked packages to prevent recursion
            self.checked_packages.add(fullname)

            # Try to find the package directly using the loader
            try:
                # Try to find the module directly
                loader = importlib.machinery.PathFinder.find_spec(fullname, path)
                if loader is not None:
                    return loader
            except Exception:
                pass

            # If not found, print a warning and return None
            print(f"WARNING: Blocked automatic installation of {fullname}", file=sys.stderr)
            return None

        # Return None to let the normal import system handle it
        return None

# Register the import hook
sys.meta_path.insert(0, PreventAutoInstallImportHook())
"""

    # Write the content to the file
    with open(sitecustomize_path, "w") as f:
        f.write(content)

    print_success(f"Fixed sitecustomize.py created at {sitecustomize_path}")

    # Additional fix for distutils on WSL
    if is_wsl:
        try:
            # Try to fix _distutils_hack.py
            distutils_hack_path = os.path.join(site_packages, "_distutils_hack", "__init__.py")
            if os.path.exists(distutils_hack_path):
                print_info(f"Fixing _distutils_hack at {distutils_hack_path}")

                # Create backup
                hack_backup_path = distutils_hack_path + ".bak"
                if not os.path.exists(hack_backup_path):
                    shutil.copy2(distutils_hack_path, hack_backup_path)
                    print_success(f"Backup created at {hack_backup_path}")

                # Read the file
                with open(distutils_hack_path) as f:
                    content = f.read()

                # Modify the content to disable the problematic parts
                content = content.replace("def do_override():", "def do_override():\n    return")

                # Write the modified content
                with open(distutils_hack_path, "w") as f:
                    f.write(content)

                print_success(f"Fixed _distutils_hack at {distutils_hack_path}")
        except Exception as e:
            print_warning(f"Could not fix _distutils_hack: {e}")

    return True


def main():
    """Main function."""
    print_info("Enhanced fix for sitecustomize.py to prevent recursion issues")

    if fix_sitecustomize():
        print_success("sitecustomize.py fixed successfully")
    else:
        print_error("Failed to fix sitecustomize.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
