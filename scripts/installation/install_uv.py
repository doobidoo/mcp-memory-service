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
Script to install UV package manager
"""

import os
import platform
import subprocess
import sys


def main():
    print("Installing UV package manager...")

    try:
        # Install UV using pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])

        print("UV installed successfully!")
        print("You can now use UV for faster dependency management:")
        print("  uv pip install -r requirements.txt")

        # Create shortcut script
        system = platform.system().lower()
        if system == "windows":
            # Create .bat file for Windows
            with open("uv-run.bat", "w") as f:
                f.write("@echo off\n")
                f.write("python -m uv run memory %*\n")
            print("Created uv-run.bat shortcut")
        else:
            # Create shell script for Unix-like systems
            with open("uv-run.sh", "w") as f:
                f.write("#!/bin/sh\n")
                f.write('python -m uv run memory "$@"\n')

            # Make it executable
            try:
                os.chmod("uv-run.sh", 0o755)
            except:
                pass
            print("Created uv-run.sh shortcut")

    except subprocess.SubprocessError as e:
        print(f"Error installing UV: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
