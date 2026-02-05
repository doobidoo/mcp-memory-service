#!/usr/bin/env python3
"""
Cross-platform service installer for MCP Memory Service.
Automatically detects the operating system and installs the appropriate service.
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_error(text):
    """Print formatted error text."""
    print(f"\n❌ ERROR: {text}")


def print_info(text):
    """Print formatted info text."""
    print(f"ℹ️  {text}")


def detect_platform():
    """Detect the current platform."""
    system = platform.system().lower()

    platforms = {"windows": "Windows", "darwin": "macOS", "linux": "Linux"}

    return system, platforms.get(system, "Unknown")


def check_python_version():
    """Check if Python version meets requirements."""


def run_platform_installer(platform_name, args):
    """Run the appropriate platform-specific installer."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    scripts_dir = script_dir / "scripts"

    installers = {
        "windows": scripts_dir / "install_windows_service.py",
        "darwin": scripts_dir / "install_macos_service.py",
        "linux": scripts_dir / "install_linux_service.py",
    }

    installer = installers.get(platform_name)

    if not installer:
        print_error(f"No installer available for platform: {platform_name}")
        sys.exit(1)

    if not installer.exists():
        print_error(f"Platform installer not found: {installer}")
        print_info("This installer may not be implemented yet.")

        # For Linux, fall back to the bash script if Python version doesn't exist
        if platform_name == "linux":
            bash_installer = script_dir / "install_service.sh"
            if bash_installer.exists():
                print_info("Falling back to bash installer for Linux...")

                # Make sure the script is executable
                bash_installer.chmod(0o755)

                # Run the bash script
                try:
                    subprocess.run([str(bash_installer)], check=True)
                    return
                except subprocess.CalledProcessError as e:
                    print_error(f"Installation failed: {e}")
                    sys.exit(1)

        sys.exit(1)

    # Build command with arguments
    cmd = [sys.executable, str(installer)]

    # Pass through command-line arguments
    if args.command:
        cmd.extend(["--command", args.command])
    if args.uninstall:
        cmd.append("--uninstall")
    if args.start:
        cmd.append("--start")
    if args.stop:
        cmd.append("--stop")
    if args.status:
        cmd.append("--status")
    if args.user:
        cmd.append("--user")
    if args.system:
        cmd.append("--system")

    # Run the platform-specific installer
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print_error(f"Installation failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print_error(f"Could not run installer: {installer}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Cross-platform service installer for MCP Memory Service")

    # Service operations
    parser.add_argument(
        "--command", choices=["install", "uninstall", "start", "stop", "restart", "status"], help="Service command to execute"
    )
    parser.add_argument("--uninstall", action="store_true", help="Uninstall the service")
    parser.add_argument("--start", action="store_true", help="Start the service after installation")
    parser.add_argument("--stop", action="store_true", help="Stop the service")
    parser.add_argument("--status", action="store_true", help="Check service status")

    # Installation options
    parser.add_argument("--user", action="store_true", help="Install as user service (default)")
    parser.add_argument("--system", action="store_true", help="Install as system service (requires admin)")

    args = parser.parse_args()

    # Print header
    print_header("MCP Memory Service - Cross-Platform Installer")

    # Check Python version
    check_python_version()

    # Detect platform
    platform_name, platform_display = detect_platform()
    print_info(f"Detected platform: {platform_display}")
    print_info(f"Python version: {sys.version.split()[0]}")

    # Check for virtual environment
    if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
        print_info(f"Virtual environment: {sys.prefix}")
    else:
        print_info("⚠️  Not running in a virtual environment (recommended)")

    # Run platform-specific installer
    print_info(f"Running {platform_display} service installer...")
    run_platform_installer(platform_name, args)


if __name__ == "__main__":
    main()
