#!/usr/bin/env python3
"""
Installation script for EchoVault Memory Service with cross-platform compatibility.
This script guides users through the installation process with the appropriate
dependencies for their platform.
"""
import os
import sys
import platform
import subprocess
import argparse
import shutil
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)

def print_step(step, text):
    """Print a formatted step."""
    print(f"\n[{step}] {text}")

def print_info(text):
    """Print formatted info text."""
    print(f"  → {text}")

def print_error(text):
    """Print formatted error text."""
    print(f"  ❌ ERROR: {text}")

def print_success(text):
    """Print formatted success text."""
    print(f"  ✅ {text}")

def print_warning(text):
    """Print formatted warning text."""
    print(f"  ⚠️  {text}")

def install_echovault_dependencies():
    """Install EchoVault-specific dependencies."""
    print_step("1", "Installing EchoVault dependencies")
    
    try:
        # Install requirements
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements-echovault.txt'
        ])
        print_success("EchoVault dependencies installed successfully")
        return True
    except subprocess.SubprocessError as e:
        print_error(f"Failed to install EchoVault dependencies: {e}")
        return False

def create_env_file():
    """Create .env file from example if it doesn't exist."""
    print_step("2", "Creating .env file")
    
    if os.path.exists(".env"):
        print_info(".env file already exists")
        return True
    
    try:
        shutil.copy(".env.example", ".env")
        print_success("Created .env file from example")
        print_info("Please edit .env file with your credentials")
        return True
    except Exception as e:
        print_error(f"Failed to create .env file: {e}")
        return False

def configure_echovault():
    """Configure EchoVault with environment variables."""
    print_step("3", "Configuring EchoVault")
    
    # Check for required environment variables
    required_vars = [
        "NEON_DSN",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "R2_ENDPOINT",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print_warning("The following environment variables are not set:")
        for var in missing_vars:
            print_info(f"  - {var}")
        print_info("Please set these variables in your .env file or environment")
        return False
    
    print_success("EchoVault configuration complete")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Install EchoVault Memory Service")
    parser.add_argument('--force', action='store_true', help='Force installation even if dependencies are already installed')
    args = parser.parse_args()
    
    print_header("EchoVault Memory Service Installation")
    
    # Install dependencies
    if args.force or not install_echovault_dependencies():
        print_warning("Failed to install EchoVault dependencies")
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Configure EchoVault
    configure_echovault()
    
    print_header("Installation Complete")
    print_info("You can now run the EchoVault Memory Service using:")
    print_info("  python -m src.mcp_memory_service.server")
    print_info("Or with the memory command if installed:")
    print_info("  memory")

if __name__ == "__main__":
    main()