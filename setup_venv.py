#!/usr/bin/env python3
"""
Virtual Environment Setup Script for EchoVault
Ensures proper environment configuration before running the project
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# Color codes for terminal output
RED = '\033[91m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_colored(text, color):
    """Print colored text to terminal."""
    print(f"{color}{text}{RESET}")

def check_virtual_env():
    """Check if we're running in a virtual environment."""
    # Check for common virtual environment indicators
    in_venv = (
        hasattr(sys, 'real_prefix') or  # virtualenv
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or  # venv
        os.environ.get('VIRTUAL_ENV') is not None or  # Both
        os.environ.get('CONDA_DEFAULT_ENV') is not None  # Conda
    )
    
    return in_venv

def get_python_version():
    """Get current Python version."""
    return sys.version_info

def check_python_compatibility():
    """Check if Python version is compatible."""
    version = get_python_version()
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        return False, f"Python {version.major}.{version.minor}"
    return True, f"Python {version.major}.{version.minor}"

def detect_existing_venv():
    """Detect existing virtual environments in the project."""
    venv_dirs = ['.venv', 'venv', 'env', '.env']
    found = []
    
    for venv_dir in venv_dirs:
        if Path(venv_dir).exists() and Path(venv_dir).is_dir():
            # Check if it's a valid venv
            activate_script = get_activate_script(venv_dir)
            if activate_script and Path(activate_script).exists():
                found.append(venv_dir)
    
    return found

def get_activate_script(venv_dir):
    """Get the activation script path based on OS."""
    system = platform.system().lower()
    
    if system == 'windows':
        return os.path.join(venv_dir, 'Scripts', 'activate.bat')
    else:
        return os.path.join(venv_dir, 'bin', 'activate')

def create_virtual_env(venv_name='.venv'):
    """Create a new virtual environment."""
    print_colored(f"\nCreating virtual environment: {venv_name}", BLUE)
    
    try:
        subprocess.check_call([sys.executable, '-m', 'venv', venv_name])
        print_colored(f"✓ Virtual environment created successfully!", GREEN)
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"✗ Failed to create virtual environment: {e}", RED)
        return False

def show_activation_instructions(venv_dir='.venv'):
    """Show how to activate the virtual environment."""
    system = platform.system().lower()
    
    print_colored("\n📋 Activation Instructions:", BLUE)
    print("─" * 50)
    
    if system == 'windows':
        print(f"Windows PowerShell:  {venv_dir}\\Scripts\\Activate.ps1")
        print(f"Windows CMD:         {venv_dir}\\Scripts\\activate.bat")
    else:
        print(f"Linux/macOS:         source {venv_dir}/bin/activate")
    
    print("\nFor Conda users:")
    print("Create:              conda create -n echovault python=3.11")
    print("Activate:            conda activate echovault")
    print("─" * 50)

def install_requirements():
    """Install project requirements."""
    requirements_files = ['requirements.txt', 'requirements-echovault.txt']
    
    print_colored("\n📦 Installing Requirements:", BLUE)
    
    for req_file in requirements_files:
        if Path(req_file).exists():
            print(f"\nInstalling {req_file}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_file])
                print_colored(f"✓ {req_file} installed successfully!", GREEN)
            except subprocess.CalledProcessError as e:
                print_colored(f"✗ Failed to install {req_file}: {e}", RED)
                return False
    
    # Install development dependencies
    dev_deps = ['pytest', 'pytest-asyncio', 'alembic']
    print("\nInstalling development dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + dev_deps)
        print_colored("✓ Development dependencies installed!", GREEN)
    except subprocess.CalledProcessError:
        print_colored("⚠ Some development dependencies failed to install", YELLOW)
    
    return True

def main():
    """Main setup function."""
    print_colored("🐍 EchoVault Virtual Environment Setup", BLUE)
    print("=" * 50)
    
    # Check Python version
    compatible, version_str = check_python_compatibility()
    if not compatible:
        print_colored(f"\n✗ {version_str} is not compatible!", RED)
        print_colored("  EchoVault requires Python 3.11 or higher", YELLOW)
        print("\nRecommended setup:")
        print("  1. Install Python 3.11+")
        print("  2. Run: python3.11 setup_venv.py")
        sys.exit(1)
    else:
        print_colored(f"\n✓ {version_str} - Compatible", GREEN)
    
    # Check if in virtual environment
    in_venv = check_virtual_env()
    
    if in_venv:
        print_colored("\n✓ Already in a virtual environment!", GREEN)
        venv_path = os.environ.get('VIRTUAL_ENV', 'Unknown')
        print(f"  Path: {venv_path}")
        
        # Ask if user wants to install requirements
        response = input("\nInstall/update requirements? (y/n): ").lower()
        if response == 'y':
            install_requirements()
    else:
        print_colored("\n⚠ Not in a virtual environment!", YELLOW)
        
        # Check for existing venv
        existing = detect_existing_venv()
        
        if existing:
            print(f"\nFound existing virtual environment(s): {', '.join(existing)}")
            show_activation_instructions(existing[0])
        else:
            # Offer to create one
            response = input("\nCreate a virtual environment? (y/n): ").lower()
            if response == 'y':
                venv_name = input("Virtual environment name (default: .venv): ").strip() or '.venv'
                if create_virtual_env(venv_name):
                    show_activation_instructions(venv_name)
                    print_colored("\n⚠ Please activate the virtual environment and run this script again to install requirements!", YELLOW)
            else:
                print_colored("\n⚠ Virtual environment is required for EchoVault!", YELLOW)
                show_activation_instructions()
    
    print_colored("\n✅ Setup check complete!", GREEN)

if __name__ == "__main__":
    main() 