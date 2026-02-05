#!/usr/bin/env python3
"""
Shared utilities for cross-platform service installation.
Provides common functionality for all platform-specific service installers.
"""

import json
import os
import platform
import secrets
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent


def get_python_executable() -> str:
    """Get the current Python executable path."""
    return sys.executable


def get_venv_path() -> Path | None:
    """Get the virtual environment path if in a venv."""
    if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
        return Path(sys.prefix)
    return None


def generate_api_key() -> str:
    """Generate a secure API key for the service."""
    return f"mcp-{secrets.token_hex(16)}"


def get_service_paths() -> dict[str, Path]:
    """Get common paths used by the service."""
    project_root = get_project_root()

    paths = {
        "project_root": project_root,
        "scripts_dir": project_root / "scripts",
        "src_dir": project_root / "src",
        "venv_dir": get_venv_path() or project_root / "venv",
        "config_dir": Path.home() / ".mcp_memory_service",
        "log_dir": Path.home() / ".mcp_memory_service" / "logs",
    }

    # Ensure config and log directories exist
    paths["config_dir"].mkdir(parents=True, exist_ok=True)
    paths["log_dir"].mkdir(parents=True, exist_ok=True)

    return paths


def get_service_environment() -> dict[str, str]:
    """Get environment variables for the service."""
    paths = get_service_paths()
    venv_path = get_venv_path()

    env = {
        "PYTHONPATH": str(paths["src_dir"]),
        "MCP_MEMORY_STORAGE_BACKEND": os.getenv("MCP_MEMORY_STORAGE_BACKEND", "sqlite_vec"),
        "MCP_HTTP_ENABLED": "true",
        "MCP_HTTP_HOST": "0.0.0.0",
        "MCP_HTTP_PORT": "8000",
        "MCP_HTTPS_ENABLED": "true",
        "MCP_MDNS_ENABLED": "true",
        "MCP_MDNS_SERVICE_NAME": "memory",
        "MCP_CONSOLIDATION_ENABLED": "true",
    }

    # Add venv to PATH if available
    if venv_path:
        bin_dir = venv_path / ("Scripts" if platform.system() == "Windows" else "bin")
        current_path = os.environ.get("PATH", "")
        env["PATH"] = f"{bin_dir}{os.pathsep}{current_path}"

    return env


def save_service_config(config: dict[str, any]) -> Path:
    """Save service configuration to file."""
    paths = get_service_paths()
    config_file = paths["config_dir"] / "service_config.json"

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    return config_file


def load_service_config() -> dict[str, any] | None:
    """Load service configuration from file."""
    paths = get_service_paths()
    config_file = paths["config_dir"] / "service_config.json"

    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return None


def check_dependencies() -> tuple[bool, str]:
    """Check if all required dependencies are installed."""
    try:
        # Check Python version

        # Check if in virtual environment (recommended)
        if not get_venv_path():
            print("WARNING: Not running in a virtual environment")

        # Check core dependencies
        required_modules = [
            "mcp",
            "chromadb",
            "sentence_transformers",
        ]

        missing = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)

        if missing:
            return False, f"Missing dependencies: {', '.join(missing)}"

        return True, "All dependencies satisfied"

    except Exception as e:
        return False, f"Error checking dependencies: {str(e)}"


def get_service_command() -> list:
    """Get the command to run the service."""
    paths = get_service_paths()
    python_exe = get_python_executable()

    # Use HTTP server script if available, otherwise fall back to main server
    http_server = paths["scripts_dir"] / "run_http_server.py"
    main_server = paths["scripts_dir"] / "run_memory_server.py"

    if http_server.exists():
        return [python_exe, str(http_server)]
    elif main_server.exists():
        return [python_exe, str(main_server)]
    else:
        # Fall back to module execution
        return [python_exe, "-m", "mcp_memory_service.server"]


def test_service_startup() -> tuple[bool, str]:
    """Test if the service can start successfully."""
    try:
        cmd = get_service_command()
        env = os.environ.copy()
        env.update(get_service_environment())

        # Try to start the service briefly
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Give it a moment to start
        import time

        time.sleep(2)

        # Check if process is still running
        if proc.poll() is None:
            # Service started successfully, terminate it
            proc.terminate()
            proc.wait(timeout=5)
            return True, "Service starts successfully"
        else:
            # Process exited, get error
            stdout, stderr = proc.communicate()
            error_msg = stderr or stdout or "Unknown error"
            return False, f"Service failed to start: {error_msg}"

    except Exception as e:
        return False, f"Error testing service: {str(e)}"


def print_service_info(api_key: str, platform_specific_info: dict[str, str] = None):
    """Print service installation information."""
    print("\n" + "=" * 60)
    print("✅ MCP Memory Service Installed Successfully!")
    print("=" * 60)

    print("\n📌 Service Information:")
    print(f"  API Key: {api_key}")
    print("  Dashboard: https://localhost:8000")
    print("  API Docs: https://localhost:8000/api/docs")
    print("  Health Check: https://localhost:8000/api/health")

    if platform_specific_info:
        print("\n🖥️  Platform-Specific Commands:")
        for key, value in platform_specific_info.items():
            print(f"  {key}: {value}")

    print("\n📝 Configuration:")
    config = load_service_config()
    if config:
        print(f"  Config File: {get_service_paths()['config_dir'] / 'service_config.json'}")
        print(f"  Log Directory: {get_service_paths()['log_dir']}")

    print("\n" + "=" * 60)


def is_admin() -> bool:
    """Check if running with administrative privileges."""
    system = platform.system()

    if system == "Windows":
        try:
            import ctypes

            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except:
            return False
    else:  # Unix-like systems
        return os.geteuid() == 0


def require_admin(message: str = None):
    """Ensure the script is running with admin privileges."""
    if not is_admin():
        system = platform.system()
        if message:
            print(f"\n❌ {message}")

        if system == "Windows":
            print("\nPlease run this script as Administrator:")
            print("  Right-click on your terminal and select 'Run as Administrator'")
        else:
            print("\nPlease run this script with sudo:")
            sys.argv[0]
            print(f"  sudo {' '.join(sys.argv)}")

        sys.exit(1)
