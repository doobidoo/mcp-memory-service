#!/usr/bin/env python3
"""
Installation script for MCP Memory Service with cross-platform compatibility.
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
from typing import Tuple, Dict, Any, Optional
import re
import importlib.util

def is_python_version_at_least(major, minor):
    """Check if current Python version is at least the specified version.

    Args:
        major: Major version number
        minor: Minor version number

    Returns:
        bool: True if current Python version >= specified version
    """
    return sys.version_info >= (major, minor)

def get_python_version_string():
    """Get Python version as a string (e.g., '3.12').

    Returns:
        str: Python version string
    """
    return f"{sys.version_info.major}.{sys.version_info.minor}"

def get_package_version():
    """Get the current package version from pyproject.toml.

    Returns:
        str: The version string from pyproject.toml or fallback version
    """
    fallback_version = "7.2.0"

    try:
        # Get path to pyproject.toml relative to this script
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"

        if not pyproject_path.exists():
            print_warning(f"pyproject.toml not found at {pyproject_path}, using fallback version {fallback_version}")
            return fallback_version

        with open(pyproject_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract version using regex - matches standard pyproject.toml format
        version_pattern = r'^version\s*=\s*["\']([^"\'\n]+)["\']'
        version_match = re.search(version_pattern, content, re.MULTILINE)

        if version_match:
            version = version_match.group(1).strip()
            if version:  # Ensure non-empty version
                return version
            else:
                print_warning("Empty version found in pyproject.toml, using fallback")
                return fallback_version
        else:
            print_warning("Version not found in pyproject.toml, using fallback")
            return fallback_version

    except (OSError, IOError) as e:
        print_warning(f"Failed to read pyproject.toml: {e}, using fallback version {fallback_version}")
        return fallback_version
    except Exception as e:
        print_warning(f"Unexpected error parsing version: {e}, using fallback version {fallback_version}")
        return fallback_version

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

def run_command_safe(cmd, success_msg=None, error_msg=None, silent=False,
                     timeout=None, fallback_in_venv=False):
    """
    Run a subprocess command with standardized error handling.

    Args:
        cmd: Command to run (list of strings)
        success_msg: Message to print on success
        error_msg: Custom error message
        silent: If True, suppress stdout/stderr
        timeout: Command timeout in seconds
        fallback_in_venv: If True and command fails, warn instead of error when in virtual environment

    Returns:
        tuple: (success: bool, result: subprocess.CompletedProcess or None)
    """
    # Validate command input
    if not cmd or not isinstance(cmd, (list, tuple)):
        print_error("Invalid command: must be a non-empty list or tuple")
        return False, None

    if not all(isinstance(arg, (str, int, float)) for arg in cmd if arg is not None):
        print_error("Invalid command arguments: all arguments must be strings, numbers, or None")
        return False, None

    # Filter out None values and convert to strings
    cmd_clean = [str(arg) for arg in cmd if arg is not None]
    if not cmd_clean:
        print_error("Command is empty after filtering")
        return False, None

    try:
        kwargs = {'capture_output': False, 'text': True}
        if silent:
            kwargs.update({'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL})
        if timeout:
            kwargs['timeout'] = timeout

        result = subprocess.run(cmd_clean, check=True, **kwargs)
        if success_msg:
            print_success(success_msg)
        return True, result
    except subprocess.TimeoutExpired as e:
        if error_msg:
            timeout_msg = error_msg
        elif hasattr(e, 'timeout') and e.timeout:
            timeout_msg = f"Command timed out after {e.timeout}s"
        else:
            timeout_msg = "Command timed out"
        print_error(timeout_msg)
        return False, None
    except subprocess.CalledProcessError as e:
        if fallback_in_venv:
            in_venv = sys.prefix != sys.base_prefix
            if in_venv:
                fallback_msg = error_msg or "Command failed, but you're in a virtual environment. If you're using an alternative package manager, this may be normal."
                print_warning(fallback_msg)
                print_warning("Note: Installation may not have succeeded. Please verify manually if needed.")
                return True, None  # Proceed anyway in venv with warning

        if error_msg:
            print_error(error_msg)
        else:
            # Safe command formatting for error messages
            cmd_str = ' '.join(f'"{arg}"' if ' ' in str(arg) else str(arg) for arg in cmd_clean)
            print_error(f"Command failed (exit code {e.returncode}): {cmd_str}")
        return False, None
    except FileNotFoundError:
        if error_msg:
            print_error(error_msg)
        else:
            print_error(f"Command not found: {cmd_clean[0]}")
        return False, None
    except PermissionError:
        permission_msg = error_msg or f"Permission denied executing: {cmd_clean[0]}"
        print_error(permission_msg)
        return False, None

def install_package_safe(package, success_msg=None, error_msg=None, fallback_in_venv=True):
    """
    Install a Python package with standardized error handling.

    Args:
        package: Package name or requirement string
        success_msg: Message to print on success
        error_msg: Custom error message
        fallback_in_venv: If True, warn instead of error when in virtual environment

    Returns:
        bool: True if installation succeeded OR if fallback was applied (see warning messages)
    """
    pip_module_available = importlib.util.find_spec("pip") is not None
    uv_path = shutil.which("uv")
    if pip_module_available:
        cmd = [sys.executable, '-m', 'pip', 'install', package]
    elif uv_path:
        # uv environments commonly omit pip; use uv to install into the current interpreter
        cmd = [uv_path, 'pip', 'install', '--python', sys.executable, package]
    else:
        print_error("Neither pip nor uv detected. Cannot install packages.")
        return False
    default_success = success_msg or f"{package} installed successfully"
    default_error = error_msg or f"Failed to install {package}"

    if fallback_in_venv:
        default_error += ". If you're using an alternative package manager like uv, please install manually."

    success, _ = run_command_safe(
        cmd,
        success_msg=default_success,
        error_msg=default_error,
        silent=True,
        fallback_in_venv=fallback_in_venv
    )
    return success

def detect_system():
    """Detect the system architecture and platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    is_windows = system == "windows"
    is_macos = system == "darwin"
    is_linux = system == "linux"
    is_arm = machine in ("arm64", "aarch64")
    is_x86 = machine in ("x86_64", "amd64", "x64")
    
    print_info(f"System: {platform.system()} {platform.release()}")
    print_info(f"Architecture: {machine}")
    print_info(f"Python: {python_version}")
    
    # Check for virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print_warning("Not running in a virtual environment. It's recommended to install in a virtual environment.")
    else:
        print_info(f"Virtual environment: {sys.prefix}")
    
    # Check for Homebrew PyTorch installation
    has_homebrew_pytorch = False
    homebrew_pytorch_version = None
    if is_macos:
        try:
            # Check if pytorch is installed via brew
            result = subprocess.run(
                ['brew', 'list', 'pytorch', '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                has_homebrew_pytorch = True
                # Extract version from output
                version_line = result.stdout.strip()
                homebrew_pytorch_version = version_line.split()[1] if len(version_line.split()) > 1 else "Unknown"
                print_info(f"Detected Homebrew PyTorch installation: {homebrew_pytorch_version}")
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    return {
        "system": system,
        "machine": machine,
        "python_version": python_version,
        "is_windows": is_windows,
        "is_macos": is_macos,
        "is_linux": is_linux,
        "is_arm": is_arm,
        "is_x86": is_x86,
        "in_venv": in_venv,
        "has_homebrew_pytorch": has_homebrew_pytorch,
        "homebrew_pytorch_version": homebrew_pytorch_version
    }

def detect_gpu():
    """Detect GPU and acceleration capabilities.

    Wrapper function that uses the shared GPU detection module.
    """
    # Lazy import to avoid module-level import issues in tests
    try:
        from mcp_memory_service.utils.gpu_detection import detect_gpu as shared_detect_gpu
    except ImportError:
        # Fallback for scripts directory context
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.mcp_memory_service.utils.gpu_detection import detect_gpu as shared_detect_gpu

    system_info = detect_system()

    # Use shared GPU detection module
    gpu_info = shared_detect_gpu(system_info)

    # Print GPU information (maintain installer output format)
    if gpu_info.get("has_cuda"):
        cuda_version = gpu_info.get("cuda_version")
        print_info(f"CUDA detected: {cuda_version or 'Unknown version'}")
    if gpu_info.get("has_rocm"):
        rocm_version = gpu_info.get("rocm_version")
        print_info(f"ROCm detected: {rocm_version or 'Unknown version'}")
    if gpu_info.get("has_mps"):
        print_info("Apple Metal Performance Shaders (MPS) detected")
    if gpu_info.get("has_directml"):
        directml_version = gpu_info.get("directml_version")
        if directml_version:
            print_info(f"DirectML detected: {directml_version}")
        else:
            print_info("DirectML detected")

    if not (gpu_info.get("has_cuda") or gpu_info.get("has_rocm") or
            gpu_info.get("has_mps") or gpu_info.get("has_directml")):
        print_info("No GPU acceleration detected, will use CPU-only mode")

    return gpu_info

def check_dependencies():
    """Check for required dependencies.
    
    Note on package managers:
    - Traditional virtual environments (venv, virtualenv) include pip by default
    - Alternative package managers like uv may not include pip or may manage packages differently
    - We attempt multiple detection methods for pip and only fail if:
      a) We're not in a virtual environment, or
      b) We can't detect pip AND can't install dependencies
    
    We proceed with installation even if pip isn't detected when in a virtual environment,
    assuming an alternative package manager (like uv) is handling dependencies.
    
    Returns:
        bool: True if all dependencies are met, False otherwise.
    """
    print_step("2", "Checking dependencies")
    
    # Check for pip
    pip_installed = False
    
    # Try subprocess check first
    success, _ = run_command_safe(
        [sys.executable, '-m', 'pip', '--version'],
        success_msg="pip is installed",
        silent=True,
        fallback_in_venv=True
    )

    if success:
        pip_installed = True
    else:
        # Fallback to import check
        try:
            import pip
            pip_installed = True
            print_info(f"pip is installed: {pip.__version__}")
        except ImportError:
            # Check if we're in a virtual environment
            in_venv = sys.prefix != sys.base_prefix
            if in_venv:
                print_warning("pip could not be detected, but you're in a virtual environment. "
                            "If you're using uv or another alternative package manager, this is normal. "
                            "Continuing installation (will use uv where needed)...")
                pip_installed = True  # Proceed anyway
            else:
                print_error("pip is not installed. Please install pip first.")
                return False
    
    # Check for setuptools
    try:
        import setuptools
        print_info(f"setuptools is installed: {setuptools.__version__}")
    except ImportError:
        print_warning("setuptools is not installed. Will attempt to install it.")
        # If pip is available, use it to install setuptools
        if pip_installed:
            success = install_package_safe("setuptools")
            if not success:
                return False
        else:
            # Should be unreachable since pip_installed would only be False if we returned earlier
            print_error("Cannot install setuptools without pip. Please install setuptools manually.")
            return False
    
    # Check for wheel
    try:
        import wheel
        print_info(f"wheel is installed: {wheel.__version__}")
    except ImportError:
        print_warning("wheel is not installed. Will attempt to install it.")
        # If pip is available, use it to install wheel
        if pip_installed:
            success = install_package_safe("wheel")
            if not success:
                return False
        else:
            # Should be unreachable since pip_installed would only be False if we returned earlier
            print_error("Cannot install wheel without pip. Please install wheel manually.")
            return False
    
    return True

def install_pytorch_platform_specific(system_info, gpu_info):
    """Install PyTorch with platform-specific configurations."""
    if system_info["is_windows"]:
        return install_pytorch_windows(gpu_info)
    elif system_info["is_macos"] and system_info["is_x86"]:
        return install_pytorch_macos_intel()
    else:
        # For other platforms, let the regular installer handle it
        return True

def install_pytorch_macos_intel():
    """Install PyTorch specifically for macOS with Intel CPUs."""
    print_step("3a", "Installing PyTorch for macOS Intel CPU")
    
    # Use the versions known to work well on macOS Intel and with Python 3.13+
    try:
        # For Python 3.13+, we need newer PyTorch versions
        python_version = sys.version_info
        
        if python_version >= (3, 13):
            # For Python 3.13+, try to install latest compatible version
            print_info(f"Installing PyTorch for macOS Intel (Python {python_version.major}.{python_version.minor})...")
            print_info("Attempting to install latest PyTorch compatible with Python 3.13...")
            
            # Try to install without version specifiers to get latest compatible version
            cmd = [
                sys.executable, '-m', 'pip', 'install',
                "torch", "torchvision", "torchaudio"
            ]
            success, _ = run_command_safe(
                cmd,
                success_msg="Latest PyTorch installed successfully",
                silent=False
            )

            if success:
                st_version = "3.0.0"  # Newer sentence-transformers for newer PyTorch
            else:
                print_warning("Failed to install latest PyTorch, trying fallback version...")
                # Fallback to a specific version
                torch_version = "2.1.0"
                torch_vision_version = "0.16.0"
                torch_audio_version = "2.1.0"
                st_version = "3.0.0"

                print_info(f"Trying fallback to PyTorch {torch_version}...")

                cmd = [
                    sys.executable, '-m', 'pip', 'install',
                    f"torch=={torch_version}",
                    f"torchvision=={torch_vision_version}",
                    f"torchaudio=={torch_audio_version}"
                ]
                success, _ = run_command_safe(
                    cmd,
                    success_msg=f"PyTorch {torch_version} installed successfully",
                    error_msg="Failed to install PyTorch fallback version",
                    silent=False
                )
                if not success:
                    return False
        else:
            # Use traditional versions for older Python
            torch_version = "1.13.1"
            torch_vision_version = "0.14.1"
            torch_audio_version = "0.13.1"
            st_version = "2.2.2"
            
            print_info(f"Installing PyTorch {torch_version} for macOS Intel (Python {python_version.major}.{python_version.minor})...")
            
            # Install PyTorch first with compatible version
            packages = [f"torch=={torch_version}", f"torchvision=={torch_vision_version}", f"torchaudio=={torch_audio_version}"]
            success, _ = run_command_safe(
                [sys.executable, '-m', 'pip', 'install'] + packages,
                success_msg=f"PyTorch {torch_version} installed successfully"
            )
            if not success:
                raise RuntimeError(f"Failed to install PyTorch {torch_version}")
        
        # Install a compatible version of sentence-transformers
        print_info(f"Installing sentence-transformers {st_version}...")
        success, _ = run_command_safe(
            [sys.executable, '-m', 'pip', 'install', f"sentence-transformers=={st_version}"],
            success_msg=f"sentence-transformers {st_version} installed successfully"
        )
        if not success:
            raise RuntimeError(f"Failed to install sentence-transformers {st_version}")
        
        print_success(f"PyTorch {torch_version} and sentence-transformers {st_version} installed successfully for macOS Intel")
        return True
    except RuntimeError as e:
        print_error(f"Failed to install PyTorch for macOS Intel: {e}")
        
        # Provide fallback instructions
        if python_version >= (3, 13):
            print_warning("You may need to manually install compatible versions for Python 3.13+ on Intel macOS:")
            print_info("pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0")
            print_info("pip install sentence-transformers==3.0.0")
        else:
            print_warning("You may need to manually install compatible versions for Intel macOS:")
            print_info("pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1")
            print_info("pip install sentence-transformers==2.2.2")
        
        return False

def install_pytorch_windows(gpu_info):
    """Install PyTorch on Windows using the appropriate index URL."""
    print_step("3a", "Installing PyTorch for Windows")
    
    # Determine the appropriate PyTorch index URL based on GPU
    if gpu_info["has_cuda"]:
        # Get CUDA version and determine appropriate index URL
        cuda_version = gpu_info.get("cuda_version", "")
        
        # Extract major version from CUDA version string
        cuda_major = None
        if cuda_version:
            # Try to extract the major version (e.g., "11.8" -> "11")
            try:
                cuda_major = cuda_version.split('.')[0]
            except (IndexError, AttributeError):
                pass
        
        # Default to cu118 if we couldn't determine the version or it's not a common one
        if cuda_major == "12":
            cuda_suffix = "cu121"  # CUDA 12.x
            print_info(f"Detected CUDA {cuda_version}, using cu121 channel")
        elif cuda_major == "11":
            cuda_suffix = "cu118"  # CUDA 11.x
            print_info(f"Detected CUDA {cuda_version}, using cu118 channel")
        elif cuda_major == "10":
            cuda_suffix = "cu102"  # CUDA 10.x
            print_info(f"Detected CUDA {cuda_version}, using cu102 channel")
        else:
            # Default to cu118 as a safe choice for newer NVIDIA GPUs
            cuda_suffix = "cu118"
            print_info(f"Using default cu118 channel for CUDA {cuda_version}")
            
        index_url = f"https://download.pytorch.org/whl/{cuda_suffix}"
    else:
        # CPU-only version
        index_url = "https://download.pytorch.org/whl/cpu"
        print_info("Using CPU-only PyTorch for Windows")
    
    # Install PyTorch with the appropriate index URL
    try:
        # Use a stable version that's known to have Windows wheels
        torch_version = "2.1.0"  # This version has Windows wheels available
        
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            f"torch=={torch_version}",
            f"torchvision=={torch_version}",
            f"torchaudio=={torch_version}",
            f"--index-url={index_url}"
        ]
        
        success, _ = run_command_safe(
            cmd,
            success_msg="PyTorch installed successfully for Windows",
            error_msg="Failed to install PyTorch for Windows",
            silent=False
        )
        if not success:
            return False
        
        # Check if DirectML is needed
        if gpu_info["has_directml"]:
            success = install_package_safe(
                "torch-directml>=0.2.0",
                success_msg="torch-directml installed successfully for DirectML support",
                error_msg="Failed to install torch-directml"
            )
            if not success:
                print_warning("DirectML support may not be available")
            
        print_success("PyTorch installed successfully for Windows")
        return True
    except RuntimeError as e:
        print_error(f"Failed to install PyTorch for Windows: {e}")
        print_warning("You may need to manually install PyTorch using instructions from https://pytorch.org/get-started/locally/")
        return False

def detect_storage_backend_compatibility(system_info, gpu_info):
    """Detect which storage backends are compatible with the current environment."""
    print_step("3a", "Analyzing storage backend compatibility")
    
    compatibility = {
        "cloudflare": {"supported": True, "issues": [], "recommendation": "production"},
        "sqlite_vec": {"supported": True, "issues": [], "recommendation": "development"},
        "chromadb": {"supported": True, "issues": [], "recommendation": "team"},
        "hybrid": {"supported": True, "issues": [], "recommendation": "recommended"}
    }
    
    # Check ChromaDB compatibility issues
    chromadb_issues = []
    
    # macOS Intel compatibility issues
    if system_info["is_macos"] and system_info["is_x86"]:
        chromadb_issues.append("ChromaDB has known installation issues on older macOS Intel systems")
        chromadb_issues.append("May require specific dependency versions")
        compatibility["chromadb"]["recommendation"] = "problematic"
        compatibility["sqlite_vec"]["recommendation"] = "recommended"
    
    # Memory constraints
    total_memory_gb = 0
    try:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # Fallback memory detection
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        total_memory_gb = int(line.split()[1]) / (1024**2)
                        break
        except (FileNotFoundError, IOError):
            pass
    
    if total_memory_gb > 0 and total_memory_gb < 4:
        chromadb_issues.append(f"System has {total_memory_gb:.1f}GB RAM - ChromaDB may consume significant memory")
        compatibility["sqlite_vec"]["recommendation"] = "recommended"
    
    # Older Python versions
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info < (3, 9):
        chromadb_issues.append(f"Python {python_version} may have ChromaDB compatibility issues")
    
    # ARM architecture considerations
    if system_info["is_arm"]:
        print_info("ARM architecture detected - both backends should work well")
    
    compatibility["chromadb"]["issues"] = chromadb_issues
    
    # Print compatibility analysis
    print_info("Storage Backend Compatibility Analysis:")
    
    for backend, info in compatibility.items():
        status = "✅" if info["supported"] else "❌"
        rec_text = {
            "production": "☁️ PRODUCTION (Cloud)",
            "development": "🪶 DEVELOPMENT (Local)",
            "team": "👥 TEAM (Multi-client)",
            "recommended": "🌟 RECOMMENDED",
            "default": "📦 Standard",
            "problematic": "⚠️  May have issues",
            "lightweight": "🪶 Lightweight"
        }.get(info["recommendation"], "")
        
        print_info(f"  {status} {backend.upper()}: {rec_text}")
        
        if info["issues"]:
            for issue in info["issues"]:
                print_info(f"    • {issue}")
    
    return compatibility

def choose_storage_backend(system_info, gpu_info, args):
    """Choose storage backend based on environment and user preferences."""
    compatibility = detect_storage_backend_compatibility(system_info, gpu_info)
    
    # Check if user specified a backend via environment
    env_backend = os.environ.get('MCP_MEMORY_STORAGE_BACKEND')
    if env_backend:
        print_info(f"Using storage backend from environment: {env_backend}")
        return env_backend
    
    # Check for command line argument (we'll add this)
    if hasattr(args, 'storage_backend') and args.storage_backend:
        print_info(f"Using storage backend from command line: {args.storage_backend}")
        return args.storage_backend
    
    # Auto-select based on compatibility
    recommended_backend = None
    for backend, info in compatibility.items():
        if info["recommendation"] == "recommended":
            recommended_backend = backend
            break
    
    if not recommended_backend:
        recommended_backend = "sqlite_vec"  # Default fallback for local development

    # Interactive backend selection
    print_step("3b", "Storage Backend Selection")
    print_info("Choose the storage backend that best fits your use case:")
    print_info("")
    print_info("Usage scenarios:")
    print_info("  1. Production/Shared (Cloudflare) - Cloud storage, multi-user access, requires credentials")
    print_info("  2. Development/Personal (SQLite-vec) - Local, lightweight, single-user")
    print_info("  3. Team/Multi-client (ChromaDB) - Local server, multiple clients")
    print_info("  4. Hybrid (Recommended) - Fast local SQLite + background Cloudflare sync")
    print_info("  5. Auto-detect - Try optimal backend based on your system")
    print_info("")

    # Show compatibility analysis
    for i, (backend, info) in enumerate(compatibility.items(), 1):
        if backend == "auto_detect":
            continue
        status = "✅" if info["supported"] else "❌"
        rec_text = {
            "production": "☁️ PRODUCTION (Cloud)",
            "development": "🪶 DEVELOPMENT (Local)",
            "team": "👥 TEAM (Multi-client)",
            "recommended": "🌟 RECOMMENDED",
            "problematic": "⚠️  May have issues"
        }.get(info["recommendation"], "")
        print_info(f"  {status} {i}. {backend.upper()}: {rec_text}")
        if info["issues"]:
            for issue in info["issues"]:
                print_info(f"     • {issue}")

    print_info("")
    default_choice = "2" if compatibility["chromadb"]["recommendation"] == "problematic" else "2"

    while True:
        try:
            choice = input(f"Choose storage backend [1-5] (default: 4 - hybrid): ").strip()
            if not choice:
                choice = "4"  # Default to hybrid

            if choice == "1":
                return "cloudflare"
            elif choice == "2":
                return "sqlite_vec"
            elif choice == "3":
                return "chromadb"
            elif choice == "4":
                return "hybrid"
            elif choice == "5":
                return "auto_detect"
            else:
                print_error("Please enter 1, 2, 3, 4, or 5")
        except (EOFError, KeyboardInterrupt):
            print_info(f"\nUsing recommended backend: hybrid")
            return "hybrid"

def setup_cloudflare_credentials():
    """Interactive setup of Cloudflare credentials."""
    print_step("3c", "Cloudflare Backend Setup")
    print_info("Cloudflare backend requires API credentials for D1 database and Vectorize index.")
    print_info("You'll need:")
    print_info("  • Cloudflare API Token (with D1 and Vectorize permissions)")
    print_info("  • Account ID")
    print_info("  • D1 Database ID")
    print_info("  • Vectorize Index name")
    print_info("")
    print_info("Visit https://dash.cloudflare.com to get these credentials.")
    print_info("")

    credentials = {}

    try:
        # Get API Token
        while True:
            token = input("Enter Cloudflare API Token: ").strip()
            if token:
                credentials['CLOUDFLARE_API_TOKEN'] = token
                break
            print_error("API token is required")

        # Get Account ID
        while True:
            account_id = input("Enter Cloudflare Account ID: ").strip()
            if account_id:
                credentials['CLOUDFLARE_ACCOUNT_ID'] = account_id
                break
            print_error("Account ID is required")

        # Get D1 Database ID
        while True:
            d1_id = input("Enter D1 Database ID: ").strip()
            if d1_id:
                credentials['CLOUDFLARE_D1_DATABASE_ID'] = d1_id
                break
            print_error("D1 Database ID is required")

        # Get Vectorize Index
        vectorize_index = input("Enter Vectorize Index name (default: mcp-memory-index): ").strip()
        if not vectorize_index:
            vectorize_index = "mcp-memory-index"
        credentials['CLOUDFLARE_VECTORIZE_INDEX'] = vectorize_index

        # Set storage backend
        credentials['MCP_MEMORY_STORAGE_BACKEND'] = 'cloudflare'

        return credentials

    except (EOFError, KeyboardInterrupt):
        print_info("\nCloudflare setup cancelled.")
        return None

def save_credentials_to_env(credentials):
    """Save credentials to .env file and current environment."""
    env_file = Path('.env')

    # Read existing .env content if it exists
    existing_lines = []
    if env_file.exists():
        with open(env_file, 'r') as f:
            existing_lines = f.readlines()

    # Filter out any existing Cloudflare variables
    filtered_lines = [
        line for line in existing_lines
        if not any(key in line for key in credentials.keys())
    ]

    # Add new credentials
    with open(env_file, 'w') as f:
        # Write existing non-Cloudflare lines
        f.writelines(filtered_lines)

        # Add separator if file wasn't empty
        if filtered_lines and not filtered_lines[-1].endswith('\n'):
            f.write('\n')
        if filtered_lines:
            f.write('\n# Cloudflare Backend Configuration\n')

        # Write Cloudflare credentials
        for key, value in credentials.items():
            f.write(f'{key}={value}\n')

    # Also set credentials in current environment for immediate use
    for key, value in credentials.items():
        os.environ[key] = value

    print_success(f"Credentials saved to .env file and current environment")

def test_cloudflare_connection(credentials):
    """Test Cloudflare API connection."""
    print_info("Testing Cloudflare API connection...")

    try:
        import requests

        headers = {
            'Authorization': f"Bearer {credentials['CLOUDFLARE_API_TOKEN']}",
            'Content-Type': 'application/json'
        }

        # Test API token validity
        response = requests.get(
            "https://api.cloudflare.com/client/v4/user/tokens/verify",
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print_success("API token is valid")
                return True
            else:
                print_error(f"API token validation failed: {data.get('errors')}")
                return False
        else:
            print_error(f"API connection failed with status {response.status_code}")
            return False

    except ImportError:
        print_warning("Could not test connection (requests not installed)")
        print_info("Connection will be tested when the service starts")
        return True
    except Exception as e:
        print_warning(f"Could not test connection: {e}")
        print_info("Connection will be tested when the service starts")
        return True

def install_storage_backend(backend, system_info):
    """Install the chosen storage backend."""
    print_step("3c", f"Installing {backend} storage backend")

    if backend == "cloudflare":
        print_info("Cloudflare backend uses cloud services - no local dependencies needed")

        # Setup credentials interactively
        credentials = setup_cloudflare_credentials()
        if not credentials:
            print_warning("Cloudflare setup cancelled. Falling back to SQLite-vec.")
            return install_storage_backend("sqlite_vec", system_info)

        # Save credentials to .env file
        save_credentials_to_env(credentials)

        # Test connection
        connection_ok = test_cloudflare_connection(credentials)
        if connection_ok:
            print_success("Cloudflare backend configured successfully")
            return "cloudflare"
        else:
            print_warning("Cloudflare connection test failed. You can continue and fix credentials later.")
            fallback = input("Continue with Cloudflare anyway? [y/N]: ").strip().lower()
            if fallback.startswith('y'):
                return "cloudflare"
            else:
                print_info("Falling back to SQLite-vec for local development.")
                return install_storage_backend("sqlite_vec", system_info)

    elif backend == "sqlite_vec":
        return install_package_safe(
            "sqlite-vec",
            success_msg="SQLite-vec installed successfully",
            error_msg="Failed to install SQLite-vec"
        )

    elif backend == "hybrid":
        print_info("Hybrid backend combines fast local SQLite with background Cloudflare sync")

        # First install SQLite-vec for local storage
        print_info("Installing SQLite-vec for local storage...")
        sqlite_success = install_package_safe(
            "sqlite-vec",
            success_msg="SQLite-vec installed successfully",
            error_msg="Failed to install SQLite-vec"
        )
        if not sqlite_success:
            print_error("Hybrid backend requires SQLite-vec. Installation failed.")
            return False

        # Setup Cloudflare credentials for cloud sync
        print_info("Configuring Cloudflare for background synchronization...")
        credentials = setup_cloudflare_credentials()
        if not credentials:
            print_warning("Cloudflare setup cancelled.")
            fallback = input("Continue with SQLite-vec only? [Y/n]: ").strip().lower()
            if not fallback or fallback.startswith('y'):
                print_info("Falling back to SQLite-vec for local-only operation.")
                return "sqlite_vec"
            else:
                return False

        # Update credentials to set hybrid backend
        credentials['MCP_MEMORY_STORAGE_BACKEND'] = 'hybrid'

        # Save credentials to .env file
        save_credentials_to_env(credentials)

        # Test connection
        connection_ok = test_cloudflare_connection(credentials)
        if connection_ok:
            print_success("Hybrid backend configured successfully")
            print_info("  • Local storage: SQLite-vec (5ms reads)")
            print_info("  • Cloud sync: Cloudflare (background)")
            return "hybrid"
        else:
            print_warning("Cloudflare connection test failed.")
            fallback = input("Continue with hybrid (will sync when connection available)? [Y/n]: ").strip().lower()
            if not fallback or fallback.startswith('y'):
                print_info("Hybrid backend will sync to Cloudflare when connection is available")
                return "hybrid"
            else:
                print_info("Falling back to SQLite-vec for local development.")
                return "sqlite_vec"

    elif backend == "chromadb":
        chromadb_version = "0.5.23"
        success = install_package_safe(
            f"chromadb=={chromadb_version}",
            success_msg=f"ChromaDB {chromadb_version} installed successfully",
            error_msg="Failed to install ChromaDB"
        )
        if not success:
            print_info("This is a known issue on some systems (especially older macOS Intel)")
        return success
            
    elif backend == "auto_detect":
        print_info("Attempting auto-detection...")
        print_info("Auto-detect will prioritize local backends (SQLite-vec, ChromaDB)")
        print_info("For production use, manually select Cloudflare backend.")

        # For auto-detect, try SQLite-vec first (most reliable)
        print_info("Trying SQLite-vec installation...")
        if install_storage_backend("sqlite_vec", system_info):
            print_success("SQLite-vec installed successfully")
            return "sqlite_vec"

        print_warning("SQLite-vec installation failed, trying ChromaDB...")
        if install_storage_backend("chromadb", system_info):
            print_success("ChromaDB installed successfully as fallback")
            return "chromadb"

        print_error("All local storage backends failed to install")
        print_info("Consider manually configuring Cloudflare backend for production use")
        return False
    
    return False

def _detect_installer_command():
    """Detect available package installer (pip or uv)."""
    # Check if pip is available
    pip_available, _ = run_command_safe(
        [sys.executable, '-m', 'pip', '--version'],
        silent=True
    )

    # Detect if uv is available
    uv_path = shutil.which("uv")
    uv_available = uv_path is not None

    # Decide installer command prefix
    if pip_available:
        return [sys.executable, '-m', 'pip']
    elif uv_available:
        print_warning("pip not found, but uv detected. Using 'uv pip' for installation.")
        return ['uv', 'pip']
    else:
        print_error("Neither pip nor uv detected. Cannot install packages.")
        return None

def _setup_storage_and_gpu_environment(args, system_info, gpu_info, env):
    """Set up storage backend and GPU environment variables."""
    # Choose and install storage backend
    chosen_backend = choose_storage_backend(system_info, gpu_info, args)

    # Check if chromadb was chosen but flag not provided
    if chosen_backend == "chromadb" and not args.with_chromadb:
        print_warning("ChromaDB backend selected but --with-chromadb flag not provided")
        print_info("ChromaDB requires heavy ML dependencies (~1-2GB).")
        print_info("To use ChromaDB, run: python scripts/installation/install.py --with-chromadb")
        print_info("Switching to SQLite-vec backend instead...")
        chosen_backend = "sqlite_vec"

    # ChromaDB automatically includes ML dependencies
    if args.with_chromadb:
        args.with_ml = True

    if chosen_backend == "auto_detect":
        # Handle auto-detection case - prefer sqlite_vec if chromadb not explicitly requested
        if not args.with_chromadb:
            print_info("Auto-detection: Using SQLite-vec (lightweight option)")
            chosen_backend = "sqlite_vec"
        else:
            actual_backend = install_storage_backend(chosen_backend, system_info)
            if not actual_backend:
                print_error("Failed to install any storage backend")
                return False
            chosen_backend = actual_backend
    else:
        # Install the chosen backend
        if not install_storage_backend(chosen_backend, system_info):
            print_error(f"Failed to install {chosen_backend} storage backend")
            return False

    # Set environment variable for chosen backend
    if chosen_backend in ["sqlite_vec", "hybrid", "cloudflare"]:
        env['MCP_MEMORY_STORAGE_BACKEND'] = chosen_backend
        if chosen_backend == "sqlite_vec":
            print_info("Configured to use SQLite-vec storage backend")
        elif chosen_backend == "hybrid":
            print_info("Configured to use Hybrid storage backend (SQLite + Cloudflare)")
        elif chosen_backend == "cloudflare":
            print_info("Configured to use Cloudflare storage backend")
    else:
        env['MCP_MEMORY_STORAGE_BACKEND'] = 'chromadb'
        print_info("Configured to use ChromaDB storage backend")

    # Set environment variables based on detected GPU
    if gpu_info.get("has_cuda"):
        print_info("Configuring for CUDA installation")
    elif gpu_info.get("has_rocm"):
        print_info("Configuring for ROCm installation")
        env['MCP_MEMORY_USE_ROCM'] = '1'
    elif gpu_info.get("has_mps"):
        print_info("Configuring for Apple Silicon MPS installation")
        env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    elif gpu_info.get("has_directml"):
        print_info("Configuring for DirectML installation")
        env['MCP_MEMORY_USE_DIRECTML'] = '1'
    else:
        print_info("Configuring for CPU-only installation")
        env['MCP_MEMORY_USE_ONNX'] = '1'

    # Check for Homebrew PyTorch installation
    using_homebrew_pytorch = False
    if system_info.get("has_homebrew_pytorch"):
        print_info(f"Using existing Homebrew PyTorch installation (version: {system_info.get('homebrew_pytorch_version')})")
        using_homebrew_pytorch = True
        # Set the environment variable to use ONNX for embeddings
        env['MCP_MEMORY_USE_ONNX'] = '1'
        # Skip the PyTorch installation step
        pytorch_installed = True
    else:
        # Handle platform-specific PyTorch installation
        pytorch_installed = install_pytorch_platform_specific(system_info, gpu_info)
        if not pytorch_installed:
            print_warning("Platform-specific PyTorch installation failed, but will continue with package installation")

    try:
        # SQLite-vec with ONNX for macOS with homebrew PyTorch or compatibility issues
        if (system_info["is_macos"] and system_info["is_x86"] and 
            (sys.version_info >= (3, 13) or using_homebrew_pytorch or args.skip_pytorch)):
            
            if using_homebrew_pytorch:
                print_info("Using Homebrew PyTorch - installing with SQLite-vec + ONNX configuration")
            elif args.skip_pytorch:
                print_info("Skipping PyTorch installation - using SQLite-vec + ONNX configuration")
            else:
                print_info("Using Python 3.13+ on macOS Intel - using SQLite-vec + ONNX configuration")
            
            # First try to install without ML dependencies
            try:
                cmd = installer_cmd + ['install']
                if len(installer_cmd) >= 2 and Path(installer_cmd[0]).stem == "uv" and installer_cmd[1] == "pip":
                    cmd += ['--python', sys.executable]
                cmd += ['--no-deps'] + install_mode + ['.']
                success, _ = run_command_safe(
                    cmd,
                    success_msg="Package installed with --no-deps successfully",
                    error_msg="Failed to install package with --no-deps",
                    silent=False
                )
                if not success:
                    raise Exception("Installation failed")
                
                # Install core dependencies except torch/sentence-transformers
                print_info("Installing core dependencies except ML libraries...")
                
                # Create a list of dependencies to install
                # Note: mcp and tokenizers are already in core dependencies
                dependencies = [
                    "onnxruntime>=1.14.1"  # ONNX runtime for lightweight embeddings
                ]

                # Add backend-specific dependencies (sqlite-vec, mcp, tokenizers are already in core)
                if args.with_chromadb:
                    dependencies.append("chromadb==0.5.23")
                
                # Install dependencies using the same installer selection as the main install
                cmd = installer_cmd + ['install']
                if len(installer_cmd) >= 2 and Path(installer_cmd[0]).stem == "uv" and installer_cmd[1] == "pip":
                    cmd += ['--python', sys.executable]
                cmd += dependencies
                success, _ = run_command_safe(
                    cmd,
                    success_msg="Core dependencies installed successfully",
                    error_msg="Failed to install core dependencies",
                    silent=False
                )
                if not success:
                    raise Exception("Core dependency installation failed")
                
                # Set environment variables for ONNX
                print_info("Configuring to use ONNX runtime for inference without PyTorch...")
                env['MCP_MEMORY_USE_ONNX'] = '1'
                if chosen_backend != "sqlite_vec":
                    print_info("Switching to SQLite-vec backend for better compatibility")
                    env['MCP_MEMORY_STORAGE_BACKEND'] = 'sqlite_vec'
                
                print_success("MCP Memory Service installed successfully (SQLite-vec + ONNX)")
                
                if using_homebrew_pytorch:
                    print_info("Using Homebrew PyTorch installation for embedding generation")
                    print_info("Environment configured to use SQLite-vec backend and ONNX runtime")
                else:
                    print_warning("ML libraries (PyTorch/sentence-transformers) were not installed due to compatibility issues")
                    print_info("The service will use ONNX runtime for inference instead")
                
                return True
            except Exception as e:
                print_error(f"Failed to install with ONNX approach: {e}")
                # Fall through to try standard installation
        
        # Standard installation with appropriate optional dependencies
        install_target = ['.']

        # Determine which optional dependencies to include based on backend and flags
        if args.with_chromadb:
            install_target = ['.[chromadb]']
            print_info("Installing with ChromaDB backend support (includes ML dependencies)")
        elif args.with_ml:
            if chosen_backend == "sqlite_vec":
                install_target = ['.[sqlite-ml]']
                print_info("Installing SQLite-vec with full ML capabilities (torch + sentence-transformers)")
            else:
                install_target = ['.[ml]']
                print_info("Installing with ML dependencies for semantic search and embeddings")
        elif chosen_backend == "sqlite_vec":
            install_target = ['.[sqlite]']
            print_info("Installing SQLite-vec with lightweight ONNX embeddings (recommended)")
            print_info("For full ML capabilities with SQLite-vec, use --with-ml flag")
        else:
            print_info("Installing lightweight version (no ML dependencies by default)")
            print_info("For full functionality, use --with-ml flag or install with: pip install mcp-memory-service[ml]")

        cmd = installer_cmd + ['install']
        if len(installer_cmd) >= 2 and Path(installer_cmd[0]).stem == "uv" and installer_cmd[1] == "pip":
            cmd += ['--python', sys.executable]
        cmd += install_mode + install_target
        success, _ = run_command_safe(
            cmd,
            success_msg="MCP Memory Service installed successfully",
            error_msg="Failed to install MCP Memory Service",
            silent=False
        )
        return success
    except Exception as e:
        print_error(f"Failed to install MCP Memory Service: {e}")
        
        # Special handling for macOS with compatibility issues
        if system_info["is_macos"] and system_info["is_x86"]:
            print_warning("Installation on macOS Intel is challenging")
            print_info("Try manually installing with:")
            print_info("1. pip install --no-deps .")
            print_info("2. pip install sqlite-vec>=0.1.0 mcp>=1.0.0,<2.0.0 onnxruntime>=1.14.1")
            print_info("3. export MCP_MEMORY_USE_ONNX=1")
            print_info("4. export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec")
            
            if system_info.get("has_homebrew_pytorch"):
                print_info("Homebrew PyTorch was detected but installation still failed.")
                print_info("Try running: python install.py --storage-backend sqlite_vec --skip-pytorch")
            
        return False

def get_platform_base_dir() -> Path:
    """Get platform-specific base directory for MCP Memory storage.

    Returns:
        Path: Platform-appropriate base directory
    """
    home_dir = Path.home()

    PLATFORM_PATHS = {
        'Darwin': home_dir / 'Library' / 'Application Support' / 'mcp-memory',
        'Windows': Path(os.environ.get('LOCALAPPDATA', '')) / 'mcp-memory',
    }

    system = platform.system()
    return PLATFORM_PATHS.get(system, home_dir / '.local' / 'share' / 'mcp-memory')


def setup_storage_directories(backend: str, base_dir: Path, args) -> Tuple[Path, Path, bool]:
    """Setup storage and backup directories for the specified backend.

    Args:
        backend: Storage backend type
        base_dir: Base directory for storage
        args: Command line arguments

    Returns:
        Tuple of (storage_path, backups_path, success)
    """
    if backend in ['sqlite_vec', 'hybrid', 'cloudflare']:
        storage_path = args.chroma_path or (base_dir / 'sqlite_vec.db')
        storage_dir = storage_path.parent if storage_path.name.endswith('.db') else storage_path
    else:  # chromadb
        storage_path = args.chroma_path or (base_dir / 'chroma_db')
        storage_dir = storage_path

    backups_path = args.backups_path or (base_dir / 'backups')

    try:
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(backups_path, exist_ok=True)

        # Test writability
        test_file = Path(storage_dir) / '.write_test'
        test_file.write_text('test')
        test_file.unlink()

        print_info(f"Storage path: {storage_path}")
        print_info(f"Backups path: {backups_path}")
        return storage_path, backups_path, True

    except (OSError, IOError, PermissionError) as e:
        print_error(f"Failed to configure storage paths: {e}")
        return storage_path, backups_path, False
    except Exception as e:
        print_error(f"Unexpected error configuring storage paths: {e}")
        return storage_path, backups_path, False


def build_mcp_env_config(storage_backend: str, storage_path: Path,
                         backups_path: Path) -> Dict[str, str]:
    """Build MCP environment configuration for Claude Desktop.

    Args:
        storage_backend: Type of storage backend
        storage_path: Path to storage directory/file
        backups_path: Path to backups directory

    Returns:
        Dict of environment variables for MCP configuration
    """
    env_config = {
        "MCP_MEMORY_BACKUPS_PATH": str(backups_path),
        "MCP_MEMORY_STORAGE_BACKEND": storage_backend
    }

    if storage_backend in ['sqlite_vec', 'hybrid']:
        env_config["MCP_MEMORY_SQLITE_PATH"] = str(storage_path)
        env_config["MCP_MEMORY_SQLITE_PRAGMAS"] = "busy_timeout=15000,cache_size=20000"

    if storage_backend in ['hybrid', 'cloudflare']:
        cloudflare_vars = [
            'CLOUDFLARE_API_TOKEN',
            'CLOUDFLARE_ACCOUNT_ID',
            'CLOUDFLARE_D1_DATABASE_ID',
            'CLOUDFLARE_VECTORIZE_INDEX'
        ]
        for var in cloudflare_vars:
            value = os.environ.get(var)
            if value:
                env_config[var] = value

    if storage_backend == 'chromadb':
        env_config["MCP_MEMORY_CHROMA_PATH"] = str(storage_path)

    return env_config


def update_claude_config_file(config_path: Path, env_config: Dict[str, str],
                              project_root: Path, is_windows: bool) -> bool:
    """Update Claude Desktop configuration file with MCP Memory settings.

    Args:
        config_path: Path to Claude config file
        env_config: Environment configuration dictionary
        project_root: Root directory of the project
        is_windows: Whether running on Windows

    Returns:
        bool: True if update succeeded
    """
    try:
        config_text = config_path.read_text()
        config = json.loads(config_text)

        if not isinstance(config, dict):
            print_warning(f"Invalid config format in {config_path}")
            return False

        if 'mcpServers' not in config:
            config['mcpServers'] = {}

        # Create server configuration
        if is_windows:
            script_path = str((project_root / "memory_wrapper.py").resolve())
            config['mcpServers']['memory'] = {
                "command": "python",
                "args": [script_path],
                "env": env_config
            }
        else:
            config['mcpServers']['memory'] = {
                "command": "uv",
                "args": [
                    "--directory",
                    str(project_root.resolve()),
                    "run",
                    "memory"
                ],
                "env": env_config
            }

        config_path.write_text(json.dumps(config, indent=2))
        print_success("Updated Claude Desktop configuration")
        return True

    except (OSError, PermissionError, json.JSONDecodeError) as e:
        print_warning(f"Failed to update Claude Desktop configuration: {e}")
        return False


def configure_paths(args):
    """Configure paths for the MCP Memory Service."""
    print_step("4", "Configuring paths")

    # Get system info
    system_info = detect_system()

    # Get platform-specific base directory
    base_dir = get_platform_base_dir()
    storage_backend = os.environ.get('MCP_MEMORY_STORAGE_BACKEND', 'chromadb')

    # Setup storage directories
    storage_path, backups_path, success = setup_storage_directories(
        storage_backend, base_dir, args
    )
    if not success:
        print_warning("Continuing with Claude Desktop configuration despite storage setup failure")

    # Test backups directory
    try:
        test_file = Path(backups_path) / '.write_test'
        test_file.write_text('test')
        test_file.unlink()
        print_success("Storage directories created and are writable")
    except (OSError, PermissionError) as e:
        print_error(f"Failed to test backups directory: {e}")
        print_warning("Continuing with Claude Desktop configuration")

    # Configure Claude Desktop
    env_config = build_mcp_env_config(storage_backend, storage_path, backups_path)
    project_root = Path(__file__).parent.parent.parent

    home_dir = Path.home()
    claude_config_paths = [
        home_dir / 'Library' / 'Application Support' / 'Claude' / 'claude_desktop_config.json',
        home_dir / '.config' / 'Claude' / 'claude_desktop_config.json',
        Path('claude_config') / 'claude_desktop_config.json'
    ]

    for config_path in claude_config_paths:
        if config_path.exists():
            print_info(f"Found Claude Desktop config at {config_path}")
            if update_claude_config_file(config_path, env_config, project_root,
                                        system_info["is_windows"]):
                break

    return True

def verify_installation():
    """Verify the installation."""
    print_step("5", "Verifying installation")
    
    # Get system info
    system_info = detect_system()
    
    # Check if the package is installed
    try:
        import mcp_memory_service
        print_success(f"MCP Memory Service is installed: {mcp_memory_service.__file__}")
    except ImportError:
        print_error("MCP Memory Service is not installed correctly")
        return False
    
    # Check if the entry point is available
    memory_script = shutil.which('memory')
    if memory_script:
        print_success(f"Memory command is available: {memory_script}")
    else:
        print_warning("Memory command is not available in PATH")
    
    # Check storage backend installation
    storage_backend = os.environ.get('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec')
    
    if storage_backend == 'sqlite_vec':
        try:
            import sqlite_vec
            print_success(f"SQLite-vec is installed: {sqlite_vec.__version__}")
        except ImportError:
            print_error("SQLite-vec is not installed correctly")
            return False
    elif storage_backend == 'chromadb':
        try:
            import chromadb
            print_success(f"ChromaDB is installed: {chromadb.__version__}")
        except ImportError:
            print_error("ChromaDB is not installed correctly")
            return False
    
    # Check for ONNX runtime
    try:
        import onnxruntime
        print_success(f"ONNX Runtime is installed: {onnxruntime.__version__}")
        use_onnx = os.environ.get('MCP_MEMORY_USE_ONNX', '').lower() in ('1', 'true', 'yes')
        if use_onnx:
            print_info("Environment configured to use ONNX runtime for embeddings")
            # Check for tokenizers (required for ONNX)
            try:
                import tokenizers
                print_success(f"Tokenizers is installed: {tokenizers.__version__}")
            except ImportError:
                print_warning("Tokenizers not installed but required for ONNX embeddings")
                print_info("Install with: pip install tokenizers>=0.20.0")
    except ImportError:
        print_warning("ONNX Runtime is not installed. This is recommended for PyTorch-free operation.")
        print_info("Install with: pip install onnxruntime>=1.14.1 tokenizers>=0.20.0")
    
    # Check for Homebrew PyTorch
    homebrew_pytorch = False
    if system_info.get("has_homebrew_pytorch"):
        homebrew_pytorch = True
        print_success(f"Homebrew PyTorch detected: {system_info.get('homebrew_pytorch_version')}")
        print_info("Using system-installed PyTorch instead of pip version")
    
    # Check ML dependencies as optional
    pytorch_installed = False
    try:
        import torch
        pytorch_installed = True
        print_info(f"PyTorch is installed: {torch.__version__}")
        
        # Check for CUDA
        if torch.cuda.is_available():
            print_success(f"CUDA is available: {torch.version.cuda}")
            print_info(f"GPU: {torch.cuda.get_device_name(0)}")
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print_success("MPS (Metal Performance Shaders) is available")
        # Check for DirectML
        else:
            try:
                import torch_directml
                version = getattr(torch_directml, '__version__', 'Unknown version')
                print_success(f"DirectML is available: {version}")
            except ImportError:
                print_info("Using CPU-only PyTorch")
        
        # For macOS Intel, verify compatibility with sentence-transformers
        if system_info["is_macos"] and system_info["is_x86"]:
            torch_version = torch.__version__.split('.')
            major, minor = int(torch_version[0]), int(torch_version[1])
            
            print_info(f"Verifying torch compatibility on macOS Intel (v{major}.{minor})")
            if major < 1 or (major == 1 and minor < 6):
                print_warning(f"PyTorch version {torch.__version__} may be too old for sentence-transformers")
            elif major > 2 or (major == 2 and minor > 1):
                print_warning(f"PyTorch version {torch.__version__} may be too new for sentence-transformers 2.2.2")
                print_info("If you encounter issues, try downgrading to torch 2.0.1")
            
    except ImportError:
        print_warning("PyTorch is not installed via pip. This is okay for basic operation with SQLite-vec backend.")
        if homebrew_pytorch:
            print_info("Using Homebrew PyTorch installation instead of pip version")
        else:
            print_info("For full functionality including embedding generation, install with: pip install 'mcp-memory-service[ml]'")
        pytorch_installed = False
    
    # Check if sentence-transformers is installed correctly (only if PyTorch is installed)
    if pytorch_installed or homebrew_pytorch:
        try:
            import sentence_transformers
            print_success(f"sentence-transformers is installed: {sentence_transformers.__version__}")
            
            if pytorch_installed:
                # Verify compatibility between torch and sentence-transformers
                st_version = sentence_transformers.__version__.split('.')
                torch_version = torch.__version__.split('.')
                
                st_major, st_minor = int(st_version[0]), int(st_version[1])
                torch_major, torch_minor = int(torch_version[0]), int(torch_version[1])
                
                # Specific compatibility check for macOS Intel
                if system_info["is_macos"] and system_info["is_x86"]:
                    if st_major >= 3 and (torch_major < 1 or (torch_major == 1 and torch_minor < 11)):
                        print_warning(f"sentence-transformers {sentence_transformers.__version__} requires torch>=1.11.0")
                        print_info("This may cause runtime issues - consider downgrading sentence-transformers to 2.2.2")
            
            # Verify by trying to load a model (minimal test)
            try:
                print_info("Testing sentence-transformers model loading...")
                test_model = sentence_transformers.SentenceTransformer('paraphrase-MiniLM-L3-v2')
                print_success("Successfully loaded test model")
            except Exception as e:
                print_warning(f"Model loading test failed: {e}")
                print_warning("There may be compatibility issues between PyTorch and sentence-transformers")
                
        except ImportError:
            print_warning("sentence-transformers is not installed. This is okay for basic operation with SQLite-vec backend.")
            print_info("For full functionality including embedding generation, install with: pip install 'mcp-memory-service[ml]'")
    
    # Check for SQLite-vec + ONNX configuration
    if storage_backend == 'sqlite_vec' and os.environ.get('MCP_MEMORY_USE_ONNX', '').lower() in ('1', 'true', 'yes'):
        print_success("SQLite-vec + ONNX configuration is set up correctly")
        print_info("This configuration can run without PyTorch dependency")
        
        try:
            # Import the key components to verify installation
            from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
            from mcp_memory_service.models.memory import Memory
            print_success("SQLite-vec + ONNX components loaded successfully")
            
            # Check paths
            sqlite_path = os.environ.get('MCP_MEMORY_SQLITE_PATH', '')
            if sqlite_path:
                print_info(f"SQLite-vec database path: {sqlite_path}")
            else:
                print_warning("MCP_MEMORY_SQLITE_PATH is not set")
            
            backups_path = os.environ.get('MCP_MEMORY_BACKUPS_PATH', '')
            if backups_path:
                print_info(f"Backups path: {backups_path}")
            else:
                print_warning("MCP_MEMORY_BACKUPS_PATH is not set")
            
        except ImportError as e:
            print_error(f"Failed to import SQLite-vec components: {e}")
            return False
            
    # Check if MCP Memory Service package is installed correctly
    try:
        import mcp_memory_service
        print_success(f"MCP Memory Service is installed correctly")
        return True
    except ImportError:
        print_error("MCP Memory Service is not installed correctly")
        return False

def install_claude_hooks(args, system_info):
    """Install Claude Code memory awareness hooks."""
    print_step("5", "Installing Claude Code Memory Awareness Hooks")

    try:
        # Check if Claude Code is available
        claude_available = shutil.which("claude") is not None
        if not claude_available:
            print_warning("Claude Code CLI not found")
            print_info("You can install hooks manually later using:")
            print_info("  cd claude-hooks && python install_hooks.py --basic")
            return

        print_info("Claude Code CLI detected")

        # Use unified Python installer for cross-platform compatibility
        claude_hooks_dir = Path(__file__).parent.parent.parent / "claude-hooks"
        unified_installer = claude_hooks_dir / "install_hooks.py"

        if not unified_installer.exists():
            print_error("Unified hook installer not found at expected location")
            print_info("Please ensure the unified installer is available:")
            print_info(f"  Expected: {unified_installer}")
            return

        # Prepare installer command with appropriate options
        version = get_package_version()
        if args.install_natural_triggers:
            print_info(f"Installing Natural Memory Triggers v{version}...")
            installer_cmd = [sys.executable, str(unified_installer), "--natural-triggers"]
        else:
            print_info("Installing standard memory awareness hooks...")
            installer_cmd = [sys.executable, str(unified_installer), "--basic"]

        # Run the unified Python installer
        print_info(f"Running unified hook installer: {unified_installer.name}")
        result = subprocess.run(
            installer_cmd,
            cwd=str(claude_hooks_dir),
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print_success("Claude Code memory awareness hooks installed successfully")
            if args.install_natural_triggers:
                print_info(f"✅ Natural Memory Triggers v{version} enabled")
                print_info("✅ Intelligent trigger detection with 85%+ accuracy")
                print_info("✅ Multi-tier performance optimization")
                print_info("✅ CLI management tools available")
                print_info("")
                print_info("Manage Natural Memory Triggers:")
                print_info("  node ~/.claude/hooks/memory-mode-controller.js status")
                print_info("  node ~/.claude/hooks/memory-mode-controller.js profile balanced")
            else:
                print_info("✅ Standard memory awareness hooks enabled")
                print_info("✅ Session-start and session-end hooks active")
        else:
            print_warning("Hook installation completed with warnings")
            if result.stdout:
                print_info("Installer output:")
                print_info(result.stdout)
            if result.stderr:
                print_warning("Installer warnings:")
                print_warning(result.stderr)

    except Exception as e:
        print_warning(f"Failed to install hooks automatically: {e}")
        print_info("You can install hooks manually later using the unified installer:")
        print_info("  cd claude-hooks && python install_hooks.py --basic")
        if args.install_natural_triggers:
            print_info("For Natural Memory Triggers:")
            print_info("  cd claude-hooks && python install_hooks.py --natural-triggers")

def detect_development_context():
    """Detect if user is likely a developer (has .git directory).

    Returns:
        bool: True if .git directory exists
    """
    git_dir = Path(".git")
    return git_dir.exists() and git_dir.is_dir()

def recommend_editable_install(args):
    """Recommend editable install for developers.

    If .git directory detected and --dev not specified, prompts user to use
    editable install mode. This prevents the common "stale venv vs source code"
    issue where MCP servers load from site-packages instead of source files.

    Args:
        args: Parsed command line arguments

    Returns:
        bool: True if editable install should be used
    """
    # If already in dev mode, nothing to do
    if args.dev:
        return True

    # Detect development context
    if detect_development_context():
        print_warning("Detected git repository - you may be a developer!")
        print("")
        print_info("For development, EDITABLE install is strongly recommended:")
        print_info("  pip install -e .")
        print("")
        print_info("Why this matters:")
        print_info("  • MCP servers load from site-packages, not source files")
        print_info("  • Without -e flag, source changes won't take effect")
        print_info("  • System restart won't help - it just relaunches stale code")
        print_info("  • Common symptom: Code shows v8.23.0 but server reports v8.5.3")
        print("")
        print_info("Editable install ensures:")
        print_info("  • Source changes take effect immediately (after server restart)")
        print_info("  • No need to reinstall package after every change")
        print_info("  • Easier debugging and development workflow")
        print("")

        response = input("Install in EDITABLE mode (recommended for development)? [Y/n]: ").lower().strip()
        if response == '' or response == 'y' or response == 'yes':
            args.dev = True
            print_success("Enabled editable install mode")
            return True
        else:
            print_warning("Proceeding with standard install (not editable)")
            print_warning("Remember: You'll need to reinstall after every source change!")
            return False

    return False

def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="Install MCP Memory Service")
    parser.add_argument('--dev', action='store_true', help='Install in development mode')
    parser.add_argument('--chroma-path', type=str, help='Path to ChromaDB storage')
    parser.add_argument('--backups-path', type=str, help='Path to backups storage')
    parser.add_argument('--force-compatible-deps', action='store_true',
                        help='Force compatible versions of PyTorch (2.0.1) and sentence-transformers (2.2.2)')
    parser.add_argument('--fallback-deps', action='store_true',
                        help='Use fallback versions of PyTorch (1.13.1) and sentence-transformers (2.2.2)')
    parser.add_argument('--storage-backend', choices=['cloudflare', 'sqlite_vec', 'hybrid', 'auto_detect'],
                        help='Choose storage backend: cloudflare (production cloud), sqlite_vec (local development), hybrid (production + local sync), or auto_detect')
    parser.add_argument('--skip-pytorch', action='store_true',
                        help='Skip PyTorch installation and use ONNX runtime with SQLite-vec backend instead')
    parser.add_argument('--use-homebrew-pytorch', action='store_true',
                        help='Use existing Homebrew PyTorch installation instead of pip version')
    parser.add_argument('--install-hooks', action='store_true',
                        help='Install Claude Code memory awareness hooks after main installation')
    parser.add_argument('--install-natural-triggers', action='store_true',
                        help='Install Natural Memory Triggers (requires Claude Code)')
    parser.add_argument('--with-ml', action='store_true',
                        help='Include ML dependencies (torch, sentence-transformers) for semantic search and embeddings')
    parser.add_argument('--with-chromadb', action='store_true',
                        help='Include ChromaDB backend support (automatically includes ML dependencies)')
    args = parser.parse_args()

    # Check if this is a development context and recommend editable install
    recommend_editable_install(args)

    print_header("MCP Memory Service Installation")
    
    # Step 1: Detect system
    print_step("1", "Detecting system")
    system_info = detect_system()
    
    # Check if user requested force-compatible dependencies for macOS Intel
    if args.force_compatible_deps:
        if system_info["is_macos"] and system_info["is_x86"]:
            print_info("Installing compatible dependencies as requested...")
            # Select versions based on Python version
            python_version = sys.version_info
            if python_version >= (3, 13):
                # Python 3.13+ compatible versions
                torch_version = "2.3.0"
                torch_vision_version = "0.18.0"
                torch_audio_version = "2.3.0"
                st_version = "3.0.0"
            else:
                # Older Python versions
                torch_version = "2.0.1"
                torch_vision_version = "2.0.1"
                torch_audio_version = "2.0.1"
                st_version = "2.2.2"
                
            cmd = [
                sys.executable, '-m', 'pip', 'install',
                f"torch=={torch_version}", f"torchvision=={torch_vision_version}", f"torchaudio=={torch_audio_version}",
                f"sentence-transformers=={st_version}"
            ]
            success, _ = run_command_safe(
                cmd,
                success_msg="Compatible dependencies installed successfully",
                error_msg="Failed to install compatible dependencies",
                silent=False
            )
        else:
            print_warning("--force-compatible-deps is only applicable for macOS with Intel CPUs")
    
    # Check if user requested fallback dependencies for troubleshooting
    if args.fallback_deps:
        print_info("Installing fallback dependencies as requested...")
        # Select versions based on Python version
        python_version = sys.version_info
        if python_version >= (3, 13):
            # Python 3.13+ compatible fallback versions
            torch_version = "2.3.0"
            torch_vision_version = "0.18.0"
            torch_audio_version = "2.3.0"
            st_version = "3.0.0"
        else:
            # Older Python fallback versions
            torch_version = "1.13.1"
            torch_vision_version = "0.14.1"
            torch_audio_version = "0.13.1"
            st_version = "2.2.2"
            
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            f"torch=={torch_version}", f"torchvision=={torch_vision_version}", f"torchaudio=={torch_audio_version}",
            f"sentence-transformers=={st_version}"
        ]
        success, _ = run_command_safe(
            cmd,
            success_msg="Fallback dependencies installed successfully",
            error_msg="Failed to install fallback dependencies",
            silent=False
        )
    
    # Step 2: Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Step 3: Install package
    if not install_package(args):
        # If installation fails and we're on macOS Intel, suggest using the force-compatible-deps option
        if system_info["is_macos"] and system_info["is_x86"]:
            print_warning("Installation failed on macOS Intel.")
            print_info("Try running the script with '--force-compatible-deps' to force compatible versions:")
            print_info("python install.py --force-compatible-deps")
        sys.exit(1)
    
    # Step 4: Configure paths
    if not configure_paths(args):
        print_warning("Path configuration failed, but installation may still work")
    
    # Step 5: Verify installation
    if not verify_installation():
        print_warning("Installation verification failed, but installation may still work")
        # If verification fails and we're on macOS Intel, suggest using the force-compatible-deps option
        if system_info["is_macos"] and system_info["is_x86"]:
            python_version = sys.version_info
            print_info("For macOS Intel compatibility issues, try these steps:")
            print_info("1. First uninstall current packages: pip uninstall -y torch torchvision torchaudio sentence-transformers")
            print_info("2. Then reinstall with compatible versions: python install.py --force-compatible-deps")
            
            if python_version >= (3, 13):
                print_info("For Python 3.13+, you may need to manually install the following:")
                print_info("pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0")
                print_info("pip install sentence-transformers==3.0.0")
    
    print_header("Installation Complete")
    
    # Get final storage backend info
    final_backend = os.environ.get('MCP_MEMORY_STORAGE_BACKEND', 'chromadb')
    use_onnx = os.environ.get('MCP_MEMORY_USE_ONNX', '').lower() in ('1', 'true', 'yes')
    
    print_info("You can now run the MCP Memory Service using the 'memory' command")
    print_info(f"Storage Backend: {final_backend.upper()}")

    if final_backend == 'sqlite_vec':
        print_info("✅ Using SQLite-vec - lightweight, fast, minimal dependencies")
        print_info("   • No complex dependencies or build issues")
        print_info("   • Excellent performance for typical use cases")
    elif final_backend == 'hybrid':
        print_info("✅ Using Hybrid Backend - RECOMMENDED for production")
        print_info("   • Fast local SQLite-vec storage (5ms reads)")
        print_info("   • Background Cloudflare sync for multi-device access")
        print_info("   • SQLite pragmas configured for concurrent access")
        print_info("   • Zero user-facing latency for cloud operations")
    elif final_backend == 'cloudflare':
        print_info("✅ Using Cloudflare Backend - cloud-only storage")
        print_info("   • Distributed edge storage with D1 database")
        print_info("   • Vectorize index for semantic search")
        print_info("   • Multi-device synchronization")
    else:
        print_info("✅ Using ChromaDB - full-featured vector database")
        print_info("   • Advanced features and extensive ecosystem")
    
    if use_onnx:
        print_info("✅ Using ONNX Runtime for inference")
        print_info("   • Compatible with Homebrew PyTorch")
        print_info("   • Reduced dependencies for better compatibility")

    # Show ML dependencies status
    if args.with_ml or args.with_chromadb:
        print_info("✅ ML Dependencies Installed")
        print_info("   • Full semantic search and embedding generation enabled")
        print_info("   • PyTorch and sentence-transformers available")
    else:
        print_info("ℹ️  Lightweight Installation (No ML Dependencies)")
        print_info("   • Basic functionality without semantic search")
        print_info("   • To enable full features: pip install mcp-memory-service[ml]")
    
    print_info("For more information, see the README.md file")

    # Install hooks if requested
    if args.install_hooks or args.install_natural_triggers:
        install_claude_hooks(args, system_info)
    
    # Print macOS Intel specific information if applicable
    if system_info["is_macos"] and system_info["is_x86"]:
        print_info("\nMacOS Intel Notes:")
        
        if system_info.get("has_homebrew_pytorch"):
            print_info("- Using Homebrew PyTorch installation: " + system_info.get("homebrew_pytorch_version", "Unknown"))
            print_info("- The MCP Memory Service is configured to use SQLite-vec + ONNX runtime")
            print_info("- To start the memory service, use:")
            print_info("  export MCP_MEMORY_USE_ONNX=1")
            print_info("  export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec")
            print_info("  memory")
        else:
            print_info("- If you encounter issues, try the --force-compatible-deps option")
            
            python_version = sys.version_info
            if python_version >= (3, 13):
                print_info("- For optimal performance on Intel Macs with Python 3.13+, torch==2.3.0 and sentence-transformers==3.0.0 are recommended")
                print_info("- You can manually install these versions with:")
                print_info("  pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 sentence-transformers==3.0.0")
            else:
                print_info("- For optimal performance on Intel Macs, torch==2.0.1 and sentence-transformers==2.2.2 are recommended")
                print_info("- You can manually install these versions with:")
                print_info("  pip install torch==2.0.1 torchvision==2.0.1 torchaudio==2.0.1 sentence-transformers==2.2.2")
                
        print_info("\nTroubleshooting Tips:")
        print_info("- If you have a Homebrew PyTorch installation, use: --use-homebrew-pytorch")
        print_info("- To completely skip PyTorch installation, use: --skip-pytorch")
        print_info("- To force the SQLite-vec backend, use: --storage-backend sqlite_vec")
        print_info("- For lightweight installation without ML, use: (default behavior)")
        print_info("- For full ML capabilities, use: --with-ml")
        print_info("- For ChromaDB with ML, use: --with-chromadb")
        print_info("- For a quick test, try running: python test_memory.py")
        print_info("- To install Claude Code hooks: --install-hooks")
        print_info("- To install Natural Memory Triggers: --install-natural-triggers")

if __name__ == "__main__":
    main()
