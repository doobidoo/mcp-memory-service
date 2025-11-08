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

# Fix Windows console encoding issues
if platform.system() == "Windows":
    # Ensure stdout uses UTF-8 on Windows to prevent character encoding issues in logs
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

# Enhanced logging system for installer
import logging
from datetime import datetime

class DualOutput:
    """Class to handle both console and file output simultaneously."""
    def __init__(self, log_file_path):
        self.console = sys.stdout
        self.log_file = None
        self.log_file_path = log_file_path
        self._setup_log_file()
    
    def _setup_log_file(self):
        """Set up the log file with proper encoding."""
        try:
            # Create log file with UTF-8 encoding
            self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
            # Write header
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Fix Windows version display in log header
            platform_info = f"{platform.system()} {platform.release()}"
            if platform.system() == "Windows":
                try:
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
                    build_number = winreg.QueryValueEx(key, "CurrentBuildNumber")[0]
                    winreg.CloseKey(key)
                    
                    # Windows 11 has build number >= 22000
                    if int(build_number) >= 22000:
                        platform_info = f"Windows 11"
                    else:
                        platform_info = f"Windows {platform.release()}"
                except (ImportError, OSError, ValueError):
                    pass  # Use default
            
            header = f"""
================================================================================
MCP Memory Service Installation Log
Started: {timestamp}
Platform: {platform_info} ({platform.machine()})
Python: {sys.version}
================================================================================

"""
            self.log_file.write(header)
            self.log_file.flush()
        except Exception as e:
            print(f"Warning: Could not create log file {self.log_file_path}: {e}")
            self.log_file = None
    
    def write(self, text):
        """Write to both console and log file."""
        # Write to console
        self.console.write(text)
        self.console.flush()
        
        # Write to log file if available
        if self.log_file:
            try:
                self.log_file.write(text)
                self.log_file.flush()
            except Exception:
                pass  # Silently ignore log file write errors
    
    def flush(self):
        """Flush both outputs."""
        self.console.flush()
        if self.log_file:
            try:
                self.log_file.flush()
            except Exception:
                pass
    
    def close(self):
        """Close the log file."""
        if self.log_file:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                footer = f"""
================================================================================
Installation completed: {timestamp}
================================================================================
"""
                self.log_file.write(footer)
                self.log_file.close()
            except Exception:
                pass

# Global dual output instance
_dual_output = None

def setup_installer_logging():
    """Set up the installer logging system."""
    global _dual_output
    
    # Create log file path
    log_file = Path.cwd() / "installation.log"
    
    # Remove old log file if it exists
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception:
            pass
    
    # Set up dual output
    _dual_output = DualOutput(str(log_file))
    
    # Redirect stdout to dual output
    sys.stdout = _dual_output
    
    print(f"Installation log will be saved to: {log_file}")
    
    return str(log_file)

def cleanup_installer_logging():
    """Clean up the installer logging system."""
    global _dual_output
    
    if _dual_output:
        # Restore original stdout
        sys.stdout = _dual_output.console
        _dual_output.close()
        _dual_output = None

# Import Claude commands utilities
try:
    from scripts.claude_commands_utils import install_claude_commands, check_claude_code_cli
except ImportError:
    # Handle case where script is run from different directory
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    try:
        from scripts.claude_commands_utils import install_claude_commands, check_claude_code_cli
    except ImportError:
        install_claude_commands = None
        check_claude_code_cli = None

# Global variable to store the uv executable path
UV_EXECUTABLE_PATH = None

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
    print(f"  -> {text}")

def print_error(text):
    """Print formatted error text."""
    print(f"  [ERROR] {text}")

def print_success(text):
    """Print formatted success text."""
    print(f"  [OK] {text}")

def print_warning(text):
    """Print formatted warning text."""
    print(f"  [WARNING]  {text}")

# Cache for system detection to avoid duplicate calls
_system_info_cache = None

def detect_system():
    """Detect the system architecture and platform."""
    global _system_info_cache
    if _system_info_cache is not None:
        return _system_info_cache
    
    system = platform.system().lower()
    machine = platform.machine().lower()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    is_windows = system == "windows"
    is_macos = system == "darwin"
    is_linux = system == "linux"
    is_arm = machine in ("arm64", "aarch64")
    is_x86 = machine in ("x86_64", "amd64", "x64")
    
    # Fix Windows version detection - Windows 11 reports as Windows 10
    if is_windows:
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
            build_number = winreg.QueryValueEx(key, "CurrentBuildNumber")[0]
            winreg.CloseKey(key)
            
            # Windows 11 has build number >= 22000
            if int(build_number) >= 22000:
                windows_version = "11"
            else:
                windows_version = platform.release()
        except (ImportError, OSError, ValueError):
            windows_version = platform.release()
        
        print_info(f"System: {platform.system()} {windows_version}")
    else:
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
            print_info("Checking for Homebrew PyTorch installation...")
            result = subprocess.run(
                ['brew', 'list', 'pytorch', '--version'],
                capture_output=True,
                text=True,
                timeout=30  # Increased timeout to prevent hanging
            )
            if result.returncode == 0:
                has_homebrew_pytorch = True
                # Extract version from output
                version_line = result.stdout.strip()
                homebrew_pytorch_version = version_line.split()[1] if len(version_line.split()) > 1 else "Unknown"
                print_info(f"Detected Homebrew PyTorch installation: {homebrew_pytorch_version}")
        except subprocess.TimeoutExpired:
            print_info("Homebrew PyTorch detection timed out - skipping")
            has_homebrew_pytorch = False
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    _system_info_cache = {
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
    return _system_info_cache

def check_sqlite_extension_support():
    """Check if Python's sqlite3 supports loading extensions."""
    import sqlite3
    
    test_conn = None
    try:
        test_conn = sqlite3.connect(":memory:")
        if not hasattr(test_conn, 'enable_load_extension'):
            return False, "Python sqlite3 module not compiled with extension support"
        
        # Test if we can actually enable extension loading
        test_conn.enable_load_extension(True)
        test_conn.enable_load_extension(False)
        return True, "Extension loading supported"
        
    except AttributeError as e:
        return False, f"enable_load_extension not available: {e}"
    except Exception as e:
        return False, f"Extension support check failed: {e}"
    finally:
        if test_conn:
            test_conn.close()

def detect_gpu():
    """Detect GPU and acceleration capabilities."""
    system_info = detect_system()
    
    # Check for CUDA
    has_cuda = False
    cuda_version = None
    if system_info["is_windows"]:
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path and os.path.exists(cuda_path):
            has_cuda = True
            try:
                # Try to get CUDA version
                nvcc_output = subprocess.check_output([os.path.join(cuda_path, 'bin', 'nvcc'), '--version'], 
                                                     stderr=subprocess.STDOUT, 
                                                     universal_newlines=True)
                for line in nvcc_output.split('\n'):
                    if 'release' in line:
                        cuda_version = line.split('release')[-1].strip().split(',')[0].strip()
                        break
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
    elif system_info["is_linux"]:
        cuda_paths = ['/usr/local/cuda', os.environ.get('CUDA_HOME')]
        for path in cuda_paths:
            if path and os.path.exists(path):
                has_cuda = True
                try:
                    # Try to get CUDA version
                    nvcc_output = subprocess.check_output([os.path.join(path, 'bin', 'nvcc'), '--version'], 
                                                         stderr=subprocess.STDOUT, 
                                                         universal_newlines=True)
                    for line in nvcc_output.split('\n'):
                        if 'release' in line:
                            cuda_version = line.split('release')[-1].strip().split(',')[0].strip()
                            break
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass
                break
    
    # Check for ROCm (AMD)
    has_rocm = False
    rocm_version = None
    if system_info["is_linux"]:
        rocm_paths = ['/opt/rocm', os.environ.get('ROCM_HOME')]
        for path in rocm_paths:
            if path and os.path.exists(path):
                has_rocm = True
                try:
                    # Try to get ROCm version
                    with open(os.path.join(path, 'bin', '.rocmversion'), 'r') as f:
                        rocm_version = f.read().strip()
                except (FileNotFoundError, IOError):
                    try:
                        rocm_output = subprocess.check_output(['rocminfo'], 
                                                            stderr=subprocess.STDOUT, 
                                                            universal_newlines=True)
                        for line in rocm_output.split('\n'):
                            if 'Version' in line:
                                rocm_version = line.split(':')[-1].strip()
                                break
                    except (subprocess.SubprocessError, FileNotFoundError):
                        pass
                break
    
    # Check for MPS (Apple Silicon)
    has_mps = False
    if system_info["is_macos"] and system_info["is_arm"]:
        try:
            # Check if Metal is supported
            print_info("Checking Metal GPU support...")
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True,
                text=True,
                timeout=30
            )
            has_mps = 'Metal' in result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    # Check for DirectML (Windows)
    has_directml = False
    if system_info["is_windows"]:
        try:
            # Check if DirectML package is installed
            import pkg_resources
            pkg_resources.get_distribution('torch-directml')
            has_directml = True
        except (ImportError, pkg_resources.DistributionNotFound):
            # Check if DirectML is available on the system
            try:
                import ctypes
                ctypes.WinDLL('DirectML.dll')
                has_directml = True
            except (ImportError, OSError):
                pass
    
    # Print GPU information
    if has_cuda:
        print_info(f"CUDA detected: {cuda_version or 'Unknown version'}")
    if has_rocm:
        print_info(f"ROCm detected: {rocm_version or 'Unknown version'}")
    if has_mps:
        print_info("Apple Metal Performance Shaders (MPS) detected")
    if has_directml:
        print_info("DirectML detected")
    
    if not (has_cuda or has_rocm or has_mps or has_directml):
        print_info("No GPU acceleration detected, will use CPU-only mode")
    
    return {
        "has_cuda": has_cuda,
        "cuda_version": cuda_version,
        "has_rocm": has_rocm,
        "rocm_version": rocm_version,
        "has_mps": has_mps,
        "has_directml": has_directml
    }

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
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
        pip_installed = True
        print_info("pip is installed")
    except subprocess.SubprocessError:
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
                            "Continuing installation...")
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
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'setuptools'], 
                                    stdout=subprocess.DEVNULL)
                print_success("setuptools installed successfully")
            except subprocess.SubprocessError:
                # Check if in virtual environment
                in_venv = sys.prefix != sys.base_prefix
                if in_venv:
                    print_warning("Failed to install setuptools with pip. If you're using an alternative package manager "
                                "like uv, please install setuptools manually using that tool (e.g., 'uv pip install setuptools').")
                else:
                    print_error("Failed to install setuptools. Please install it manually.")
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
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'wheel'], 
                                    stdout=subprocess.DEVNULL)
                print_success("wheel installed successfully")
            except subprocess.SubprocessError:
                # Check if in virtual environment
                in_venv = sys.prefix != sys.base_prefix
                if in_venv:
                    print_warning("Failed to install wheel with pip. If you're using an alternative package manager "
                                "like uv, please install wheel manually using that tool (e.g., 'uv pip install wheel').")
                else:
                    print_error("Failed to install wheel. Please install it manually.")
                    return False
        else:
            # Should be unreachable since pip_installed would only be False if we returned earlier
            print_error("Cannot install wheel without pip. Please install wheel manually.")
            return False
    
    return True

def install_pytorch_platform_specific(system_info, gpu_info, args=None):
    """Install PyTorch with platform-specific configurations."""
    # Check if PyTorch installation should be skipped
    if args and args.skip_pytorch:
        print_info("Skipping PyTorch installation as requested")
        return True
        
    if system_info["is_windows"]:
        return install_pytorch_windows(gpu_info)
    elif system_info["is_macos"] and system_info["is_x86"]:
        return install_pytorch_macos_intel()
    elif system_info["is_macos"] and system_info["is_arm"]:
        return install_pytorch_macos_arm64()
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
            
            try:
                # Try to install without version specifiers to get latest compatible version
                cmd = [
                    sys.executable, '-m', 'pip', 'install',
                    "torch", "torchvision", "torchaudio"
                ]
                print_info(f"Running: {' '.join(cmd)}")
                subprocess.check_call(cmd)
                st_version = "3.0.0"  # Newer sentence-transformers for newer PyTorch
            except subprocess.SubprocessError as e:
                print_warning(f"Failed to install latest PyTorch: {e}")
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
                print_info(f"Running: {' '.join(cmd)}")
                subprocess.check_call(cmd)
        else:
            # Use traditional versions for older Python
            torch_version = "1.13.1"
            torch_vision_version = "0.14.1"
            torch_audio_version = "0.13.1"
            st_version = "2.2.2"
            
            print_info(f"Installing PyTorch {torch_version} for macOS Intel (Python {python_version.major}.{python_version.minor})...")
            
            # Install PyTorch first with compatible version
            cmd = [
                sys.executable, '-m', 'pip', 'install',
                f"torch=={torch_version}",
                f"torchvision=={torch_vision_version}",
                f"torchaudio=={torch_audio_version}"
            ]
            
            print_info(f"Running: {' '.join(cmd)}")
            subprocess.check_call(cmd)
        
        # Install a compatible version of sentence-transformers
        print_info(f"Installing sentence-transformers {st_version}...")
        
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            f"sentence-transformers=={st_version}"
        ]
        
        print_info(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        print_success(f"PyTorch {torch_version} and sentence-transformers {st_version} installed successfully for macOS Intel")
        return True
    except subprocess.SubprocessError as e:
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

def install_pytorch_macos_arm64():
    """Install PyTorch specifically for macOS with ARM64 (Apple Silicon)."""
    print_step("3a", "Installing PyTorch for macOS ARM64 (Apple Silicon)")
    
    try:
        # For Apple Silicon, we can use the latest PyTorch with MPS support
        print_info("Installing PyTorch with Metal Performance Shaders (MPS) support...")
        
        # Install PyTorch with MPS support - let pip choose the best compatible version
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch>=2.0.0',
            'torchvision',
            'torchaudio'
        ]
        
        print_info(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        # Install sentence-transformers
        print_info("Installing sentence-transformers (for embedding generation)...")
        print_info("Note: Models will be downloaded on first use (~25MB)")
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            'sentence-transformers>=2.2.2'
        ]
        
        print_info(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        print_success("PyTorch and sentence-transformers installed successfully for macOS ARM64")
        print_info("MPS (Metal Performance Shaders) acceleration is available for GPU compute")
        
        return True
    except subprocess.SubprocessError as e:
        print_error(f"Failed to install PyTorch for macOS ARM64: {e}")
        
        # Provide fallback instructions
        print_warning("You may need to manually install PyTorch for Apple Silicon:")
        print_info("pip install torch torchvision torchaudio")
        print_info("pip install sentence-transformers")
        print_info("")
        print_info("If you encounter issues, try:")
        print_info("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu")
        
        return False

def install_pytorch_windows(gpu_info):
    """Install PyTorch on Windows using the appropriate index URL."""
    print_step("3a", "Installing PyTorch for Windows")
    
    # Check if PyTorch is already installed and compatible
    pytorch_installed = False
    torch_version_installed = None
    directml_compatible = False
    
    try:
        import torch
        torch_version_installed = torch.__version__
        pytorch_installed = True
        print_info(f"PyTorch {torch_version_installed} is already installed")
        
        # Check if version is compatible with DirectML (2.4.x works, 2.5.x doesn't)
        version_parts = torch_version_installed.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if gpu_info["has_directml"]:
            if major == 2 and minor == 4:
                directml_compatible = True
                print_success(f"PyTorch {torch_version_installed} is compatible with DirectML")
                
                # Check if torch-directml is also installed
                try:
                    import torch_directml
                    directml_version = getattr(torch_directml, '__version__', 'Unknown version')
                    print_success(f"torch-directml {directml_version} is already installed")
                    return True  # Everything is compatible, no need to reinstall
                except ImportError:
                    print_info("torch-directml not found, will install it")
                    # Install torch-directml only
                    try:
                        subprocess.check_call([
                            sys.executable, '-m', 'pip', 'install', 'torch-directml==0.2.5.dev240914'
                        ])
                        print_success("torch-directml installed successfully")
                        return True
                    except subprocess.SubprocessError:
                        print_warning("Failed to install torch-directml - DirectML support will be limited")
                        return True  # Still return True since PyTorch works
                        
            elif major == 2 and minor >= 5:
                print_warning(f"PyTorch {torch_version_installed} is not compatible with torch-directml")
                print_info("torch-directml requires PyTorch 2.4.x, but 2.5.x is installed")
                print_info("Keeping existing PyTorch installation - DirectML support will be limited")
                return True  # Don't break existing installation
            else:
                print_info(f"PyTorch {torch_version_installed} compatibility with DirectML is unknown")
        else:
            # No DirectML needed, check if current version is reasonable
            if major == 2 and minor >= 4:
                print_success(f"PyTorch {torch_version_installed} is acceptable for CPU usage")
                return True  # Keep existing installation
                
    except ImportError:
        print_info("PyTorch not found, will install compatible version")
    
    # If we get here, we need to install PyTorch
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
        # Use versions compatible with DirectML if needed
        if gpu_info["has_directml"]:
            # Use PyTorch 2.4.x which is compatible with torch-directml
            torch_version = "2.4.1"
            torchvision_version = "0.19.1"  # Compatible with torch 2.4.1
            torchaudio_version = "2.4.1"
            print_info("Using PyTorch 2.4.1 for DirectML compatibility")
        else:
            # Use latest version for non-DirectML systems
            torch_version = "2.5.1"
            torchvision_version = "0.20.1"  # Compatible with torch 2.5.1
            torchaudio_version = "2.5.1"
            print_info("Using PyTorch 2.5.1 for optimal performance")
        
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            f"torch=={torch_version}",
            f"torchvision=={torchvision_version}",
            f"torchaudio=={torchaudio_version}",
            f"--index-url={index_url}"
        ]
        
        print_info(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        # Check if DirectML is needed
        if gpu_info["has_directml"]:
            print_info("Installing torch-directml for DirectML support")
            try:
                # Try the latest dev version since stable versions aren't available
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', 'torch-directml==0.2.5.dev240914'
                ])
            except subprocess.SubprocessError:
                print_warning("Failed to install torch-directml - DirectML support will be limited")
                print_info("You can install manually later with: pip install torch-directml==0.2.5.dev240914")
            
        print_success("PyTorch installed successfully for Windows")
        return True
    except subprocess.SubprocessError as e:
        print_error(f"Failed to install PyTorch for Windows: {e}")
        print_warning("You may need to manually install PyTorch using instructions from https://pytorch.org/get-started/locally/")
        return False

def detect_storage_backend_compatibility(system_info, gpu_info):
    """Detect which storage backends are compatible with the current environment."""
    print_step("3a", "Analyzing storage backend compatibility")
    
    compatibility = {
        "chromadb": {"supported": True, "issues": [], "recommendation": "legacy"},
        "sqlite_vec": {"supported": True, "issues": [], "recommendation": "default"}
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
        status = "[OK]" if info["supported"] else "[X]"
        rec_text = {
            "recommended": "[*] RECOMMENDED",
            "default": "[+] Standard",
            "problematic": "[!] May have issues",
            "lightweight": "[-] Lightweight"
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
        recommended_backend = "sqlite_vec"  # Default fallback
    
    # Interactive selection if no auto-recommendation is clear
    if compatibility["chromadb"]["recommendation"] == "problematic":
        print_step("3b", "Storage Backend Selection")
        print_info("Based on your system, ChromaDB may have installation issues.")
        print_info("SQLite-vec is recommended as a lightweight, compatible alternative.")
        print_info("")
        print_info("Available options:")
        print_info("  1. SQLite-vec (Recommended) - Lightweight, fast, minimal dependencies")
        print_info("  2. ChromaDB (Standard) - Full-featured but may have issues on your system")
        print_info("  3. Auto-detect - Try ChromaDB first, fallback to SQLite-vec if it fails")
        print_info("")
        
        while True:
            try:
                if args.non_interactive:
                    print_info("Non-interactive mode: using default storage backend (SQLite-vec)")
                    choice = "1"
                else:
                    print("\n" + "=" * 60)
                    print("⚠️  USER INPUT REQUIRED")
                    print("=" * 60)
                    choice = input("Choose storage backend [1-3] (default: 1, press Enter for default): ").strip()
                    print("=" * 60 + "\n")
                    if not choice:
                        choice = "1"
                
                if choice == "1":
                    return "sqlite_vec"
                elif choice == "2":
                    return "chromadb"
                elif choice == "3":
                    return "auto_detect"
                else:
                    print_error("Please enter 1, 2, or 3")
            except (EOFError, KeyboardInterrupt):
                print_info("\nUsing recommended backend: sqlite_vec")
                return "sqlite_vec"
    
    return recommended_backend

def install_storage_backend(backend, system_info):
    """Install the chosen storage backend."""
    print_step("3c", f"Installing {backend} storage backend")
    
    if backend == "sqlite_vec":
        # Check extension support before attempting installation
        extension_supported, extension_message = check_sqlite_extension_support()
        if not extension_supported:
            print_warning(f"SQLite extension support not available: {extension_message}")
            
            # Provide platform-specific guidance
            if platform.system().lower() == "darwin":  # macOS
                print_info("This is common on macOS with system Python.")
                print_info("SOLUTIONS:")
                print_info("  • Install Python via Homebrew: brew install python")
                print_info("  • Use pyenv with extensions: PYTHON_CONFIGURE_OPTS='--enable-loadable-sqlite-extensions' pyenv install 3.12.0")
                print_info("  • Switch to ChromaDB backend: --storage-backend chromadb")
                
                # Ask user what they want to do
                if not system_info.get('non_interactive'):
                    print("\n" + "=" * 60)
                    print("⚠️  USER INPUT REQUIRED")
                    print("=" * 60)
                    print("sqlite-vec requires SQLite extension support, which is not available.")
                    response = input("Switch to ChromaDB backend instead? (y/N): ").strip().lower()
                    print("=" * 60 + "\n")
                    if response in ['y', 'yes']:
                        print_info("Switching to ChromaDB backend...")
                        return install_storage_backend("chromadb", system_info)
                    else:
                        print_info("Continuing with sqlite-vec installation (may fail at runtime)...")
                else:
                    print_info("Non-interactive mode: attempting sqlite-vec installation anyway")
            else:
                print_info("Consider switching to ChromaDB backend for better compatibility")
        
        # Special handling for Python 3.13
        if sys.version_info >= (3, 13):
            print_info("Detected Python 3.13+ - using special installation method for sqlite-vec")
            return install_sqlite_vec_python313(system_info)
        
        # Standard installation for older Python versions
        try:
            print_info("Installing SQLite-vec...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sqlite-vec'])
            print_success("SQLite-vec installed successfully")
            return True
        except subprocess.SubprocessError as e:
            print_error(f"Failed to install SQLite-vec: {e}")
            return False
            
    elif backend == "chromadb":
        print_error("ChromaDB backend has been removed in v8.0.0")
        print_info("Please use one of the supported backends:")
        print_info("  - 'hybrid': Local speed + cloud persistence (recommended)")
        print_info("  - 'sqlite_vec': Fast local storage")
        print_info("  - 'cloudflare': Cloud storage only")
        print_info("\nTo migrate from ChromaDB, run: python scripts/migration/migrate_to_sqlite_vec.py")
        return False
            
    elif backend == "auto_detect":
        print_info("Attempting auto-detection...")
        
        # Try ChromaDB first
        print_info("Trying ChromaDB installation...")
        if install_storage_backend("chromadb", system_info):
            print_success("ChromaDB installed successfully")
            return "chromadb"
        
        print_warning("ChromaDB installation failed, falling back to SQLite-vec...")
        if install_storage_backend("sqlite_vec", system_info):
            print_success("SQLite-vec installed successfully as fallback")
            return "sqlite_vec"
        
        print_error("Both storage backends failed to install")
        return False
    
    return False

def initialize_sqlite_vec_database(storage_path):
    """Initialize SQLite-vec database during installation."""
    try:
        print_info("Initializing SQLite-vec database...")
        
        # Add src to path for imports
        src_path = str(Path(__file__).parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Import required modules
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        from mcp_memory_service.models.memory import Memory
        from mcp_memory_service.utils.hashing import generate_content_hash
        import asyncio
        
        async def init_db():
            # Create storage instance
            storage = SqliteVecMemoryStorage(str(storage_path))
            
            # Initialize the database
            await storage.initialize()
            
            # Create a test memory to verify the database works
            test_content = "Database initialization successful"
            test_memory = Memory(
                content=test_content,
                content_hash=generate_content_hash(test_content),
                tags=["init", "system"],
                memory_type="system"
            )
            
            # Store test memory
            success, message = await storage.store(test_memory)
            return success, message
        
        # Run initialization
        success, message = asyncio.run(init_db())
        
        if success:
            print_success(f"SQLite-vec database initialized: {storage_path}")
            return True
        else:
            print_warning(f"Database initialization partially failed: {message}")
            return True  # Database exists even if test failed
            
    except ImportError as e:
        print_warning(f"Could not initialize database (dependencies missing): {e}")
        print_info("Database will be initialized on first use")
        return True  # Not a critical failure
    except Exception as e:
        print_warning(f"Database initialization failed: {e}")
        print_info("Database will be initialized on first use")
        return True  # Not a critical failure

def install_sqlite_vec_python313(system_info):
    """Special installation method for sqlite-vec on Python 3.13+."""
    print_info("Python 3.13+ detected - sqlite-vec may not have pre-built wheels yet")
    
    # Check if uv is available
    uv_path = shutil.which("uv")
    use_uv = uv_path is not None
    
    # Try multiple installation strategies
    strategies = []
    
    if use_uv:
        # Strategy 1: Try with uv pip
        strategies.append({
            'name': 'uv pip install',
            'cmd': [uv_path, 'pip', 'install', 'sqlite-vec'],
            'description': 'Installing with uv package manager'
        })
        
        # Strategy 2: Try with uv pip and no-binary flag
        strategies.append({
            'name': 'uv pip install (source build)',
            'cmd': [uv_path, 'pip', 'install', '--no-binary', ':all:', 'sqlite-vec'],
            'description': 'Building from source with uv'
        })
    
    # Strategy 3: Standard pip install
    strategies.append({
        'name': 'pip install',
        'cmd': [sys.executable, '-m', 'pip', 'install', 'sqlite-vec'],
        'description': 'Installing with pip'
    })
    
    # Strategy 4: pip with no-binary flag to force compilation
    strategies.append({
        'name': 'pip install (source build)',
        'cmd': [sys.executable, '-m', 'pip', 'install', '--no-binary', ':all:', 'sqlite-vec'],
        'description': 'Building from source with pip'
    })
    
    # Strategy 5: Install from GitHub directly
    strategies.append({
        'name': 'GitHub install',
        'cmd': [sys.executable, '-m', 'pip', 'install', 'git+https://github.com/asg017/sqlite-vec.git#subdirectory=python'],
        'description': 'Installing directly from GitHub'
    })
    
    # Try each strategy
    for i, strategy in enumerate(strategies, 1):
        try:
            print_info(f"Attempt {i}/{len(strategies)}: {strategy['description']}...")
            subprocess.check_call(strategy['cmd'], stderr=subprocess.PIPE)
            print_success(f"SQLite-vec installed successfully using {strategy['name']}")
            return True
        except subprocess.CalledProcessError as e:
            print_warning(f"{strategy['name']} failed: {e}")
            if i < len(strategies):
                print_info("Trying next installation method...")
            continue
        except Exception as e:
            print_warning(f"{strategy['name']} failed with unexpected error: {e}")
            continue
    
    # All strategies failed - provide manual instructions
    print_error("Failed to install sqlite-vec with all automatic methods")
    print_info("")
    print_info("MANUAL INSTALLATION OPTIONS:")
    print_info("")
    print_info("Option 1: Use Python 3.12 (recommended)")
    print_info("  1. Install Python 3.12: brew install python@3.12")
    print_info("  2. Create venv: python3.12 -m venv .venv")
    print_info("  3. Activate: source .venv/bin/activate")
    print_info("  4. Re-run: python install.py")
    print_info("")
    print_info("Option 2: Install pysqlite3-binary (alternative)")
    print_info("  pip install pysqlite3-binary")
    print_info("")
    print_info("Option 3: Wait for sqlite-vec Python 3.13 support")
    print_info("  Check: https://github.com/asg017/sqlite-vec/issues")
    print_info("")
    print_info("Option 4: Use hybrid or cloudflare backend")
    print_info("  python install.py --storage-backend hybrid")
    print_info("")

    # Note: ChromaDB option removed in v8.0.0
    
    return False

def install_uv():
    """Install uv package manager if not already installed."""
    uv_path = shutil.which("uv")
    if uv_path:
        print_info(f"uv is already installed at: {uv_path}")
        return uv_path
    
    print_info("Installing uv package manager...")
    
    try:
        # Determine the installation directory
        if platform.system() == 'Windows':
            # On Windows, install to user's AppData/Local
            install_dir = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'uv')
        else:
            # On Unix-like systems, install to ~/.local/bin
            install_dir = os.path.expanduser("~/.local/bin")
        
        # Create installation directory if it doesn't exist
        os.makedirs(install_dir, exist_ok=True)
        
        # Download and install uv
        if platform.system() == 'Windows':
            # Windows installation
            install_script = "powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
            subprocess.check_call(install_script, shell=True)
        else:
            # Unix-like installation
            install_script = "curl -LsSf https://astral.sh/uv/install.sh | sh"
            subprocess.check_call(install_script, shell=True)
        
        # Check if uv was installed successfully
        uv_path = shutil.which("uv")
        if not uv_path:
            # Try common installation paths
            if platform.system() == 'Windows':
                possible_paths = [
                    os.path.join(install_dir, 'uv.exe'),
                    os.path.join(os.environ.get('USERPROFILE', ''), '.cargo', 'bin', 'uv.exe')
                ]
            else:
                possible_paths = [
                    os.path.join(install_dir, 'uv'),
                    os.path.expanduser("~/.cargo/bin/uv")
                ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    uv_path = path
                    break
        
        if uv_path:
            print_success(f"uv installed successfully at: {uv_path}")
            return uv_path
        else:
            print_error("uv installation completed but executable not found in PATH")
            print_info("You may need to add the installation directory to your PATH")
            return None
            
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install uv: {e}")
        return None
    except Exception as e:
        print_error(f"Unexpected error installing uv: {e}")
        return None

def install_package(args):
    """Install the package with the appropriate dependencies, supporting pip or uv."""
    print_step("3", "Installing MCP Memory Service")

    # Determine installation mode
    install_mode = []
    if args.dev:
        install_mode = ['-e']
        print_info("Installing in development mode")

    # Set environment variables for installation
    env = os.environ.copy()

    # Detect if pip is available
    pip_available = False
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        pip_available = True
    except subprocess.SubprocessError:
        pip_available = False

    # Try to install uv if pip is not available
    if not pip_available:
        print_info("pip not found, attempting to install uv package manager...")
        uv_path = install_uv()
    else:
        # Check if uv is already available
        uv_path = shutil.which("uv")
        if uv_path:
            print_info(f"uv package manager found at: {uv_path}")
        else:
            print_info("uv package manager not found (will use pip for installation)")
    
    # Store the uv path globally for config generation
    global UV_EXECUTABLE_PATH
    UV_EXECUTABLE_PATH = uv_path

    # Decide installer command prefix
    if pip_available:
        installer_cmd = [sys.executable, '-m', 'pip']
    elif uv_path:
        installer_cmd = [uv_path, 'pip']
        print_info(f"Using uv for installation: {uv_path}")
    else:
        print_error("Neither pip nor uv could be found or installed. Cannot install packages.")
        return False

    # Get system and GPU info
    system_info = detect_system()
    gpu_info = detect_gpu()

    # Choose and install storage backend
    chosen_backend = choose_storage_backend(system_info, gpu_info, args)
    if chosen_backend == "auto_detect":
        # Handle auto-detection case
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
    if chosen_backend == "sqlite_vec":
        env['MCP_MEMORY_STORAGE_BACKEND'] = 'sqlite_vec'
        os.environ['MCP_MEMORY_STORAGE_BACKEND'] = 'sqlite_vec'
        print_info("Configured to use SQLite-vec storage backend")
    else:
        env['MCP_MEMORY_STORAGE_BACKEND'] = 'chromadb'
        os.environ['MCP_MEMORY_STORAGE_BACKEND'] = 'chromadb'
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
        pytorch_installed = install_pytorch_platform_specific(system_info, gpu_info, args)
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
                cmd = installer_cmd + ['install', '--no-deps'] + install_mode + ['.']
                print_info(f"Running: {' '.join(cmd)}")
                subprocess.check_call(cmd, env=env)
                
                # Install core dependencies except torch/sentence-transformers
                print_info("Installing core dependencies (without ML libraries for compatibility)...")
                print_info("Note: First run will download embedding models automatically (~25MB)")
                
                # Create a list of dependencies to install
                dependencies = [
                    "mcp>=1.0.0,<2.0.0",
                    "onnxruntime>=1.14.1",  # ONNX runtime for embeddings
                    "tokenizers>=0.20.0",  # Required for ONNX tokenization
                    "httpx>=0.24.0",  # For downloading ONNX models
                    "aiohttp>=3.8.0"  # Required for MCP server functionality
                ]
                
                # Add backend-specific dependencies
                if chosen_backend == "sqlite_vec":
                    dependencies.append("sqlite-vec>=0.1.0")
                else:
                    dependencies.append("chromadb==0.5.23")
                    dependencies.append("tokenizers==0.20.3")
                
                # Install dependencies
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install'] + dependencies
                )
                
                # Set environment variables for ONNX
                print_info("Configuring to use ONNX runtime for inference without PyTorch...")
                env['MCP_MEMORY_USE_ONNX'] = '1'
                os.environ['MCP_MEMORY_USE_ONNX'] = '1'  # Also set in the main process
                if chosen_backend != "sqlite_vec":
                    print_info("Switching to SQLite-vec backend for better compatibility")
                    env['MCP_MEMORY_STORAGE_BACKEND'] = 'sqlite_vec'
                    os.environ['MCP_MEMORY_STORAGE_BACKEND'] = 'sqlite_vec'  # Also set in the main process
                    chosen_backend = 'sqlite_vec'  # Update the chosen_backend for consistency
                
                print_success("MCP Memory Service installed successfully (SQLite-vec + ONNX)")
                
                if using_homebrew_pytorch:
                    print_info("Using Homebrew PyTorch installation for embedding generation")
                    print_info("Environment configured to use SQLite-vec backend and ONNX runtime")
                else:
                    print_warning("ML libraries (PyTorch/sentence-transformers) were not installed due to compatibility issues")
                    print_info("The service will use ONNX runtime for inference instead")
                
                return True
            except subprocess.SubprocessError as e:
                print_error(f"Failed to install with ONNX approach: {e}")
                # Fall through to try standard installation
        
        # Standard installation
        cmd = installer_cmd + ['install'] + install_mode + ['.']
        print_info(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd, env=env)
        print_success("MCP Memory Service installed successfully")
        return True
    except subprocess.SubprocessError as e:
        print_error(f"Failed to install MCP Memory Service: {e}")
        
        # Special handling for macOS with compatibility issues
        if system_info["is_macos"] and system_info["is_x86"]:
            print_warning("Installation on macOS Intel is challenging")
            print_info("Try manually installing with:")
            print_info("1. pip install --no-deps .")
            print_info("2. pip install sqlite-vec>=0.1.0 mcp>=1.0.0,<2.0.0 onnxruntime>=1.14.1 aiohttp>=3.8.0")
            print_info("3. export MCP_MEMORY_USE_ONNX=1")
            print_info("4. export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec")
            
            if system_info.get("has_homebrew_pytorch"):
                print_info("Homebrew PyTorch was detected but installation still failed.")
                print_info("Try running: python install.py --storage-backend sqlite_vec --skip-pytorch")
            
        return False

def configure_paths(args):
    """Configure paths for the MCP Memory Service."""
    print_step("4", "Configuring paths")
    
    # Get system info
    system_info = detect_system()
    
    # Determine home directory
    home_dir = Path.home()
    
    # Determine base directory based on platform
    if platform.system() == 'Darwin':  # macOS
        base_dir = home_dir / 'Library' / 'Application Support' / 'mcp-memory'
    elif platform.system() == 'Windows':  # Windows
        base_dir = Path(os.environ.get('LOCALAPPDATA', '')) / 'mcp-memory'
    else:  # Linux and others
        base_dir = home_dir / '.local' / 'share' / 'mcp-memory'
    
    # Create directories based on storage backend
    storage_backend = args.storage_backend or os.environ.get('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec')
    
    if storage_backend == 'sqlite_vec':
        # For sqlite-vec, we need a database file path
        storage_path = args.chroma_path or (base_dir / 'sqlite_vec.db')
        storage_dir = storage_path.parent if storage_path.name.endswith('.db') else storage_path
        backups_path = args.backups_path or (base_dir / 'backups')
        
        try:
            os.makedirs(storage_dir, exist_ok=True)
            os.makedirs(backups_path, exist_ok=True)
            print_info(f"SQLite-vec database: {storage_path}")
            print_info(f"Backups path: {backups_path}")
            
            # Test if directory is writable
            test_file = os.path.join(storage_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            
            # Initialize the SQLite-vec database
            if not initialize_sqlite_vec_database(storage_path):
                print_warning("SQLite-vec database initialization failed, but continuing...")
                
        except Exception as e:
            print_error(f"Failed to configure SQLite-vec paths: {e}")
            return False
    else:
        # ChromaDB configuration
        chroma_path = args.chroma_path or (base_dir / 'chroma_db')
        backups_path = args.backups_path or (base_dir / 'backups')
        storage_path = chroma_path
        
        try:
            os.makedirs(chroma_path, exist_ok=True)
            os.makedirs(backups_path, exist_ok=True)
            print_info(f"ChromaDB path: {chroma_path}")
            print_info(f"Backups path: {backups_path}")
            
            # Test if directories are writable
            test_file = os.path.join(chroma_path, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            print_error(f"Failed to configure ChromaDB paths: {e}")
            return False
    
    # Test backups directory for both backends
    try:
        test_file = os.path.join(backups_path, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print_success("Storage directories created and are writable")
    except Exception as e:
        print_error(f"Failed to test backups directory: {e}")
        return False
    
    # Configure Claude Desktop if available
    claude_config_paths = [
            home_dir / 'Library' / 'Application Support' / 'Claude' / 'claude_desktop_config.json',
            home_dir / '.config' / 'Claude' / 'claude_desktop_config.json',
            Path('claude_config') / 'claude_desktop_config.json'
        ]
    
    for config_path in claude_config_paths:
        if config_path.exists():
            print_info(f"Found Claude Desktop config at {config_path}")
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Update or add MCP Memory configuration
                if 'mcpServers' not in config:
                    config['mcpServers'] = {}
                
                # Create environment configuration based on storage backend
                env_config = {
                    "MCP_MEMORY_BACKUPS_PATH": str(backups_path),
                    "MCP_MEMORY_STORAGE_BACKEND": storage_backend
                }
                
                if storage_backend == 'sqlite_vec':
                    env_config["MCP_MEMORY_SQLITE_PATH"] = str(storage_path)
                else:
                    env_config["MCP_MEMORY_CHROMA_PATH"] = str(storage_path)
                
                # Create or update the memory server configuration
                if system_info["is_windows"]:
                    # Use the memory_wrapper.py script for Windows
                    script_path = os.path.abspath("memory_wrapper.py")
                    config['mcpServers']['memory'] = {
                        "command": "python",
                        "args": [script_path],
                        "env": env_config
                    }
                    print_info("Configured Claude Desktop to use memory_wrapper.py for Windows")
                else:
                    # Use the standard configuration for other platforms
                    config['mcpServers']['memory'] = {
                        "command": UV_EXECUTABLE_PATH or "uv",
                        "args": [
                            "--directory",
                            os.path.abspath("."),
                            "run",
                            "memory"
                        ],
                        "env": env_config
                    }
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print_success("Updated Claude Desktop configuration")
            except Exception as e:
                print_warning(f"Failed to update Claude Desktop configuration: {e}")
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

def is_legacy_hardware(system_info):
    """Detect legacy hardware that needs special handling."""
    if system_info["is_macos"] and system_info["is_x86"]:
        # Check if it's likely an older Intel Mac
        # This is a heuristic based on common patterns
        try:
            # Try to get more detailed system info
            print_info("Detecting hardware configuration (this may take a moment)...")
            result = subprocess.run(
                ['system_profiler', 'SPHardwareDataType'],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                output = result.stdout.lower()
                # Look for indicators of older hardware
                if any(year in output for year in ['2013', '2014', '2015', '2016', '2017']):
                    return True
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass
    
    return False

def detect_memory_gb():
    """Detect available system memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # Fallback detection methods
        try:
            if platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    return int(result.stdout.strip()) / (1024**3)
            elif platform.system() == "Linux":
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            return int(line.split()[1]) / (1024**2)
        except (subprocess.SubprocessError, FileNotFoundError, IOError):
            pass
    
    return 0  # Unknown

def recommend_backend_intelligent(system_info, gpu_info, memory_gb, args):
    """Intelligent backend recommendation based on hardware analysis."""
    # User explicitly chose backend
    if hasattr(args, 'storage_backend') and args.storage_backend:
        return args.storage_backend
    
    # Legacy hardware mode
    if args.legacy_hardware or is_legacy_hardware(system_info):
        print_info("[DETECT] Legacy hardware detected - optimizing for compatibility")
        return "sqlite_vec"
    
    # Server mode
    if args.server_mode:
        print_info("[SERVER] Server mode - selecting lightweight backend")
        return "sqlite_vec"
    
    # Low memory systems
    if memory_gb > 0 and memory_gb < 4:
        print_info(f"[MEMORY] Limited memory detected ({memory_gb:.1f}GB) - using efficient backend")
        return "sqlite_vec"
    
    # macOS Intel with known ChromaDB issues
    if system_info["is_macos"] and system_info["is_x86"]:
        compatibility = detect_storage_backend_compatibility(system_info, gpu_info)
        if compatibility["chromadb"]["recommendation"] == "problematic":
            print_info("[WARNING] macOS Intel compatibility issues detected - using SQLite-vec")
            # Set environment variables for consistent backend selection
            os.environ['MCP_MEMORY_STORAGE_BACKEND'] = 'sqlite_vec'
            # For Intel Macs, also enable ONNX runtime for better compatibility
            if system_info.get("has_homebrew_pytorch") or sys.version_info >= (3, 13):
                print_info("[CONFIG] Enabling ONNX runtime for better compatibility")
                os.environ['MCP_MEMORY_USE_ONNX'] = '1'
            return "sqlite_vec"
    
    # Hardware with GPU acceleration - SQLite-vec still recommended for simplicity
    if gpu_info.get("has_cuda") or gpu_info.get("has_mps") or gpu_info.get("has_directml"):
        gpu_type = "CUDA" if gpu_info.get("has_cuda") else "MPS" if gpu_info.get("has_mps") else "DirectML"
        print_info(f"[GPU] {gpu_type} acceleration detected - SQLite-vec recommended for simplicity and speed")
        return "sqlite_vec"
    
    # High memory systems without GPU - explain the choice
    if memory_gb >= 16:
        print_info("[CHOICE] High-memory system without GPU detected")
        print_info("  -> SQLite-vec: Faster startup, simpler setup, no network dependencies")
        print_info("  -> ChromaDB: Legacy option, being deprecated in v6.0.0")
        print_info("  -> Defaulting to SQLite-vec (recommended for all users)")
        return "sqlite_vec"
    
    # Default recommendation for most users
    print_info("[DEFAULT] Recommending SQLite-vec for optimal user experience")
    return "sqlite_vec"

def show_detailed_help():
    """Show detailed hardware-specific installation help."""
    print_header("MCP Memory Service - Hardware-Specific Installation Guide")
    
    # Detect current system
    system_info = detect_system()
    gpu_info = detect_gpu()
    memory_gb = detect_memory_gb()
    
    # Check SQLite extension support
    extension_supported, extension_message = check_sqlite_extension_support()
    
    print_info("Your System Configuration:")
    print_info(f"  Platform: {platform.system()} {platform.release()}")
    print_info(f"  Architecture: {platform.machine()}")
    print_info(f"  Python: {sys.version_info.major}.{sys.version_info.minor}")
    if memory_gb > 0:
        print_info(f"  Memory: {memory_gb:.1f}GB")
    print_info(f"  SQLite Extensions: {'✅ Supported' if extension_supported else '❌ Not Supported'}")
    
    # Warn about potential sqlite-vec issues
    if not extension_supported and platform.system().lower() == "darwin":
        print_warning("SQLite extension support not available - this may cause issues with sqlite-vec backend")
        print_info("Consider using Homebrew Python: brew install python")
    
    # Hardware-specific recommendations
    print_step("Recommendations", "Based on your hardware")
    
    if is_legacy_hardware(system_info):
        print_success("Legacy Hardware Path (2013-2017 Intel Mac)")
        print_info("  Recommended: python install.py --legacy-hardware")
        print_info("  This will:")
        print_info("    • Use SQLite-vec backend (avoids ChromaDB compatibility issues)")
        print_info("    • Configure ONNX runtime for CPU-only inference")
        print_info("    • Use Homebrew PyTorch for better compatibility")
        print_info("    • Optimize resource usage for older hardware")
    elif system_info["is_macos"] and system_info["is_arm"]:
        print_success("Apple Silicon Mac - Modern Hardware Path")
        print_info("  Recommended: python install.py")
        print_info("  This will:")
        print_info("    • Use SQLite-vec backend (fast and efficient)")
        print_info("    • Enable MPS acceleration")
        print_info("    • Zero network dependencies")
    elif system_info["is_windows"] and gpu_info.get("has_cuda"):
        print_success("Windows with CUDA GPU - High Performance Path")
        print_info("  Recommended: python install.py")
        print_info("  This will:")
        print_info("    • Use SQLite-vec backend (fast and efficient)")
        print_info("    • Enable CUDA acceleration")
        print_info("    • Zero network dependencies")
    elif memory_gb > 0 and memory_gb < 4:
        print_success("Low-Memory System")
        print_info("  Recommended: python install.py --storage-backend sqlite_vec")
        print_info("  This will:")
        print_info("    • Use lightweight SQLite-vec backend")
        print_info("    • Minimize memory usage")
        print_info("    • Enable ONNX runtime for efficiency")
    elif memory_gb >= 16 and not (gpu_info.get("has_cuda") or gpu_info.get("has_mps") or gpu_info.get("has_directml")):
        print_success("High-Memory System (No GPU) - Choose Your Path")
        print_info("  Option 1 (Recommended): python install.py")
        print_info("    • SQLite-vec: Fast startup, simple setup, same features")
        print_info("  Option 2: python install.py --storage-backend chromadb")
        print_info("    • ChromaDB: Better for 10K+ memories, production deployments")
        print_info("  Most users benefit from SQLite-vec's simplicity")
    elif gpu_info.get("has_cuda") or gpu_info.get("has_mps") or gpu_info.get("has_directml"):
        gpu_type = "CUDA" if gpu_info.get("has_cuda") else "MPS" if gpu_info.get("has_mps") else "DirectML"
        print_success(f"GPU-Accelerated System ({gpu_type}) - High Performance Path")
        print_info("  Recommended: python install.py")
        print_info("  This will:")
        print_info(f"    • Use SQLite-vec backend (fast and efficient)")
        print_info(f"    • Enable {gpu_type} hardware acceleration")
        print_info("    • Zero network dependencies")
    else:
        print_success("Standard Installation")
        print_info("  Recommended: python install.py")
        print_info("  This will:")
        print_info("    • Use SQLite-vec backend (optimal for most users)")
        print_info("    • Fast startup and simple setup")
        print_info("    • Full semantic search capabilities")
    
    print_step("Available Options", "Command-line flags you can use")
    print_info("  --legacy-hardware     : Optimize for 2013-2017 Intel Macs")
    print_info("  --server-mode         : Headless server installation")
    print_info("  --storage-backend X   : Force backend (chromadb/sqlite_vec)")
    print_info("  --enable-http-api     : Enable HTTP/SSE web interface")
    print_info("  --use-homebrew-pytorch: Use existing Homebrew PyTorch")
    
    print_step("Documentation", "Hardware-specific guides")
    print_info("  Legacy Mac Guide: docs/platforms/macos-intel-legacy.md")
    print_info("  Backend Comparison: docs/guides/STORAGE_BACKENDS.md")
    print_info("  Master Guide: docs/guides/INSTALLATION_MASTER.md")

def generate_personalized_docs():
    """Generate personalized setup documentation."""
    print_header("Generating Personalized Setup Guide")
    
    # Detect system
    system_info = detect_system()
    gpu_info = detect_gpu()
    memory_gb = detect_memory_gb()
    
    # Create personalized guide
    guide_content = f"""# Your Personalized MCP Memory Service Setup Guide

Generated on: {platform.node()} at {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Your System Configuration

- **Platform**: {platform.system()} {platform.release()}
- **Architecture**: {platform.machine()}
- **Python Version**: {sys.version_info.major}.{sys.version_info.minor}
- **Memory**: {memory_gb:.1f}GB (detected)
- **GPU**: {'Yes (' + ('CUDA' if gpu_info.get('has_cuda') else 'MPS' if gpu_info.get('has_mps') else 'DirectML' if gpu_info.get('has_directml') else 'Unknown') + ')' if gpu_info.get('has_cuda') or gpu_info.get('has_mps') or gpu_info.get('has_directml') else 'No'}

## Recommended Installation Command

```bash
"""
    
    # Generate recommendation
    class Args:
        storage_backend = None
        legacy_hardware = False
        server_mode = False
    
    args = Args()
    recommended_backend = recommend_backend_intelligent(system_info, gpu_info, memory_gb, args)
    
    if is_legacy_hardware(system_info):
        guide_content += "python install.py --legacy-hardware\n"
    elif memory_gb < 4:
        guide_content += "python install.py --storage-backend sqlite_vec\n"
    else:
        guide_content += "python install.py\n"
    
    guide_content += f"""```

## Why This Configuration?

Based on your {platform.system()} system with {memory_gb:.1f}GB RAM:
"""
    
    if is_legacy_hardware(system_info):
        guide_content += """
- [OK] **Hardware Compatibility**: SQLite-vec avoids ChromaDB installation issues on older Intel Macs
- [OK] **Homebrew PyTorch**: Better compatibility with older systems and reduced dependencies
- [OK] **ONNX Runtime**: CPU-optimized inference for systems without GPU acceleration
- [OK] **Memory Efficient**: Optimized resource usage for systems with limited RAM
- [OK] **Full Feature Set**: Complete semantic search, tagging, and time-based recall capabilities
"""
    elif recommended_backend == "sqlite_vec":
        if memory_gb >= 16 and not (gpu_info.get("has_cuda") or gpu_info.get("has_mps") or gpu_info.get("has_directml")):
            guide_content += """
- [OK] **Smart Choice**: SQLite-vec recommended for high-memory systems without GPU
- [OK] **No GPU Needed**: ChromaDB's advantages require GPU acceleration you don't have
- [OK] **Faster Startup**: Database ready in 2-3 seconds vs ChromaDB's 15-30 seconds
- [OK] **Simpler Setup**: Single-file database, no complex dependencies
- [OK] **Full Feature Set**: Complete semantic search, tagging, and time-based recall capabilities
- [INFO] **Alternative**: Use `--storage-backend chromadb` if you plan 10K+ memories
"""
        else:
            guide_content += """
- [OK] **SQLite-vec Backend**: Lightweight with complete vector search capabilities
- [OK] **Low Memory Usage**: Optimized for systems with limited RAM
- [OK] **Quick Startup**: Database ready in seconds
- [OK] **Full Feature Set**: Semantic search, tagging, time-based recall
"""
    else:
        guide_content += """
- [OK] **ChromaDB Backend**: Production-grade with advanced HNSW indexing and rich ecosystem
- [OK] **Hardware Acceleration**: Takes advantage of your GPU/MPS acceleration
- [OK] **Scalable Performance**: Optimized for large datasets (10K+ memories) and complex metadata queries
- [OK] **Full Feature Set**: Complete semantic search, tagging, and time-based recall capabilities
"""
    
    guide_content += f"""
## Next Steps

1. **Run the installation**:
   ```bash
   cd mcp-memory-service
   {guide_content.split('```bash')[1].split('```')[0].strip()}
   ```

2. **Test the installation**:
   ```bash
   python scripts/test_memory_simple.py
   ```

3. **Configure Claude Desktop**:
   The installer will generate the optimal configuration for your system.

## Troubleshooting

If you encounter issues, see the platform-specific guide:
- **Legacy Mac Issues**: docs/platforms/macos-intel-legacy.md
- **General Issues**: docs/guides/troubleshooting.md
- **Backend Selection**: docs/guides/STORAGE_BACKENDS.md

## Support

Generated configuration ID: {hash(str(system_info) + str(gpu_info))}-{int(__import__('time').time())}
Include this ID when requesting support for faster assistance.
"""
    
    # Write the guide
    guide_path = "YOUR_PERSONALIZED_SETUP_GUIDE.md"
    with open(guide_path, 'w') as f:
        f.write(guide_content)
    
    print_success(f"Personalized setup guide created: {guide_path}")
    print_info("This guide contains hardware-specific recommendations for your system")
    print_info("Keep this file for future reference and troubleshooting")

def configure_claude_code_integration(system_info):
    """Configure Claude Code MCP integration with optimized settings."""
    print_step("6", "Configuring Claude Code Integration")
    
    # Check if Claude Code is installed
    try:
        result = subprocess.run(['claude', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print_warning("Claude Code CLI not found. Please install it first:")
            print_info("curl -fsSL https://claude.ai/install.sh | sh")
            return False
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        print_warning("Claude Code CLI not found. Please install it first:")
        print_info("curl -fsSL https://claude.ai/install.sh | sh")
        return False
    
    print_success("Claude Code CLI detected")
    
    # Load template and create personalized .mcp.json
    template_path = Path('.mcp.json.template')
    if not template_path.exists():
        print_error("Template file .mcp.json.template not found")
        return False
    
    try:
        import json
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Replace placeholders with actual values
        user_home = str(Path.home())
        personalized_content = template_content.replace('{{USER_HOME}}', user_home)
        
        # Create .mcp.json
        mcp_config_path = Path('.mcp.json')
        with open(mcp_config_path, 'w') as f:
            f.write(personalized_content)
        
        print_success(f"Created personalized .mcp.json configuration")
        print_info(f"Configuration file: {mcp_config_path.absolute()}")
        
        # Add to .gitignore if it exists
        gitignore_path = Path('.gitignore')
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
            
            if '.mcp.json' not in gitignore_content:
                with open(gitignore_path, 'a') as f:
                    f.write('\n# MCP configuration (contains personal paths)\n.mcp.json\n')
                print_success("Added .mcp.json to .gitignore")
        
        # Verify Claude Code can see the configuration
        try:
            result = subprocess.run(['claude', 'mcp', 'list'], 
                                  capture_output=True, text=True, timeout=10, cwd='.')
            if 'memory-service' in result.stdout:
                print_success("Claude Code MCP integration configured successfully!")
                print_info("You can now use memory functions in Claude Code")
            else:
                print_warning("Configuration created but memory-service not detected")
                print_info("You may need to restart Claude Code or check the configuration")
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            print_warning("Could not verify Claude Code configuration")
            print_info("Configuration file created - restart Claude Code to use memory functions")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to configure Claude Code integration: {e}")
        return False

def detect_mcp_clients():
    """Detect installed MCP-compatible applications."""
    clients = {}
    
    # Check for Claude Desktop
    claude_config_paths = [
        Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json",  # Windows
        Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",  # macOS
        Path.home() / ".config" / "Claude" / "claude_desktop_config.json"  # Linux
    ]
    for path in claude_config_paths:
        if path.exists():
            clients['claude_desktop'] = path
            break
    
    # Check for Claude Code CLI
    try:
        result = subprocess.run(['claude', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            clients['claude_code'] = True
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check for VS Code with MCP extension
    vscode_settings_paths = [
        Path.home() / "AppData" / "Roaming" / "Code" / "User" / "settings.json",  # Windows
        Path.home() / "Library" / "Application Support" / "Code" / "User" / "settings.json",  # macOS
        Path.home() / ".config" / "Code" / "User" / "settings.json"  # Linux
    ]
    for path in vscode_settings_paths:
        if path.exists():
            try:
                import json
                with open(path, 'r') as f:
                    settings = json.load(f)
                # Check for MCP-related extensions or configurations
                if any('mcp' in str(key).lower() or 'model-context-protocol' in str(key).lower() 
                       for key in settings.keys()):
                    clients['vscode_mcp'] = path
                    break
            except (json.JSONDecodeError, IOError):
                pass
    
    # Check for Continue IDE
    continue_paths = [
        Path.home() / ".continue" / "config.json",
        Path.home() / ".config" / "continue" / "config.json",
        Path.home() / "AppData" / "Roaming" / "continue" / "config.json"  # Windows
    ]
    for path in continue_paths:
        if path.exists():
            clients['continue'] = path
            break
    
    # Check for generic MCP configurations
    generic_mcp_paths = [
        Path.home() / ".mcp.json",
        Path.home() / ".config" / "mcp" / "config.json",
        Path.cwd() / ".mcp.json"
    ]
    for path in generic_mcp_paths:
        if path.exists():
            clients['generic_mcp'] = path
            break
    
    # Check for Cursor IDE (similar to VS Code)
    cursor_settings_paths = [
        Path.home() / "AppData" / "Roaming" / "Cursor" / "User" / "settings.json",  # Windows
        Path.home() / "Library" / "Application Support" / "Cursor" / "User" / "settings.json",  # macOS
        Path.home() / ".config" / "Cursor" / "User" / "settings.json"  # Linux
    ]
    for path in cursor_settings_paths:
        if path.exists():
            try:
                import json
                with open(path, 'r') as f:
                    settings = json.load(f)
                # Check for MCP-related configurations
                if any('mcp' in str(key).lower() or 'model-context-protocol' in str(key).lower() 
                       for key in settings.keys()):
                    clients['cursor'] = path
                    break
            except (json.JSONDecodeError, IOError):
                pass
    
    return clients

def print_detected_clients(clients):
    """Print information about detected MCP clients."""
    if clients:
        print_info("Detected MCP Clients:")
        for client_type, config_path in clients.items():
            client_names = {
                'claude_desktop': 'Claude Desktop',
                'claude_code': 'Claude Code CLI',
                'vscode_mcp': 'VS Code with MCP',
                'continue': 'Continue IDE',
                'cursor': 'Cursor IDE',
                'generic_mcp': 'Generic MCP Client'
            }
            client_name = client_names.get(client_type, client_type.title())
            config_info = config_path if isinstance(config_path, (str, Path)) else "CLI detected"
            print_info(f"  [*] {client_name}: {config_info}")
    else:
        print_info("No MCP clients detected - configuration will work with any future MCP client")

def should_offer_multi_client_setup(args, final_backend):
    """Determine if multi-client setup should be offered."""
    # Only offer if using SQLite-vec backend (requirement for multi-client)
    if final_backend != "sqlite_vec":
        return False
    
    # Don't offer in pure server mode
    if args.server_mode:
        return False
    
    # Skip if user explicitly requested to skip
    if args.skip_multi_client_prompt:
        return False
    
    # Always beneficial for development environments - any future MCP client can benefit
    return True

def configure_detected_clients(clients, system_info, storage_backend="sqlite_vec"):
    """Configure each detected client for multi-client access."""
    success_count = 0
    
    for client_type, config_path in clients.items():
        try:
            if client_type == 'claude_desktop':
                if configure_claude_desktop_multi_client(config_path, system_info, storage_backend):
                    success_count += 1
            elif client_type == 'vscode_mcp' or client_type == 'cursor':
                if configure_vscode_like_multi_client(config_path, client_type, storage_backend):
                    success_count += 1
            elif client_type == 'continue':
                if configure_continue_multi_client(config_path, storage_backend):
                    success_count += 1
            elif client_type == 'generic_mcp':
                if configure_generic_mcp_multi_client(config_path, storage_backend):
                    success_count += 1
            elif client_type == 'claude_code':
                # Claude Code uses project-level .mcp.json, handle separately
                print_info(f"  -> Claude Code: Configure via project .mcp.json (see instructions below)")
                success_count += 1
        except Exception as e:
            print_warning(f"  -> Failed to configure {client_type}: {e}")
    
    return success_count

def configure_claude_desktop_multi_client(config_path, system_info, storage_backend="sqlite_vec"):
    """Configure Claude Desktop for multi-client access."""
    try:
        import json
        
        # Read existing configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Ensure mcpServers section exists
        if 'mcpServers' not in config:
            config['mcpServers'] = {}
        
        # Update memory server configuration with multi-client settings
        repo_path = str(Path.cwd()).replace('\\', '\\\\')  # Escape backslashes for JSON
        
        # Build environment configuration based on storage backend
        env_config = {
            "MCP_MEMORY_STORAGE_BACKEND": storage_backend,
            "LOG_LEVEL": "INFO"
        }
        
        # Add backend-specific configuration
        if storage_backend == "sqlite_vec":
            env_config["MCP_MEMORY_SQLITE_PRAGMAS"] = "busy_timeout=15000,cache_size=20000"
            # SQLite path will be auto-determined by the service
        else:  # chromadb
            # ChromaDB path will be auto-determined by the service
            pass
        
        config['mcpServers']['memory'] = {
            "command": UV_EXECUTABLE_PATH or "uv",
            "args": ["--directory", repo_path, "run", "memory"],
            "env": env_config
        }
        
        # Write updated configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print_info(f"  [OK] Claude Desktop: Updated configuration for multi-client access")
        return True
        
    except Exception as e:
        print_warning(f"  -> Claude Desktop configuration failed: {e}")
        return False

def configure_vscode_like_multi_client(config_path, client_type, storage_backend="sqlite_vec"):
    """Configure VS Code or Cursor for multi-client access."""
    try:
        import json
        
        # For VS Code/Cursor, we provide instructions rather than modifying settings directly
        # since MCP configuration varies by extension
        
        client_name = "VS Code" if client_type == 'vscode_mcp' else "Cursor"
        print_info(f"  -> {client_name}: MCP extension detected")
        print_info(f"    Add memory server to your MCP extension with these settings:")
        print_info(f"    - Backend: {storage_backend}")
        if storage_backend == "sqlite_vec":
            print_info(f"    - Database: shared SQLite-vec database")
        else:
            print_info(f"    - Database: shared ChromaDB database")
        print_info(f"    - See generic configuration below for details")
        return True
        
    except Exception as e:
        print_warning(f"  -> {client_type} configuration failed: {e}")
        return False

def configure_continue_multi_client(config_path, storage_backend="sqlite_vec"):
    """Configure Continue IDE for multi-client access."""
    try:
        import json
        
        # Read existing Continue configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Add or update MCP server configuration for Continue
        if 'mcpServers' not in config:
            config['mcpServers'] = {}
        
        repo_path = str(Path.cwd())
        
        # Build environment configuration based on storage backend
        env_config = {
            "MCP_MEMORY_STORAGE_BACKEND": storage_backend,
            "LOG_LEVEL": "INFO"
        }
        
        # Add backend-specific configuration
        if storage_backend == "sqlite_vec":
            env_config["MCP_MEMORY_SQLITE_PRAGMAS"] = "busy_timeout=15000,cache_size=20000"
        
        config['mcpServers']['memory'] = {
            "command": UV_EXECUTABLE_PATH or "uv",
            "args": ["--directory", repo_path, "run", "memory"],
            "env": env_config
        }
        
        # Write updated configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print_info(f"  [OK] Continue IDE: Updated configuration for multi-client access")
        return True
        
    except Exception as e:
        print_warning(f"  -> Continue IDE configuration failed: {e}")
        return False

def configure_generic_mcp_multi_client(config_path, storage_backend="sqlite_vec"):
    """Configure generic MCP client for multi-client access."""
    try:
        import json
        
        # Read existing configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Add memory server if not present
        if 'mcpServers' not in config:
            config['mcpServers'] = {}
        
        repo_path = str(Path.cwd())
        
        # Build environment configuration based on storage backend
        env_config = {
            "MCP_MEMORY_STORAGE_BACKEND": storage_backend,
            "LOG_LEVEL": "INFO"
        }
        
        # Add backend-specific configuration
        if storage_backend == "sqlite_vec":
            env_config["MCP_MEMORY_SQLITE_PRAGMAS"] = "busy_timeout=15000,cache_size=20000"
        
        config['mcpServers']['memory'] = {
            "command": UV_EXECUTABLE_PATH or "uv",
            "args": ["--directory", repo_path, "run", "memory"],
            "env": env_config
        }
        
        # Write updated configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print_info(f"  [OK] Generic MCP Client: Updated configuration")
        return True
        
    except Exception as e:
        print_warning(f"  -> Generic MCP client configuration failed: {e}")
        return False

async def test_wal_mode_coordination():
    """Test WAL mode storage coordination for multi-client access."""
    try:
        # Add src to path for import
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        from mcp_memory_service.models.memory import Memory
        from mcp_memory_service.utils.hashing import generate_content_hash
        
        import tempfile
        import asyncio
        
        # Create a temporary database for testing
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            test_db_path = tmp.name
        
        try:
            # Test direct SQLite-vec storage with WAL mode
            print_info("  -> Testing WAL mode coordination...")
            storage = SqliteVecMemoryStorage(test_db_path)
            await storage.initialize()
            
            # Test storing a memory
            content = "Multi-client setup test - WAL mode verification"
            test_memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=["setup", "wal-test", "multi-client"],
                memory_type="test"
            )
            
            # Store memory
            success, message = await storage.store(test_memory)
            if not success:
                print_warning(f"  -> Memory storage failed: {message}")
                return False
            
            # Test concurrent access simulation
            storage2 = SqliteVecMemoryStorage(test_db_path)
            await storage2.initialize()
            
            # Both should be able to read
            results1 = await storage.search_by_tag(["setup"])
            results2 = await storage2.search_by_tag(["setup"])
            
            if len(results1) != len(results2) or len(results1) == 0:
                print_warning("  -> Concurrent read access test failed")
                return False
            
            # Test concurrent write
            content2 = "Second client test memory"
            memory2 = Memory(
                content=content2,
                content_hash=generate_content_hash(content2),
                tags=["setup", "client2"],
                memory_type="test"
            )
            
            success2, _ = await storage2.store(memory2)
            if not success2:
                print_warning("  -> Concurrent write access test failed")
                return False
            
            # Verify both clients can see both memories
            all_results = await storage.search_by_tag(["setup"])
            if len(all_results) < 2:
                print_warning("  -> Multi-client data sharing test failed")
                return False
            
            storage.close()
            storage2.close()
            
            print_info("  [OK] WAL mode coordination test passed")
            return True
            
        finally:
            # Cleanup test files
            try:
                os.unlink(test_db_path)
                for ext in ["-wal", "-shm"]:
                    try:
                        os.unlink(test_db_path + ext)
                    except:
                        pass
            except:
                pass
                
    except Exception as e:
        print_warning(f"  -> WAL mode test failed: {e}")
        return False

def setup_shared_environment():
    """Set up shared environment variables for multi-client access."""
    try:
        print_info("  -> Configuring shared environment variables...")
        
        # Set environment variables in current process (for testing)
        os.environ['MCP_MEMORY_STORAGE_BACKEND'] = 'sqlite_vec'
        os.environ['MCP_MEMORY_SQLITE_PRAGMAS'] = 'busy_timeout=15000,cache_size=20000'
        os.environ['LOG_LEVEL'] = 'INFO'
        
        print_info("  [OK] Environment variables configured")
        
        # Provide instructions for permanent setup
        system_info = detect_system()
        if system_info["is_windows"]:
            print_info("  -> For permanent setup, run these PowerShell commands as Administrator:")
            print_info("    [System.Environment]::SetEnvironmentVariable('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec', [System.EnvironmentVariableTarget]::User)")
            print_info("    [System.Environment]::SetEnvironmentVariable('MCP_MEMORY_SQLITE_PRAGMAS', 'busy_timeout=15000,cache_size=20000', [System.EnvironmentVariableTarget]::User)")
            print_info("    [System.Environment]::SetEnvironmentVariable('LOG_LEVEL', 'INFO', [System.EnvironmentVariableTarget]::User)")
        else:
            print_info("  -> For permanent setup, add to your shell profile:")
            print_info("    export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec")
            print_info("    export MCP_MEMORY_SQLITE_PRAGMAS='busy_timeout=15000,cache_size=20000'")
            print_info("    export LOG_LEVEL=INFO")
        
        return True
        
    except Exception as e:
        print_warning(f"  -> Environment setup failed: {e}")
        return False

def provide_generic_configuration(storage_backend="sqlite_vec"):
    """Provide configuration instructions for any MCP client."""
    print_info("")
    print_info("Universal MCP Client Configuration:")
    print_info("=" * 50)
    print_info("For any MCP-compatible client, use these settings:")
    print_info("")
    print_info("MCP Server Configuration:")
    
    repo_path = str(Path.cwd())
    
    # Windows-style path
    uv_cmd = UV_EXECUTABLE_PATH or "uv"
    if platform.system() == 'Windows':
        print_info(f"  Command: {uv_cmd} --directory \"{repo_path}\" run memory")
        print_info(f"  Alternative: python -m mcp_memory_service.server")
    else:
        print_info(f"  Command: {uv_cmd} --directory {repo_path} run memory")
        print_info(f"  Alternative: python -m mcp_memory_service.server")
    
    print_info("")
    print_info("Environment Variables:")
    print_info(f"  MCP_MEMORY_STORAGE_BACKEND={storage_backend}")
    if storage_backend == "sqlite_vec":
        print_info("  MCP_MEMORY_SQLITE_PRAGMAS=busy_timeout=15000,cache_size=20000")
    print_info("  LOG_LEVEL=INFO")
    
    print_info("")
    print_info("Shared Database Location:")
    if storage_backend == "sqlite_vec":
        if platform.system() == 'Windows':
            print_info("  %LOCALAPPDATA%\\mcp-memory\\sqlite_vec.db")
        elif platform.system() == 'Darwin':
            print_info("  ~/Library/Application Support/mcp-memory/sqlite_vec.db")
        else:
            print_info("  ~/.local/share/mcp-memory/sqlite_vec.db")
    else:  # chromadb
        if platform.system() == 'Windows':
            print_info("  %LOCALAPPDATA%\\mcp-memory\\chroma_db")
        elif platform.system() == 'Darwin':
            print_info("  ~/Library/Application Support/mcp-memory/chroma_db")
        else:
            print_info("  ~/.local/share/mcp-memory/chroma_db")
    
    print_info("")
    print_info("Claude Code Project Configuration (.mcp.json):")
    print_info("  {")
    print_info("    \"mcpServers\": {")
    print_info("      \"memory\": {")
    print_info(f"        \"command\": \"{UV_EXECUTABLE_PATH or 'uv'}\",")
    print_info(f"        \"args\": [\"--directory\", \"{repo_path}\", \"run\", \"memory\"],")
    print_info("        \"env\": {")
    print_info(f"          \"MCP_MEMORY_STORAGE_BACKEND\": \"{storage_backend}\",")
    if storage_backend == "sqlite_vec":
        print_info("          \"MCP_MEMORY_SQLITE_PRAGMAS\": \"busy_timeout=15000,cache_size=20000\",")
    print_info("          \"LOG_LEVEL\": \"INFO\"")
    print_info("        }")
    print_info("      }")
    print_info("    }")
    print_info("  }")

def setup_universal_multi_client_access(system_info, args, storage_backend="sqlite_vec"):
    """Configure multi-client access for any MCP-compatible clients."""
    print_step("7", "Configuring Universal Multi-Client Access")
    
    print_info("Setting up multi-client coordination for all MCP applications...")
    print_info("Benefits:")
    print_info("  • Share memories between Claude Desktop, VS Code, Continue, and other MCP clients")
    print_info("  • Seamless context sharing across development environments")
    print_info("  • Single source of truth for all your project memories")
    print_info("")
    
    # Test WAL mode coordination only for sqlite_vec
    if storage_backend == "sqlite_vec":
        try:
            import asyncio
            wal_success = asyncio.run(test_wal_mode_coordination())
            if not wal_success:
                print_error("WAL mode coordination test failed")
                return False
        except Exception as e:
            print_error(f"Failed to test WAL mode coordination: {e}")
            return False
    
    # Detect available MCP clients
    detected_clients = detect_mcp_clients()
    print_detected_clients(detected_clients)
    print_info("")
    
    # Configure each detected client
    print_info("Configuring detected clients...")
    success_count = configure_detected_clients(detected_clients, system_info, storage_backend)
    
    # Set up shared environment variables
    setup_shared_environment()
    
    # Provide generic configuration for manual setup
    provide_generic_configuration(storage_backend)
    
    print_info("")
    print_success(f"Multi-client setup complete! {success_count} clients configured automatically.")
    print_info("")
    print_info("Next Steps:")
    print_info("  1. Restart your applications (Claude Desktop, VS Code, etc.)")
    print_info("  2. All clients will share the same memory database")
    print_info("  3. Test: Store memory in one app, access from another")
    print_info("  4. For Claude Code: Create .mcp.json in your project directory")
    
    return True

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
    parser.add_argument('--storage-backend', choices=['chromadb', 'sqlite_vec', 'auto_detect'],
                        help='Choose storage backend: chromadb (default), sqlite_vec (lightweight), or auto_detect (try chromadb, fallback to sqlite_vec)')
    parser.add_argument('--skip-pytorch', action='store_true',
                        help='Skip PyTorch installation and use ONNX runtime with SQLite-vec backend instead')
    parser.add_argument('--use-homebrew-pytorch', action='store_true',
                        help='Use existing Homebrew PyTorch installation instead of pip version')
    parser.add_argument('--force-pytorch', action='store_true',
                        help='Force PyTorch installation even when using SQLite-vec (overrides auto-skip)')
    
    # New intelligent installer options
    parser.add_argument('--legacy-hardware', action='store_true',
                        help='Optimize installation for legacy hardware (2013-2017 Intel Macs)')
    parser.add_argument('--server-mode', action='store_true',
                        help='Install for server/headless deployment (minimal UI dependencies)')
    parser.add_argument('--enable-http-api', action='store_true',
                        help='Enable HTTP/SSE API functionality')
    parser.add_argument('--migrate-from-chromadb', action='store_true',
                        help='Migrate existing ChromaDB installation to selected backend')
    parser.add_argument('--configure-claude-code', action='store_true',
                        help='Automatically configure Claude Code MCP integration with optimized settings')
    parser.add_argument('--help-detailed', action='store_true',
                        help='Show detailed hardware-specific installation recommendations')
    parser.add_argument('--generate-docs', action='store_true',
                        help='Generate personalized setup documentation for your hardware')
    parser.add_argument('--setup-multi-client', action='store_true',
                        help='Configure multi-client access for any MCP-compatible applications (Claude, VS Code, Continue, etc.)')
    parser.add_argument('--skip-multi-client-prompt', action='store_true',
                        help='Skip the interactive prompt for multi-client setup')
    parser.add_argument('--install-claude-commands', action='store_true',
                        help='Install Claude Code commands for memory operations')
    parser.add_argument('--skip-claude-commands-prompt', action='store_true',
                        help='Skip the interactive prompt for Claude Code commands')
    parser.add_argument('--non-interactive', action='store_true',
                        help='Run in non-interactive mode using default values for all prompts')
    
    args = parser.parse_args()
    
    # Handle special help modes
    if args.help_detailed:
        show_detailed_help()
        sys.exit(0)
    
    if args.generate_docs:
        generate_personalized_docs()
        sys.exit(0)
    
    # Set up logging system to capture installation output
    try:
        log_file_path = setup_installer_logging()
    except Exception as e:
        print(f"Warning: Could not set up logging: {e}")
        log_file_path = None
    
    print_header("MCP Memory Service Installation")
    
    # Step 1: Detect system
    print_step("1", "Detecting system")
    system_info = detect_system()
    gpu_info = detect_gpu()
    memory_gb = detect_memory_gb()
    
    # Show memory info if detected
    if memory_gb > 0:
        print_info(f"System memory: {memory_gb:.1f}GB")
    
    # Intelligent backend recommendation
    if not args.storage_backend:
        recommended_backend = recommend_backend_intelligent(system_info, gpu_info, memory_gb, args)
        args.storage_backend = recommended_backend
        print_info(f"Recommended backend: {recommended_backend}")
    
    # Handle legacy hardware special case
    if args.legacy_hardware or is_legacy_hardware(system_info):
        print_step("1a", "Legacy Hardware Optimization")
        args.storage_backend = "sqlite_vec"
        args.use_homebrew_pytorch = True
        print_success("Configuring for legacy hardware compatibility")
        print_info("• SQLite-vec backend selected")
        print_info("• Homebrew PyTorch integration enabled")
        print_info("• ONNX runtime will be configured")
    
    # Handle server mode
    if args.server_mode:
        print_step("1b", "Server Mode Configuration")
        args.storage_backend = "sqlite_vec"
        print_success("Configuring for server deployment")
        print_info("• Lightweight SQLite-vec backend")
        print_info("• Minimal UI dependencies")
    
    # Handle HTTP API
    if args.enable_http_api:
        print_step("1c", "HTTP/SSE API Configuration")
        if args.storage_backend == "chromadb":
            print_warning("HTTP/SSE API works best with SQLite-vec backend")
            if args.non_interactive:
                print_info("Non-interactive mode: switching to SQLite-vec for HTTP API compatibility")
                args.storage_backend = "sqlite_vec"
            else:
                print("\n" + "=" * 60)
                print("⚠️  USER INPUT REQUIRED")
                print("=" * 60)
                response = input("Switch to SQLite-vec for optimal HTTP API experience? (y/N, press Enter for N): ")
                print("=" * 60 + "\n")
                if response.lower().startswith('y'):
                    args.storage_backend = "sqlite_vec"
    
    # Handle migration
    if args.migrate_from_chromadb:
        print_step("1d", "Migration Setup")
        print_info("Preparing to migrate from existing ChromaDB installation")
        
        # Check if ChromaDB data exists
        chromadb_paths = [
            os.path.expanduser("~/.mcp_memory_chroma"),
            os.path.expanduser("~/Library/Application Support/mcp-memory/chroma_db"),
            os.path.expanduser("~/.local/share/mcp-memory/chroma_db")
        ]
        
        chromadb_found = None
        for path in chromadb_paths:
            if os.path.exists(path):
                chromadb_found = path
                break
        
        if chromadb_found:
            print_success(f"Found ChromaDB data at: {chromadb_found}")
            args.storage_backend = "sqlite_vec"  # Migration target
            args.chromadb_found = chromadb_found  # Store for later use
            print_info("Migration will run after installation completes")
        else:
            print_warning("No ChromaDB data found at standard locations")
            if args.non_interactive:
                print_info("Non-interactive mode: skipping ChromaDB migration")
                args.migrate_from_chromadb = False
            else:
                print("\n" + "=" * 60)
                print("⚠️  USER INPUT REQUIRED")
                print("=" * 60)
                manual_path = input("Enter ChromaDB path manually (or press Enter to skip): ").strip()
                print("=" * 60 + "\n")
                if manual_path and os.path.exists(manual_path):
                    chromadb_found = manual_path
                    args.storage_backend = "sqlite_vec"
                    args.chromadb_found = chromadb_found
                else:
                    print_info("Skipping migration - no valid ChromaDB path provided")
                    args.migrate_from_chromadb = False
    
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
                
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install',
                    f"torch=={torch_version}", f"torchvision=={torch_vision_version}", f"torchaudio=={torch_audio_version}",
                    f"sentence-transformers=={st_version}"
                ])
                print_success("Compatible dependencies installed successfully")
            except subprocess.SubprocessError as e:
                print_error(f"Failed to install compatible dependencies: {e}")
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
            
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                f"torch=={torch_version}", f"torchvision=={torch_vision_version}", f"torchaudio=={torch_audio_version}",
                f"sentence-transformers=={st_version}"
            ])
            print_success("Fallback dependencies installed successfully")
        except subprocess.SubprocessError as e:
            print_error(f"Failed to install fallback dependencies: {e}")
    
    # Apply intelligent PyTorch skipping for sqlite_vec (default behavior)
    if (args.storage_backend == "sqlite_vec" and 
        not args.skip_pytorch and 
        not args.force_pytorch):
        print_step("1d", "Optimizing for SQLite-vec setup")
        args.skip_pytorch = True
        print_success("Auto-skipping PyTorch installation for SQLite-vec backend")
        print_info("• SQLite-vec uses SQLite for vector storage (lighter than ChromaDB)")
        print_info("• Note: Embedding models still require PyTorch/SentenceTransformers")
        print_info("• Add --force-pytorch if you want PyTorch installed here")
        print_warning("• You'll need PyTorch available for embedding functionality")
    
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
    
    # Execute migration if requested
    if args.migrate_from_chromadb and hasattr(args, 'chromadb_found') and args.chromadb_found:
        print_step("6", "Migrating from ChromaDB")
        try:
            migration_script = "scripts/migrate_chroma_to_sqlite.py"
            if os.path.exists(migration_script):
                print_info("Running migration script...")
                subprocess.check_call([sys.executable, migration_script, "--auto-confirm"])
                print_success("Migration completed successfully!")
            else:
                print_warning("Migration script not found - manual migration required")
                print_info("Run: python scripts/migrate_chroma_to_sqlite.py")
        except subprocess.SubprocessError as e:
            print_error(f"Migration failed: {e}")
            print_info("You can run migration manually later with:")
            print_info("python scripts/migrate_chroma_to_sqlite.py")
    
    # Step 6: Configure Claude Code integration if requested
    if args.configure_claude_code:
        if not configure_claude_code_integration(system_info):
            print_warning("Claude Code integration configuration failed")
            print_info("You can configure it manually later using the documentation")
    
    # Step 7: Install Claude Code commands if requested or available
    should_install_commands = args.install_claude_commands
    
    # If not explicitly requested, check if we should prompt the user
    if not should_install_commands and not args.skip_claude_commands_prompt:
        if install_claude_commands is not None and check_claude_code_cli is not None:
            claude_available, _ = check_claude_code_cli()
            if claude_available:
                print_step("7", "Optional Claude Code Commands")
                print_info("Claude Code CLI detected! You can install memory operation commands.")
                print_info("Commands would include: /memory-store, /memory-recall, /memory-search, /memory-health")
                
                if args.non_interactive:
                    print_info("Non-interactive mode: skipping Claude Code commands installation")
                    should_install_commands = False
                else:
                    print("\n" + "=" * 60)
                    print("⚠️  USER INPUT REQUIRED")
                    print("=" * 60)
                    response = input("Install Claude Code memory commands? (y/N, press Enter for N): ")
                    print("=" * 60 + "\n")
                    should_install_commands = response.lower().startswith('y')
    
    if should_install_commands:
        if install_claude_commands is not None:
            print_step("7", "Installing Claude Code Commands")
            try:
                if install_claude_commands(verbose=True):
                    print_success("Claude Code commands installed successfully!")
                else:
                    print_warning("Claude Code commands installation had issues")
                    print_info("You can install them manually later with:")
                    print_info("python scripts/claude_commands_utils.py")
            except Exception as e:
                print_error(f"Failed to install Claude Code commands: {str(e)}")
                print_info("You can install them manually later with:")
                print_info("python scripts/claude_commands_utils.py")
        else:
            print_warning("Claude commands utilities not available")
            print_info("Commands installation skipped")
    
    print_header("Installation Complete")
    
    # First-time setup notice
    print_info("")
    print_info("⚠️  FIRST-TIME SETUP EXPECTATIONS:")
    print_info("On first run, you may see these NORMAL warnings:")
    print_info("  • 'No snapshots directory' - Model will download automatically (~25MB)")
    print_info("  • 'TRANSFORMERS_CACHE deprecated' - Informational, doesn't affect operation")
    print_info("  • Model download progress - One-time download (1-2 minutes)")
    print_info("")
    print_info("These warnings disappear after the first successful run.")
    print_info("See docs/first-time-setup.md for details.")
    print_info("")
    
    # Get final storage backend info for multi-client setup determination
    if system_info["is_macos"] and system_info["is_x86"] and system_info.get("has_homebrew_pytorch"):
        final_backend = 'sqlite_vec'
        # Ensure environment variable is set for future use
        os.environ['MCP_MEMORY_STORAGE_BACKEND'] = 'sqlite_vec'
    else:
        final_backend = os.environ.get('MCP_MEMORY_STORAGE_BACKEND', 'chromadb')
    
    # Multi-client setup integration
    if args.setup_multi_client or (should_offer_multi_client_setup(args, final_backend) and not args.setup_multi_client):
        if args.setup_multi_client:
            # User explicitly requested multi-client setup
            try:
                setup_universal_multi_client_access(system_info, args, final_backend)
            except Exception as e:
                print_error(f"Multi-client setup failed: {e}")
                print_info("You can set up multi-client access manually using:")
                print_info("python setup_multi_client_complete.py")
        else:
            # Interactive prompt for multi-client setup
            print_info("")
            print_info("Multi-Client Access Available!")
            print_info("")
            print_info("The MCP Memory Service can be configured for multi-client access, allowing")
            print_info("multiple applications and IDEs to share the same memory database.")
            print_info("")
            print_info("Benefits:")
            print_info("  • Share memories between Claude Desktop, VS Code, Continue, and other MCP clients")
            print_info("  • Seamless context sharing across development environments")
            print_info("  • Single source of truth for all your project memories")
            print_info("")
            
            try:
                if args.non_interactive:
                    print_info("Non-interactive mode: skipping multi-client configuration")
                    response = 'n'
                else:
                    print("\n" + "=" * 60)
                    print("⚠️  USER INPUT REQUIRED")
                    print("=" * 60)
                    response = input("Would you like to configure multi-client access? (y/N, press Enter for N): ").strip().lower()
                    print("=" * 60 + "\n")
                if response in ['y', 'yes']:
                    print_info("")
                    try:
                        setup_universal_multi_client_access(system_info, args, final_backend)
                    except Exception as e:
                        print_error(f"Multi-client setup failed: {e}")
                        print_info("You can set up multi-client access manually using:")
                        print_info("python setup_multi_client_complete.py")
                else:
                    print_info("Skipping multi-client setup. You can configure it later using:")
                    print_info("python setup_multi_client_complete.py")
            except (EOFError, KeyboardInterrupt):
                print_info("\nSkipping multi-client setup. You can configure it later using:")
                print_info("python setup_multi_client_complete.py")
            
            print_info("")
    
    # Final storage backend info (already determined above for multi-client setup)
    
    use_onnx = os.environ.get('MCP_MEMORY_USE_ONNX', '').lower() in ('1', 'true', 'yes')
    
    print_info("You can now run the MCP Memory Service using the 'memory' command")
    print_info(f"Storage Backend: {final_backend.upper()}")
    
    if final_backend == 'sqlite_vec':
        print_success("Using SQLite-vec - lightweight, fast, minimal dependencies")
        print_info("   • No complex dependencies or build issues")
        print_info("   • Excellent performance for typical use cases")
    else:
        print_success("Using ChromaDB - full-featured vector database")
        print_info("   • Advanced features and extensive ecosystem")
    
    if use_onnx:
        print_info("[OK] Using ONNX Runtime for inference")
        print_info("   • Compatible with Homebrew PyTorch")
        print_info("   • Reduced dependencies for better compatibility")
    
    print_info("For more information, see:")
    print_info("  • Installation Guide: docs/guides/INSTALLATION_MASTER.md")
    print_info("  • Backend Comparison: docs/guides/STORAGE_BACKENDS.md")
    if system_info["is_macos"] and system_info["is_x86"] and is_legacy_hardware(system_info):
        print_info("  • Legacy Mac Guide: docs/platforms/macos-intel-legacy.md")
    print_info("  • Main README: README.md")
    
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
        print_info("- For a quick test, try running: python test_memory.py")
    
    # Clean up logging system
    try:
        cleanup_installer_logging()
        if log_file_path:
            print(f"\nInstallation log saved to: {log_file_path}")
    except Exception:
        pass  # Silently ignore cleanup errors

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInstallation interrupted by user")
        cleanup_installer_logging()
        sys.exit(1)
    except Exception as e:
        print(f"\nInstallation failed with error: {e}")
        cleanup_installer_logging()
        sys.exit(1)