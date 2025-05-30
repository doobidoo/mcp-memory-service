#!/usr/bin/env python3
"""
Enhanced Wrapper script for MCP Memory Service with UV integration.
This script ensures that PyTorch is properly installed before running the memory server,
with improved debugging, error handling, and dependency management using UV.
"""
# Disable sitecustomize.py and other import hooks to prevent recursion issues
# These must be set before importing any other modules
import os
os.environ["PYTHONNOUSERSITE"] = "1"  # Disable user site-packages
os.environ["PYTHONPATH"] = ""  # Clear PYTHONPATH

import sys
import subprocess
import importlib.util
import argparse
import traceback
import site
import platform
import time
from pathlib import Path

# Configure logging level
DEBUG = False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MCP Memory Service Wrapper with UV")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-auto-install", action="store_true", help="Disable automatic PyTorch installation")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU-only mode even if GPU is available")
    parser.add_argument("--chroma-path", type=str, help="Path to ChromaDB storage")
    parser.add_argument("--backups-path", type=str, help="Path to backups storage")
    parser.add_argument("--use-echovault", action="store_true", help="Enable EchoVault features")
    return parser.parse_args()

def print_debug(text):
    """Print formatted debug text to stderr."""
    if DEBUG:
        print(f"[DEBUG] {text}", file=sys.stderr, flush=True)

def print_info(text):
    """Print formatted info text to stderr."""
    print(f"[INFO] {text}", file=sys.stderr, flush=True)

def print_error(text):
    """Print formatted error text to stderr."""
    print(f"[ERROR] {text}", file=sys.stderr, flush=True)

def print_success(text):
    """Print formatted success text to stderr."""
    print(f"[SUCCESS] {text}", file=sys.stderr, flush=True)

def print_warning(text):
    """Print formatted warning text to stderr."""
    print(f"[WARNING] {text}", file=sys.stderr, flush=True)

def print_environment_info():
    """Print detailed environment information for debugging."""
    print_debug("=== Environment Information ===")
    print_debug(f"Python version: {sys.version}")
    print_debug(f"Python executable: {sys.executable}")
    print_debug(f"Platform: {platform.platform()}")
    print_debug(f"System: {platform.system()} {platform.release()}")
    print_debug(f"Architecture: {platform.machine()}")
    print_debug(f"Current working directory: {os.getcwd()}")
    print_debug(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Print Python path
    print_debug("Python path:")
    for path in sys.path:
        print_debug(f"  - {path}")
    
    # Print environment variables
    print_debug("Environment variables:")
    for key, value in os.environ.items():
        if key.startswith("PYTHON") or key.startswith("PATH") or key.startswith("MCP") or "TORCH" in key or "UV" in key:
            print_debug(f"  - {key}={value}")
    
    # Check if running in virtual environment
    in_venv = sys.prefix != sys.base_prefix
    print_debug(f"Running in virtual environment: {in_venv}")
    if in_venv:
        print_debug(f"Virtual environment path: {sys.prefix}")

def check_uv():
    """Check if UV is installed and properly configured."""
    print_info("Checking UV installation")
    try:
        subprocess.check_call([sys.executable, '-m', 'uv', '--version'], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
        print_success("UV is installed")
        return True
    except subprocess.SubprocessError:
        print_error("UV is not installed")
        return False

def install_uv():
    """Install UV package manager."""
    print_info("Installing UV package manager")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'uv'])
        print_success("UV installed successfully")
        return True
    except subprocess.SubprocessError as e:
        print_error(f"Failed to install UV: {e}")
        return False

def check_installed_packages():
    """Check and print information about installed packages using UV."""
    print_debug("=== Installed Packages ===")
    
    try:
        # First try with UV if available
        if check_uv():
            print_debug("Checking installed packages with UV")
            try:
                result = subprocess.check_output([sys.executable, '-m', 'uv', 'pip', 'list'], 
                                               universal_newlines=True)
                
                for line in result.splitlines():
                    if any(pkg in line.lower() for pkg in ["torch", "torchvision", "torchaudio", 
                                                          "sentence-transformers", "chromadb", "mcp"]):
                        print_debug(f"  - {line.strip()}")
                return
            except subprocess.SubprocessError:
                print_debug("Failed to list packages with UV, falling back to pip")
        
        # Fallback to pip
        try:
            import pkg_resources
            for package in pkg_resources.working_set:
                if package.key in ["torch", "torchvision", "torchaudio", "sentence-transformers",
                                  "transformers", "chromadb", "mcp", "mcp-memory-service"]:
                    print_debug(f"  - {package.key}=={package.version}")
        except ImportError:
            print_debug("Could not import pkg_resources to check installed packages")
    except Exception as e:
        print_debug(f"Error checking installed packages: {e}")

def check_pytorch():
    """Check if PyTorch is installed and properly configured."""
    print_info("Checking PyTorch installation")
    
    # First attempt to check using UV if available
    if check_uv():
        try:
            # Check if torch is installed using UV
            result = subprocess.check_output([sys.executable, '-m', 'uv', 'pip', 'list'], 
                                           universal_newlines=True)
            if "torch" in result:
                print_success("PyTorch is installed (verified by UV)")
                
                # Continue with regular check to get version details
                pass
            else:
                print_warning("PyTorch not found in UV package list")
                return False
        except subprocess.SubprocessError:
            print_debug("Failed to verify PyTorch with UV, continuing with direct import check")
    
    # Save original sys.path
    original_sys_path = sys.path.copy()
    
    # Temporarily disable import hooks
    original_meta_path = sys.meta_path
    sys.meta_path = [finder for finder in sys.meta_path
                    if not hasattr(finder, 'find_spec') or
                    not hasattr(finder, 'blocked_packages')]
    
    try:
        # Add site-packages to sys.path
        site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')
        if site_packages not in sys.path:
            sys.path.insert(0, site_packages)
        
        # Try to find torch directly in site-packages
        torch_path = os.path.join(site_packages, 'torch')
        if os.path.exists(torch_path) and torch_path not in sys.path:
            sys.path.insert(0, torch_path)
        
        # Try to import torch directly
        print_debug("Attempting to import torch directly")
        
        # Use importlib to import torch
        import importlib.util
        import importlib.machinery
        
        # Find the torch module spec
        torch_spec = importlib.machinery.PathFinder.find_spec('torch', [site_packages])
        if torch_spec is None:
            print_error("Could not find torch module spec")
            return False
        
        # Load the torch module
        torch = importlib.util.module_from_spec(torch_spec)
        torch_spec.loader.exec_module(torch)
        
        print_success(f"PyTorch is installed (version {torch.__version__})")
        print_debug(f"PyTorch location: {torch.__file__}")
        
        # Check if PyTorch C extensions are available
        if hasattr(torch, '_C'):
            print_debug("PyTorch C extensions are available")
        else:
            print_warning("PyTorch C extensions might not be properly installed")
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            print_success(f"CUDA is available (version {torch.version.cuda})")
            print_info(f"GPU: {torch.cuda.get_device_name(0)}")
            print_debug(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            
            # Test a simple CUDA operation
            try:
                x = torch.rand(5, 3).cuda()
                y = torch.rand(5, 3).cuda()
                z = x + y
                print_debug("Basic CUDA tensor operations work correctly")
            except Exception as e:
                print_warning(f"CUDA tensor operations failed: {e}")
                print_warning("Falling back to CPU mode")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            print_info("CUDA is not available, using CPU-only mode")
            
            # Check if DirectML is available
            try:
                directml_spec = importlib.machinery.PathFinder.find_spec('torch_directml', [site_packages])
                if directml_spec is not None:
                    torch_directml = importlib.util.module_from_spec(directml_spec)
                    directml_spec.loader.exec_module(torch_directml)
                    print_success(f"DirectML is available (version {torch_directml.__version__})")
            except Exception:
                print_debug("DirectML is not available")
            
        # Test a simple tensor operation
        try:
            x = torch.rand(5, 3)
            y = torch.rand(5, 3)
            z = x + y
            print_debug("Basic tensor operations work correctly")
        except Exception as e:
            print_error(f"Failed to perform basic tensor operations: {e}")
            return False
        
        # Add torch to sys.modules to make it available to other modules
        sys.modules['torch'] = torch
        
        return True
    except ImportError as e:
        print_error(f"PyTorch is not installed: {e}")
        return False
    except Exception as e:
        print_error(f"Error checking PyTorch installation: {e}")
        print_debug(traceback.format_exc())
        return False
    finally:
        # Restore original sys.path and meta_path
        sys.path = original_sys_path
        sys.meta_path = original_meta_path

def check_sentence_transformers():
    """Check if sentence-transformers is installed and properly configured."""
    print_info("Checking sentence-transformers installation")
    try:
        import sentence_transformers
        print_success(f"sentence-transformers is installed (version {sentence_transformers.__version__})")
        print_debug(f"sentence-transformers location: {sentence_transformers.__file__}")
        return True
    except ImportError as e:
        print_error(f"sentence-transformers is not installed: {e}")
        return False
    except Exception as e:
        print_error(f"Error checking sentence-transformers installation: {e}")
        print_debug(traceback.format_exc())
        return False

def check_chromadb():
    """Check if ChromaDB is installed and properly configured."""
    print_info("Checking ChromaDB installation")
    try:
        import chromadb
        print_success(f"ChromaDB is installed (version {chromadb.__version__})")
        print_debug(f"ChromaDB location: {chromadb.__file__}")
        return True
    except ImportError as e:
        print_error(f"ChromaDB is not installed: {e}")
        return False
    except Exception as e:
        print_error(f"Error checking ChromaDB installation: {e}")
        print_debug(traceback.format_exc())
        return False

def check_mcp():
    """Check if MCP is installed and properly configured."""
    print_info("Checking MCP installation")
    try:
        import mcp
        print_success(f"MCP is installed (version {mcp.__version__})")
        print_debug(f"MCP location: {mcp.__file__}")
        return True
    except ImportError as e:
        print_error(f"MCP is not installed: {e}")
        return False
    except Exception as e:
        print_error(f"Error checking MCP installation: {e}")
        print_debug(traceback.format_exc())
        return False

def check_echovault_dependencies():
    """Check if EchoVault dependencies are installed."""
    print_info("Checking EchoVault dependencies")
    
    # List of required packages
    echovault_packages = [
        "asyncpg",
        "qdrant_client",
        "boto3",
        "opentelemetry.api",
        "prometheus_client"
    ]
    
    missing_packages = []
    
    for package in echovault_packages:
        try:
            # Try to import the package
            __import__(package.split('.')[0])
            print_success(f"{package} is installed")
        except ImportError:
            print_warning(f"{package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print_error(f"Missing EchoVault dependencies: {', '.join(missing_packages)}")
        print_info("Install with: pip install -r requirements-echovault.txt")
        return False
    
    return True

def install_pytorch_with_uv(no_auto_install=False):
    """Install PyTorch using UV with the platform-specific installation method."""
    if no_auto_install:
        print_warning("Automatic PyTorch installation is disabled")
        return False
    
    # Check if UV is installed, install it if not
    if not check_uv():
        print_info("UV is not installed. Installing UV...")
        if not install_uv():
            print_error("Failed to install UV, falling back to pip for PyTorch installation")
            return False
    
    # Get the system information
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # Determine the appropriate installation method based on the platform
    if system == "windows":
        print_info("Installing PyTorch for Windows using UV")
        
        # Check for CUDA
        has_cuda = False
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path and os.path.exists(cuda_path):
            has_cuda = True
            print_info("CUDA detected, installing PyTorch with CUDA support")
            
            # Install PyTorch with CUDA support
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'uv', 'pip', 'install', 
                    'torch==2.1.0', 'torchvision==2.1.0', 'torchaudio==2.1.0',
                    '--extra-index-url', 'https://download.pytorch.org/whl/cu118'
                ])
                print_success("PyTorch installed successfully with CUDA support")
                return True
            except subprocess.SubprocessError as e:
                print_error(f"Failed to install PyTorch with CUDA: {e}")
                print_info("Falling back to CPU-only installation")
        
        # Install PyTorch CPU-only version
        try:
            subprocess.check_call([
                sys.executable, '-m', 'uv', 'pip', 'install',
                'torch==2.1.0', 'torchvision==2.1.0', 'torchaudio==2.1.0',
                '--extra-index-url', 'https://download.pytorch.org/whl/cpu'
            ])
            print_success("PyTorch installed successfully (CPU-only)")
            return True
        except subprocess.SubprocessError as e:
            print_error(f"Failed to install PyTorch with UV: {e}")
            return False
            
    elif system == "darwin":  # macOS
        print_info("Installing PyTorch for macOS using UV")
        try:
            # Install PyTorch for macOS - Use the specific versions you need
            print_debug("Installing PyTorch for macOS using UV")
            subprocess.check_call([
                sys.executable, '-m', 'uv', 'pip', 'install',
                "torch==1.13.1",
                "torchvision==0.14.1",
                "torchaudio==0.13.1"
            ])
            
            print_success("PyTorch installed successfully for macOS")
            return True
        except subprocess.SubprocessError as e:
            print_error(f"Failed to install PyTorch for macOS with UV: {e}")
            return False
    else:  # Linux or other platforms
        print_info("Installing PyTorch for Linux/other platform using UV")
        try:
            # Generic PyTorch installation for Linux
            subprocess.check_call([
                sys.executable, '-m', 'uv', 'pip', 'install',
                "torch==1.13.1",
                "torchvision==0.14.1", 
                "torchaudio==0.13.1"
            ])
            
            print_success("PyTorch installed successfully with UV")
            return True
        except subprocess.SubprocessError as e:
            print_error(f"Failed to install PyTorch with UV: {e}")
            return False

def setup_environment(args):
    """Set up environment variables for the memory server."""
    print_info("Setting up environment variables")
    
    # Mark that we're using UV
    os.environ["UV_ACTIVE"] = "1"
    
    # Set environment variables for better cross-platform compatibility
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # For Windows with limited GPU memory, use smaller chunks
    if platform.system().lower() == "windows":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Force CPU mode if requested
    if args.force_cpu:
        print_info("Forcing CPU-only mode")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["MCP_MEMORY_FORCE_CPU"] = "1"
    
    # Set ChromaDB path if provided
    if args.chroma_path:
        print_info(f"Using custom ChromaDB path: {args.chroma_path}")
        os.environ["MCP_MEMORY_CHROMA_PATH"] = args.chroma_path
    
    # Set backups path if provided
    if args.backups_path:
        print_info(f"Using custom backups path: {args.backups_path}")
        os.environ["MCP_MEMORY_BACKUPS_PATH"] = args.backups_path
    
    # Enable EchoVault if requested
    if args.use_echovault:
        print_info("Enabling EchoVault features")
        os.environ["USE_ECHOVAULT"] = "true"
    
    # Print all environment variables for debugging
    if DEBUG:
        print_debug("Environment variables after setup:")
        for key, value in sorted(os.environ.items()):
            if key.startswith("PYTHON") or key.startswith("PATH") or key.startswith("MCP") or "TORCH" in key or "UV" in key:
                print_debug(f"  - {key}={value}")

def run_memory_server_with_uv():
    """Run the MCP Memory Service with UV."""
    print_info("Starting MCP Memory Service with UV")
    
    try:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Run memory server using UV
        uv_cmd = [sys.executable, '-m', 'uv', 'run', 'memory']
        
        if DEBUG:
            uv_cmd.append('--debug')
            
        print_debug(f"Running command: {' '.join(uv_cmd)}")
        subprocess.run(uv_cmd, check=True)
    except subprocess.SubprocessError as e:
        print_error(f"Error running memory server with UV: {e}")
        print_debug(traceback.format_exc())
        
        # Fallback to classic run method
        print_warning("Falling back to classic run method without UV")
        run_memory_server_classic()
    except Exception as e:
        print_error(f"Unexpected error running memory server: {e}")
        print_debug(traceback.format_exc())
        sys.exit(1)

def run_memory_server_classic():
    """Run the MCP Memory Service with classic method (fallback)."""
    print_info("Starting MCP Memory Service (fallback method)")
    
    # Save original sys.path and meta_path
    original_sys_path = sys.path.copy()
    original_meta_path = sys.meta_path
    
    # Temporarily disable import hooks
    sys.meta_path = [finder for finder in sys.meta_path
                    if not hasattr(finder, 'find_spec') or
                    not hasattr(finder, 'blocked_packages')]
    
    try:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Add src directory to path if it exists
        src_dir = os.path.join(script_dir, "src")
        if os.path.exists(src_dir) and src_dir not in sys.path:
            print_debug(f"Adding {src_dir} to sys.path")
            sys.path.insert(0, src_dir)
        
        # Add site-packages to sys.path
        site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')
        if site_packages not in sys.path:
            sys.path.insert(0, site_packages)
        
        # Try to import using importlib
        print_debug("Attempting to import mcp_memory_service.server using importlib")
        import importlib.util
        import importlib.machinery
        
        # First try to find the module in site-packages
        server_spec = importlib.machinery.PathFinder.find_spec('mcp_memory_service.server', [site_packages])
        
        # If not found, try to find it in src directory
        if server_spec is None and os.path.exists(src_dir):
            server_spec = importlib.machinery.PathFinder.find_spec('mcp_memory_service.server', [src_dir])
        
        if server_spec is None:
            print_error("Could not find mcp_memory_service.server module spec")
            sys.exit(1)
        
        # Load the server module
        server = importlib.util.module_from_spec(server_spec)
        server_spec.loader.exec_module(server)
        
        print_debug("Successfully imported mcp_memory_service.server")
        
        # Run the memory server with error handling
        try:
            print_debug("Calling mcp_memory_service.server.main()")
            server.main()
        except Exception as e:
            print_error(f"Error running memory server: {e}")
            print_debug(traceback.format_exc())
            sys.exit(1)
    except ImportError as e:
        print_error(f"Failed to import mcp_memory_service.server: {e}")
        print_debug(traceback.format_exc())
        sys.exit(1)
    except Exception as e:
        print_error(f"Error setting up memory server: {e}")
        print_debug(traceback.format_exc())
        sys.exit(1)
    finally:
        # Restore original sys.path and meta_path
        sys.path = original_sys_path
        sys.meta_path = original_meta_path

def main():
    """Main function with enhanced error handling and debugging."""
    # Parse command line arguments
    args = parse_args()
    
    # Set global debug flag
    global DEBUG
    DEBUG = args.debug
    
    print_info("Enhanced Memory wrapper script with UV starting")
    
    # Print detailed environment information for debugging
    if DEBUG:
        print_environment_info()
        check_installed_packages()
    
    # Set up environment variables
    setup_environment(args)
    
    # Check if UV is installed, install it if not
    if not check_uv():
        print_info("UV is not installed. Installing UV...")
        if not install_uv():
            print_error("Failed to install UV, falling back to classic mode")
            
            # Import and run the classic memory wrapper
            from memory_wrapper import main as classic_main
            return classic_main()
    
    # Check if PyTorch is installed
    pytorch_ok = check_pytorch()
    if not pytorch_ok:
        print_info("PyTorch is not installed or not working properly")
        if not args.no_auto_install:
            print_info("Attempting to install PyTorch with UV")
            if not install_pytorch_with_uv(args.no_auto_install):
                print_error("Failed to install PyTorch with UV, cannot continue")
                sys.exit(1)
        else:
            print_error("Automatic PyTorch installation is disabled, cannot continue")
            sys.exit(1)
    
    # Check other dependencies
    st_ok = check_sentence_transformers()
    chroma_ok = check_chromadb()
    mcp_ok = check_mcp()
    
    if not (st_ok and chroma_ok and mcp_ok):
        print_warning("Some dependencies are missing or not working properly")
        print_warning("Will attempt to continue anyway, but errors may occur")
    
    # Check EchoVault dependencies if enabled
    if args.use_echovault:
        echovault_ok = check_echovault_dependencies()
        if not echovault_ok:
            print_warning("EchoVault dependencies are missing")
            print_warning("Will continue with standard memory service")
    
    # Add a small delay to ensure all imports are properly initialized
    print_debug("Waiting 1 second before starting the memory server")
    time.sleep(1)
    
    # Run the memory server with UV
    run_memory_server_with_uv()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_info("Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unhandled exception: {e}")
        print_debug(traceback.format_exc())
        sys.exit(1)