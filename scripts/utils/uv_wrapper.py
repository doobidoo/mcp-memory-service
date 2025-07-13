#!/usr/bin/env python3
"""
UV wrapper for MCP Memory Service
This wrapper ensures UV is properly configured and runs the memory service.
"""
import os
import sys
import subprocess
import importlib.util
import importlib.machinery
import traceback

# Disable sitecustomize.py and other import hooks to prevent recursion issues
os.environ["PYTHONNOUSERSITE"] = "1"  # Disable user site-packages
os.environ["PYTHONPATH"] = ""  # Clear PYTHONPATH

# Set environment variables to prevent pip from installing dependencies
os.environ["PIP_NO_DEPENDENCIES"] = "1"
os.environ["PIP_NO_INSTALL"] = "1"

# Set environment variables for better cross-platform compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# For Windows with limited GPU memory, use smaller chunks
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def print_info(text):
    """Print formatted info text."""
    print(f"[INFO] {text}", file=sys.stderr, flush=True)

def print_error(text):
    """Print formatted error text."""
    print(f"[ERROR] {text}", file=sys.stderr, flush=True)

def print_success(text):
    """Print formatted success text."""
    print(f"[SUCCESS] {text}", file=sys.stderr, flush=True)

def print_warning(text):
    """Print formatted warning text."""
    print(f"[WARNING] {text}", file=sys.stderr, flush=True)

def check_uv_installed():
    """Check if UV is installed."""
    try:
        result = subprocess.run([sys.executable, '-m', 'uv', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def install_uv():
    """Install UV if not already installed."""
    print_info("Installing UV package manager...")
    try:
        # Try regular pip install first
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'uv'])
        print_success("UV installed successfully!")
        return True
    except subprocess.SubprocessError as e:
        print_warning(f"Failed to install UV with pip: {e}")
        
        # Try with --user flag for user installation
        try:
            print_info("Trying user installation...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'uv'])
            print_success("UV installed successfully with --user flag!")
            return True
        except subprocess.SubprocessError as e2:
            print_error(f"Failed to install UV with --user: {e2}")
            return False

def run_with_uv():
    """Run the memory service using UV."""
    print_info("Starting MCP Memory Service with UV...")
    
    # Set ChromaDB path if provided via environment variables
    if "MCP_MEMORY_CHROMA_PATH" in os.environ:
        print_info(f"Using ChromaDB path: {os.environ['MCP_MEMORY_CHROMA_PATH']}")
    
    # Set backups path if provided via environment variables
    if "MCP_MEMORY_BACKUPS_PATH" in os.environ:
        print_info(f"Using backups path: {os.environ['MCP_MEMORY_BACKUPS_PATH']}")
    
    # Check if running in Docker
    running_in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)
    if running_in_docker:
        print_info("Running in Docker container - ensuring proper process handling")
        
    # Check if running in standalone mode
    standalone_mode = os.environ.get('MCP_STANDALONE_MODE', '').lower() == '1'
    if standalone_mode:
        print_info("Running in standalone mode - server will stay alive without active client")
    
    try:
        # Try to run using UV
        cmd = [sys.executable, '-m', 'uv', 'run', 'memory']
        cmd.extend(sys.argv[1:])  # Pass through any additional arguments
        
        print_info(f"Running command: {' '.join(cmd)}")
        
        # Use subprocess.run with proper handling for Docker
        if running_in_docker and not standalone_mode:
            # In Docker with MCP client mode, handle stdin properly
            result = subprocess.run(cmd, check=False, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
        else:
            # Normal execution
            result = subprocess.run(cmd, check=False)
        
        if result.returncode != 0:
            print_warning(f"UV run exited with code {result.returncode}")
            # Only raise error if not in Docker or if it's a real error (not exit code 0)
            if not running_in_docker or result.returncode != 0:
                raise subprocess.SubprocessError(f"UV run failed with exit code {result.returncode}")
        
    except subprocess.SubprocessError as e:
        print_error(f"UV run failed: {e}")
        print_info("Falling back to direct execution...")
        
        # Fallback to direct execution
        try:
            from mcp_memory_service.server import main
            main()
        except ImportError:
            # Try to import from source directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.join(script_dir, "src")
            
            if os.path.exists(src_dir):
                sys.path.insert(0, src_dir)
                try:
                    from mcp_memory_service.server import main
                    main()
                except ImportError as import_error:
                    print_error(f"Failed to import memory service: {import_error}")
                    sys.exit(1)
            else:
                print_error("Could not find memory service source code")
                sys.exit(1)
    except Exception as e:
        print_error(f"Error running memory service: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

def main():
    """Main entry point."""
    try:
        # Check if UV is installed
        if not check_uv_installed():
            print_warning("UV not found, installing...")
            if not install_uv():
                print_error("Failed to install UV, exiting")
                sys.exit(1)
        
        # Run the memory service with UV
        run_with_uv()
        
    except KeyboardInterrupt:
        print_info("Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unhandled exception: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
