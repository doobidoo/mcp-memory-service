#!/usr/bin/env python3
"""
Smithery wrapper for MCP Memory Service
This wrapper is specifically designed for Smithery installations.
It doesn't rely on UV and works with the installed package.
"""
import os
import sys
import subprocess
import traceback
import importlib.util

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

def setup_environment():
    """Set up the environment for proper MCP Memory Service operation."""
    # Set environment variables for better cross-platform compatibility
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    
    # For systems with limited GPU memory, use smaller chunks
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    
    # Ensure proper Python path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, "src")
    if os.path.exists(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)

def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = ["mcp", "chromadb", "sentence_transformers"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_info(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print_warning(f"✗ {package} is missing")
    
    return missing_packages

def install_missing_packages(packages):
    """Try to install missing packages."""
    if not packages:
        return True
    
    print_warning("Missing packages detected. For Smithery installations, dependencies should be pre-installed.")
    print_info("Attempting to install missing packages with --break-system-packages flag...")
    
    for package in packages:
        try:
            # Try user installation first
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', package])
            print_success(f"Successfully installed {package}")
        except subprocess.SubprocessError:
            try:
                # Try with --break-system-packages for externally managed environments
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--break-system-packages', package])
                print_success(f"Successfully installed {package}")
            except subprocess.SubprocessError as e:
                print_error(f"Failed to install {package}: {e}")
                print_warning("Continuing anyway - dependencies might be available in different location")
                continue
    
    return True

def run_memory_service():
    """Run the memory service."""
    print_info("Starting MCP Memory Service...")
    
    # Display environment configuration
    if "MCP_MEMORY_CHROMA_PATH" in os.environ:
        print_info(f"Using ChromaDB path: {os.environ['MCP_MEMORY_CHROMA_PATH']}")
    
    if "MCP_MEMORY_BACKUPS_PATH" in os.environ:
        print_info(f"Using backups path: {os.environ['MCP_MEMORY_BACKUPS_PATH']}")
    
    try:
        # Try to import and run the server directly
        from mcp_memory_service.server import main
        print_success("Successfully imported memory service")
        main()
    except ImportError as e:
        print_warning(f"Failed to import from installed package: {e}")
        
        # Fallback to source directory import
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(script_dir, "src")
        
        if os.path.exists(src_dir):
            print_info("Trying to import from source directory...")
            sys.path.insert(0, src_dir)
            try:
                from mcp_memory_service.server import main
                print_success("Successfully imported from source directory")
                main()
            except ImportError as import_error:
                print_error(f"Failed to import from source directory: {import_error}")
                sys.exit(1)
        else:
            print_error("Could not find memory service source code")
            sys.exit(1)
    except Exception as e:
        print_error(f"Error running memory service: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

def main():
    """Main entry point for Smithery wrapper."""
    print_info("MCP Memory Service - Smithery Wrapper")
    
    try:
        # Set up environment
        setup_environment()
        
        # Check dependencies (informational only)
        missing_packages = check_dependencies()
        
        if missing_packages:
            print_warning(f"Some packages appear missing: {', '.join(missing_packages)}")
            print_info("Attempting to proceed anyway - packages might be available in different location")
        
        # Run the memory service
        run_memory_service()
        
    except KeyboardInterrupt:
        print_info("Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unhandled exception: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()