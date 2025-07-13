#!/usr/bin/env python3
"""
UV-specific memory wrapper for MCP Memory Service
This wrapper is specifically designed for UV-based installations.
"""
import os
import sys
import subprocess
import traceback

# Set environment variables for better cross-platform compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
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

def main():
    """Main entry point for UV-based memory service."""
    print_info("Starting MCP Memory Service with UV...")
    
    # Set ChromaDB path if provided via environment variables
    if "MCP_MEMORY_CHROMA_PATH" in os.environ:
        print_info(f"Using ChromaDB path: {os.environ['MCP_MEMORY_CHROMA_PATH']}")
    
    # Set backups path if provided via environment variables
    if "MCP_MEMORY_BACKUPS_PATH" in os.environ:
        print_info(f"Using backups path: {os.environ['MCP_MEMORY_BACKUPS_PATH']}")
    
    try:
        # Use UV to run the memory service
        cmd = [sys.executable, '-m', 'uv', 'run', 'memory']
        cmd.extend(sys.argv[1:])  # Pass through any additional arguments
        
        print_info(f"Running command: {' '.join(cmd)}")
        
        # Execute the command
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
        
    except subprocess.SubprocessError as e:
        print_error(f"UV run failed: {e}")
        print_info("Falling back to direct module execution...")
        
        # Fallback to direct execution
        try:
            # Add the source directory to path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.join(script_dir, "src")
            
            if os.path.exists(src_dir):
                sys.path.insert(0, src_dir)
            
            # Import and run the server
            from mcp_memory_service.server import main as server_main
            server_main()
            
        except ImportError as import_error:
            print_error(f"Failed to import memory service: {import_error}")
            sys.exit(1)
        except Exception as fallback_error:
            print_error(f"Fallback execution failed: {fallback_error}")
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    
    except Exception as e:
        print_error(f"Error running memory service: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_info("Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unhandled exception: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
