#!/usr/bin/env python3
"""
Test script to verify the memory service can be imported and run.
"""
import sys
import os

# Add the src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
sys.path.insert(0, src_dir)

try:
    from mcp_memory_service.server import main
    print("SUCCESS: Successfully imported mcp_memory_service.server.main")
    
    # Test basic configuration
    from mcp_memory_service.config import SERVER_NAME, SERVER_VERSION, CHROMA_PATH
    print(f"SUCCESS: Server name: {SERVER_NAME}")
    print(f"SUCCESS: Server version: {SERVER_VERSION}")
    print(f"SUCCESS: ChromaDB path: {CHROMA_PATH}")
    
    print("SUCCESS: All imports successful - the memory service is ready to use!")
    
except ImportError as e:
    print(f"ERROR: Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Error: {e}")
    sys.exit(1)
