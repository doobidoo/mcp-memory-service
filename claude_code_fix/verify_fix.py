#!/usr/bin/env python3
"""
Quick test to verify that the server.py file no longer contains oneOf or anyOf constraints
that break Claude Code compatibility.
"""

import json
import re

def check_schema_compatibility(file_path):
    """Check if the server.py file has been fixed for Claude Code compatibility."""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for oneOf constraints
    oneOf_matches = re.findall(r'"oneOf"\s*:', content)
    anyOf_matches = re.findall(r'"anyOf"\s*:', content)
    
    print("=== Claude Code Compatibility Check ===")
    print(f"Checking file: {file_path}")
    print()
    
    if oneOf_matches:
        print(f"❌ Found {len(oneOf_matches)} 'oneOf' constraint(s)")
        print("   Claude Code does not support oneOf schemas!")
    else:
        print("✅ No 'oneOf' constraints found")
    
    if anyOf_matches:
        print(f"❌ Found {len(anyOf_matches)} 'anyOf' constraint(s)")
        print("   Claude Code may have issues with anyOf schemas!")
    else:
        print("✅ No 'anyOf' constraints found")
    
    print()
    
    if not oneOf_matches and not anyOf_matches:
        print("🎉 SUCCESS: The file appears to be compatible with Claude Code!")
        return True
    else:
        print("⚠️  WARNING: The file may still have compatibility issues with Claude Code.")
        return False

if __name__ == "__main__":
    import os
    import sys
    
    # Get the path to server.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(
        os.path.dirname(script_dir), 
        "src", 
        "mcp_memory_service", 
        "server.py"
    )
    
    if not os.path.exists(server_path):
        print(f"Error: Could not find server.py at {server_path}")
        sys.exit(1)
    
    success = check_schema_compatibility(server_path)
    sys.exit(0 if success else 1)
