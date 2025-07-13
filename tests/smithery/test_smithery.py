#!/usr/bin/env python3
"""
Test script to verify Smithery configuration works correctly.
This simulates how Smithery would invoke the service.
"""
import os
import sys
import subprocess
import tempfile
import json

def test_smithery_config():
    """Test the Smithery configuration by simulating the expected command."""
    print("Testing Smithery configuration...")
    
    # Create temporary paths for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        chroma_path = os.path.join(temp_dir, "chroma_db")
        backups_path = os.path.join(temp_dir, "backups")
        
        # Create directories
        os.makedirs(chroma_path, exist_ok=True)
        os.makedirs(backups_path, exist_ok=True)
        
        # Set environment variables as Smithery would
        test_env = os.environ.copy()
        test_env.update({
            'MCP_MEMORY_CHROMA_PATH': chroma_path,
            'MCP_MEMORY_BACKUPS_PATH': backups_path,
            'PYTHONUNBUFFERED': '1',
            'PYTORCH_ENABLE_MPS_FALLBACK': '1'
        })
        
        # Command that Smithery would run
        cmd = [sys.executable, 'smithery_wrapper.py', '--version']
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"Environment: {json.dumps({k: v for k, v in test_env.items() if k.startswith('MCP_') or k in ['PYTHONUNBUFFERED', 'PYTORCH_ENABLE_MPS_FALLBACK']}, indent=2)}")
        
        try:
            result = subprocess.run(
                cmd, 
                env=test_env,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
                
            if result.returncode == 0:
                print("✅ SUCCESS: Smithery configuration test passed!")
                return True
            else:
                print("❌ FAILED: Smithery configuration test failed!")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ FAILED: Command timed out")
            return False
        except Exception as e:
            print(f"❌ FAILED: Exception occurred: {e}")
            return False

if __name__ == "__main__":
    success = test_smithery_config()
    sys.exit(0 if success else 1)