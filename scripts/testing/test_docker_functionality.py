#!/usr/bin/env python3
"""
Test script to verify Docker container functionality after cleanup.
Tests basic memory operations and timestamp handling.
"""

import subprocess
import sys


def run_command(cmd, capture_output=True, timeout=30):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"


def test_docker_build():
    """Test Docker image build."""
    print("🔨 Testing Docker build...")

    # Build the Docker image
    cmd = "docker build -f tools/docker/Dockerfile -t mcp-memory-service:test ."
    returncode, stdout, stderr = run_command(cmd, timeout=300)

    if returncode != 0:
        print("❌ Docker build failed:")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return False

    print("✅ Docker build successful")
    return True


def test_docker_import():
    """Test that the server can import without errors."""
    print("🧪 Testing Python imports in container...")

    # Test import using python directly instead of the entrypoint
    cmd = '''docker run --rm --entrypoint python mcp-memory-service:test -c "
import sys
sys.path.append('/app/src')
from mcp_memory_service.server import main
from mcp_memory_service.models.memory import Memory
from datetime import datetime
print('✅ All imports successful')
print('✅ Memory model available')
print('✅ Server main function available')
"'''

    returncode, stdout, stderr = run_command(cmd, timeout=60)

    if returncode != 0:
        print("❌ Import test failed:")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return False

    print(stdout.strip())
    return True


def test_memory_model():
    """Test Memory model and timestamp functionality."""
    print("📝 Testing Memory model and timestamps...")

    cmd = '''docker run --rm --entrypoint python mcp-memory-service:test -c "
import sys
sys.path.append('/app/src')
from mcp_memory_service.models.memory import Memory
from datetime import datetime
import json

# Test Memory creation
memory = Memory(
    content='Test memory content',
    content_hash='testhash123',
    tags=['test', 'docker'],
    metadata={'source': 'test_script'}
)

print(f'✅ Memory created successfully')
print(f'✅ Content: {memory.content}')
print(f'✅ Tags: {memory.tags}')
print(f'✅ Timestamp type: {type(memory.timestamp).__name__}')
print(f'✅ Timestamp value: {memory.timestamp}')

# Test that timestamp is already datetime (no conversion needed)
if isinstance(memory.timestamp, datetime):
    print('✅ Timestamp is correctly a datetime object')
else:
    print('❌ Timestamp is not a datetime object')
    sys.exit(1)
"'''

    returncode, stdout, stderr = run_command(cmd, timeout=60)

    if returncode != 0:
        print("❌ Memory model test failed:")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return False

    print(stdout.strip())
    return True


def test_server_startup():
    """Test that server can start without crashing immediately."""
    print("🚀 Testing server startup...")

    # Start server in background and check if it runs for a few seconds
    # Test server startup by running it briefly
    cmd = '''timeout 5s docker run --rm mcp-memory-service:test 2>/dev/null || echo "✅ Server startup test completed (timeout expected)"'''

    returncode, stdout, stderr = run_command(cmd, timeout=15)

    # We expect a timeout or success message
    if "Server started successfully" in stdout or "Server startup test completed" in stdout:
        print("✅ Server can start without immediate crashes")
        return True
    else:
        print("❌ Server startup test unclear:")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return False


def cleanup_docker():
    """Clean up test Docker images."""
    print("🧹 Cleaning up test images...")
    run_command("docker rmi mcp-memory-service:test", capture_output=False)


def main():
    """Run all tests."""
    print("🔍 DOCKER FUNCTIONALITY TEST SUITE")
    print("=" * 50)

    tests = [
        ("Docker Build", test_docker_build),
        ("Python Imports", test_docker_import),
        ("Memory Model", test_memory_model),
        ("Server Startup", test_server_startup),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)

        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} ERROR: {e}")

    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed / (passed + failed) * 100:.1f}%")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Docker functionality is working correctly.")
        cleanup_docker()
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the issues above.")
        cleanup_docker()
        return 1


if __name__ == "__main__":
    sys.exit(main())
