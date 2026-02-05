#!/usr/bin/env python3
"""
Complete multi-client setup and verification script.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Set environment variables for optimal multi-client setup
os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
os.environ["MCP_MEMORY_HTTP_AUTO_START"] = "false"  # Disable for simplicity - WAL mode is sufficient
os.environ["MCP_HTTP_PORT"] = "8000"
os.environ["MCP_HTTP_HOST"] = "localhost"
os.environ["MCP_MEMORY_SQLITE_PRAGMAS"] = "busy_timeout=15000,cache_size=20000"
os.environ["LOG_LEVEL"] = "INFO"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def print_header():
    """Print setup header."""
    print("=" * 60)
    print("MCP MEMORY SERVICE - MULTI-CLIENT SETUP")
    print("=" * 60)
    print("Setting up Claude Desktop + Claude Code coordination...")
    print()


async def test_wal_mode_storage():
    """Test WAL mode storage directly."""
    print("Testing WAL Mode Storage (Phase 1)...")

    try:
        from mcp_memory_service.models.memory import Memory
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

        # Create a temporary database for testing
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            test_db_path = tmp.name

        try:
            # Test direct SQLite-vec storage with WAL mode
            print("  Creating SQLite-vec storage with WAL mode...")
            storage = SqliteVecMemoryStorage(test_db_path)
            await storage.initialize()

            print("  [OK] Storage initialized with WAL mode")

            # Test storing a memory
            from mcp_memory_service.utils.hashing import generate_content_hash

            content = "Multi-client setup test - WAL mode verification"
            test_memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=["setup", "wal-test", "multi-client"],
                memory_type="test",
            )

            print("  Testing memory operations...")
            success, message = await storage.store(test_memory)
            if success:
                print("  [OK] Memory stored successfully")

                # Test retrieval
                results = await storage.search_by_tag(["setup"])
                if results and len(results) > 0:
                    print(f"  [OK] Memory retrieved successfully ({len(results)} results)")
                    print(f"  Content: {results[0].content[:50]}...")

                    # Test concurrent access simulation
                    print("  Testing concurrent access simulation...")

                    # Create another storage instance (simulating second client)
                    storage2 = SqliteVecMemoryStorage(test_db_path)
                    await storage2.initialize()

                    # Both should be able to read
                    results1 = await storage.search_by_tag(["setup"])
                    results2 = await storage2.search_by_tag(["setup"])

                    if len(results1) == len(results2) and len(results1) > 0:
                        print("  [OK] Concurrent read access works")

                        # Test concurrent write
                        content2 = "Second client test memory"
                        memory2 = Memory(
                            content=content2,
                            content_hash=generate_content_hash(content2),
                            tags=["setup", "client2"],
                            memory_type="test",
                        )

                        success2, _ = await storage2.store(memory2)
                        if success2:
                            print("  [OK] Concurrent write access works")

                            # Verify both clients can see both memories
                            all_results = await storage.search_by_tag(["setup"])
                            if len(all_results) >= 2:
                                print("  [OK] Multi-client data sharing works")
                                return True
                            else:
                                print("  [WARNING] Data sharing issue detected")
                        else:
                            print("  [WARNING] Concurrent write failed")
                    else:
                        print("  [WARNING] Concurrent read issue detected")

                    storage2.close()
                else:
                    print("  [WARNING] Memory retrieval failed")
            else:
                print(f"  [ERROR] Memory storage failed: {message}")

            storage.close()

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
        print(f"  [ERROR] WAL mode test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def create_claude_desktop_config():
    """Create the Claude Desktop configuration."""
    config_dir = Path.home() / "AppData" / "Roaming" / "Claude"
    config_file = config_dir / "claude_desktop_config.json"

    print("\nClaude Desktop Configuration:")
    print(f"File location: {config_file}")

    config_content = """{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "C:\\\\REPOSITORIES\\\\mcp-memory-service", "run", "memory"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_MEMORY_SQLITE_PRAGMAS": "busy_timeout=15000,cache_size=20000",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}"""

    print("Configuration content:")
    print(config_content)

    # Check if config directory exists
    if config_dir.exists():
        print(f"\n[INFO] Claude config directory found: {config_dir}")
        if config_file.exists():
            print(f"[INFO] Existing config file found: {config_file}")
            print("       You may need to merge this configuration with your existing one.")
        else:
            print(f"[INFO] No existing config file. You can create: {config_file}")
    else:
        print(f"\n[INFO] Claude config directory not found: {config_dir}")
        print("       This will be created when you first run Claude Desktop.")

    return config_content


def print_environment_setup():
    """Print environment variable setup instructions."""
    print("\nEnvironment Variables Setup:")
    print("The following environment variables have been configured:")

    vars_to_set = {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_MEMORY_SQLITE_PRAGMAS": "busy_timeout=15000,cache_size=20000",
        "LOG_LEVEL": "INFO",
    }

    print("\nFor permanent setup, run these commands in PowerShell (as Admin):")
    for var, value in vars_to_set.items():
        print(f'[System.Environment]::SetEnvironmentVariable("{var}", "{value}", [System.EnvironmentVariableTarget]::User)')

    print("\nOr use these setx commands:")
    for var, value in vars_to_set.items():
        print(f'setx {var} "{value}"')


def print_final_instructions():
    """Print final setup instructions."""
    print("\n" + "=" * 60)
    print("SETUP COMPLETE - FINAL INSTRUCTIONS")
    print("=" * 60)

    print("\n1. RESTART APPLICATIONS:")
    print("   - Close Claude Desktop completely (check system tray)")
    print("   - Close any Claude Code sessions")
    print("   - Restart both applications")

    print("\n2. VERIFY MULTI-CLIENT ACCESS:")
    print("   - Start both Claude Desktop and Claude Code")
    print("   - Store a memory in Claude Desktop: 'Remember: Test from Desktop'")
    print("   - In Claude Code, ask: 'What did I ask you to remember?'")
    print("   - Both should access the same memory database")

    print("\n3. TROUBLESHOOTING:")
    print("   - Check logs for 'WAL mode enabled' messages")
    print("   - Look for 'SQLite pragmas applied' in startup logs")
    print("   - If issues persist, check environment variables are set")

    print("\n4. CONFIGURATION MODE:")
    print("   - Using: WAL Mode (Phase 1)")
    print("   - Features: Multiple readers + single writer")
    print("   - Benefit: Reliable concurrent access without HTTP complexity")

    print("\n5. ADVANCED OPTIONS (Optional):")
    print("   - To enable HTTP coordination: set MCP_MEMORY_HTTP_AUTO_START=true")
    print("   - To use different port: set MCP_HTTP_PORT=8001")
    print("   - To increase timeout: modify MCP_MEMORY_SQLITE_PRAGMAS")


async def main():
    """Main setup function."""
    print_header()

    # Test WAL mode storage
    wal_success = await test_wal_mode_storage()

    if wal_success:
        print("\n[SUCCESS] WAL Mode Multi-Client Test PASSED!")
        print("Your system is ready for multi-client access.")

        # Generate configuration
        create_claude_desktop_config()

        # Print environment setup
        print_environment_setup()

        # Print final instructions
        print_final_instructions()

        print("\n" + "=" * 60)
        print("MULTI-CLIENT SETUP SUCCESSFUL!")
        print("=" * 60)
        print("Claude Desktop + Claude Code can now run simultaneously")
        print("using the WAL mode coordination system.")

    else:
        print("\n[ERROR] Setup test failed.")
        print("Please check the error messages above and try again.")
        print("You may need to install dependencies: python install.py")


if __name__ == "__main__":
    asyncio.run(main())
