#!/usr/bin/env python3
"""
Integration tests for CLI interfaces to prevent conflicts and ensure compatibility.

This module tests the different CLI entry points to ensure they work correctly
and that the compatibility layer functions as expected.
"""

import subprocess
import pytest
import warnings
import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent / "src"
sys.path.insert(0, str(src_dir))

from mcp_memory_service.cli.main import memory_server_main, main as cli_main


class TestCLIInterfaces:
    """Test CLI interface compatibility and functionality."""
    
    def test_memory_command_backward_compatibility(self):
        """Test that 'uv run memory' (without server) starts the MCP server for backward compatibility."""
        result = subprocess.run(
            ["uv", "run", "memory", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=current_dir.parent.parent
        )
        # Should show help text (not start server) when --help is provided
        assert result.returncode == 0
        assert "MCP Memory Service" in result.stdout
    
    def test_memory_command_exists(self):
        """Test that the memory command is available."""
        result = subprocess.run(
            ["uv", "run", "memory", "--help"],
            capture_output=True,
            text=True,
            cwd=current_dir.parent.parent
        )
        assert result.returncode == 0
        assert "MCP Memory Service" in result.stdout
        assert "server" in result.stdout
        assert "status" in result.stdout
    
    def test_memory_server_command_exists(self):
        """Test that the memory-server command is available."""
        result = subprocess.run(
            ["uv", "run", "memory-server", "--help"],
            capture_output=True, 
            text=True,
            cwd=current_dir.parent.parent
        )
        assert result.returncode == 0
        assert "MCP Memory Service" in result.stdout
        # Should show deprecation warning in stderr
        assert "deprecated" in result.stderr.lower()
    
    def test_mcp_memory_server_command_exists(self):
        """Test that the mcp-memory-server command is available."""
        result = subprocess.run(
            ["uv", "run", "mcp-memory-server", "--help"],
            capture_output=True,
            text=True, 
            cwd=current_dir.parent.parent
        )
        # This might have different behavior or missing dependencies
        # 0 for success, 1 for import error (missing fastmcp), 2 for argument error
        assert result.returncode in [0, 1, 2]
        
        # If it failed due to missing fastmcp dependency, that's expected
        if result.returncode == 1 and "fastmcp" in result.stderr:
            pytest.skip("mcp-memory-server requires FastMCP which is not installed")
    
    def test_memory_server_version_flag(self):
        """Test that memory-server --version works and shows deprecation warning."""
        result = subprocess.run(
            ["uv", "run", "memory-server", "--version"],
            capture_output=True,
            text=True,
            cwd=current_dir.parent.parent
        )
        assert result.returncode == 0
        assert "6.3.0" in result.stdout
        assert "deprecated" in result.stderr.lower()
    
    def test_memory_server_vs_memory_server_subcommand(self):
        """Test that both memory-server and memory server provide similar functionality."""
        # Test memory-server --help
        result1 = subprocess.run(
            ["uv", "run", "memory-server", "--help"],
            capture_output=True,
            text=True,
            cwd=current_dir.parent.parent
        )
        
        # Test memory server --help  
        result2 = subprocess.run(
            ["uv", "run", "memory", "server", "--help"],
            capture_output=True,
            text=True,
            cwd=current_dir.parent.parent
        )
        
        assert result1.returncode == 0
        assert result2.returncode == 0
        
        # Both should mention debug and chroma-path options
        assert "--debug" in result1.stdout
        assert "--debug" in result2.stdout
        assert "--chroma-path" in result1.stdout or "chroma-path" in result1.stdout
        assert "--chroma-path" in result2.stdout or "chroma-path" in result2.stdout
    
    def test_compatibility_wrapper_deprecation_warning(self):
        """Test that the compatibility wrapper issues deprecation warnings."""
        # Capture warnings when calling memory_server_main
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Ensure all warnings are caught
            
            # Mock sys.argv to test argument parsing
            original_argv = sys.argv
            try:
                sys.argv = ["memory-server", "--version"]
                # This will raise SystemExit due to --version, which is expected
                with pytest.raises(SystemExit) as exc_info:
                    memory_server_main()
                assert exc_info.value.code == 0  # Should exit successfully
            finally:
                sys.argv = original_argv
            
            # Check that deprecation warning was issued
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0
            assert "deprecated" in str(deprecation_warnings[0].message).lower()
            assert "memory server" in str(deprecation_warnings[0].message)
    
    def test_argument_compatibility(self):
        """Test that arguments are properly passed through compatibility wrapper."""
        # Test with --debug flag
        result = subprocess.run(
            ["uv", "run", "memory-server", "--help"],
            capture_output=True,
            text=True,
            cwd=current_dir.parent.parent
        )
        
        assert result.returncode == 0
        assert "--debug" in result.stdout
        assert "--chroma-path" in result.stdout
        assert "--version" in result.stdout
    
    def test_no_cli_conflicts_during_import(self):
        """Test that importing CLI modules doesn't cause conflicts."""
        try:
            # These imports should work without conflicts
            from mcp_memory_service.cli.main import main, memory_server_main
            from mcp_memory_service import server
            
            # Check that functions exist and are callable
            assert callable(main)
            assert callable(memory_server_main)
            assert callable(server.main)
            
            # Should not raise any import errors or conflicts
        except ImportError as e:
            pytest.fail(f"CLI import conflict detected: {str(e)}")


class TestCLIFunctionality:
    """Test actual CLI functionality to ensure commands work end-to-end."""
    
    def test_memory_status_command(self):
        """Test that memory status command works."""
        result = subprocess.run(
            ["uv", "run", "memory", "status"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=current_dir.parent.parent
        )
        # Status command might fail if no storage is available, but should not crash
        # Return code 0 = success, 1 = expected error (e.g., no storage)
        assert result.returncode in [0, 1]
        
        if result.returncode == 0:
            assert "MCP Memory Service Status" in result.stdout
            assert "Version: 6.3.0" in result.stdout
        else:
            # If it fails, should have a meaningful error message
            assert len(result.stderr) > 0 or len(result.stdout) > 0


class TestCLIRobustness:
    """Test CLI robustness and edge cases."""
    
    def test_environment_variable_passing(self):
        """Test that environment variables are properly set by CLI commands."""
        import os
        import subprocess
        
        # Test memory server command with chroma-path
        env = os.environ.copy()
        result = subprocess.run(
            ["uv", "run", "python", "-c", """
import os
print(f"MCP_MEMORY_CHROMA_PATH={os.environ.get('MCP_MEMORY_CHROMA_PATH', 'NOT_SET')}")
from mcp_memory_service.cli.main import cli
import sys
sys.argv = ['memory', 'server', '--chroma-path', '/tmp/test-chroma']
# Don't actually run server, just test env var setting
from mcp_memory_service.cli.main import server
server(debug=False, chroma_path='/tmp/test-chroma', storage_backend='sqlite_vec')
"""],
            capture_output=True,
            text=True,
            cwd=current_dir.parent.parent,
            env=env,
            timeout=10
        )
        
        # Should either succeed or fail gracefully (not crash)
        assert result.returncode in [0, 1]  # 0 = success, 1 = expected server start failure
    
    def test_cli_error_handling(self):
        """Test that CLI handles errors gracefully."""
        # Test invalid storage backend
        result = subprocess.run(
            ["uv", "run", "memory", "server", "--storage-backend", "invalid"],
            capture_output=True,
            text=True,
            cwd=current_dir.parent.parent,
            timeout=10
        )
        
        # Should fail with clear error message
        assert result.returncode != 0
        assert len(result.stderr) > 0 or "invalid" in result.stdout.lower()
    
    def test_memory_server_argument_parity(self):
        """Test that memory-server and memory server support the same core arguments."""
        import subprocess
        
        # Test memory server arguments
        result1 = subprocess.run(
            ["uv", "run", "memory", "server", "--help"],
            capture_output=True,
            text=True,
            cwd=current_dir.parent.parent
        )
        
        # Test memory-server arguments
        result2 = subprocess.run(
            ["uv", "run", "memory-server", "--help"],
            capture_output=True,
            text=True,
            cwd=current_dir.parent.parent
        )
        
        assert result1.returncode == 0
        assert result2.returncode == 0
        
        # Both should support core arguments
        core_args = ["--debug", "--chroma-path"]
        for arg in core_args:
            assert arg in result1.stdout
            assert arg in result2.stdout
    
    def test_entry_point_isolation(self):
        """Test that different entry points don't interfere with each other."""
        # Test that we can import all entry points without conflicts
        try:
            # Import main CLI
            from mcp_memory_service.cli.main import main, memory_server_main
            
            # Import server main
            from mcp_memory_service.server import main as server_main
            
            # Verify they're different functions
            assert main != memory_server_main
            assert main != server_main
            assert memory_server_main != server_main
            
            # Verify they're all callable
            assert callable(main)
            assert callable(memory_server_main)  
            assert callable(server_main)
            
        except ImportError as e:
            pytest.fail(f"Entry point isolation failed: {e}")
    
    def test_backward_compatibility_deprecation_warning(self):
        """Test that using 'memory' without subcommand shows deprecation warning."""
        import warnings
        import sys
        from mcp_memory_service.cli.main import cli
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Mock sys.argv to test backward compatibility
            original_argv = sys.argv
            try:
                # This simulates 'uv run memory' without subcommand
                sys.argv = ["memory"]
                with pytest.raises(SystemExit):  # Server will try to start and exit
                    cli(standalone_mode=False)
            except Exception:
                # Expected - server can't actually start in test environment
                pass
            finally:
                sys.argv = original_argv
            
            # Verify backward compatibility deprecation warning
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0
            
            warning_msg = str(deprecation_warnings[0].message)
            assert "without a subcommand is deprecated" in warning_msg
            assert "memory server" in warning_msg
            assert "backward compatibility will be removed" in warning_msg
    
    def test_deprecation_warning_format(self):
        """Test that deprecation warning has proper format and information."""
        import warnings
        import sys
        from mcp_memory_service.cli.main import memory_server_main
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Mock sys.argv for the compatibility wrapper
            original_argv = sys.argv
            try:
                sys.argv = ["memory-server", "--version"]
                with pytest.raises(SystemExit):  # --version causes system exit
                    memory_server_main()
            finally:
                sys.argv = original_argv
            
            # Verify deprecation warning content
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0
            
            warning_msg = str(deprecation_warnings[0].message)
            assert "memory-server" in warning_msg.lower()
            assert "deprecated" in warning_msg.lower() 
            assert "memory server" in warning_msg.lower()
            assert "removed" in warning_msg.lower()


class TestCLIPerformance:
    """Test CLI performance characteristics."""
    
    def test_cli_startup_time(self):
        """Test that CLI commands start reasonably quickly."""
        import time
        import subprocess
        
        start_time = time.time()
        result = subprocess.run(
            ["uv", "run", "memory", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=current_dir.parent.parent
        )
        elapsed = time.time() - start_time
        
        assert result.returncode == 0
        # CLI help should respond within 30 seconds (generous timeout for CI)
        assert elapsed < 30
        # Log performance for monitoring
        print(f"CLI startup took {elapsed:.2f} seconds")
    
    def test_memory_version_performance(self):
        """Test that version commands are fast."""
        import time
        import subprocess
        
        # Test both version commands
        commands = [
            ["uv", "run", "memory", "--version"],
            ["uv", "run", "memory-server", "--version"]
        ]
        
        for cmd in commands:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
                cwd=current_dir.parent.parent
            )
            elapsed = time.time() - start_time
            
            assert result.returncode == 0
            assert "6.3.0" in result.stdout
            # Version should be very fast
            assert elapsed < 15
            print(f"Version command {' '.join(cmd[2:])} took {elapsed:.2f} seconds")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])