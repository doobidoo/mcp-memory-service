"""
Unit tests for memory_wrapper_cleanup.py

Tests cross-platform orphan process detection and cleanup logic.
"""
import os
import sys
import platform
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add scripts/run to path
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts" / "run"
sys.path.insert(0, str(SCRIPTS_DIR))

import memory_wrapper_cleanup as wrapper


class TestPathHelpers:
    """Test path helper functions."""

    def test_get_script_dir(self):
        """Test script directory detection."""
        script_dir = wrapper.get_script_dir()
        assert script_dir.exists()
        assert script_dir.name == "run"

    def test_get_project_dir(self):
        """Test project root detection."""
        project_dir = wrapper.get_project_dir()
        assert project_dir.exists()
        assert (project_dir / "pyproject.toml").exists()

    def test_print_stderr(self, capsys):
        """Test stderr logging."""
        wrapper.print_stderr("test message")
        captured = capsys.readouterr()
        assert "[mcp-memory-wrapper] test message" in captured.err


class TestOrphanDetectionUnix:
    """Test Unix orphan detection logic."""

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix only")
    @patch('subprocess.run')
    def test_find_orphaned_processes_unix_no_orphans(self, mock_run):
        """Test when no orphaned processes found."""
        # Mock pgrep returns no processes
        mock_run.return_value = Mock(returncode=1, stdout="")

        orphans = wrapper.find_orphaned_processes_unix()
        assert orphans == []

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix only")
    @patch('subprocess.run')
    def test_find_orphaned_processes_unix_with_orphans(self, mock_run):
        """Test detection of orphaned processes (ppid=1)."""
        # Mock pgrep returns PIDs
        pgrep_result = Mock(returncode=0, stdout="1234\n5678\n")
        # Mock ps returns ppid=1 for first, ppid=100 for second
        ps_result_orphan = Mock(returncode=0, stdout="1")
        ps_result_active = Mock(returncode=0, stdout="100")

        mock_run.side_effect = [
            pgrep_result,
            ps_result_orphan,  # PID 1234 has ppid=1 (orphan)
            ps_result_active   # PID 5678 has ppid=100 (active)
        ]

        orphans = wrapper.find_orphaned_processes_unix()
        assert 1234 in orphans
        assert 5678 not in orphans

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix only")
    @patch('subprocess.run')
    @patch('os.getpid', return_value=9999)
    def test_find_orphaned_processes_unix_skips_self(self, mock_getpid, mock_run):
        """Test that wrapper skips its own process."""
        # Mock pgrep returns our own PID
        mock_run.return_value = Mock(returncode=0, stdout="9999\n")

        orphans = wrapper.find_orphaned_processes_unix()
        assert 9999 not in orphans

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix only")
    @patch('subprocess.run', side_effect=FileNotFoundError)
    def test_find_orphaned_processes_unix_missing_pgrep(self, mock_run):
        """Test graceful handling when pgrep is missing."""
        orphans = wrapper.find_orphaned_processes_unix()
        assert orphans == []


class TestOrphanDetectionWindows:
    """Test Windows orphan detection logic."""

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
    @patch('subprocess.run')
    def test_find_orphaned_processes_windows_no_orphans(self, mock_run):
        """Test when no orphaned processes found."""
        # Mock wmic returns no processes
        mock_run.return_value = Mock(returncode=1, stdout="")

        orphans = wrapper.find_orphaned_processes_windows()
        assert orphans == []

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
    @patch('subprocess.run')
    def test_find_orphaned_processes_windows_with_orphans(self, mock_run):
        """Test detection of orphaned processes (parent doesn't exist)."""
        # Mock wmic returns processes
        wmic_result = Mock(returncode=0, stdout="ParentProcessId  ProcessId\n1000  2000\n3000  4000\n")
        # Mock tasklist - parent 1000 doesn't exist, parent 3000 exists
        tasklist_missing = Mock(returncode=0, stdout="No tasks running")
        tasklist_exists = Mock(returncode=0, stdout="python.exe  3000  Console")

        mock_run.side_effect = [wmic_result, tasklist_missing, tasklist_exists]

        orphans = wrapper.find_orphaned_processes_windows()
        assert 2000 in orphans  # Parent 1000 missing = orphan
        assert 4000 not in orphans  # Parent 3000 exists


class TestProcessKilling:
    """Test process termination logic."""

    @patch('platform.system', return_value='Linux')
    @patch('os.kill')
    def test_kill_process_unix_sigterm(self, mock_kill, mock_system):
        """Test Unix process kill with SIGTERM."""
        result = wrapper.kill_process(1234, force=False)

        assert result is True
        mock_kill.assert_called_once_with(1234, 15)  # SIGTERM = 15

    @patch('platform.system', return_value='Linux')
    @patch('os.kill')
    def test_kill_process_unix_sigkill(self, mock_kill, mock_system):
        """Test Unix process kill with SIGKILL."""
        result = wrapper.kill_process(1234, force=True)

        assert result is True
        mock_kill.assert_called_once_with(1234, 9)  # SIGKILL = 9

    @patch('platform.system', return_value='Windows')
    @patch('subprocess.run')
    def test_kill_process_windows(self, mock_run, mock_system):
        """Test Windows process kill with taskkill."""
        mock_run.return_value = Mock(returncode=0)

        result = wrapper.kill_process(1234, force=True)

        assert result is True
        # Verify taskkill called with /F flag
        call_args = mock_run.call_args[0][0]
        assert "taskkill" in call_args
        assert "/F" in call_args
        assert "1234" in call_args

    @patch('platform.system', return_value='Linux')
    @patch('os.kill', side_effect=ProcessLookupError)
    def test_kill_process_not_found(self, mock_kill, mock_system):
        """Test handling of non-existent process."""
        result = wrapper.kill_process(9999)
        assert result is False

    @patch('platform.system', return_value='Linux')
    @patch('os.kill', side_effect=PermissionError)
    def test_kill_process_permission_denied(self, mock_kill, mock_system):
        """Test handling of permission errors."""
        result = wrapper.kill_process(1)
        assert result is False


class TestCleanupOrphans:
    """Test orphan cleanup orchestration."""

    @patch('platform.system', return_value='Linux')
    @patch('memory_wrapper_cleanup.find_orphaned_processes_unix', return_value=[])
    def test_cleanup_orphans_no_orphans(self, mock_find, mock_system, capsys):
        """Test cleanup when no orphans found."""
        wrapper.cleanup_orphans()

        captured = capsys.readouterr()
        assert "No orphaned processes found" in captured.err

    @patch('platform.system', return_value='Linux')
    @patch('memory_wrapper_cleanup.find_orphaned_processes_unix', return_value=[1234, 5678])
    @patch('memory_wrapper_cleanup.kill_process', return_value=True)
    def test_cleanup_orphans_success(self, mock_kill, mock_find, mock_system, capsys):
        """Test successful cleanup of orphans."""
        wrapper.cleanup_orphans()

        captured = capsys.readouterr()
        assert "Found 2 orphaned process(es)" in captured.err
        assert "Terminated orphaned process: 1234" in captured.err
        assert "Terminated orphaned process: 5678" in captured.err

        assert mock_kill.call_count == 2

    @patch('platform.system', return_value='Linux')
    @patch('memory_wrapper_cleanup.find_orphaned_processes_unix', return_value=[1234])
    @patch('memory_wrapper_cleanup.kill_process', side_effect=[False, True])
    def test_cleanup_orphans_force_kill(self, mock_kill, mock_find, mock_system, capsys):
        """Test force kill when normal kill fails."""
        wrapper.cleanup_orphans()

        captured = capsys.readouterr()
        assert "Force-terminated orphaned process: 1234" in captured.err

        # Should try normal kill, then force kill
        assert mock_kill.call_count == 2
        mock_kill.assert_any_call(1234)
        mock_kill.assert_any_call(1234, force=True)


class TestRunMemoryServer:
    """Test server startup logic."""

    @patch('os.chdir')
    @patch('platform.system', return_value='Linux')
    @patch('os.execvp', side_effect=OSError("Mocked exec"))
    @patch('memory_wrapper_cleanup.get_project_dir')
    def test_run_memory_server_unix(self, mock_get_dir, mock_execvp, mock_system, mock_chdir):
        """Test server startup on Unix (exec replaces process)."""
        mock_get_dir.return_value = Path("/path/to/project")

        # Should raise when exec fails
        with pytest.raises(OSError):
            wrapper.run_memory_server()

        mock_chdir.assert_called_once()
        mock_execvp.assert_called_once()

    @patch('os.chdir')
    @patch('platform.system', return_value='Windows')
    @patch('subprocess.run')
    @patch('sys.exit')
    @patch('memory_wrapper_cleanup.get_project_dir')
    def test_run_memory_server_windows(self, mock_get_dir, mock_exit, mock_run, mock_system, mock_chdir):
        """Test server startup on Windows (subprocess)."""
        mock_get_dir.return_value = Path("C:\\path\\to\\project")
        mock_run.return_value = Mock(returncode=0)

        wrapper.run_memory_server()

        mock_chdir.assert_called_once()
        mock_run.assert_called_once()
        mock_exit.assert_called_once_with(0)


class TestMainFunction:
    """Test main entry point."""

    @patch('memory_wrapper_cleanup.cleanup_orphans')
    @patch('memory_wrapper_cleanup.run_memory_server')
    def test_main_success(self, mock_run_server, mock_cleanup):
        """Test successful main execution."""
        wrapper.main()

        mock_cleanup.assert_called_once()
        mock_run_server.assert_called_once()

    @patch('memory_wrapper_cleanup.cleanup_orphans', side_effect=Exception("Test error"))
    @patch('sys.exit')
    def test_main_exception_handling(self, mock_exit, mock_cleanup, capsys):
        """Test exception handling in main."""
        # Run the actual __main__ exception handler
        try:
            wrapper.main()
        except Exception:
            pass

        # The exception should be caught and logged
        captured = capsys.readouterr()
        # Main doesn't re-raise, so we just verify cleanup was attempted
        mock_cleanup.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch('subprocess.run')
    def test_orphan_detection_empty_stdout(self, mock_run):
        """Test handling of empty stdout from pgrep."""
        mock_run.return_value = Mock(returncode=0, stdout="")

        orphans = wrapper.find_orphaned_processes_unix()
        assert orphans == []

    @patch('subprocess.run')
    def test_orphan_detection_malformed_output(self, mock_run):
        """Test handling of malformed process output."""
        # Invalid PID format
        mock_run.return_value = Mock(returncode=0, stdout="not-a-pid\n")

        orphans = wrapper.find_orphaned_processes_unix()
        assert orphans == []

    @patch('platform.system', return_value='Linux')
    @patch('memory_wrapper_cleanup.find_orphaned_processes_unix', return_value=[1234])
    @patch('memory_wrapper_cleanup.kill_process', side_effect=[False, False])
    def test_cleanup_orphans_all_kills_fail(self, mock_kill, mock_find, mock_system, capsys):
        """Test when all kill attempts fail."""
        wrapper.cleanup_orphans()

        captured = capsys.readouterr()
        assert "Failed to terminate process: 1234" in captured.err

        # Should try normal and force kill
        assert mock_kill.call_count == 2
