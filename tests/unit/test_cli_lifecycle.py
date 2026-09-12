# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for CLI lifecycle management commands.

Tests verify the new launch/stop/restart/info/health/logs commands
are properly registered and functional.
"""

import os
import socket
import subprocess
import sys
import textwrap
from unittest.mock import MagicMock

from click.testing import CliRunner

from mcp_memory_service.cli import lifecycle


_REAL_KILL_PROCESS = lifecycle._kill_process

# The _kill_process real-signal guard and MCP_HTTP_PORT/HOST pinning live
# in conftest.py's autouse _lifecycle_process_safety_net fixture -- it
# applies here automatically, no per-file setup needed.


def test_lifecycle_commands_registered():
    """Test that all lifecycle commands are registered with the CLI group."""
    from mcp_memory_service.cli.main import cli, LAZY_COMMANDS
    
    # Get available commands
    ctx = cli.make_context('memory', [])
    commands = cli.list_commands(ctx)
    
    # Lifecycle commands
    lifecycle = ['launch', 'stop', 'restart', 'info', 'health', 'logs']
    for cmd in lifecycle:
        assert cmd in commands, f"Lifecycle command {cmd!r} not in {commands}"
    


def test_launch_command_structure():
    """Test that the launch command has expected options."""
    from mcp_memory_service.cli.main import launch
    
    # Launch should be a Click command
    assert hasattr(launch, 'callback')
    assert hasattr(launch, 'params')
    
    # Check for expected parameters
    param_names = [p.name for p in launch.params]
    expected = ['http_host', 'http_port', 'detach', 'storage_backend', 'debug']
    for param in expected:
        assert param in param_names, f"launch command missing param {param!r}"
    


def test_stop_command_structure():
    """Test that the stop command has expected options."""
    from mcp_memory_service.cli.main import stop
    
    assert hasattr(stop, 'callback')
    param_names = [p.name for p in stop.params]
    assert 'http_host' in param_names
    assert 'http_port' in param_names
    assert 'force' in param_names


def _unused_local_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _start_listener(command, env=None):
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "ready"
    return proc


def _foreign_listener(port):
    code = textwrap.dedent(
        """
        import socket
        import time

        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", PORT))
        listener.listen()
        print("ready", flush=True)
        time.sleep(60)
        """
    ).replace("PORT", str(port))
    return _start_listener([sys.executable, "-c", code])


def _memory_server_listener(tmp_path, port):
    module_dir = tmp_path / "uvicorn"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")
    (module_dir / "__main__.py").write_text(
        textwrap.dedent(
            """
            import socket
            import sys
            import time

            port = int(sys.argv[sys.argv.index("--port") + 1])
            listener = socket.socket()
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", port))
            listener.listen()
            print("ready", flush=True)
            time.sleep(60)
            """
        )
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    return _start_listener(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "mcp_memory_service.web.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
    )


def test_memory_server_command_requires_launcher_app_position():
    """An unrelated Uvicorn process must not match on a later argument."""
    assert lifecycle._is_memory_server_command(
        [sys.executable, "-m", "uvicorn", "mcp_memory_service.web.app:app"]
    )
    assert not lifecycle._is_memory_server_command(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "unrelated.web:app",
            "--app-dir",
            "mcp_memory_service.fake",
        ]
    )


def test_stop_refuses_to_kill_foreign_listener_without_force(monkeypatch, tmp_path):
    """A missing PID file must not make stop kill an unrelated service."""
    port = _unused_local_port()
    proc = _foreign_listener(port)
    monkeypatch.setattr(lifecycle, "_pid_file", lambda: tmp_path / "server.pid")

    try:
        result = CliRunner().invoke(lifecycle.stop, ["--port", str(port)])

        assert result.exit_code != 0
        assert str(proc.pid) in result.output
        assert sys.executable in result.output
        assert "Refusing to stop" in result.output
        assert "--force" in result.output
        assert proc.poll() is None
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_stop_kills_untracked_memory_server_listener(monkeypatch, tmp_path):
    """The lost-PID recovery path still stops a launcher-shaped process."""
    port = _unused_local_port()
    proc = _memory_server_listener(tmp_path, port)
    monkeypatch.setattr(lifecycle, "_pid_file", lambda: tmp_path / "server.pid")
    monkeypatch.setattr(lifecycle, "_kill_process", _REAL_KILL_PROCESS)

    try:
        result = CliRunner().invoke(lifecycle.stop, ["--port", str(port)])

        assert result.exit_code == 0, result.output
        proc.wait(timeout=5)
        assert "Server stopped" in result.output
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_stop_force_kills_foreign_listener(monkeypatch, tmp_path):
    """The explicit force escape hatch can terminate a foreign listener."""
    port = _unused_local_port()
    proc = _foreign_listener(port)
    monkeypatch.setattr(lifecycle, "_pid_file", lambda: tmp_path / "server.pid")
    monkeypatch.setattr(lifecycle, "_kill_process", _REAL_KILL_PROCESS)

    try:
        result = CliRunner().invoke(
            lifecycle.stop, ["--port", str(port), "--force"]
        )

        assert result.exit_code == 0, result.output
        proc.wait(timeout=5)
        assert "Server stopped" in result.output
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
    


def test_info_command_structure():
    """Test that the info command has expected options."""
    from mcp_memory_service.cli.main import info
    
    assert hasattr(info, 'callback')
    param_names = [p.name for p in info.params]
    assert 'http_host' in param_names
    assert 'http_port' in param_names
    


def test_health_command_structure():
    """Test that the health command has expected options."""
    from mcp_memory_service.cli.main import health_cmd
    
    assert hasattr(health_cmd, 'callback')
    param_names = [p.name for p in health_cmd.params]
    assert 'http_host' in param_names
    assert 'http_port' in param_names
    


def test_logs_command_structure():
    """Test that the logs command has expected options."""
    from mcp_memory_service.cli.main import logs
    
    assert hasattr(logs, 'callback')
    param_names = [p.name for p in logs.params]
    assert 'lines' in param_names
    


def test_lifecycle_commands_use_lifecycle_module():
    """Test that lifecycle commands are properly defined in the CLI module."""
    from mcp_memory_service.cli.main import launch, stop, restart, info, health_cmd, logs
    from click import Command
    
    # All lifecycle commands should be Click Command objects
    assert isinstance(launch, Command), f"launch is {type(launch)}, not Command"
    assert isinstance(stop, Command), f"stop is {type(stop)}, not Command"
    assert isinstance(info, Command), f"info is {type(info)}, not Command"
    assert isinstance(health_cmd, Command), f"health_cmd is {type(health_cmd)}, not Command"
    assert isinstance(logs, Command), f"logs is {type(logs)}, not Command"
    
    # Verify each command has a callback
    for cmd in [launch, stop, restart, info, health_cmd, logs]:
        assert hasattr(cmd, 'callback'), f"Command {cmd.name} has no callback attribute"
        assert callable(cmd.callback), f"Command {cmd.name} callback is not callable"
    


# ─── Security and Fixes Tests ─────────────────────────────────────────────────

def test_launch_command_no_c_style_injection():
    """Test that launch command builds safe subprocess args (no -c injection pattern).
    
    The fix replaces the vulnerable pattern:
        [sys.executable, "-c", "from ...; uvicorn.run(..., host='...')"]
    
    With safe argument list:
        [sys.executable, "-m", "uvicorn", "app:app", "--host", host, "--port", str(port)]
    """
    from mcp_memory_service.cli import lifecycle
    from unittest.mock import patch, MagicMock
    import subprocess
    
    # Mock subprocess.Popen to capture the command that would be executed
    with patch('subprocess.Popen') as mock_popen:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_popen.return_value = mock_proc
        
        # Mock _probe_health to return healthy status immediately
        with patch.object(lifecycle, '_probe_health', return_value=({"status": "healthy"}, False)):
            with patch.object(lifecycle, '_write_pid'):
                with (
                    # Never let this test depend on real machine state or
                    # touch the real filesystem: a live PID file would
                    # make launch take the "already running" early return
                    # before Popen is ever called; --port 8000 reaching
                    # the real _find_process_on_port/_kill_process path on
                    # a machine actually running `memory launch` would
                    # send a real SIGTERM/SIGKILL to the running server;
                    # and without mocking _log_file/open, launch does a
                    # real `open()` against this user's actual data
                    # directory (which only "worked" locally because that
                    # directory already existed from prior real use --
                    # a fresh checkout/CI runner has no such directory and
                    # fails with FileNotFoundError).
                    patch.object(lifecycle, '_read_pid', return_value=None),
                    patch.object(lifecycle, '_find_process_on_port', return_value=None),
                    patch('mcp_memory_service.cli.lifecycle.sys.executable', new='/usr/bin/python3'),
                    patch.object(lifecycle, '_ensure_dirs'),
                    patch.object(lifecycle, '_log_file', return_value=MagicMock()),
                    patch('builtins.open'),
                ):
                    # Simulate launch call with a potentially malicious host
                    malicious_host = "127.0.0.1'; echo pwned #"

                    # We need to call through the click command entry point
                    from click.testing import CliRunner
                    from mcp_memory_service.cli.lifecycle import launch

                    runner = CliRunner()
                    result = runner.invoke(launch, ['--host', malicious_host, '--port', '8000', '--detach'])
                    assert result.exit_code == 0, f"{result.output}\n{result.exception}"

                    # Check that Popen was called
                    assert mock_popen.called, "subprocess.Popen was not called"

                    # Get the args passed to Popen
                    call_args = mock_popen.call_args
                    cmd = call_args[0][0] if call_args[0] else call_args[1].get('args', [])

                    # Verify no '-c' flag in command args (the fix)
                    assert '-c' not in cmd, f"Found '-c' in command args: {cmd} - command injection vulnerability still present"

                    # Verify host is NOT interpolated into a code string
                    if len(cmd) > 0:
                        cmd_str = ' '.join(str(x) for x in cmd)
                        # The malicious host should not appear inside a Python code string
                        assert f"host='{malicious_host}'" not in cmd_str, f"Unsafe string interpolation detected in: {cmd_str}"


def test_launch_command_sends_excluded_handles():
    """Test that launch command closes file handles in parent after spawn.
    
    The fix ensures parent closes stdout/stderr handles after Popen,
    preventing file handle leaks in detached mode.
    """
    from mcp_memory_service.cli import lifecycle
    from unittest.mock import patch, MagicMock, call
    import subprocess
    
    with patch('subprocess.Popen') as mock_popen:
        with patch('builtins.open') as mock_open:
            mock_out_handle = MagicMock()
            mock_err_handle = MagicMock()
            mock_out_handle.close = MagicMock()
            mock_err_handle.close = MagicMock()
            mock_open.side_effect = [mock_out_handle, mock_err_handle]
            
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_popen.return_value = mock_proc
            
            with patch.object(lifecycle, '_probe_health', return_value=({"status": "healthy"}, False)):
                with patch.object(lifecycle, '_write_pid'):
                    with (
                        # Same real-machine-state hazard as the sibling
                        # test above -- a live PID file would skip Popen
                        # entirely, and --port 8000 reaching the real
                        # _find_process_on_port/_kill_process call would
                        # send a real SIGTERM/SIGKILL to whatever is
                        # actually running on this service's default port.
                        patch.object(lifecycle, '_read_pid', return_value=None),
                        patch.object(lifecycle, '_find_process_on_port', return_value=None),
                        patch('mcp_memory_service.cli.lifecycle.sys.executable', new='/usr/bin/python3'),
                        patch.object(lifecycle, '_ensure_dirs'),
                        # Mock _log_file to return a valid Path
                        patch.object(lifecycle, '_log_file', return_value=MagicMock()),
                    ):
                        from click.testing import CliRunner
                        from mcp_memory_service.cli.lifecycle import launch

                        runner = CliRunner()
                        result = runner.invoke(launch, ['--host', '127.0.0.1', '--port', '8000', '--detach'])
                        assert result.exit_code == 0, f"{result.output}\n{result.exception}"

                        # Verify close() was called on both handles after Popen
                        assert mock_out_handle.close.called, "stdout handle was not closed in parent"
                        assert mock_err_handle.close.called, "stderr handle was not closed in parent"


def test_logs_command_uses_streaming_tail():
    """Test that logs command uses streaming tail (deque) instead of full-file read.
    
    The fix replaces log_path.read_text().splitlines() with:
        list(deque(f, maxlen=lines))
    """
    import tempfile
    import os
    from collections import deque
    
    # Create a test log file with multiple lines
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.log")
        with open(log_file, 'w') as f:
            for i in range(100):
                f.write(f"Line {i}\n")
        
        # Test streaming tail directly
        with open(log_file, 'r', errors='replace') as f:
            result = list(deque(f, maxlen=10))
        
        assert len(result) == 10, f"Expected 10 lines, got {len(result)}"
        assert "Line 90" in result[0], f"First line should be 'Line 90', got {result[0]}"
        assert "Line 99" in result[-1], f"Last line should be 'Line 99', got {result[-1]}"
        
        # Verify it loaded only last 10 lines, not all 100
        # (Deque with maxlen only keeps last N items, no full file load)


def test_logs_command_handle_missing_log_file():
    """Test that logs command gracefully handles missing log file."""
    from click.testing import CliRunner
    from mcp_memory_service.cli.lifecycle import logs
    from unittest.mock import patch
    
    runner = CliRunner()
    
    # Mock _log_file to return a nonexistent path
    with patch('mcp_memory_service.cli.lifecycle._log_file') as mock_log_file:
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_log_file.return_value = mock_path
        
        result = runner.invoke(logs, ['--lines', '10'])
        assert result.exit_code == 0
        assert "No log file found" in result.output or "No log file found" in str(result.output)


def test_launch_help_text_security_warning():
    """Test that launch command help text contains security warning about host binding."""
    from click.testing import CliRunner
    from mcp_memory_service.cli.lifecycle import launch
    
    runner = CliRunner()
    result = runner.invoke(launch, ['--help'])
    
    # Should contain security warning
    assert '--host' in result.output, "Help should show --host option"
    # Check that the help text mentions security concerns
    # The fix adds a warning about binding to non-loopback hosts



def test_read_pid_accepts_json_metadata_for_live_process(tmp_path, monkeypatch):
    """A live managed child must survive JSON PID metadata validation."""
    import json
    import subprocess

    from mcp_memory_service.cli import lifecycle

    # Match the launcher's stable command-line prefix without importing the
    # real uvicorn application: a tiny temporary module keeps the child alive.
    (tmp_path / "uvicorn.py").write_text("import time\ntime.sleep(60)\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "mcp_memory_service.web.app:app"],
        cwd=tmp_path,
        env=env,
    )

    pid_path = tmp_path / "server.pid"
    monkeypatch.setattr(lifecycle, "_pid_file", lambda: pid_path)
    monkeypatch.setattr(lifecycle, "_ensure_dirs", lambda: None)

    try:
        lifecycle._write_pid(proc.pid)
        metadata_text = pid_path.read_text()
        metadata = json.loads(metadata_text)
        assert metadata["pid"] == proc.pid
        assert lifecycle._is_stale_pid(pid_path) is False
        assert lifecycle._read_pid() == proc.pid

        # The structured format must not regress support for pre-metadata PID files.
        pid_path.write_text(str(proc.pid))
        assert lifecycle._read_pid() == proc.pid
        pid_path.write_text(metadata_text)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    assert lifecycle._is_stale_pid(pid_path) is True
