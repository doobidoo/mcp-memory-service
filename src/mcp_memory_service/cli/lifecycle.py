"""
Server lifecycle management for MCP Memory Service.

Provides cross-platform commands to launch, stop, restart, and monitor
the HTTP server as a background daemon process.

This module uses ONLY absolute imports from stdlib + click, so it can
be loaded without triggering the heavy mcp_memory_service.__init__
(which loads torch/transformers and takes 20+ seconds).
"""

import os
import sys
import json
import signal
import time
import logging
import subprocess
from pathlib import Path
from collections import deque

import click

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

def _data_dir() -> Path:
    """Return the platform-appropriate data directory for runtime files."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "mcp-memory"


def _pid_file() -> Path:
    return _data_dir() / "server.pid"


def _log_dir() -> Path:
    return _data_dir() / "logs"


def _log_file() -> Path:
    return _log_dir() / "server.log"


def _ensure_dirs() -> None:
    _data_dir().mkdir(parents=True, exist_ok=True)
    _log_dir().mkdir(parents=True, exist_ok=True)


# ─── PID management ───────────────────────────────────────────────────────────

def _read_pid() -> int | None:
    pid_path = _pid_file()
    if not pid_path.exists():
        return None
    try:
        content = pid_path.read_text().strip()
        # Support both old format (just an int) and new format (JSON with metadata)
        try:
            pid_info = json.loads(content)
            pid = pid_info.get("pid", int(content)) if isinstance(pid_info, dict) else int(pid_info)
        except (ValueError, TypeError):
            pid = int(content)
    except (ValueError, OSError):
        return None
    if _is_process_alive(pid):
        # Validate against stale PID files (PID reuse after reboot)
        if _is_stale_pid(pid_path):
            pid_path.unlink(missing_ok=True)
            return None
        return pid
    pid_path.unlink(missing_ok=True)
    return None


def _write_pid(pid: int) -> None:
    _ensure_dirs()
    # Record PID alongside process creation time and cmdline hint to detect
    # stale PID files after reboot or PID reuse.
    pid_info = {"pid": pid}
    try:
        import psutil
        proc = psutil.Process(pid)
        pid_info["create_time"] = proc.create_time()
        pid_info["cmdline_hint"] = " ".join(proc.cmdline()[:3]) if proc.cmdline() else ""
    except Exception:
        pass  # psutil not available — fall back to PID-only (less safe)
    _pid_file().write_text(json.dumps(pid_info))


def _remove_pid() -> None:
    _pid_file().unlink(missing_ok=True)


def _is_stale_pid(pid_path: Path) -> bool:
    """Check if the PID file points to a different process than the original.
    
    Handles PID reuse after reboot or counter rollover by comparing
    the recorded create_time and cmdline hint against the current process.
    Returns True if the PID is stale (different process).
    """
    try:
        pid_info = json.loads(pid_path.read_text().strip())
    except (ValueError, OSError):
        return True  # Can't parse — treat as stale
    
    if isinstance(pid_info, int):
        return False  # Old format (just an int) — can't validate, assume valid
    
    pid = pid_info.get("pid")
    if pid is None:
        return True
    
    recorded_create_time = pid_info.get("create_time")
    recorded_cmdline = pid_info.get("cmdline_hint", "")
    
    if recorded_create_time is not None:
        try:
            import psutil
            proc = psutil.Process(pid)
            current_create_time = proc.create_time()
            # If create times differ by more than 1 second, it's a different process
            if abs(current_create_time - recorded_create_time) > 1.0:
                return True  # Stale — PID was reused by a different process
            # Extra check: cmdline hint should match
            if recorded_cmdline:
                current_cmdline = " ".join(proc.cmdline()[:3]) if proc.cmdline() else ""
                if recorded_cmdline != current_cmdline:
                    return True  # Stale — different command
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return True  # Process doesn't exist — stale
        except Exception:
            pass  # psutil failed — can't validate, assume valid (benign)
    
    return False  # Not stale, or can't determine (psutil unavailable)


def _is_process_alive(pid: int) -> bool:
    """Check whether a process with the given PID is alive (cross-platform)."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            try:
                os.kill(pid, 0)
                return True
            except (ProcessLookupError, PermissionError):
                return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False


# ─── Port scanning ────────────────────────────────────────────────────────────

def _find_process_on_port(port: int) -> int | None:
    """Find PID of the process listening on the given port (cross-platform)."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["netstat", "-aon"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if f":{port} " in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        return int(parts[-1])
        except Exception:
            # netstat unavailable, timed out, or returned unparseable output —
            # treat as "no process found" so the caller can fall back to the
            # PID-file path. Logging would be noise during normal `memory info`.
            pass
    else:
        try:
            result = subprocess.run(
                ["lsof", "-i", f":{port}", "-t"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                return int(result.stdout.strip().splitlines()[0])
        except Exception:
            # lsof not installed, timed out, or returned no listener — treat
            # as "no process found" so the caller can fall back to the
            # PID-file path. Logging would be noise during normal `memory info`.
            pass
    return None


def _kill_process(pid: int) -> bool:
    """Terminate a process gracefully, then forcefully if needed."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                capture_output=True, timeout=5,
            )
        else:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.5)
            if _is_process_alive(pid):
                os.kill(pid, signal.SIGKILL)
        return True
    except Exception:
        return False


# ─── Health check ─────────────────────────────────────────────────────────────

def _http_get_json(url: str, timeout: int = 3) -> dict | None:
    """GET a JSON endpoint, return parsed dict or None on failure."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


# ─── Log reading with streaming tail (no full-file load) ──────────────────────

def _read_log_tail(lines: int = 30) -> list[str]:
    """Read the last N lines of log file using streaming tail (deque).
    
    Uses collections.deque with maxlen to efficiently read only the last N lines
    without loading the entire file into memory.
    """
    log_path = _log_file()
    if not log_path.exists():
        return []
    
    try:
        with open(log_path, 'r', errors='replace') as f:
            # Stream through file, keeping only last N lines in deque
            return list(deque(f, maxlen=lines))
    except Exception:
        return []


# ─── Click commands ───────────────────────────────────────────────────────────

@click.command()
@click.option("--host", "http_host", default=None,
              help="HTTP server host (default: 127.0.0.1)")
@click.option("--port", "http_port", default=None, type=int,
              help="HTTP server port (default: 8000 or MCP_HTTP_PORT)")
@click.option("--detach/--foreground", "detach", default=True,
              help="Run server in background (default) or foreground")
@click.option("--storage-backend", "-s", default=None,
              type=click.Choice(["sqlite_vec", "sqlite-vec", "cloudflare", "hybrid"]),
              help="Storage backend to use")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def launch(ctx, http_host, http_port, detach, storage_backend, debug):
    """Start the HTTP memory server (background by default).
    
    ⚠️  SECURITY WARNING: Binding to non-loopback hosts (e.g., 0.0.0.0)
    exposes the API to your network. Use authentication and/or firewall
    rules in production. Intended for development or trusted networks only.
    
    Equivalent to 'memory server --http' but with lifecycle management:
    PID tracking, log redirection, and automatic health-check polling.
    
    Use --foreground to run attached (same as 'memory server --http').
    """
    # Resolve host and port
    host = http_host or os.environ.get("MCP_HTTP_HOST", "127.0.0.1")
    port = http_port or int(os.environ.get("MCP_HTTP_PORT", "8000"))
    base_url = f"http://{host}:{port}"

    # Apply env overrides
    os.environ["MCP_HTTP_HOST"] = host
    os.environ["MCP_HTTP_PORT"] = str(port)
    # Pass through MCP_ALLOW_ANONYMOUS_ACCESS unchanged — do NOT force a default.
    # If the user set it explicitly (or via .env), respect that.
    # If unset, the server's own default applies (which does NOT set it at all,
    # requiring authentication). This is consistent with `memory server --http`.
    if storage_backend:
        os.environ["MCP_MEMORY_STORAGE_BACKEND"] = storage_backend
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    # Check if already running
    existing_pid = _read_pid()
    if existing_pid:
        health = _http_get_json(f"{base_url}/api/health")
        if health and health.get("status") == "healthy":
            click.echo(f"Server already running (PID {existing_pid})")
            click.echo(f"  Dashboard: {base_url}")
            return

    # Kill stale process on the port if any
    port_pid = _find_process_on_port(port)
    if port_pid and port_pid != existing_pid:
        click.echo(f"Freeing port {port} (stale PID {port_pid})...")
        _kill_process(port_pid)
        time.sleep(0.5)

    if not detach:
        # Foreground: import the heavy stuff and run directly
        click.echo(f"Starting HTTP server on {host}:{port}...")
        from mcp_memory_service.web.app import app  # heavy import
        import uvicorn
        uvicorn.run(app, host=host, port=port,
                    log_level="debug" if debug else "info")
        return

    # ─── Background (detached) mode ──────────────────────────────────────
    _ensure_dirs()
    log_out = _log_file()
    log_err = log_out.with_suffix(".err")

    click.echo(f"Starting memory server on port {port}...")

    # Build safe command arguments (no string interpolation of user-controlled host)
    # Use sys.executable -m uvicorn directly with separate args
    cmd = [
        sys.executable,
        "-m", "uvicorn",
        "mcp_memory_service.web.app:app",
        "--host", host,
        "--port", str(port),
        "--log-level", "info"
    ]

    # Open log files for the child process
    log_out_handle = open(log_out, "a")
    log_err_handle = open(log_err, "a")
    
    # Build child env: pass through current env with host/port overrides.
    # Do NOT force MCP_ALLOW_ANONYMOUS_ACCESS — respect user's explicit setting.
    child_env = {**os.environ, "MCP_HTTP_PORT": str(port), "MCP_HTTP_HOST": host}

    # Close handles immediately in parent after spawning child (fixes file handle leak)
    try:
        popen_kwargs = {
            "stdout": log_out_handle,
            "stderr": log_err_handle,
            "stdin": subprocess.DEVNULL,
            "env": child_env,
        }

        if sys.platform == "win32":
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NO_WINDOW", 0x08000000
            )
        else:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(cmd, **popen_kwargs)
        
        # Close the parent's file handles immediately after spawn
        # (child process has its own copy via dup2)
        log_out_handle.close()
        log_err_handle.close()
        
    except Exception:
        # If Popen fails, make sure to close handles
        log_out_handle.close()
        log_err_handle.close()
        raise

    _write_pid(proc.pid)

    # Poll health endpoint until server is ready
    click.echo("Waiting for server to start...")
    for i in range(60):
        time.sleep(0.5)
        health = _http_get_json(f"{base_url}/api/health")
        if health and health.get("status") == "healthy":
            click.echo(f"Server started (PID {proc.pid})")
            click.echo(f"  Dashboard: {base_url}")
            click.echo(f"  API docs:   {base_url}/docs")
            if health.get("version"):
                click.echo(f"  Version:    {health['version']}")
            if health.get("storage_backend"):
                click.echo(f"  Backend:    {health['storage_backend']}")
            return

    click.echo(f"Server process started (PID {proc.pid}) but health check timed out.")
    click.echo(f"Check logs: {_log_file()}")
    click.echo("Verify with: memory health")


@click.command()
@click.option("--host", "http_host", default=None, help="Host to check")
@click.option("--port", "http_port", default=None, type=int, help="Port to check")
def stop(http_host, http_port):
    """Stop a background memory server."""
    host = http_host or os.environ.get("MCP_HTTP_HOST", "127.0.0.1")
    port = http_port or int(os.environ.get("MCP_HTTP_PORT", "8000"))

    pid = _read_pid()
    port_pid = _find_process_on_port(port)
    stopped = False

    if pid:
        click.echo(f"Stopping PID {pid}...")
        if _kill_process(pid):
            click.echo("Process terminated.")
        else:
            click.echo(f"Could not terminate PID {pid}.", err=True)
        _remove_pid()
        stopped = True

    if port_pid and port_pid != pid:
        click.echo(f"Freeing port {port} (PID {port_pid})...")
        if _kill_process(port_pid):
            click.echo("Process terminated.")
        stopped = True

    if stopped:
        time.sleep(0.5)
        click.echo("Server stopped.")
    else:
        base_url = f"http://{host}:{port}"
        health = _http_get_json(f"{base_url}/api/health")
        if health:
            click.echo("Server responds but no managed PID found. Force-stopping by port...")
            port_pid_now = _find_process_on_port(port)
            if port_pid_now:
                _kill_process(port_pid_now)
                click.echo("Server stopped.")
            else:
                click.echo("Could not find process on port. Stop manually.")
        else:
            click.echo("Server is not running.")


@click.command()
@click.option("--host", "http_host", default=None, help="Host to check")
@click.option("--port", "http_port", default=None, type=int, help="Port to check")
@click.option("--storage-backend", "-s", default=None,
              type=click.Choice(["sqlite_vec", "sqlite-vec", "cloudflare", "hybrid"]),
              help="Storage backend to use (reads from running server if omitted)")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def restart(ctx, http_host, http_port, storage_backend, debug):
    """Restart the memory server (stop + launch).
    
    Preserves --storage-backend and --debug flags. If --storage-backend is
    not provided, attempts to read the current backend from the running
    server's health endpoint before restarting.
    """
    host = http_host or os.environ.get("MCP_HTTP_HOST", "127.0.0.1")
    port = http_port or int(os.environ.get("MCP_HTTP_PORT", "8000"))
    base_url = f"http://{host}:{port}"
    
    # If storage_backend not specified, try to read it from the running server
    if storage_backend is None:
        health = _http_get_json(f"{base_url}/api/health")
        if health and health.get("storage_backend"):
            storage_backend = health["storage_backend"]
            # Normalize: server may return "sqlite_vec" or "sqlite-vec"
            if storage_backend not in ("sqlite_vec", "sqlite-vec", "cloudflare", "hybrid"):
                storage_backend = None  # Unknown backend, let launch use its default
    
    click.echo("Restarting server...")
    ctx.invoke(stop, http_host=http_host, http_port=http_port)
    time.sleep(1)
    ctx.invoke(launch, http_host=http_host, http_port=http_port,
               detach=True, storage_backend=storage_backend, debug=debug)


@click.command()
@click.option("--host", "http_host", default=None, help="Host to check")
@click.option("--port", "http_port", default=None, type=int, help="Port to check")
def status(http_host, http_port):
    """Show server status (running/stopped, PID, backend info)."""
    host = http_host or os.environ.get("MCP_HTTP_HOST", "127.0.0.1")
    port = http_port or int(os.environ.get("MCP_HTTP_PORT", "8000"))
    base_url = f"http://{host}:{port}"

    pid = _read_pid()
    health = _http_get_json(f"{base_url}/api/health")

    click.echo()
    click.echo("  MCP Memory Service")
    click.echo("  ==========================")
    click.echo()

    if health and health.get("status") == "healthy":
        click.echo("  Status:    ACTIVE")
        click.echo(f"  Port:      {port}")
        click.echo(f"  Dashboard: {base_url}")
        if pid:
            click.echo(f"  PID:       {pid}")
        if health.get("version"):
            click.echo(f"  Version:   {health['version']}")
        if health.get("storage_backend"):
            click.echo(f"  Backend:   {health['storage_backend']}")
    else:
        click.echo("  Status:    INACTIVE")
        click.echo(f"  Port:      {port}")
        click.echo()
        click.echo("  Start with: memory launch")

    log_path = _log_file()
    if log_path.exists():
        size_kb = log_path.stat().st_size / 1024
        click.echo(f"  Log:       {log_path} ({size_kb:.1f} KB)")

    click.echo()


@click.command()
@click.option("--host", "http_host", default=None, help="Host to check")
@click.option("--port", "http_port", default=None, type=int, help="Port to check")
def health_cmd(http_host, http_port):
    """Check if the memory server HTTP API is reachable (detailed)."""
    host = http_host or os.environ.get("MCP_HTTP_HOST", "127.0.0.1")
    port = http_port or int(os.environ.get("MCP_HTTP_PORT", "8000"))
    base_url = f"http://{host}:{port}"

    health = _http_get_json(f"{base_url}/api/health")
    if health:
        click.echo(json.dumps(health, indent=2))
    else:
        detailed = _http_get_json(f"{base_url}/api/health/detailed")
        if detailed:
            click.echo(json.dumps(detailed, indent=2))
        else:
            click.echo("Server is not reachable.")
            click.echo("Start with: memory launch")
            sys.exit(1)


@click.command()
@click.option("--lines", "-n", default=30, type=int, help="Number of lines to show")
def logs(lines):
    """Show recent server log entries.
    
    Reads only the last N lines from the log file using streaming tail,
    avoiding loading the entire file into memory.
    """
    log_lines = _read_log_tail(lines)
    if not log_lines:
        click.echo("No log file found.")
        click.echo(f"Expected at: {_log_file()}")
        return
        
    for line in log_lines:
        click.echo(line.rstrip('\n\r'))
