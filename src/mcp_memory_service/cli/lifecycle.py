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
from pathlib import Path

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
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return None
    if _is_process_alive(pid):
        return pid
    pid_path.unlink(missing_ok=True)
    return None


def _write_pid(pid: int) -> None:
    _ensure_dirs()
    _pid_file().write_text(str(pid))


def _remove_pid() -> None:
    _pid_file().unlink(missing_ok=True)


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
    import subprocess
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
            pass
    return None


def _kill_process(pid: int) -> bool:
    """Terminate a process gracefully, then forcefully if needed."""
    import subprocess
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
    os.environ["MCP_ALLOW_ANONYMOUS_ACCESS"] = os.environ.get(
        "MCP_ALLOW_ANONYMOUS_ACCESS", "true"
    )
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

    import subprocess

    env = {**os.environ, "MCP_HTTP_PORT": str(port),
           "MCP_HTTP_HOST": host,
           "MCP_ALLOW_ANONYMOUS_ACCESS": "true"}

    # Run the server as a subprocess using the entry point script
    # This avoids needing to import the heavy package in the launcher
    cmd = [sys.executable, "-c",
           "from mcp_memory_service.web.app import app; "
           "import uvicorn, os; "
           f"uvicorn.run(app, host={repr(host)}, port={port}, "
           "log_level='info')"]

    popen_kwargs = {
        "stdout": open(log_out, "a"),
        "stderr": open(log_err, "a"),
        "stdin": subprocess.DEVNULL,
        "env": env,
    }

    if sys.platform == "win32":
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NO_WINDOW", 0x08000000
        )
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)

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
@click.pass_context
def restart(ctx, http_host, http_port):
    """Restart the memory server (stop + launch)."""
    click.echo("Restarting server...")
    ctx.invoke(stop, http_host=http_host, http_port=http_port)
    time.sleep(1)
    ctx.invoke(launch, http_host=http_host, http_port=http_port,
               detach=True, storage_backend=None, debug=False)


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
    """Show recent server log entries."""
    log_path = _log_file()
    if not log_path.exists():
        click.echo("No log file found.")
        click.echo(f"Expected at: {log_path}")
        return

    log_lines = log_path.read_text(errors="replace").splitlines()
    for line in log_lines[-lines:]:
        click.echo(line)
