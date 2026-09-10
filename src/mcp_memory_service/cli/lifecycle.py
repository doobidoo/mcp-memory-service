"""
Server lifecycle management for MCP Memory Service.

Provides cross-platform commands to launch, stop, restart, and monitor
the HTTP server as a background daemon process.

This module uses ONLY absolute imports from stdlib + click, so it can
be loaded without triggering the heavy mcp_memory_service.__init__
(which loads torch/transformers and takes 20+ seconds).

⚠️  Do not add an import of mcp_memory_service.config (or any submodule,
e.g. config.transport) to this file. That import triggers load_dotenv
as a side effect, populating os.environ from a stray .env file in
whatever directory happens to be cwd. _is_https_enabled() and
_cli_allow_self_signed_certs() below both deliberately read only
os.environ for security-relevant decisions (which scheme to probe,
whether to disable TLS verification) -- a config import anywhere in
this file would silently reintroduce a .env fallback both functions
are designed to exclude.
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
from typing import NamedTuple

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
        except (ValueError, TypeError):
            pid_info = content

        if isinstance(pid_info, dict):
            pid_value = pid_info.get("pid")
            if pid_value is None:
                return None
        else:
            pid_value = pid_info
        pid = int(pid_value)
    except (ValueError, TypeError, OSError):
        return None
    if _is_process_alive(pid):
        # Validate against stale PID files (PID reuse after reboot)
        if _is_stale_pid(pid_path):
            pid_path.unlink(missing_ok=True)
            return None
        return pid
    pid_path.unlink(missing_ok=True)
    return None


def _write_pid(pid: int, scheme: str = "http") -> None:
    _ensure_dirs()
    # Record PID alongside process creation time and cmdline hint to detect
    # stale PID files after reboot or PID reuse. The scheme records what the
    # server was actually launched with, so later commands probe the right
    # one instead of re-deriving it -- see _recorded_scheme().
    pid_info = {"pid": pid, "scheme": scheme}
    try:
        import psutil  # inline import: deferred so this third-party dependency doesn't load at module import time, matching this module's fast-load design
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
                # -sTCP:LISTEN: without it, lsof matches every socket
                # touching this port -- including established client
                # connections -- and splitlines()[0] (process-table
                # order, not connection role) can return a client's PID
                # instead of the actual listener's. -n -P skip DNS/service
                # name resolution, which otherwise risk blocking toward
                # the 5s timeout on a host with slow/broken DNS.
                ["lsof", "-n", "-P", "-sTCP:LISTEN", "-i", f":{port}", "-t"],
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

def _is_https_enabled() -> bool:
    """Return True only if MCP_HTTPS_ENABLED is explicitly set in the
    environment. Deliberately reads only os.environ, not a .env file:
    this decides which scheme the health check probes with, and a
    stray .env file in whatever directory happens to be cwd shouldn't
    be able to silently redirect that probe onto HTTPS (or off it).
    Same discipline as _cli_allow_self_signed_certs() below, for the
    same reason -- previously this function had its own .env fallback,
    deliberately left untouched and out of scope when that sibling
    function was introduced; this closes that gap.

    See the module docstring for why this file must not import
    mcp_memory_service.config -- doing so would silently reintroduce
    the .env fallback removed here.

    Accepted values match config.base.safe_get_bool_env()'s truthy set
    (true/1/yes/on/enabled) -- kept in sync manually since this
    function can't import that module. See run_server.py and
    check_http_server.py for the same duplication under the same
    constraint."""
    return os.environ.get("MCP_HTTPS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on", "enabled")  # sync with config.base.safe_get_bool_env


def _dotenv_candidates() -> list:
    """The .env locations the server's own config will search, in the same
    priority order. Mirrors config.base._find_and_load_dotenv() -- kept in
    sync manually, since importing that module here would trigger its
    load_dotenv side effect (see the module docstring)."""
    here = Path(__file__)
    return [
        Path.cwd() / ".env",
        here.parent.parent.parent.parent / ".env",
        *[p.parent / ".env" for p in here.parents if (p / "pyproject.toml").exists()],
        Path("C:/REPOSITORIES/personal/mcp-memory-service/.env"),
        Path("C:/REPOSITORIES/mcp-memory-service/.env"),
        Path.home() / ".mcp-memory" / ".env",
    ]


def _server_env(name: str) -> str | None:
    """Read a server-side setting the way the spawned child will see it:
    os.environ first, then the .env file config.base would load.

    Deliberately uses dotenv_values(), which returns a dict, rather than
    load_dotenv(), which writes into os.environ. Nothing here may leak .env
    into the process environment -- _is_https_enabled() and
    _cli_allow_self_signed_certs() read os.environ precisely because a stray
    .env must not be able to steer them. Precedence matches config.base's
    load_dotenv(override=False): a real environment variable wins."""
    value = os.environ.get(name)
    if value is not None:
        return value
    from dotenv import dotenv_values  # inline import: third-party, kept out of module load per this module's fast-load design
    for env_file in _dotenv_candidates():
        try:
            if env_file.exists():
                return dotenv_values(env_file).get(name)
        except (OSError, PermissionError):
            continue
    return None


class _ServerTls(NamedTuple):
    """What the spawned server needs in order to serve the configured scheme.

    Carries ready-to-splice arguments rather than a bare on/off flag, so the
    two spawn sites stay branch-free: the background path appends cli_args to
    the uvicorn command line, the foreground path expands uvicorn_kwargs into
    uvicorn.run(). Both are empty for plain HTTP."""
    scheme: str
    cli_args: list
    uvicorn_kwargs: dict


_NO_TLS = _ServerTls("http", [], {})


def _resolve_server_tls() -> _ServerTls:
    """Return the TLS configuration for the server to be spawned.

    `memory launch` used to build a fixed uvicorn command line with no SSL
    arguments at all, so a deployment configured for HTTPS came back up on
    plain HTTP after `memory restart` -- silently, and with `memory info`
    reporting http:// in agreement. scripts/server/run_http_server.py did
    honour the same settings, so the two entry points disagreed.

    Raises ClickException rather than falling back to HTTP: a server that
    was asked for TLS and cannot provide it must not come up unencrypted
    while looking healthy. Certificate auto-generation is deliberately not
    reimplemented here -- explicit MCP_SSL_CERT_FILE/MCP_SSL_KEY_FILE paths
    are required.
    """
    raw = (_server_env("MCP_HTTPS_ENABLED") or "").strip().lower()
    # sync with config.base.safe_get_bool_env
    if raw not in ("1", "true", "yes", "on", "enabled"):
        return _NO_TLS

    cert_file = _server_env("MCP_SSL_CERT_FILE")
    key_file = _server_env("MCP_SSL_KEY_FILE")
    for name, value in (("MCP_SSL_CERT_FILE", cert_file), ("MCP_SSL_KEY_FILE", key_file)):
        if not value:
            raise click.ClickException(
                f"MCP_HTTPS_ENABLED is set but {name} is not. Set both "
                "certificate paths, or unset MCP_HTTPS_ENABLED to serve HTTP."
            )
        if not Path(value).is_file():
            raise click.ClickException(
                f"{name}={value!r} does not exist or is not a file. "
                "Refusing to start on plain HTTP with TLS requested."
            )
    return _ServerTls(
        "https",
        ["--ssl-certfile", cert_file, "--ssl-keyfile", key_file],
        {"ssl_certfile": cert_file, "ssl_keyfile": key_file},
    )


def _recorded_scheme() -> str | None:
    """Return the scheme the running server was actually launched with, as
    recorded in the PID file, or None if there is nothing to go on.

    Ground truth beats inference: _is_https_enabled() answers "should this
    CLI *probe* over HTTPS" from os.environ alone, on purpose. But the
    server's own TLS config legitimately comes from .env (see
    _resolve_server_tls), so those two can disagree -- and when they do, an
    HTTPS server gets probed over HTTP, the health check fails, and launch
    tears down a child that started perfectly well. Recording what was
    launched removes the guess without loosening _is_https_enabled()."""
    pid_path = _pid_file()
    try:
        pid_info = json.loads(pid_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(pid_info, dict):
        return None
    scheme = pid_info.get("scheme")
    return scheme if scheme in ("http", "https") else None


def _base_url(host: str, port: int, scheme: str | None = None) -> str:
    scheme = scheme or _recorded_scheme() or ("https" if _is_https_enabled() else "http")
    return f"{scheme}://{host}:{port}"


def _cli_allow_self_signed_certs() -> bool:
    """Return True only if MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS is explicitly
    enabled in the environment. Defaults to False (verify). Deliberately
    reads only os.environ, not a .env file: a flag that disables this
    CLI's only line of MITM defense should require an explicit, visible
    environment variable, not a value read from a stray .env file in
    whatever directory happens to be the current working directory.
    Matches the name examples/http-mcp-bridge.js already uses for the
    identical opt-in. Not shared with opencode (its own
    OPENCODE_MEMORY_ALLOW_SELF_SIGNED_CERTS) or claude-hooks (a
    config-file `allowSelfSignedCerts` boolean, not an env var) -- a
    homelab deployment pointing all of these at one self-signed endpoint
    still needs to set each of the three separately."""
    return os.environ.get("MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS", "").strip().lower() in (
        "1", "true", "yes",
    )


_self_signed_warning_shown = False
_ssl_failure_warning_shown = False


def _warn_self_signed_bypass_once() -> None:
    """Print the MITM warning at most once per process -- callers like
    `launch` poll _probe_health dozens of times per invocation, and a
    security warning that scrolls by 60 times trains users to ignore it."""
    global _self_signed_warning_shown
    if not _self_signed_warning_shown:
        _self_signed_warning_shown = True
        click.echo(
            "WARNING: TLS certificate validation DISABLED "
            "(MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS). This leaves the connection "
            "vulnerable to MITM -- use only for local development with "
            "self-signed certs. To trust a real internal CA instead, set "
            "SSL_CERT_FILE to its bundle path.",
            err=True,
        )


def _warn_ssl_failure_once(url: str, err: Exception) -> None:
    """Print a TLS-failure diagnostic at most once per process -- same
    reasoning as _warn_self_signed_bypass_once. Only recommends the
    self-signed opt-in for an actual certificate-verification failure
    (ssl.SSLCertVerificationError); a plain ssl.SSLError from something
    else (protocol mismatch, e.g. MCP_HTTPS_ENABLED=true against a server
    that's actually speaking plain HTTP) gets a different message, since
    that opt-in cannot fix it and would be reached with it already on
    (CERT_NONE makes SSLCertVerificationError impossible once opted in)."""
    import ssl  # inline import: matches this module's stdlib-only, no-heavy-import-at-load-time design
    global _ssl_failure_warning_shown
    if _ssl_failure_warning_shown:
        return
    _ssl_failure_warning_shown = True
    if isinstance(err, ssl.SSLCertVerificationError):
        click.echo(
            f"WARNING: TLS certificate verification failed for {url} ({err}). "
            "If this is a local self-signed certificate, set "
            "MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS=true -- development only, "
            "it disables MITM protection.",
            err=True,
        )
    else:
        click.echo(
            f"WARNING: TLS error talking to {url} ({err}). If the server "
            "isn't actually running HTTPS, unset MCP_HTTPS_ENABLED; "
            "otherwise this may be a network issue or an incompatible TLS "
            "configuration.",
            err=True,
        )


def _probe_health(url: str, timeout: int = 3) -> tuple[dict | None, bool]:
    """GET a JSON endpoint. Returns (parsed_body_or_None, tls_blocked),
    where tls_blocked is True only for a certificate-verification failure
    specifically (not any other TLS or connection error) -- the one case
    where a None result likely means "something is listening and probably
    healthy, just unverifiable" rather than "not running." Callers that
    decide whether to kill and restart a process based on a failed health
    check need that distinction: killing a working server because its
    self-signed cert wasn't in the trust store would be a regression, not
    a fix.

    Certificate verification stays on unless MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS
    opts out. Warns when the bypass actually fires -- this
    disables certificate validation and is this CLI's only line of MITM
    defense against its own configured HTTPS endpoint, so it should never
    be silent. A verification failure that reaches here without the opt-in
    also gets a one-line pointer to that flag, instead of the generic
    "server not responding" a caller would otherwise see.

    Note: urllib wraps every failure raised inside urlopen's connection
    handling -- including ssl.SSLError -- in a urllib.error.URLError, so
    the real exception is on `.reason`, not the caught object itself.
    Catching bare ssl.SSLError here would never match a real handshake
    failure.
    """
    import ssl  # inline import: matches this module's stdlib-only, no-heavy-import-at-load-time design
    import urllib.error  # inline import: see above
    import urllib.request  # inline import: see above
    ctx = None
    if url.startswith("https:") and _cli_allow_self_signed_certs():
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        _warn_self_signed_bypass_once()
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode()), False
    except (ssl.SSLError, urllib.error.URLError) as e:
        reason = e.reason if isinstance(e, urllib.error.URLError) else e
        if isinstance(reason, ssl.SSLError):
            _warn_ssl_failure_once(url, reason)
            return None, isinstance(reason, ssl.SSLCertVerificationError)
        return None, False
    except Exception:
        return None, False


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
    # Resolve TLS before anything probes or spawns: this decides both what the
    # child gets and which scheme every health check in this invocation uses.
    tls = _resolve_server_tls()
    base_url = _base_url(host, port, scheme=tls.scheme)

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
        health, tls_blocked = _probe_health(f"{base_url}/api/health")
        if health and health.get("status") == "healthy":
            click.echo(f"Server already running (PID {existing_pid})")
            click.echo(f"  Dashboard: {base_url}")
            return
        if tls_blocked:
            # _read_pid() already filters out dead PIDs, so existing_pid
            # being truthy means this process is alive. A cert-verification
            # failure then means something is listening and likely healthy
            # -- killing and relaunching it here would be a regression for
            # a self-signed-cert user who upgraded, not a helpful recovery
            # from a genuinely stopped server.
            click.echo(
                f"Process {existing_pid} appears to be running, but its "
                "certificate could not be verified, so health could not be "
                "confirmed."
            )
            click.echo(
                "If this is a known self-signed certificate, set "
                "MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS=true and re-run, "
                f"verify manually at {base_url}/api/health, or run "
                "'memory stop' first if you want to force a restart."
            )
            click.echo(
                "Not restarting an already-running process based on an "
                "unverifiable health check."
            )
            return

    # Kill stale process on the port if any
    port_pid = _find_process_on_port(port)
    if port_pid and port_pid != existing_pid:
        click.echo(f"Freeing port {port} (stale PID {port_pid})...")
        _kill_process(port_pid)
        time.sleep(0.5)

    if not detach:
        # Foreground: import the heavy stuff and run directly
        click.echo(f"Starting {tls.scheme.upper()} server on {host}:{port}...")
        from mcp_memory_service.web.app import app  # heavy import
        import uvicorn  # inline import: heavy dependency, avoided at module load time
        uvicorn.run(app, host=host, port=port,
                    log_level="debug" if debug else "info", **tls.uvicorn_kwargs)
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

    # uvicorn's CLI knows nothing about MCP_HTTPS_ENABLED -- translating the
    # config into these flags is exactly the step that was missing, and its
    # absence downgraded HTTPS deployments to plain HTTP on every restart.
    # Empty for plain HTTP, so no branch is needed here.
    cmd += tls.cli_args

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

    _write_pid(proc.pid, scheme=tls.scheme)

    # Poll health endpoint until server is ready
    click.echo("Waiting for server to start...")
    saw_tls_block = False
    for i in range(60):
        time.sleep(0.5)
        health, tls_blocked = _probe_health(f"{base_url}/api/health")
        if health and health.get("status") == "healthy":
            click.echo(f"Server started (PID {proc.pid})")
            click.echo(f"  Dashboard: {base_url}")
            click.echo(f"  API docs:   {base_url}/docs")
            if health.get("version"):
                click.echo(f"  Version:    {health['version']}")
            if health.get("storage_backend"):
                click.echo(f"  Backend:    {health['storage_backend']}")
            return
        saw_tls_block = saw_tls_block or tls_blocked
        if proc.poll() is not None:
            # Child has exited -- further polling is pointless. Break
            # immediately rather than waiting out the remaining ~29s to
            # reach the same "check logs" conclusion.
            break
        if tls_blocked and _find_process_on_port(port) == proc.pid:
            # A cert-verification failure will never resolve itself by
            # polling again -- stop wasting the remaining time. Confirmed
            # via port ownership, not just "child hasn't exited": uvicorn
            # does its heavy import (config.load()) before binding the
            # socket, so during that window the child is alive but not
            # yet listening, and a probe hitting a *different* process
            # already on the port would also read as tls_blocked. Only
            # trust the message once our own child is the one that
            # actually answered. _find_process_on_port requires lsof/
            # netstat; on a host without either it always returns None,
            # so this never fires and polling continues to the timeout
            # below -- saw_tls_block still gets that case a useful hint.
            click.echo(f"Server process started (PID {proc.pid}), but its certificate "
                       "could not be verified.")
            click.echo("If this is a known self-signed certificate, set "
                       "MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS=true and check again with "
                       "'memory health'.")
            return

    if proc.poll() is not None:
        # A dead child could have exited for any reason (e.g. EADDRINUSE
        # because a pre-existing, unrelated service already owns the
        # port) -- a cert failure seen while polling may belong to that
        # other service, not to anything of ours, so the self-signed hint
        # below is intentionally reserved for the genuine-timeout case.
        click.echo(f"Server process (PID {proc.pid}) exited with code {proc.returncode} "
                   "before becoming healthy.")
    else:
        click.echo(f"Server process started (PID {proc.pid}) but health check timed out.")
        if saw_tls_block:
            click.echo("A certificate-verification failure was seen while polling. If "
                       "this is a known self-signed certificate, set "
                       "MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS=true and check again with "
                       "'memory health'.")
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
        base_url = _base_url(host, port)
        health, tls_blocked = _probe_health(f"{base_url}/api/health")
        if health:
            click.echo("Server responds but no managed PID found. Force-stopping by port...")
            port_pid_now = _find_process_on_port(port)
            if port_pid_now:
                _kill_process(port_pid_now)
                click.echo("Server stopped.")
            else:
                click.echo("Could not find process on port. Stop manually.")
        elif tls_blocked:
            click.echo(
                "Something is listening on this port, but its certificate "
                "could not be verified, so it can't be confirmed as the "
                "memory server."
            )
            click.echo(
                "If this is a known self-signed certificate, set "
                "MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS=true and re-run, or "
                "stop it manually if it's not the memory server."
            )
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
    base_url = _base_url(host, port)
    
    # If storage_backend not specified, try to read it from the running server
    if storage_backend is None:
        health, tls_blocked = _probe_health(f"{base_url}/api/health")
        if health and health.get("storage_backend"):
            storage_backend = health["storage_backend"]
            # Normalize: server may return "sqlite_vec" or "sqlite-vec"
            if storage_backend not in ("sqlite_vec", "sqlite-vec", "cloudflare", "hybrid"):
                storage_backend = None  # Unknown backend, let launch use its default
        elif tls_blocked:
            click.echo(
                "Could not confirm the current storage backend because the "
                "server's certificate could not be verified. Restarting "
                "without an explicit backend -- the server will fall back "
                "to its own configuration (.env or environment), which may "
                "not be what you expect. Pass --storage-backend explicitly "
                "to be sure.",
                err=True,
            )
    
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
    base_url = _base_url(host, port)

    pid = _read_pid()
    health, tls_blocked = _probe_health(f"{base_url}/api/health")

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
    elif tls_blocked:
        click.echo("  Status:    UNKNOWN (certificate unverifiable)")
        click.echo(f"  Port:      {port}")
        click.echo(f"  Dashboard: {base_url}")
        if pid:
            click.echo(f"  PID:       {pid}")
        click.echo()
        click.echo("  Something is listening on this port, but its certificate")
        click.echo("  could not be verified. If this is a known self-signed")
        click.echo("  certificate, set MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS=true")
        click.echo("  and check again.")
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
    base_url = _base_url(host, port)

    health, tls_blocked = _probe_health(f"{base_url}/api/health")
    if health:
        click.echo(json.dumps(health, indent=2))
    elif tls_blocked:
        # The /health/detailed probe would fail the identical way for the
        # identical reason -- skip it rather than wait out a second doomed
        # request before showing the same diagnostic.
        click.echo("Something is listening, but its certificate could not be verified.")
        click.echo("If this is a known self-signed certificate, set "
                   "MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS=true and check again.")
        sys.exit(1)
    else:
        detailed, detailed_tls_blocked = _probe_health(f"{base_url}/api/health/detailed")
        if detailed:
            click.echo(json.dumps(detailed, indent=2))
        elif detailed_tls_blocked:
            click.echo("Something is listening, but its certificate could not be verified.")
            click.echo("If this is a known self-signed certificate, set "
                       "MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS=true and check again.")
            sys.exit(1)
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
