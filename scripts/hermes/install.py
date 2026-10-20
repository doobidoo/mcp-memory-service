#!/usr/bin/env python3
"""Register the MCP memory service with Hermes and prove the round trip works.

Scope is deliberately narrow: this registers the server and verifies that a write
followed by a read actually round-trips. Starting the server is `memory launch`'s
job, and lifecycle hooks / usage skills are separate work.

Three Hermes CLI behaviours shape the design (measured on Hermes 0.22.x, Linux):

1. `hermes mcp add` is interactive — it asks which tools to enable. With no stdin
   answered it prints `Cancelled.` and exits 0, so the exit code is not evidence
   that the server was registered.
2. `hermes config get` does not exist (only show/edit/set/path/env-path/check/migrate).
3. `hermes config unset` does not exist.

Usage:
    python scripts/hermes/install.py --port 8000
    python scripts/hermes/install.py --dry-run
    python scripts/hermes/install.py --uninstall
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

DEFAULT_PORT = 8000
SERVER_NAME = "mms"
PROBE_TAG = "mms-install-probe"


# --- location ----------------------------------------------------------------


def resolve_hermes_home(env: dict, home: Path) -> Path:
    """The Hermes profile directory.

    An explicit path is used literally: `--hermes-home /profiles/work` means that
    directory, not `/profiles/work/.hermes`. Only the implicit default appends
    `.hermes` to the user home.
    """
    explicit = env.get("HERMES_HOME")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_dir():
            raise RuntimeError(f"HERMES_HOME points at a directory that does not exist: {path}")
        return path
    candidate = home / ".hermes"
    if not candidate.is_dir():
        raise RuntimeError(f"no Hermes profile at {candidate}; pass --hermes-home")
    return candidate


def env_file_port(path: Path) -> int | None:
    """Port declared in a service env file (`MCP_HTTP_PORT` / `MCP_SSE_PORT`)."""
    if not path.is_file():
        return None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() in {"MCP_HTTP_PORT", "MCP_SSE_PORT"}:
            try:
                return int(value.strip())
            except ValueError:
                return None
    return None


def api_key() -> str:
    """Bearer token when the streamable-http transport is key protected."""
    return os.environ.get("MCP_API_KEY", "").strip()


def health_ok(port: int) -> tuple[bool, str]:
    """Is the service reachable? Returns (ok, message) so 401 is explicit."""
    request = urllib.request.Request(f"http://localhost:{port}/health")
    key = api_key()
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    try:
        urllib.request.urlopen(request, timeout=5)
        return True, ""
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return False, "the server requires a key: set MCP_API_KEY"
        return False, f"/health returned {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return False, f"service not answering on :{port} ({exc})"


def write_env_key(hermes_home: Path, key: str) -> None:
    """Record the key under the env var name Hermes reads for this server.

    `hermes mcp add` sets `headers.Authorization = Bearer ${MCP_MMS_API_KEY}` and
    reads that variable from the profile's .env. Without it the registration is
    created but cannot authenticate, so the probe would pass while Hermes fails.
    """
    env_path = hermes_home / ".env"
    name = "MCP_MMS_API_KEY"
    lines = []
    if env_path.is_file():
        lines = [
            line
            for line in env_path.read_text().splitlines()
            if not line.strip().startswith(f"{name}=")
        ]
    lines.append(f"{name}={key}")
    env_path.write_text("\n".join(lines) + "\n")
    os.environ[name] = key


# --- registration ------------------------------------------------------------


@dataclass
class Action:
    argv: list[str]
    description: str

    def render(self) -> str:
        return " ".join(self.argv)


def mcp_is_registered(port: int, runner: Callable[[Sequence[str]], tuple[int, str]] | None = None) -> bool:
    """True only if `hermes mcp list` shows mms pointing at THIS port.

    A name match alone accepts a stale entry for another host on the same port
    and a disabled one; both would skip registration while the probe checks the
    local server, so the endpoint and the enabled state are checked too.
    """
    run = runner or default_runner
    try:
        rc, out = run(["hermes", "mcp", "list"])
    except Exception:
        return False
    if rc != 0:
        return False
    expected = f":{port}/mcp"
    for line in out.splitlines():
        tokens = line.split()
        if tokens and tokens[0] == SERVER_NAME:
            lowered = line.lower()
            if "disabled" in lowered or "✗" in line:
                return False
            return expected in line
    return False


def plan(port: int, installed: bool) -> list[Action]:
    """Registration only, and only when it is missing."""
    if installed:
        return []
    return [
        Action(
            argv=["hermes", "mcp", "add", SERVER_NAME, "--url", f"http://localhost:{port}/mcp"],
            description=f"register {SERVER_NAME} with Hermes",
        )
    ]


def default_runner(argv: Sequence[str]) -> tuple[int, str]:
    """Run one command, answering the interactive prompts of `hermes mcp add`."""
    proc = subprocess.run(
        argv, capture_output=True, text=True, timeout=180, input="Y\n" * 5, check=False
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


# --- round trip --------------------------------------------------------------


def _mcp_call(port: int, tool: str, arguments: dict) -> dict:
    payload = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}
    ).encode()
    request = urllib.request.Request(
        f"http://localhost:{port}/mcp",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **({"Authorization": f"Bearer {api_key()}"} if api_key() else {}),
        },
    )
    body = urllib.request.urlopen(request, timeout=30).read().decode("utf-8", "replace")
    for line in body.splitlines():
        if line.startswith("data:"):
            body = line[5:].strip()
            break
    return json.loads(body)


def _tool_text(result: dict) -> str:
    blocks = result.get("result", {}).get("content") or []
    return " ".join(b.get("text", "") for b in blocks if isinstance(b, dict))


def verify_round_trip(
    port: int,
    token: str,
    call: Callable[[int, str, dict], dict] | None = None,
) -> str:
    """Write a uniquely tagged probe, read it back, and return its hash.

    Both calls are inspected. `memory_search` always returns exactly one text
    block — the results, `No memories found ...`, or `Error searching memories: ...`
    — so counting blocks proves nothing. The probe carries a unique token because
    a constant one is rejected as an exact duplicate on the second run, and the
    search would then find the PREVIOUS run's probe and never exercise this write.
    """
    do_call = call or _mcp_call
    content = f"install probe {token}"

    stored = do_call(port, "memory_store", {"content": content, "metadata": {"tags": PROBE_TAG}})
    stored_text = _tool_text(stored).lower()
    if stored.get("error") or "error" in stored_text or "duplicate" in stored_text:
        raise RuntimeError(f"memory_store did not accept the probe: {_tool_text(stored)[:200]}")

    found = do_call(port, "memory_search", {"query": token, "limit": 1, "mode": "exact"})
    found_text = _tool_text(found)
    if found.get("error") or "error searching" in found_text.lower():
        raise RuntimeError(f"memory_search failed: {found_text[:200]}")
    if "no memories found" in found_text.lower() or token not in found_text:
        raise RuntimeError(
            "the probe was stored but the search did not return it; the service is "
            "reachable but not retrieving"
        )

    return _extract_hash(found_text)


def _extract_hash(text: str) -> str:
    for token in text.replace(":", " ").split():
        cleaned = token.strip("[](),")
        if len(cleaned) == 64 and all(c in "0123456789abcdef" for c in cleaned.lower()):
            return cleaned
    return ""


def delete_probe(port: int, probe_hash: str, call: Callable[[int, str, dict], dict] | None = None) -> None:
    """Remove the probe we created; an installer must not leave data behind."""
    if not probe_hash:
        return
    do_call = call or _mcp_call
    try:
        do_call(port, "memory_delete", {"content_hash": probe_hash})
    except Exception:
        pass  # best effort; the install already succeeded


# --- manifest ----------------------------------------------------------------


def write_manifest(path: Path, probe_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"server": SERVER_NAME, "probe_hash": probe_hash}, indent=2) + "\n")


def read_manifest(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# --- main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-home", default=None, help="Hermes profile directory")
    parser.add_argument("--port", type=int, default=None, help="service port (default: env file, else 8000)")
    parser.add_argument("--env-file", default=None, help="service env file declaring the port")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args(argv)

    env = dict(os.environ)
    home = Path(args.hermes_home) if args.hermes_home else Path.home()
    if args.hermes_home:
        env["HERMES_HOME"] = args.hermes_home

    try:
        hermes_home = resolve_hermes_home(env, home)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    port = args.port
    if port is None and args.env_file:
        port = env_file_port(Path(args.env_file))
    if port is None:
        port = DEFAULT_PORT

    manifest = hermes_home / "mms-install.json"

    if args.uninstall:
        if args.dry_run:
            print(f"would remove {SERVER_NAME} and the install probe from {hermes_home}")
            return 0
        rc, out = default_runner(["hermes", "mcp", "remove", SERVER_NAME])
        probe_hash = read_manifest(manifest).get("probe_hash") or ""
        delete_probe(port, probe_hash)
        if manifest.is_file():
            manifest.unlink()
        if rc != 0:
            print(f"error: could not remove the registration: {out.strip()[:200]}", file=sys.stderr)
            return 1
        print(f"removed {SERVER_NAME} and the install probe")
        return 0

    actions = plan(port, mcp_is_registered(port))
    if not actions:
        print("nothing to do: the server is already registered for this port")
        return 0

    for action in actions:
        print(action.render())
    if args.dry_run:
        return 0

    ok, reason = health_ok(port)
    if not ok:
        print(f"error: {reason}; nothing was written", file=sys.stderr)
        return 1

    if api_key():
        write_env_key(hermes_home, api_key())

    rc, out = default_runner(actions[0].argv)
    if rc != 0:
        print(f"error: `{actions[0].render()}` failed: {out.strip()[:200]}", file=sys.stderr)
        return 1

    if not mcp_is_registered(port):
        print(
            f"error: `hermes mcp list` does not show {SERVER_NAME} on :{port}/mcp after "
            "registration; the command reported success without registering",
            file=sys.stderr,
        )
        return 1

    try:
        probe_hash = verify_round_trip(port, token=os.urandom(6).hex())
    except urllib.error.HTTPError as exc:
        hint = " (the server requires a key: set MCP_API_KEY)" if exc.code == 401 else ""
        print(f"error: round trip failed with HTTP {exc.code}{hint}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: round trip failed: {exc}", file=sys.stderr)
        return 1

    delete_probe(port, probe_hash)
    write_manifest(manifest, probe_hash)
    print(f"registered {SERVER_NAME} on :{port}/mcp and verified a store/read round trip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
