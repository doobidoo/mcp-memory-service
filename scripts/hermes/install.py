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


PORT_KEYS = {"MCP_HTTP_PORT", "MCP_SSE_PORT"}


def _parse_env_line(raw: str) -> tuple[str, str] | None:
    """`KEY=value` from one env-file line, or None when it is not an assignment."""
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, _, value = line.partition("=")
    return key.strip(), value.strip()


def env_file_port(path: Path) -> int | None:
    """Port declared in a service env file (`MCP_HTTP_PORT` / `MCP_SSE_PORT`)."""
    if not path.is_file():
        return None
    for raw in path.read_text().splitlines():
        parsed = _parse_env_line(raw)
        if parsed is None or parsed[0] not in PORT_KEYS:
            continue
        try:
            return int(parsed[1])
        except ValueError:
            return None
    return None


def api_key() -> str:
    """Bearer token when the streamable-http transport is key protected."""
    return os.environ.get("MCP_API_KEY", "").strip()


HEALTH_ROUTES = ("/health", "/api/health")


def health_ok(port: int) -> tuple[bool, str]:
    """Is the service reachable? Tries both documented health routes.

    `memory server --streamable-http` serves `/health`; the HTTP server started by
    `memory launch` serves `/api/health`. Checking only one aborted installation
    against a perfectly healthy server of the other kind.
    """
    key = api_key()
    last = ""
    for route in HEALTH_ROUTES:
        request = urllib.request.Request(f"http://localhost:{port}{route}")
        if key:
            request.add_header("Authorization", f"Bearer {key}")
        try:
            urllib.request.urlopen(request, timeout=5)
            return True, ""
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                return False, "the server requires a key: set MCP_API_KEY"
            last = f"{route} returned {exc.code}"
        except Exception as exc:  # noqa: BLE001
            last = f"service not answering on :{port} ({exc})"
    return False, last


def write_env_key(hermes_home: Path, key: str) -> str | None:
    """Record the key under the env var name Hermes reads for this server.

    `hermes mcp add` sets `headers.Authorization = Bearer ${MCP_MMS_API_KEY}` and
    reads that variable from the profile's .env. Without it the registration is
    created but cannot authenticate, so the probe would pass while Hermes fails.
    """
    env_path = hermes_home / ".env"
    name = "MCP_MMS_API_KEY"
    lines: list[str] = []
    previous: str | None = None
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith(f"{name}="):
                previous = line.split("=", 1)[1].strip()
                continue
            lines.append(line)
    lines.append(f"{name}={key}")
    env_path.write_text("\n".join(lines) + "\n")
    os.environ[name] = key
    return previous


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
    return _line_matches_server(out.splitlines(), port)


def _line_matches_server(lines: Sequence[str], port: int) -> bool:
    """Find the `mms` row and require it to be enabled and on `port`."""
    expected = f":{port}/mcp"
    for line in lines:
        tokens = line.split()
        if not tokens or tokens[0] != SERVER_NAME:
            continue
        if "disabled" in line.lower() or "✗" in line:
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
    _require_store_accepted(
        do_call(port, "memory_store", {"content": f"install probe {token}", "metadata": {"tags": PROBE_TAG}})
    )
    found_text = _require_search_returns_probe(
        do_call(port, "memory_search", {"query": token, "limit": 1, "mode": "exact"}), token
    )
    return _extract_hash(found_text)


def _require_store_accepted(stored: dict) -> None:
    """A write is only accepted when it reports neither an error nor a duplicate."""
    text = _tool_text(stored).lower()
    if stored.get("error") or "error" in text or "duplicate" in text:
        raise RuntimeError(f"memory_store did not accept the probe: {_tool_text(stored)[:200]}")


def _require_search_returns_probe(found: dict, token: str) -> str:
    """The search must return THIS run's probe, not merely some text."""
    text = _tool_text(found)
    if found.get("error") or "error searching" in text.lower():
        raise RuntimeError(f"memory_search failed: {text[:200]}")
    if "no memories found" in text.lower() or token not in text:
        raise RuntimeError(
            "the probe was stored but the search did not return it; the service is "
            "reachable but not retrieving"
        )
    return text


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


def write_manifest(path: Path, probe_hash: str, *, added_env_key: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "server": SERVER_NAME,
        "probe_hash": probe_hash,
        "added_env_key": added_env_key,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def remove_env_key(hermes_home: Path) -> None:
    """Undo a key this installer added, leaving later user edits alone."""
    env_path = hermes_home / ".env"
    if not env_path.is_file():
        return
    kept = [
        line
        for line in env_path.read_text().splitlines()
        if not line.strip().startswith("MCP_MMS_API_KEY=")
    ]
    env_path.write_text("\n".join(kept) + ("\n" if kept else ""))


def read_manifest(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# --- main --------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-home", default=None, help="Hermes profile directory")
    parser.add_argument("--port", type=int, default=None, help="service port (env file, else 8000)")
    parser.add_argument("--env-file", default=None, help="service env file declaring the port")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    return parser


def _resolve_target(args: argparse.Namespace) -> tuple[Path, int, Path]:
    """Profile directory, service port and manifest path for this run."""
    env = dict(os.environ)
    home = Path(args.hermes_home) if args.hermes_home else Path.home()
    if args.hermes_home:
        env["HERMES_HOME"] = args.hermes_home
    hermes_home = resolve_hermes_home(env, home)

    port = args.port
    if port is None and args.env_file:
        port = env_file_port(Path(args.env_file))
    return hermes_home, port or DEFAULT_PORT, hermes_home / "mms-install.json"


def _uninstall(hermes_home: Path, port: int, manifest: Path) -> int:
    rc, out = default_runner(["hermes", "mcp", "remove", SERVER_NAME])
    recorded = read_manifest(manifest)
    delete_probe(port, recorded.get("probe_hash") or "")
    if recorded.get("added_env_key"):
        remove_env_key(hermes_home)
    if manifest.is_file():
        manifest.unlink()
    if rc != 0:
        print(f"error: could not remove the registration: {out.strip()[:200]}", file=sys.stderr)
        return 1
    print(f"removed {SERVER_NAME} and the install probe")
    return 0


def _register(action: "Action") -> int:
    """Run the registration command and require the effect to be visible."""
    rc, out = default_runner(action.argv)
    if rc != 0:
        print(f"error: `{action.render()}` failed: {out.strip()[:200]}", file=sys.stderr)
        return 1
    return 0


def _verified_round_trip(port: int) -> tuple[int, str]:
    """Prove store/read works and clean the probe up. Returns (rc, probe_hash)."""
    try:
        probe_hash = verify_round_trip(port, token=os.urandom(6).hex())
    except urllib.error.HTTPError as exc:
        hint = " (the server requires a key: set MCP_API_KEY)" if exc.code == 401 else ""
        print(f"error: round trip failed with HTTP {exc.code}{hint}", file=sys.stderr)
        return 1, ""
    except Exception as exc:  # noqa: BLE001
        print(f"error: round trip failed: {exc}", file=sys.stderr)
        return 1, ""
    delete_probe(port, probe_hash)
    return 0, probe_hash


def _install(hermes_home: Path, port: int, manifest: Path, actions: list["Action"]) -> int:
    ok, reason = health_ok(port)
    if not ok:
        print(f"error: {reason}; nothing was written", file=sys.stderr)
        return 1

    added_env_key = write_env_key(hermes_home, api_key()) is None if api_key() else False

    failure = _register(actions[0])
    if failure:
        return failure

    if not mcp_is_registered(port):
        print(
            f"error: `hermes mcp list` does not show {SERVER_NAME} on :{port}/mcp after "
            "registration; the command reported success without registering",
            file=sys.stderr,
        )
        return 1

    failure, probe_hash = _verified_round_trip(port)
    if failure:
        return failure

    write_manifest(manifest, probe_hash, added_env_key=added_env_key)
    print(f"registered {SERVER_NAME} on :{port}/mcp and verified a store/read round trip")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        hermes_home, port, manifest = _resolve_target(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.uninstall:
        if args.dry_run:
            print(f"would remove {SERVER_NAME} and the install probe from {hermes_home}")
            return 0
        return _uninstall(hermes_home, port, manifest)

    actions = plan(port, mcp_is_registered(port))
    for action in actions:
        print(action.render())
    if args.dry_run:
        if not actions:
            print(f"{SERVER_NAME} is already registered on :{port}/mcp; nothing to do")
        return 0

    if not actions:
        return _verify_existing(port)

    return _install(hermes_home, port, manifest, actions)


def _verify_existing(port: int) -> int:
    """Already registered is not the same as working: verify it anyway.

    A stopped server or broken retrieval used to pass a repeat install as
    "nothing to do", so the promise to verify the connection held only on the
    first run.
    """
    ok, reason = health_ok(port)
    if not ok:
        print(f"error: {SERVER_NAME} is registered but {reason}", file=sys.stderr)
        return 1

    failure, _ = _verified_round_trip(port)
    if failure:
        return failure

    print("nothing to register: the server is already registered and the round trip works")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
