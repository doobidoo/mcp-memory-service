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

"""`mms-doctor` — diagnostic, degradation, and proof-by-host CLI.

Spec: docs/rfc/planned/spec-d3-mms-doctor.md

Runs six boundary-injectable checks (health, port, version, sync, author,
embeddings) and reports pass/fail with cited evidence. stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

HEALTH_TIMEOUT_SECONDS = 2.0


@dataclass
class CheckResult:
    """Outcome of a single doctor check."""

    name: str
    ok: bool
    evidence: str
    repaired: bool = False


def _default_health_get(port: int = 0) -> dict:
    """Real boundary: GET http://127.0.0.1:<port>/health with a short timeout.

    Returns a dict shaped like {"status": <http status>, "port": <port>, ...}.
    Raises on connection failure (caller is responsible for catching).
    """
    url = f"http://127.0.0.1:{port}/health"
    with urllib.request.urlopen(url, timeout=HEALTH_TIMEOUT_SECONDS) as resp:
        status = resp.status
        try:
            body = json.loads(resp.read().decode("utf-8"))
        except Exception:
            body = {}
    return {"status": status, "port": port, "body": body}


def _default_restart_service() -> None:
    """Real boundary: restart the memory-service unit, Linux-only.

    On non-Linux platforms this is a deliberate no-op — the spec forbids
    invoking systemctl outside Linux (same principle as D7/R14).
    """
    if platform.system() != "Linux":
        return
    # R3-bis: never restart a unit that is still coming up. The service needs
    # ~30 s to become ready; a restart during startup prevents it from ever
    # completing (observed 2026-10-20: systemd start-limit-hit, service down).
    # Tolerant: a host without user systemd (container, WSL, Alpine) must not
    # blow up the diagnostic tool (same pattern as _default_journal_lines).
    try:
        state = subprocess.run(
            ["systemctl", "--user", "is-active", "memory-service"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        if state in ("activating", "reloading", "active"):
            return
        subprocess.run(
            ["systemctl", "--user", "restart", "memory-service"],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except Exception:
        return


def _default_journal_lines() -> list:
    """Real boundary: recent journal lines for the memory-service unit.

    Linux-only (journalctl); returns [] elsewhere or on failure so the sync
    check degrades gracefully rather than raising.
    """
    if platform.system() != "Linux":
        return []
    try:
        proc = subprocess.run(
            ["journalctl", "--user", "-u", "memory-service", "-n", "50", "--no-pager"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.stdout.splitlines()
    except Exception:
        return []


def _default_unit_state() -> str:
    """Real boundary: current systemd state of the memory-service unit.

    Linux-only (systemctl); returns "unknown" elsewhere or on failure so
    the health check degrades gracefully rather than raising.
    """
    if platform.system() != "Linux":
        return "unknown"
    try:
        return subprocess.run(
            ["systemctl", "--user", "is-active", "memory-service"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _check_health(
    *,
    port: int,
    health_get: Callable[..., dict],
    restart_service: Callable[[], None],
    sleep: Callable[[float], None] = time.sleep,
    ready_timeout: float = 45.0,
    poll_initial: float = 1.0,
    poll_max: float = 8.0,
    unit_state: Optional[Callable[[], str]] = None,
) -> tuple:
    """Check 1: GET /health; on failure, restart (R3-bis) + readiness poll
    with backoff (R3-bis b), then report the intervention (R3-ter).

    Returns (CheckResult, last_raw_result_or_None). The raw result is
    reused by the port check so we never call health_get() more times
    than needed to establish readiness.
    """
    if unit_state is None:
        unit_state = _default_unit_state

    status_before: Optional[str] = None
    try:
        result = health_get()
        status_before = str(result.get("status", "unknown"))
        return (
            CheckResult(
                name="health",
                ok=True,
                evidence=f"health responded with status={status_before}",
                repaired=False,
            ),
            result,
        )
    except Exception as exc:
        status_before = f"error: {exc}"

    # R3-bis: consult unit state BEFORE deciding to restart. If the guard
    # inside _default_restart_service would skip the restart (unit still
    # coming up), say so explicitly instead of silently retrying forever.
    state_before = unit_state()
    if state_before in ("activating", "reloading", "active"):
        evidence = (
            f"health status before={status_before}; restart evitado "
            f"(unit={state_before})"
        )
        return (
            CheckResult(name="health", ok=False, evidence=evidence, repaired=False),
            None,
        )

    try:
        restart_service()
    except Exception:
        pass

    state_after_restart = unit_state()

    # R3-bis b: readiness poll with backoff — 1s, 2s, 4s, 8s(cap), until
    # ready or ready_timeout is exhausted.
    elapsed = 0.0
    wait = poll_initial
    attempt = 0
    while True:
        attempt += 1
        try:
            result = health_get()
            status_after = str(result.get("status", "unknown"))
            evidence = (
                f"unit={state_before} antes, {state_after_restart} depois; "
                f"health status before={status_before}, after={status_after} "
                f"(tentativa={attempt})"
            )
            return (
                CheckResult(name="health", ok=True, evidence=evidence, repaired=True),
                result,
            )
        except Exception as exc2:
            status_after = f"error: {exc2}"

        if elapsed + wait > ready_timeout:
            evidence = (
                f"unit={state_before} antes, {state_after_restart} depois; "
                f"health status before={status_before}, after={status_after}; "
                f"esgotado apos {attempt} tentativas (ready_timeout={ready_timeout}s)"
            )
            return (
                CheckResult(name="health", ok=False, evidence=evidence, repaired=False),
                None,
            )

        sleep(wait)
        elapsed += wait
        wait = min(wait * 2, poll_max)


DEFAULT_PORT = 8765
DEFAULT_ENV_FILE_PATH = "~/dtp/ai-configs/services/env/memory-service.env"


def _resolve_ready_timeout(explicit: Optional[float]) -> float:
    """Readiness window in seconds: argument > env > 45.

    A malformed env value must not raise into the report (CA6): fall back to the
    default, same care already taken by the sync threshold resolver.
    """
    if explicit is not None:
        return float(explicit)
    try:
        return float(os.environ.get("MCP_DOCTOR_READY_TIMEOUT", "45"))
    except (TypeError, ValueError):
        return 45.0


def _read_port_from_env_file(env_file_path: str) -> Optional[int]:
    """Parse MCP_SSE_PORT=<n> out of a service env file, if present/readable.

    Tolerant: missing file, unreadable file, or missing key all yield None
    rather than raising — this is a best-effort fallback in the port
    resolution chain, not a required source.
    """
    path = Path(env_file_path).expanduser()
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("MCP_SSE_PORT="):
            value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _resolve_port(
    *,
    explicit_port: Optional[int] = None,
    env_file_path: Optional[str] = None,
    fallback_port: int = DEFAULT_PORT,
) -> int:
    """Resolve the declared SSE port with the documented precedence.

    Order: (1) explicit_port (--port) > (2) MCP_SSE_PORT env var >
    (3) MCP_SSE_PORT parsed from the service env file, ONLY when
    env_file_path was explicitly provided by the caller > (4) fallback_port.

    env_file_path is opt-in on purpose: run_checks()'s existing callers
    (tests, library users) never pass it, so by default the declared port
    falls back to the `port` argument they already supply — unchanged
    behavior. Only `main()` (the real CLI) passes --env-file explicitly,
    which is where host-dependent env-file lookups belong.
    """
    if explicit_port is not None:
        return explicit_port

    env_value = os.environ.get("MCP_SSE_PORT")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass

    if env_file_path is not None:
        from_file = _read_port_from_env_file(env_file_path)
        if from_file is not None:
            return from_file

    return fallback_port


def _check_port(*, declared_port: int, listening_port: Optional[int]) -> CheckResult:
    """Check 2: declared port (resolved via precedence chain) vs health reply.

    When the health probe never answered, we don't know the real listening
    port — the evidence says so explicitly rather than implying the service
    is dead (it may simply be answering on a different, undeclared port).
    """
    if listening_port is None:
        return CheckResult(
            name="port",
            ok=False,
            evidence=(
                f"sem resposta em declarado={declared_port} "
                "(porta errada ou servico fora; porta real desconhecida)"
            ),
        )
    ok = declared_port == listening_port
    evidence = f"declarado={declared_port} ouvido={listening_port}"
    return CheckResult(name="port", ok=ok, evidence=evidence)


def _check_version(*, repo_path: str) -> CheckResult:
    """Check 3: installed mcp_memory_service.__version__ vs pyproject.toml."""
    import mcp_memory_service as _pkg

    installed_version = getattr(_pkg, "__version__", "unknown")

    pyproject_path = Path(repo_path) / "pyproject.toml"
    repo_version = "unknown"
    try:
        text = pyproject_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("version"):
                parts = stripped.split("=", 1)
                if len(parts) == 2:
                    repo_version = parts[1].strip().strip('"').strip("'")
                    break
    except Exception:
        repo_version = "unreadable"

    ok = installed_version == repo_version
    evidence = (
        f"instalada={installed_version} pyproject={repo_version} ({pyproject_path})"
    )
    return CheckResult(name="version", ok=ok, evidence=evidence)


DEFAULT_SYNC_MAX_AGE_SECONDS = 3600.0  # 1 hour: delta-sync cron runs every 30 min


def _resolve_sync_max_age(sync_max_age: Optional[float]) -> float:
    """Resolve the sync staleness threshold: explicit arg > env > default.

    The real delta-sync cron runs every 30 minutes, so a 12-minute-old
    cursor is normal, not stale. Default is 1h; override via
    --sync-max-age or the MCP_DOCTOR_SYNC_MAX_AGE env var (seconds).
    """
    if sync_max_age is not None:
        return float(sync_max_age)
    env_value = os.environ.get("MCP_DOCTOR_SYNC_MAX_AGE")
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            pass
    return DEFAULT_SYNC_MAX_AGE_SECONDS


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Tolerant schema check: True iff a table with this exact name exists.

    Used so every check degrades to a CheckResult(ok=False, evidence=...)
    on a zeroed/schemaless database instead of raising sqlite3.OperationalError.
    """
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _check_sync(
    *,
    db_path: str,
    journal_lines: Callable[[], list],
    sync_max_age: Optional[float] = None,
) -> CheckResult:
    """Check 4: sync_cursor freshness + recent journal errors.

    Staleness threshold is configurable (default 1h) because the real
    delta-sync cron runs every 30 minutes — a 12-minute-old cursor is
    normal, not a failure.
    """
    threshold = _resolve_sync_max_age(sync_max_age)

    conn = sqlite3.connect(db_path)
    try:
        if not _table_exists(conn, "sync_cursor"):
            return CheckResult(
                name="sync",
                ok=False,
                evidence="banco sem schema: tabela sync_cursor ausente",
            )
        row = conn.execute(
            "SELECT peer_id, last_seq_seen, last_hlc_physical, last_hlc_logical, "
            "updated_at FROM sync_cursor ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return CheckResult(
            name="sync", ok=False, evidence="sync_cursor vazio: nenhum cursor registrado"
        )

    peer_id, last_seq_seen, last_hlc_physical, last_hlc_logical, updated_at = row

    try:
        lines = journal_lines() or []
    except Exception:
        lines = []

    error_lines = [ln for ln in lines if "ERROR" in ln or "error" in ln.lower()]

    age = time.time() - float(updated_at)
    is_stale = age > threshold

    if is_stale and error_lines:
        evidence = (
            f"cursor parado: peer={peer_id} updated_at={updated_at} "
            f"(idade={age:.0f}s, limite={threshold:.0f}s); erro no journal: {error_lines[0]}"
        )
        return CheckResult(name="sync", ok=False, evidence=evidence)

    if is_stale:
        evidence = (
            f"cursor parado: peer={peer_id} updated_at={updated_at} "
            f"(idade={age:.0f}s, limite={threshold:.0f}s)"
        )
        return CheckResult(name="sync", ok=False, evidence=evidence)

    evidence = (
        f"peer={peer_id} last_seq_seen={last_seq_seen} updated_at={updated_at} "
        f"(idade={age:.0f}s, limite={threshold:.0f}s) journal_errors={len(error_lines)}"
    )
    return CheckResult(name="sync", ok=True, evidence=evidence)


def _check_author(*, db_path: str) -> CheckResult:
    """Check 5: latest live memory must have a non-empty metadata.agent_id."""
    conn = sqlite3.connect(db_path)
    try:
        if not _table_exists(conn, "memories"):
            return CheckResult(
                name="author",
                ok=False,
                evidence="banco sem schema: tabela memories ausente",
            )
        row = conn.execute(
            "SELECT content_hash, metadata, created_at FROM memories "
            "WHERE deleted_at IS NULL ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return CheckResult(
            name="author", ok=False, evidence="nenhuma memoria viva encontrada"
        )

    content_hash, metadata_raw, created_at = row
    try:
        metadata = json.loads(metadata_raw) if metadata_raw else {}
    except Exception:
        metadata = {}

    agent_id = metadata.get("agent_id") if isinstance(metadata, dict) else None
    ok = bool(agent_id)
    evidence = (
        f"hash={content_hash} created_at={created_at} agent_id={agent_id!r}"
    )
    return CheckResult(name="author", ok=ok, evidence=evidence)


def _check_embeddings(*, db_path: str) -> CheckResult:
    """Check 6: live memories (deleted_at IS NULL) that are missing an embedding.

    CRITICAL: never query memory_embeddings (vec0 virtual table) with plain
    sqlite3 — use memory_embeddings_rowids (a normal table) instead.

    Comparing raw COUNT(*) of memory_embeddings_rowids against live-memory
    COUNT(*) is wrong: soft-deleted (tombstoned) memories legitimately keep
    their embedding row, so that comparison flags normal tombstones as a
    failure. What actually matters is live memories that have NO embedding
    at all.
    """
    conn = sqlite3.connect(db_path)
    try:
        if not _table_exists(conn, "memories") or not _table_exists(
            conn, "memory_embeddings_rowids"
        ):
            return CheckResult(
                name="embeddings",
                ok=False,
                evidence="banco sem schema: tabela memories ou "
                "memory_embeddings_rowids ausente",
            )
        (live_count,) = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL"
        ).fetchone()
        (missing_count,) = conn.execute(
            "SELECT COUNT(*) FROM memories m WHERE m.deleted_at IS NULL "
            "AND m.rowid NOT IN (SELECT rowid FROM memory_embeddings_rowids)"
        ).fetchone()
        (tombstoned_with_embedding,) = conn.execute(
            "SELECT COUNT(*) FROM memories m WHERE m.deleted_at IS NOT NULL "
            "AND m.rowid IN (SELECT rowid FROM memory_embeddings_rowids)"
        ).fetchone()
    finally:
        conn.close()

    ok = missing_count == 0
    evidence = (
        f"vivas={live_count} com_embedding={live_count - missing_count} "
        f"faltando={missing_count} "
        f"tombstones_com_embedding={tombstoned_with_embedding} (informativo)"
    )
    return CheckResult(name="embeddings", ok=ok, evidence=evidence)


def run_checks(
    port: int,
    db_path: str,
    repo_path: str,
    health_get: Optional[Callable[..., dict]] = None,
    journal_lines: Optional[Callable[[], list]] = None,
    restart_service: Optional[Callable[[], None]] = None,
    sync_max_age: Optional[float] = None,
    env_file_path: Optional[str] = None,
    explicit_port: Optional[int] = None,
    sleep: Optional[Callable[[float], None]] = None,
    ready_timeout: Optional[float] = None,
    unit_state: Optional[Callable[[], str]] = None,
) -> list:
    """Run the six doctor checks and return their results.

    All I/O boundaries (health_get, journal_lines, restart_service) are
    injectable for testing; defaults hit real systems (HTTP, journalctl,
    systemctl).

    `port` remains the port used to probe /health (unchanged contract).
    New optional params:
      - sync_max_age: override the sync-staleness threshold in seconds
        (default 1h; also readable from MCP_DOCTOR_SYNC_MAX_AGE).
      - env_file_path: path to the service env file used as a fallback
        source for the *declared* port (see _resolve_port).
      - explicit_port: an operator-provided --port value, which takes
        precedence over everything else when resolving the declared port
        shown in the port check's evidence.
    """
    if health_get is None:
        health_get = lambda: _default_health_get(port)  # noqa: E731
    if journal_lines is None:
        journal_lines = _default_journal_lines
    # B1.1 (G5): the restart decision must always see the REAL unit state in
    # production. Tying it to `injected` made the guard inside `_check_health`
    # dead code on the CLI path (main() always passes health_get), leaving the
    # protection to a side effect of `_default_restart_service`. Tests that must
    # not touch systemd declare the state explicitly via `unit_state=`.
    if restart_service is None:
        restart_service = _default_restart_service
    if unit_state is None:
        unit_state = _default_unit_state

    health_result, health_raw = _check_health(
        port=port,
        health_get=health_get,
        restart_service=restart_service,
        sleep=sleep if sleep is not None else time.sleep,
        ready_timeout=_resolve_ready_timeout(ready_timeout),
        unit_state=unit_state,
    )

    # Derive the listening port from the health probe's own result — do NOT
    # call health_get() again (R3/contract: health_get is called at most
    # twice total, by _check_health alone).
    listening_port: Optional[int] = None
    if health_raw is not None:
        listening_port = health_raw.get("port")

    declared_port = _resolve_port(
        explicit_port=explicit_port, env_file_path=env_file_path, fallback_port=port
    )

    port_result = _check_port(declared_port=declared_port, listening_port=listening_port)
    version_result = _check_version(repo_path=repo_path)
    sync_result = _check_sync(
        db_path=db_path, journal_lines=journal_lines, sync_max_age=sync_max_age
    )
    author_result = _check_author(db_path=db_path)
    embeddings_result = _check_embeddings(db_path=db_path)

    return [
        health_result,
        port_result,
        version_result,
        sync_result,
        author_result,
        embeddings_result,
    ]


def _format_line(result: CheckResult) -> str:
    """Render one check result as an <=80-column terminal line."""
    status = "OK" if result.ok else "FAIL"
    repaired_marker = " (repaired)" if result.repaired else ""
    line = f"[{status}] {result.name}{repaired_marker}: {result.evidence}"
    if len(line) > 80:
        line = line[:77] + "..."
    return line


def main(argv: Optional[list] = None) -> int:
    """CLI entry point for `mms-doctor`. Returns 0 if all checks pass, else 1."""
    parser = argparse.ArgumentParser(prog="mms-doctor", description="MCP Memory Service doctor")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=(
            "Declared SSE port. If omitted, resolved in this order: "
            "MCP_SSE_PORT env var, then MCP_SSE_PORT from --env-file, "
            f"then the built-in default ({DEFAULT_PORT})."
        ),
    )
    parser.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE_PATH,
        help=(
            "Path to the service env file consulted as a port fallback "
            f"(default: {DEFAULT_ENV_FILE_PATH})"
        ),
    )
    parser.add_argument(
        "--sync-max-age",
        type=float,
        default=None,
        help=(
            "Max age in seconds before the sync cursor is considered stale "
            "(default: MCP_DOCTOR_SYNC_MAX_AGE env or "
            f"{DEFAULT_SYNC_MAX_AGE_SECONDS:.0f}s)"
        ),
    )
    parser.add_argument("--db-path", required=True, help="Path to the sqlite_vec database")
    parser.add_argument("--repo-path", required=True, help="Path to the repo running the service")
    args = parser.parse_args(argv)

    resolved_port = _resolve_port(explicit_port=args.port, env_file_path=args.env_file)

    def _bound_health_get():
        # Looked up dynamically (not bound at import time) so test
        # monkeypatches of `_default_health_get` take effect. Tests may
        # monkeypatch it as a zero-arg callable, while the real default
        # takes an optional `port` — try with the port first, fall back
        # to zero-arg for monkeypatched stand-ins.
        try:
            return _default_health_get(resolved_port)
        except TypeError:
            return _default_health_get()

    try:
        results = run_checks(
            port=resolved_port,
            db_path=args.db_path,
            repo_path=args.repo_path,
            health_get=_bound_health_get,
            sync_max_age=args.sync_max_age,
            env_file_path=args.env_file,
            explicit_port=args.port,
        )
    except Exception as exc:
        print(f"[FAIL] doctor: unexpected error: {exc}")
        return 1

    for result in results:
        print(_format_line(result))
        if result.name == "health" and result.repaired:
            print("[INFO] health: intervencao realizada (restart + readiness poll)")

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
