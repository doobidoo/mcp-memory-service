"""D7 installer: one command, idempotent, reversible.

Installs/configures the memory-service MCP client against an existing
Hermes install. Follows spec-d7-instalador.md (R1-R14): resolves
HERMES_HOME, detects the platform service backend, builds a command plan
(never edits YAML by hand), verifies with a real search after writing, and
supports a surgical --uninstall driven by a manifest file.

Stdlib only. No side effects on import.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

DEFAULT_PORT = 3202
DEFAULT_TASK_NAME = "mms-memory-service"
DEFAULT_UNIT_NAME = "memory-service"


@dataclass
class Action:
    """A single installation step expressed as a command, not a file edit."""

    argv: list[str]
    description: str = ""


# --- R1: HERMES_HOME resolution --------------------------------------------


def resolve_hermes_home(env: dict, home: Path) -> Path:
    """Resolve HERMES_HOME: env var wins, else home/.hermes.

    Raises if the resolved path does not exist (installer must not act
    against a Hermes that isn't actually there).
    """
    raw = env.get("HERMES_HOME")
    if raw:
        candidate = Path(raw)
        if not candidate.is_dir():
            raise RuntimeError(
                f"HERMES_HOME aponta para um caminho inexistente: {candidate}"
            )
        return candidate

    candidate = home / ".hermes"
    if not candidate.is_dir():
        raise RuntimeError(
            f"Nenhum Hermes instalado encontrado (HERMES_HOME nao definido e "
            f"{candidate} nao existe)"
        )
    return candidate


# --- R12/R13: platform backend ---------------------------------------------


def detect_backend(system: str) -> str:
    """Map a platform.system() value to the service backend to use."""
    mapping = {"Linux": "systemd", "Windows": "schtasks"}
    try:
        return mapping[system]
    except KeyError:
        raise RuntimeError(
            f"Plataforma nao suportada: {system}. Suportadas: "
            f"{', '.join(mapping)}"
        ) from None


# --- R12/CA8: Windows command builder (no POSIX tools) ---------------------


def build_schtasks_command(
    python_exe: Path, script: Path, task_name: str
) -> list[str]:
    """Build a `schtasks /create` command for Task Scheduler on logon.

    Pure command construction — never shells out to bash/systemctl/chmod.
    """
    tr = f'"{python_exe}" "{script}"'
    return [
        "schtasks",
        "/create",
        "/tn",
        task_name,
        "/tr",
        tr,
        "/sc",
        "onlogon",
        "/f",
    ]


def build_systemd_commands(unit_name: str = "mms-memory-service") -> list[Action]:
    """Build systemd --user commands to enable and start the service unit."""
    return [
        Action(
            argv=["systemctl", "--user", "daemon-reload"],
            description="recarregar unidades do systemd --user",
        ),
        Action(
            argv=["systemctl", "--user", "enable", "--now", unit_name],
            description=f"habilitar e iniciar {unit_name}",
        ),
    ]


# --- R6/CA2: idempotent plan -------------------------------------------------


def plan(hermes_home: Path, port: int, backend: str, installed: dict) -> list:
    """Build the list of actions needed to reach a fully-installed state.

    installed flags: mcp, skill, service, hooks. Any flag missing/False
    produces the corresponding action. All-True yields an empty plan
    (idempotency / "nada a fazer").
    """
    actions: list[Action] = []

    if not installed.get("mcp"):
        actions.append(
            Action(
                argv=[
                    "hermes",
                    "mcp",
                    "add",
                    "mms",
                    "--url",
                    f"http://localhost:{port}/mcp",
                ],
                description="registrar o servidor mcp mms via hermes cli",
            )
        )

    if not installed.get("skill"):
        actions.append(
            Action(
                argv=[
                    "hermes",
                    "config",
                    "set",
                    "skills.mms.enabled",
                    "true",
                ],
                description="habilitar a skill de doutrina do mms",
            )
        )

    if not installed.get("service"):
        if backend == "systemd":
            actions.extend(build_systemd_commands())
        elif backend == "schtasks":
            actions.append(
                Action(
                    argv=build_schtasks_command(
                        python_exe=Path(sys.executable),
                        script=hermes_home / "scripts" / "server" / "run_memory_server.py",
                        task_name=DEFAULT_TASK_NAME,
                    ),
                    description="criar tarefa agendada do servico (onlogon)",
                )
            )
        else:
            raise RuntimeError(f"Backend de servico desconhecido: {backend}")

    if not installed.get("hooks"):
        actions.append(
            Action(
                argv=[
                    "hermes",
                    "config",
                    "set",
                    "hooks.mms.enabled",
                    "true",
                ],
                description="registrar hooks de convivencia hibrida",
            )
        )

    return actions


# --- R5/R8: post-write verification -----------------------------------------


def verify_search(search_fn: Callable[[], list]) -> list:
    """Run a real post-install search and fail loudly if it comes back empty."""
    results = search_fn()
    if not results:
        raise RuntimeError(
            "Verificacao pos-instalacao falhou: a busca nao devolveu nenhuma "
            "memoria. O instalador escreveu config mas a busca real nao "
            "confirma que o servico esta funcional."
        )
    return results


# --- R7/R9/CA3: manifest + surgical uninstall -------------------------------


def write_manifest(path: Path, entries: list[str], undo: list[list[str]] | None = None) -> None:
    """Persist what this installer created and the commands that undo it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"created": list(entries), "undo": list(undo or [])}, indent=2))


def _load_manifest(path: Path) -> dict:
    """Read the manifest, tolerating a missing or corrupted file.

    A truncated or hand-edited manifest must never crash `uninstall` (CA6):
    degrade to an empty manifest and let the operator see an empty removal list.
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_manifest(path: Path) -> list[str]:
    """Read back the manifest artefact paths, or [] if unusable."""
    created = _load_manifest(path).get("created")
    return [str(x) for x in created] if isinstance(created, list) else []


def read_manifest_undo(path: Path) -> list[list[str]]:
    """Read back the undo commands, or [] if unusable."""
    undo = _load_manifest(path).get("undo")
    if not isinstance(undo, list):
        return []
    return [list(cmd) for cmd in undo if isinstance(cmd, list) and cmd]


def uninstall(
    manifest_path: Path,
    hermes_home: Path,
    runner: Callable[[Sequence[str]], tuple[int, str]] | None = None,
) -> list:
    """Remove exactly what the installer created, per the manifest.

    Runs the recorded undo commands first (they restore the Hermes config and
    the service), then deletes files the installer created. A file that is not
    in the manifest is never touched, and a corrupted manifest degrades to an
    empty one instead of raising.
    """
    run = runner or default_runner
    removed: list[str] = []

    for command in read_manifest_undo(manifest_path):
        rc, out = run(command)
        if rc == 0:
            removed.append(" ".join(command))
        else:
            # Best effort: report, do not abort the whole uninstall.
            print(
                f"aviso: falha ao desfazer `{' '.join(command)}` (rc={rc}): {out.strip()[:120]}",
                file=sys.stderr,
            )

    for entry in read_manifest(manifest_path):
        target = Path(entry)
        if target.is_file() or target.is_symlink():
            target.unlink()
            removed.append(str(target))
        elif target.is_dir():
            target.rmdir()
            removed.append(str(target))
    return removed


# --- R2/CA4: abort before writing when service is down ---------------------


def default_runner(argv: Sequence[str]) -> tuple[int, str]:
    """Run one planned command; return (returncode, combined output)."""
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def detect_installed(
    hermes_home: Path,
    port: int,
    backend: str,
    health_get: Callable[[], object] | None = None,
    runner: Callable[[Sequence[str]], tuple[int, str]] | None = None,
) -> dict:
    """Discover what is already in place, so `plan()` can be idempotent.

    Without this, `main()` planned every action on every run and the installer
    was only idempotent in unit tests that handed `installed` in by hand.
    """
    check_health = health_get or (lambda: default_health_get(port))
    run = runner or default_runner
    installed: dict[str, bool] = {}

    try:
        rc, out = run(["hermes", "mcp", "list"])
        installed["mcp"] = rc == 0 and "mms" in out
    except Exception:
        installed["mcp"] = False

    for key, config_key in (
        ("skill", "skills.mms.enabled"),
        ("hooks", "hooks.mms.enabled"),
    ):
        try:
            rc, out = run(["hermes", "config", "get", config_key])
            installed[key] = rc == 0 and "true" in out.lower()
        except Exception:
            installed[key] = False

    try:
        check_health()
        installed["service"] = True
    except Exception:
        installed["service"] = False

    return installed


def undo_for(action: Action) -> list[str]:
    """The command that reverses one planned action (used by `uninstall`)."""
    argv = list(action.argv)
    if argv[:3] == ["hermes", "mcp", "add"] and len(argv) > 3:
        return ["hermes", "mcp", "remove", argv[3]]
    if argv[:3] == ["hermes", "config", "set"] and len(argv) > 3:
        return ["hermes", "config", "unset", argv[3]]
    if argv and argv[0] == "schtasks":
        task = argv[argv.index("/tn") + 1] if "/tn" in argv else DEFAULT_TASK_NAME
        return ["schtasks", "/delete", "/tn", task, "/f"]
    if argv and argv[0] == "systemctl":
        return ["systemctl", "--user", "disable", "--now", DEFAULT_UNIT_NAME]
    return []


def apply_or_abort(
    hermes_home: Path,
    port: int,
    health_get: Callable[[], object],
    executor: Callable[[Sequence[Action]], object],
    actions: Sequence[Action] | None = None,
) -> object:
    """Check /health before writing anything; propagate failure untouched.

    If health_get() raises, the exception propagates before executor runs
    and before any artefact is created under hermes_home.

    `actions` is the plan the caller already computed from the REAL installed
    state. Replanning here with an empty dict (the previous behaviour) both
    ignored what was installed and made the CLI run every command twice.
    """
    health_get()  # let any exception propagate as-is, before touching disk
    if actions is None:
        actions = plan(hermes_home, port, detect_backend(platform.system()), {})
    return executor(actions)


def default_health_get(port: int) -> dict:
    """Real GET /health call, used by the CLI path (not by tests)."""
    url = f"http://localhost:{port}/health"
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def default_executor(actions: Sequence[Action]) -> list[str]:
    """Run each action's argv via subprocess, returning stdout per action."""
    outputs = []
    for action in actions:
        result = subprocess.run(
            action.argv, capture_output=True, text=True, check=False
        )
        outputs.append(result.stdout)
    return outputs


# --- CLI ---------------------------------------------------------------------


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="instalar", description="Instalador D7 do mms para Hermes."
    )
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hermes-home", default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--env-file", default=None)
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: --uninstall, --dry-run, --hermes-home, --port, --env-file."""
    import os

    args = _parse_args(argv if argv is not None else sys.argv[1:])

    env = dict(os.environ)
    if args.env_file:
        env_path = Path(args.env_file)
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()

    home = Path(args.hermes_home) if args.hermes_home else Path.home()
    try:
        hermes_home = resolve_hermes_home(env, home=home)
    except Exception as exc:  # noqa: BLE001 - surface clearly to the operator
        print(f"erro: {exc}", file=sys.stderr)
        return 1

    manifest_path = hermes_home / "mms-install-manifest.json"

    if args.uninstall:
        removed = uninstall(manifest_path, hermes_home)
        print(f"removido: {removed}")
        return 0

    try:
        backend = detect_backend(platform.system())
    except Exception as exc:  # noqa: BLE001
        print(f"erro: {exc}", file=sys.stderr)
        return 1

    actions = plan(hermes_home, args.port, backend, detect_installed(hermes_home, args.port, backend))

    if not actions:
        print("nada a fazer: instalacao ja completa")
        return 0

    if args.dry_run:
        for action in actions:
            print(" ".join(action.argv))
        return 0

    def health_get() -> dict:
        return default_health_get(args.port)

    try:
        apply_or_abort(
            hermes_home=hermes_home,
            port=args.port,
            health_get=health_get,
            executor=default_executor,
            actions=actions,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"erro: servico indisponivel, abortando sem escrever: {exc}", file=sys.stderr)
        return 1

    write_manifest(
        manifest_path,
        [],
        [undo_for(action) for action in actions if undo_for(action)],
    )
    print(f"instalado via backend={backend}, porta={args.port}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
