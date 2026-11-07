#!/usr/bin/env python3
"""Install Codex memory hooks without replacing unrelated hook definitions."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


EVENTS = {"SessionStart": 10, "Stop": 10, "PreCompact": 10, "SessionEnd": 3}
SOURCE = Path(__file__).resolve().parent


def atomic_write(destination, data):
    descriptor, temporary = tempfile.mkstemp(prefix=".memory-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(data)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def hook_group(script, event, timeout):
    return {"hooks": [{
        "type": "command",
        "command": "node " + shlex.quote(str(script)),
        "commandWindows": subprocess.list2cmdline(["node", str(script)]),
        "timeout": timeout,
        "statusMessage": "Memory Service",
        **({"additionalContextLimit": 2000} if event == "SessionStart" else {}),
    }]}


def read_document(hooks_path):
    """Read and validate the hook document without modifying the destination."""
    original = hooks_path.read_text(encoding="utf-8") if hooks_path.exists() else None
    document = json.loads(original) if original is not None else {}
    if not isinstance(document, dict) or not isinstance(document.get("hooks", {}), dict):
        raise ValueError("hooks.json must contain an object with a hooks object")
    return document, original


def preserve_other_handlers(group, command, event):
    """Remove our command while retaining other handlers and matcher settings."""
    if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
        raise ValueError(f"Invalid matcher group in hooks.{event}")
    remaining = [handler for handler in group["hooks"] if not (
        isinstance(handler, dict) and handler.get("type") == "command"
        and handler.get("command") == command
    )]
    if remaining or not group["hooks"]:
        return {**group, "hooks": remaining}
    return None


def merge_hooks(document, script):
    """Replace only the integration's own handlers, once per supported event."""
    hooks = document.setdefault("hooks", {})
    for event, timeout in EVENTS.items():
        groups = hooks.setdefault(event, [])
        if not isinstance(groups, list):
            raise ValueError(f"hooks.{event} must be an array")
        group = hook_group(script, event, timeout)
        preserved = [preserve_other_handlers(existing, group["hooks"][0]["command"], event)
                     for existing in groups]
        hooks[event] = [*[existing for existing in preserved if existing is not None], group]


def install_files(script):
    """Copy runtime files and create configuration only on the first install."""
    script.parent.mkdir(parents=True, exist_ok=True)
    utilities = script.parent.parent / "claude-hooks" / "utilities"
    utilities.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / "memory-hook.js", script)
    shutil.copy2(SOURCE.parent / "claude-hooks" / "utilities" / "auto-capture-patterns.js", utilities)
    config_path = script.parent.parent / "config.json"
    if not config_path.exists():
        atomic_write(config_path, (SOURCE / "config.example.json").read_text(encoding="utf-8"))
    return config_path


def install(destination, dry_run=False):
    """Validate, merge and install a user or project hook configuration."""
    destination = destination.expanduser().resolve()
    hooks_path = destination / "hooks.json"
    document, original = read_document(hooks_path)
    script = destination / "memory-service" / "codex" / "memory-hook.js"
    merge_hooks(document, script)

    rendered = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    changed = original != rendered
    if dry_run:
        print(f"Would install scripts in {script.parent.parent}")
        print(f"Would {'update' if changed else 'keep'} {hooks_path}")
        print("Events: " + ", ".join(EVENTS))
        return

    # Validate the entire configuration before creating any files.
    config_path = install_files(script)
    if changed:
        if original is not None:
            backup = hooks_path.with_name(f"hooks.json.memory-service-{time.time_ns()}.bak")
            atomic_write(backup, original)
            print(f"Backup: {backup}")
        atomic_write(hooks_path, rendered)
    print(f"Installed: {hooks_path}")
    print(f"Configuration: {config_path}")
    print("Restart Codex, then review and trust these hooks with /hooks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--codex-home", type=Path, help="Codex configuration directory (default: CODEX_HOME or ~/.codex)")
    scope.add_argument("--project", type=Path, help="Install in PROJECT/.codex instead of the user configuration")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show destinations without writing files")
    args = parser.parse_args()
    destination = args.project / ".codex" if args.project else args.codex_home or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    try:
        install(destination, args.dry_run)
    except (ValueError, OSError) as error:
        # Do not print configuration content, which may include credentials.
        parser.exit(1, f"Installation failed ({type(error).__name__}); check the destination and hooks.json structure.\n")


if __name__ == "__main__":
    main()
