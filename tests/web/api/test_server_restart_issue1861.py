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
Tests for server process restart command rebuilding (#1861).

When the server is started with `python -m uvicorn ...`, sys.argv[0] points
to uvicorn/__main__.py. Restarting via `[sys.executable] + sys.argv` executes
__main__.py as a script instead of running `-m uvicorn`, placing the package's
directory at the head of sys.path. Any internal module named `logging.py` then
shadows standard library logging and causes circular import failures.

Rebuilding the restart command from `sys.orig_argv` (Python 3.10+) preserves the
original `-m uvicorn ...` invocation and any interpreter flags.
"""

import os
import sys
import subprocess
from pathlib import Path

import pytest

from mcp_memory_service.web.api.server import _get_restart_args


def test_get_restart_args_with_orig_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """sys.orig_argv is used to preserve `-m uvicorn` arguments."""
    simulated_orig = [
        "python",
        "-m",
        "uvicorn",
        "mcp_memory_service.web.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    monkeypatch.setattr(sys, "orig_argv", simulated_orig, raising=False)
    args = _get_restart_args()

    assert args == [sys.executable] + simulated_orig[1:]
    assert args[1:3] == ["-m", "uvicorn"]


def test_get_restart_args_preserves_interpreter_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interpreter options before `-m` are preserved from sys.orig_argv."""
    simulated_orig = ["python", "-W", "ignore", "-O", "-m", "uvicorn", "app:app"]
    monkeypatch.setattr(sys, "orig_argv", simulated_orig, raising=False)
    args = _get_restart_args()

    assert args == [sys.executable, "-W", "ignore", "-O", "-m", "uvicorn", "app:app"]


def test_get_restart_args_fallback_when_orig_argv_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """When sys.orig_argv is None or missing, fall back to sys.argv."""
    monkeypatch.setattr(sys, "orig_argv", None, raising=False)
    simulated_argv = ["server_entry.py", "--port", "8000"]
    monkeypatch.setattr(sys, "argv", simulated_argv)

    args = _get_restart_args()
    assert args == [sys.executable] + simulated_argv


def test_subprocess_restart_preserves_module_execution_with_shadow_logging(tmp_path: Path) -> None:
    """
    Drive the real restart helper in a real subprocess (#1861).

    Create a test package with a `logging.py` submodule. When started via
    `python -m <pkg>`, restarting with the old argument list (`[sys.executable] + sys.argv`)
    executes `__main__.py` as a script, placing `<pkg>` first on `sys.path` and
    causing `logging.py` to shadow stdlib `logging`.

    With `_get_restart_args()`, the process restarts via `-m <pkg>`, sys.path is not
    polluted with the package directory, and stdlib logging loads cleanly.
    """
    pkg_dir = tmp_path / "shadow_pkg"
    pkg_dir.mkdir()

    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    # This logging.py mimics uvicorn/logging.py. If shadow_pkg is on sys.path,
    # `import logging` loads this file, triggering circular import / missing Formatter.
    (pkg_dir / "logging.py").write_text(
        "import logging\n\n"
        "class ColourizedFormatter(logging.Formatter):\n"
        "    pass\n",
        encoding="utf-8",
    )

    (pkg_dir / "__main__.py").write_text(
        "import os\n"
        "import sys\n"
        "from mcp_memory_service.web.api.server import _get_restart_args\n\n"
        "if '--restarted' in sys.argv:\n"
        "    import logging\n"
        "    _ = logging.Formatter\n"
        "    print('RESTARTED_SUCCESS')\n"
        "    sys.exit(0)\n\n"
        "# First invocation: restart through the real helper\n"
        "args = _get_restart_args() + ['--restarted']\n"
        "if sys.platform == 'win32':\n"
        "    import subprocess\n"
        "    proc = subprocess.Popen(args)\n"
        "    proc.wait()\n"
        "    sys.exit(proc.returncode)\n"
        "else:\n"
        "    os.execv(args[0], args)\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    # Ensure current python path / venv can import mcp_memory_service
    repo_root = str(Path(__file__).resolve().parents[3] / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{existing_pythonpath}" if existing_pythonpath else repo_root

    proc = subprocess.run(
        [sys.executable, "-m", "shadow_pkg"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert proc.returncode == 0, f"Process failed: stdout={proc.stdout!r}, stderr={proc.stderr!r}"
    assert "RESTARTED_SUCCESS" in proc.stdout
