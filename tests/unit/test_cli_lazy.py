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
Tests for CLI lazy command loading.

Tests verify that:
1. CLI help shows lazy-loaded commands without heavy imports
2. Ingestion commands are available via lazy resolution
3. Lifecycle commands are properly registered

Subprocess isolation is used for the heavy-import assertions to avoid
mutating the parent process's sys.modules — that pollution previously
broke any later test relying on FastAPI dependency_overrides bound to
specific module objects (the override key references the original
function; deleting modules forces a re-import that yields fresh function
objects, and overrides silently miss).
"""

import subprocess
import sys
import textwrap

from click.testing import CliRunner


def _run_in_subprocess(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_help_does_not_trigger_heavy_imports():
    """Running `memory --help` must not import torch/transformers/sentence_transformers."""
    result = _run_in_subprocess("""
        import sys
        from click.testing import CliRunner
        from mcp_memory_service.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        if result.exit_code != 0:
            print(f'FAIL: --help exit_code={result.exit_code}: {result.output}')
            sys.exit(1)
        if 'memory' not in result.output.lower() and 'MCP Memory Service' not in result.output:
            print(f'FAIL: unexpected help output: {result.output[:500]}')
            sys.exit(1)
        heavy = ['torch', 'transformers', 'sentence_transformers']
        loaded = [m for m in heavy if m in sys.modules]
        if loaded:
            print(f'FAIL: heavy modules loaded during --help: {loaded}')
            sys.exit(1)
        print('OK')
    """)
    assert result.returncode == 0, (
        f"CLI --help triggered heavy imports.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_cli_has_lifecycle_commands():
    """Lifecycle commands (launch/stop/restart/info/health/logs) must be registered."""
    from mcp_memory_service.cli.main import cli

    ctx = cli.make_context('memory', [])
    commands = cli.list_commands(ctx)

    for cmd in ['launch', 'stop', 'restart', 'info', 'health', 'logs']:
        assert cmd in commands, f"Lifecycle command {cmd!r} not in {commands}"


def test_cli_has_ingestion_commands():
    """Ingestion commands must appear in the lazy command map and group listing."""
    from mcp_memory_service.cli.main import cli, LAZY_COMMANDS

    for cmd in ['ingest-document', 'ingest-directory', 'list-formats']:
        assert cmd in LAZY_COMMANDS, f"{cmd!r} not in LAZY_COMMANDS"

    ctx = cli.make_context('memory', [])
    commands = cli.list_commands(ctx)
    for cmd in ['ingest-document', 'ingest-directory', 'list-formats']:
        assert cmd in commands, f"Ingestion command {cmd!r} not found in {commands}"


def test_lazy_get_command_for_ingest():
    """get_command must lazily resolve ingestion commands to a Click command object."""
    from mcp_memory_service.cli.main import cli

    ctx = cli.make_context('memory', ['ingest-document', '--help'])
    cmd = cli.get_command(ctx, 'ingest-document')

    assert cmd is not None, "ingest-document command not found"
    assert hasattr(cmd, 'callback') or hasattr(cmd, 'name')


def test_cli_ingestion_command_help():
    """Help for a lazy-loaded ingestion command must work end-to-end."""
    from mcp_memory_service.cli.main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['ingest-document', '--help'])

    assert result.exit_code == 0, f"ingest-document --help failed: {result.output}"
    assert 'ingest' in result.output.lower() or 'document' in result.output.lower()
