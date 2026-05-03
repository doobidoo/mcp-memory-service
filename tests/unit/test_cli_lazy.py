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
"""

import sys
from click.testing import CliRunner


def test_cli_help_does_not_trigger_heavy_imports():
    """Test that running CLI help doesn't trigger torch/transformers imports."""
    # Store baseline
    initial_modules = set(sys.modules.keys())
    
    # Run CLI help
    from mcp_memory_service.cli.main import cli
    
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    
    # Should succeed
    assert result.exit_code == 0, f"CLI help failed: {result.output}"
    
    # Check help output contains expected content
    assert 'memory' in result.output.lower() or 'MCP Memory Service' in result.output
    
    # Heavy imports should NOT be loaded
    heavy_imports = ['torch', 'transformers', 'sentence_transformers']
    for heavy in heavy_imports:
        assert heavy not in sys.modules, f"Heavy import {heavy!r} loaded during CLI help"
    
    # Clean up
    for mod_name in list(sys.modules.keys()):
        if 'mcp_memory_service' in mod_name:
            del sys.modules[mod_name]


def test_cli_has_lifecycle_commands():
    """Test that lifecycle commands are available in the CLI group."""
    from mcp_memory_service.cli.main import cli
    
    # Get available commands - use an empty context so --help doesn't trigger exit
    ctx = cli.make_context('memory', [])
    commands = cli.list_commands(ctx)
    
    # Lifecycle commands
    lifecycle = ['launch', 'stop', 'restart', 'info', 'health', 'logs']
    for cmd in lifecycle:
        assert cmd in commands, f"Lifecycle command {cmd!r} not in {commands}"
    
    # Clean up
    for mod_name in list(sys.modules.keys()):
        if 'mcp_memory_service' in mod_name:
            del sys.modules[mod_name]


def test_cli_has_ingestion_commands():
    """Test that ingestion commands are available via lazy resolution."""
    from mcp_memory_service.cli.main import cli, LAZY_COMMANDS
    
    # Lazy commands should be defined
    assert 'ingest-document' in LAZY_COMMANDS
    assert 'ingest-directory' in LAZY_COMMANDS
    assert 'list-formats' in LAZY_COMMANDS
    
    # Get actual command names from the group
    ctx = cli.make_context('memory', [])
    commands = cli.list_commands(ctx)
    
    # Ingestion commands should appear in help
    ingestion_commands = ['ingest-document', 'ingest-directory', 'list-formats']
    for cmd in ingestion_commands:
        assert cmd in commands, f"Ingestion command {cmd!r} not found"


def test_lazy_get_command_for_ingest():
    """Test that get_command lazily resolves ingestion commands without importing heavy deps."""
    # Remove CLI module from cache
    for mod_name in list(sys.modules.keys()):
        if 'mcp_memory_service' in mod_name:
            del sys.modules[mod_name]
    
    from mcp_memory_service.cli.main import cli, LAZY_COMMANDS
    
    # Heavy modules should not be loaded
    assert 'torch' not in sys.modules
    
    # Try to get an ingestion command - this should trigger lazy import
    ctx = cli.make_context('memory', ['ingest-document', '--help'])
    
    # The command should be resolvable
    cmd = cli.get_command(ctx, 'ingest-document')
    
    # Should return a Click command object
    assert cmd is not None, "ingest-document command not found"
    assert hasattr(cmd, 'callback') or hasattr(cmd, 'name')
    
    # Clean up
    for mod_name in list(sys.modules.keys()):
        if 'mcp_memory_service' in mod_name:
            del sys.modules[mod_name]


def test_cli_ingestion_command_help():
    """Test that getting help for lazy-loaded commands works."""
    from mcp_memory_service.cli.main import cli
    
    runner = CliRunner()
    
    # Test ingest-document help (lazy-loaded)
    result = runner.invoke(cli, ['ingest-document', '--help'])
    
    # Should succeed without errors
    assert result.exit_code == 0, f"ingest-document --help failed: {result.output}"
    assert 'ingest' in result.output.lower() or 'document' in result.output.lower()
    
    # Clean up
    for mod_name in list(sys.modules.keys()):
        if 'mcp_memory_service' in mod_name:
            del sys.modules[mod_name]
