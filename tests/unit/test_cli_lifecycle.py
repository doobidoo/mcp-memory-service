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
Tests for CLI lifecycle management commands.

Tests verify the new launch/stop/restart/info/health/logs commands
are properly registered and functional.
"""

import sys
import os
from unittest.mock import patch, MagicMock
from click.testing import CliRunner


def test_lifecycle_commands_registered():
    """Test that all lifecycle commands are registered with the CLI group."""
    from mcp_memory_service.cli.main import cli, LAZY_COMMANDS
    
    # Get available commands
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


def test_launch_command_structure():
    """Test that the launch command has expected options."""
    from mcp_memory_service.cli.main import launch
    
    # Launch should be a Click command
    assert hasattr(launch, 'callback')
    assert hasattr(launch, 'params')
    
    # Check for expected parameters
    param_names = [p.name for p in launch.params]
    expected = ['http_host', 'http_port', 'detach', 'storage_backend', 'debug']
    for param in expected:
        assert param in param_names, f"launch command missing param {param!r}"
    
    # Clean up
    for mod_name in list(sys.modules.keys()):
        if 'mcp_memory_service' in mod_name:
            del sys.modules[mod_name]


def test_stop_command_structure():
    """Test that the stop command has expected options."""
    from mcp_memory_service.cli.main import stop
    
    assert hasattr(stop, 'callback')
    param_names = [p.name for p in stop.params]
    assert 'http_host' in param_names
    assert 'http_port' in param_names
    
    # Clean up
    for mod_name in list(sys.modules.keys()):
        if 'mcp_memory_service' in mod_name:
            del sys.modules[mod_name]


def test_info_command_structure():
    """Test that the info command has expected options."""
    from mcp_memory_service.cli.main import info
    
    assert hasattr(info, 'callback')
    param_names = [p.name for p in info.params]
    assert 'http_host' in param_names
    assert 'http_port' in param_names
    
    # Clean up
    for mod_name in list(sys.modules.keys()):
        if 'mcp_memory_service' in mod_name:
            del sys.modules[mod_name]


def test_health_command_structure():
    """Test that the health command has expected options."""
    from mcp_memory_service.cli.main import health_cmd
    
    assert hasattr(health_cmd, 'callback')
    param_names = [p.name for p in health_cmd.params]
    assert 'http_host' in param_names
    assert 'http_port' in param_names
    
    # Clean up
    for mod_name in list(sys.modules.keys()):
        if 'mcp_memory_service' in mod_name:
            del sys.modules[mod_name]


def test_logs_command_structure():
    """Test that the logs command has expected options."""
    from mcp_memory_service.cli.main import logs
    
    assert hasattr(logs, 'callback')
    param_names = [p.name for p in logs.params]
    assert 'lines' in param_names
    
    # Clean up
    for mod_name in list(sys.modules.keys()):
        if 'mcp_memory_service' in mod_name:
            del sys.modules[mod_name]


def test_lifecycle_commands_use_lifecycle_module():
    """Test that lifecycle commands are properly defined in the CLI module."""
    from mcp_memory_service.cli.main import launch, stop, restart, info, health_cmd, logs
    from click import Command
    
    # All lifecycle commands should be Click Command objects
    assert isinstance(launch, Command), f"launch is {type(launch)}, not Command"
    assert isinstance(stop, Command), f"stop is {type(stop)}, not Command"
    assert isinstance(info, Command), f"info is {type(info)}, not Command"
    assert isinstance(health_cmd, Command), f"health_cmd is {type(health_cmd)}, not Command"
    assert isinstance(logs, Command), f"logs is {type(logs)}, not Command"
    
    # Verify each command has a callback
    for cmd in [launch, stop, restart, info, health_cmd, logs]:
        assert hasattr(cmd, 'callback'), f"Command {cmd.name} has no callback attribute"
        assert callable(cmd.callback), f"Command {cmd.name} callback is not callable"
    
    # Clean up
    for mod_name in list(sys.modules.keys()):
        if 'mcp_memory_service' in mod_name:
            del sys.modules[mod_name]
