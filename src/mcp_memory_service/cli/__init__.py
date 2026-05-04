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
Command Line Interface for MCP Memory Service.

NOTE: This package uses lazy imports to avoid loading torch/transformers
at import time. Use ``from mcp_memory_service.cli.main import cli`` directly,
or call ``mcp_memory_service.cli.add_ingestion_commands()`` when needed.
"""

__all__ = ['add_ingestion_commands']


def add_ingestion_commands(cli_group):
    """Lazily import and register ingestion commands."""
    from .ingestion import add_ingestion_commands as _add
    _add(cli_group)
