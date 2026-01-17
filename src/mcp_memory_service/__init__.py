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

"""MCP Memory Service - Semantic memory with persistent storage."""

# Load version from _version.py or package metadata
try:
    from ._version import __version__
except ImportError:
    try:
        from importlib.metadata import version as _get_version
        __version__ = _get_version("mcp-memory-service")
    except Exception:
        __version__ = "0.0.0.dev0"

# Import core classes to anchor package structure (required for pytest package recognition)
# Without at least one import, pytest fails with "'mcp_memory_service' is not a package"
from .models.memory import Memory, MemoryQueryResult  # noqa: F401

# Export main classes
__all__ = ['Memory', 'MemoryQueryResult', 'MemoryStorage', 'generate_content_hash', 'SqliteVecMemoryStorage']

