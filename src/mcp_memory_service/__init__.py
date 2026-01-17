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

# Set version immediately - no imports, no function calls, just assignment
__version__ = "0.0.0.dev0"

# Try to load actual version without any complex logic
try:
    from ._version import __version__
except Exception:
    try:
        from importlib.metadata import version
        __version__ = version("mcp-memory-service")
    except Exception:
        pass  # Keep default

# Export main classes - delay all other initialization
__all__ = ['Memory', 'MemoryQueryResult', 'MemoryStorage', 'generate_content_hash', 'SqliteVecMemoryStorage']

