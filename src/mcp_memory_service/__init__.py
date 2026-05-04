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
        __version__ = "10.13.0"

# Ensure Python recognizes mcp_memory_service as a package (not a plain module)
# in all install contexts (uvx, editable, wheel). Without this, a module-level
# __getattr__ can cause Python to treat this as a non-package on some paths,
# breaking subpackage imports like from mcp_memory_service.ingestion.csv_loader import CSVLoader
__path__ = __path__  # noqa: F821 – always defined for packages; self-assignment is intentional


def __getattr__(name):
    """Lazy-load heavy classes on first access to avoid loading
    torch/transformers (~22s) at package import time.
    This keeps CLI commands like 'memory launch', 'memory stop', 'memory info'
    fast and responsive.
    """
    _lazy_map = {
        "Memory": ".models",
        "MemoryQueryResult": ".models",
        "MemoryStorage": ".storage",
        "SqliteVecMemoryStorage": ".storage",
        "generate_content_hash": ".utils",
    }
    if name in _lazy_map:
        import importlib
        module = importlib.import_module(_lazy_map[name], __name__)
        value = getattr(module, name)
        # Cache in globals so __getattr__ is not called again
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'Memory',
    'MemoryQueryResult',
    'MemoryStorage',
    'generate_content_hash',
]
