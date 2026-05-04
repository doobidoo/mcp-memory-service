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

# Eagerly import lightweight subpackages and core types.
# This anchors package discovery for `from mcp_memory_service.X import ...`
# in all install contexts (editable, wheel, uvx). A no-op `__path__ = __path__`
# is not enough — pytest/import paths that use `sys.path.insert(0, 'src')`
# can otherwise resolve this as a plain module and break subpackage imports.
# Heavy imports (torch/transformers via storage backends) stay deferred via
# `__getattr__` below.
from .models import Memory, MemoryQueryResult
from .utils import generate_content_hash


def __getattr__(name):
    """Lazy-load heavy storage classes on first access to avoid loading
    torch/transformers (~22s) at package import time.
    Keeps CLI commands like 'memory launch', 'memory stop', 'memory info' fast.
    """
    _lazy_map = {
        "MemoryStorage": ".storage",
        "SqliteVecMemoryStorage": ".storage",
    }
    if name in _lazy_map:
        import importlib
        module = importlib.import_module(_lazy_map[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'Memory',
    'MemoryQueryResult',
    'MemoryStorage',
    'generate_content_hash',
]
