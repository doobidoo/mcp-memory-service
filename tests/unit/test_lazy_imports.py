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
Tests for lazy import behavior in mcp_memory_service package.

Lazy imports prevent heavy dependencies (torch, transformers) from being
loaded at package import time, improving CLI startup performance.

These tests run in subprocess to guarantee a clean import state — mutating
the parent's sys.modules to "reset" lazy state pollutes global state and
breaks any later test that uses FastAPI dependency_overrides on web.api.*
modules (the route holds a reference to the original module; deleting it
from sys.modules forces a re-import on next access, and the override keys
no longer match the live function objects).
"""

import subprocess
import sys
import textwrap


def _run_in_subprocess(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_import_does_not_eagerly_load_heavy_modules():
    """Importing the package should not pull torch/transformers/sentence_transformers."""
    result = _run_in_subprocess("""
        import sys
        import mcp_memory_service
        heavy = ['torch', 'transformers', 'sentence_transformers']
        loaded = [m for m in heavy if m in sys.modules]
        if loaded:
            print(f'FAIL: heavy modules loaded: {loaded}')
            sys.exit(1)
        print('OK')
    """)
    assert result.returncode == 0, (
        f"Heavy imports leaked at package import time.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_lazy_getattr_is_defined_on_package():
    """The package should expose a module-level __getattr__ for lazy attribute access."""
    result = _run_in_subprocess("""
        import sys
        import mcp_memory_service
        assert hasattr(mcp_memory_service, '__getattr__'), 'no __getattr__'
        assert callable(mcp_memory_service.__getattr__), '__getattr__ not callable'
        # Heavy modules must remain unloaded until a lazy attribute is accessed
        assert 'torch' not in sys.modules, 'torch loaded prematurely'
        print('OK')
    """)
    assert result.returncode == 0, (
        f"Lazy __getattr__ scaffolding broken.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
