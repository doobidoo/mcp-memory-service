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
Regression tests for import issues.

Ensures all required imports are present to prevent issues like the
'import time' bug fixed in v8.57.0 (Issue #295, Phase 1).
"""

import pytest


def test_server_impl_imports():
    """
    Regression test for missing 'import time' bug (v8.57.0).

    Ensures server_impl.py has all required imports, particularly the
    'time' module which was missing and caused NameError in 27+ tests.

    Related: PR #294, v8.57.0 Phase 1 fixes
    """
    import mcp_memory_service.server_impl as si

    # Verify time module is imported and accessible
    assert hasattr(si, 'time'), "server_impl.py must import 'time' module"

    # Verify other critical standard library imports
    assert hasattr(si, 'asyncio'), "server_impl.py must import 'asyncio'"
    assert hasattr(si, 'logging'), "server_impl.py must import 'logging'"
    assert hasattr(si, 'json'), "server_impl.py must import 'json'"


def test_memory_service_imports():
    """Ensure memory_service.py has all required imports."""
    import mcp_memory_service.services.memory_service as ms

    # Verify critical imports
    assert hasattr(ms, 'logging'), "memory_service.py must import 'logging'"

    # Verify model imports
    from mcp_memory_service.models.memory import Memory, MemoryQueryResult
    assert Memory is not None
    assert MemoryQueryResult is not None
