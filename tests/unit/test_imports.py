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
    # Read server_impl.py source to verify imports are present
    import os

    server_impl_path = os.path.join(
        os.path.dirname(__file__),
        '../../src/mcp_memory_service/server_impl.py'
    )
    server_impl_path = os.path.abspath(server_impl_path)

    with open(server_impl_path, 'r') as f:
        source = f.read()

    # Verify critical imports are present in source
    assert 'import time' in source, "server_impl.py must import 'time' module"
    assert 'import asyncio' in source, "server_impl.py must import 'asyncio'"
    assert 'import logging' in source, "server_impl.py must import 'logging'"
    assert 'import json' in source, "server_impl.py must import 'json'"


def test_memory_service_imports():
    """Ensure memory_service.py has all required imports."""
    import mcp_memory_service.services.memory_service as ms

    # Verify critical imports
    assert hasattr(ms, 'logging'), "memory_service.py must import 'logging'"

    # Verify model imports
    from mcp_memory_service.models.memory import Memory, MemoryQueryResult
    assert Memory is not None
    assert MemoryQueryResult is not None
