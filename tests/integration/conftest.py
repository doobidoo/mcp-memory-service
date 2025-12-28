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
Shared fixtures and configuration for integration tests.

This conftest.py sets up test-wide authentication disabling to ensure
all integration tests can access the FastAPI endpoints without authentication.

CRITICAL: Environment variables MUST be set BEFORE any mcp_memory_service
modules are imported. This is because config.py reads env vars at module-level
import time, not at runtime.

Why session-scoped autouse fixture works:
- pytest loads conftest.py BEFORE importing test modules
- autouse=True ensures this runs BEFORE any test collection
- Session scope means it runs once at the very start
- os.environ changes affect subsequent imports of config.py
"""

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def disable_auth_for_integration_tests():
    """
    Disable authentication globally for all integration tests.

    Sets environment variables BEFORE any app imports happen.
    This must be session-scoped and autouse=True to ensure it runs
    before FastAPI app initialization.

    Technical note for uvx CI compatibility:
    - config.py reads env vars at import time: OAUTH_ENABLED = os.getenv('MCP_OAUTH_ENABLED', True)
    - If config.py is imported before this fixture, auth remains enabled
    - Session-scope + autouse ensures this runs FIRST
    - Works in both local pytest and uvx CI environments
    """
    # Set env vars BEFORE any imports
    os.environ['MCP_API_KEY'] = ''
    os.environ['MCP_OAUTH_ENABLED'] = 'false'
    os.environ['MCP_ALLOW_ANONYMOUS_ACCESS'] = 'true'

    yield

    # Cleanup not needed - test session ends after this
