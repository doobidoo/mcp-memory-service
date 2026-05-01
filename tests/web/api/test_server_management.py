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
Tests for /api/server management endpoints (#729).

Covers the silent-failure modes the original issue described:
- /server/update used to return HTTP 200 with {"status": "error"} on failure,
  so the dashboard never noticed and showed a success toast.
- /server/update used to run `git pull` directly, with no dirty-tree check,
  so a conflicting working tree silently aborted the pull.
- /server/restart used to return without any way for the client to verify
  the process actually restarted.

These tests pin the new contract: failures raise HTTPException with the
real reason in `detail`, and both endpoints return pre_restart_pid +
pre_restart_version so the client can verify rollover.
"""

import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Import the route module + auth result type at module load so monkeypatch /
# unittest.mock.patch can target attributes via the live module objects.
# Dotted-string targets (e.g. "mcp_memory_service.web.api.server._foo") fail
# under non-editable installs (uvx CI) because pytest's monkeypatch resolves
# each path segment via getattr on the parent package, and the `web` submodule
# isn't always set as an attribute on `mcp_memory_service` at that point.
# Holding a direct module reference dodges that resolution path entirely.
from mcp_memory_service.web.api import server as server_module
from mcp_memory_service.web.oauth.middleware import AuthenticationResult


@pytest.fixture
def test_app(monkeypatch):
    """FastAPI test client with admin auth bypassed and restart task neutered."""
    # Note: Auth is bypassed entirely via FastAPI dependency_overrides below,
    # so we deliberately do NOT touch MCP_ALLOW_ANONYMOUS_ACCESS — the OAuth
    # middleware module reads that env var at import time and other test files
    # in the suite assert on the production default (auth required).
    from mcp_memory_service.web.app import app
    # CRITICAL: Override using the SAME function objects the route holds in its
    # Depends(...) — the route imported these at module-load time. If we
    # imported them again from middleware here, a reload elsewhere in the test
    # suite would have rebound middleware's symbols and our overrides would
    # miss, leaving the real auth path active (HTTP 403 in tests).

    async def mock_admin_user():
        return AuthenticationResult(
            authenticated=True,
            client_id="test_admin",
            scope="read write admin",
            auth_method="test",
        )

    app.dependency_overrides[server_module.require_read_access] = mock_admin_user
    app.dependency_overrides[server_module.require_admin_access] = mock_admin_user

    # Replace the background restart task with a no-op so tests don't try to
    # exec a new Python process / kill the test runner.
    async def noop_restart():
        return None

    monkeypatch.setattr(server_module, '_restart_server_delayed', noop_restart)

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


@pytest.mark.integration
def test_update_refuses_dirty_working_tree(test_app):
    """Dirty working tree → HTTP 409 with the offending paths in detail."""
    with patch.object(
        server_module,
        '_check_working_tree_clean',
        return_value=(False, ['M  src/foo.py', '?? config/local.yaml']),
    ):
        response = test_app.post('/api/server/update', json={'confirm': True})

    assert response.status_code == 409
    detail = response.json()['detail']
    assert 'uncommitted changes' in detail
    assert 'force=true' in detail
    assert 'src/foo.py' in detail


@pytest.mark.integration
def test_update_with_force_skips_dirty_check(test_app):
    """force=true bypasses the dirty-tree refusal and proceeds to git pull."""
    with patch.object(
        server_module,
        '_check_working_tree_clean',
        return_value=(False, ['M  src/foo.py']),
    ) as mock_check, patch.object(
        server_module,
        '_run_git_command',
        return_value=('Already up to date.', True),
    ), patch.object(
        server_module,
        '_run_pip_command',
        return_value=('Successfully installed', True),
    ):
        response = test_app.post(
            '/api/server/update',
            json={'confirm': True, 'force': True},
        )

    # Dirty check should not even be called when force=true
    assert mock_check.call_count == 0
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'success'
    assert body['restart_scheduled'] is True


@pytest.mark.integration
def test_update_returns_500_when_git_pull_fails(test_app):
    """git pull failure → HTTP 500 with stderr in detail (not silent 200)."""
    with patch.object(
        server_module,
        '_check_working_tree_clean',
        return_value=(True, []),
    ), patch.object(
        server_module,
        '_run_git_command',
        return_value=('error: Your local changes would be overwritten by merge', False),
    ):
        response = test_app.post('/api/server/update', json={'confirm': True})

    assert response.status_code == 500
    detail = response.json()['detail']
    assert 'Git pull failed' in detail
    assert 'local changes would be overwritten' in detail


@pytest.mark.integration
def test_update_returns_500_when_pip_install_fails(test_app):
    """pip install failure → HTTP 500 (not silent 200) with output in detail."""
    with patch.object(
        server_module,
        '_check_working_tree_clean',
        return_value=(True, []),
    ), patch.object(
        server_module,
        '_run_git_command',
        return_value=('Updating abc..def', True),
    ), patch.object(
        server_module,
        '_run_pip_command',
        return_value=('ERROR: Could not find a version that satisfies the requirement foo', False),
    ):
        response = test_app.post('/api/server/update', json={'confirm': True})

    assert response.status_code == 500
    detail = response.json()['detail']
    assert 'Pip install failed' in detail
    assert 'Could not find a version' in detail


@pytest.mark.integration
def test_update_success_returns_pre_restart_identity(test_app):
    """Successful update returns pid + version captured before restart."""
    with patch.object(
        server_module,
        '_check_working_tree_clean',
        return_value=(True, []),
    ), patch.object(
        server_module,
        '_run_git_command',
        return_value=('Updating abc..def', True),
    ), patch.object(
        server_module,
        '_run_pip_command',
        return_value=('Successfully installed', True),
    ):
        response = test_app.post('/api/server/update', json={'confirm': True})

    assert response.status_code == 200
    body = response.json()
    assert body['restart_scheduled'] is True
    assert body['pre_restart_pid'] == os.getpid()
    assert isinstance(body['pre_restart_version'], str)
    assert body['pre_restart_version']


@pytest.mark.integration
def test_restart_returns_pre_restart_identity(test_app):
    """Restart endpoint exposes pre-restart pid + version for verification."""
    response = test_app.post('/api/server/restart', json={'confirm': True})

    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'accepted'
    assert body['pre_restart_pid'] == os.getpid()
    assert isinstance(body['pre_restart_version'], str)
    assert body['pre_restart_version']


@pytest.mark.integration
def test_restart_requires_confirm(test_app):
    """Restart without confirm=true → HTTP 400."""
    response = test_app.post('/api/server/restart', json={'confirm': False})
    assert response.status_code == 400


@pytest.mark.integration
def test_update_requires_confirm(test_app):
    """Update without confirm=true → HTTP 400."""
    response = test_app.post('/api/server/update', json={'confirm': False})
    assert response.status_code == 400
