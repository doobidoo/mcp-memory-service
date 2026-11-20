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

import asyncio
import importlib.util
import os
import shutil
import sys
from types import SimpleNamespace

import pytest
from unittest.mock import patch
from fastapi import BackgroundTasks, HTTPException
from fastapi.testclient import TestClient

# Import the route module + auth result type at module load so monkeypatch /
# unittest.mock.patch can target attributes via the live module objects.
# Dotted-string targets (e.g. "mcp_memory_service.web.api.server._foo") fail
# under non-editable installs (uvx CI) because pytest's monkeypatch resolves
# each path segment via getattr on the parent package, and the `web` submodule
# isn't always set as an attribute on `mcp_memory_service` at that point.
# Holding a direct module reference dodges that resolution path entirely.
from mcp_memory_service.backup import scheduler as scheduler_module
from mcp_memory_service.web.api import server as server_module
from mcp_memory_service.web.oauth.middleware import AuthenticationResult


class FakeBackupService:
    """Stands in for BackupService; records each backup in `events`."""

    def __init__(self, events=None, db_path='/data/sqlite_vec.db', result=None, error=None):
        self.events = events if events is not None else []
        self.db_path = db_path
        self.result = result or {'success': True, 'filename': 'memory_backup_20261120_120000.db'}
        self.error = error

    async def create_backup(self, description=None):
        self.events.append(('backup', description))
        if self.error:
            raise self.error
        return self.result


def _use_backup_service(monkeypatch, service):
    # server.py imports get_backup_service from the scheduler at call time,
    # the way web/api/backup.py does, so patch it at the source.
    monkeypatch.setattr(scheduler_module, 'get_backup_service', lambda: service)
    return service


@pytest.fixture(autouse=True)
def fake_backup(monkeypatch):
    """Every update in this module backs up into a fake, never the real database."""
    return _use_backup_service(monkeypatch, FakeBackupService())


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

    # The update endpoint resolves an installer before git pull; pin it so the
    # tests do not depend on whether the dev environment has pip or uv.
    monkeypatch.setattr(
        server_module, '_pip_command',
        lambda args: ([sys.executable, '-m', 'pip', *args], 'Pip'),
    )

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


def _no_pip(monkeypatch):
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None)


def test_pip_command_prefers_pip_when_importable(monkeypatch):
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: object())
    monkeypatch.setattr(shutil, 'which', lambda name: 'C:/bin/uv.exe')
    assert server_module._pip_command(['install', '-e', '.']) == (
        [sys.executable, '-m', 'pip', 'install', '-e', '.'], 'Pip',
    )


def test_pip_command_falls_back_to_uv_pinned_to_this_interpreter(monkeypatch):
    _no_pip(monkeypatch)
    monkeypatch.delenv('UV', raising=False)
    monkeypatch.setattr(shutil, 'which', lambda name: 'C:/bin/uv.exe')

    with patch.object(server_module, '_run_command', return_value=('ok', True)) as mock_run:
        server_module._run_pip_command(['install', '-e', '.'])
    assert mock_run.call_args.args[0] == [
        'C:/bin/uv.exe', 'pip', 'install', '--python', sys.executable, '-e', '.',
    ]
    assert mock_run.call_args.args[2] == 'uv'


def test_pip_command_uses_the_uv_that_launched_the_server(monkeypatch, tmp_path):
    # `uv run` exports UV with its own path; a scheduled task's PATH may lack uv.
    uv_exe = tmp_path / 'uv.exe'
    uv_exe.write_bytes(b'')
    _no_pip(monkeypatch)
    monkeypatch.setenv('UV', str(uv_exe))
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    command, tool = server_module._pip_command(['install', '-e', '.'])
    assert command[:2] == [str(uv_exe), 'pip']
    assert tool == 'uv'


def test_pip_command_ignores_a_stale_uv_variable(monkeypatch, tmp_path):
    _no_pip(monkeypatch)
    monkeypatch.setenv('UV', str(tmp_path / 'missing' / 'uv.exe'))
    monkeypatch.setattr(shutil, 'which', lambda name: 'C:/bin/uv.exe')
    command, _ = server_module._pip_command(['install'])
    assert command[0] == 'C:/bin/uv.exe'


def test_pip_command_none_without_pip_or_uv(monkeypatch):
    _no_pip(monkeypatch)
    monkeypatch.delenv('UV', raising=False)
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    assert server_module._pip_command(['install']) is None
    output, success = server_module._run_pip_command(['install', '-e', '.'])
    assert not success
    assert 'Neither pip nor uv' in output


@pytest.mark.integration
def test_update_without_installer_fails_before_git_pull(test_app, monkeypatch):
    """No pip and no uv -> 500 and the checkout is never touched."""
    monkeypatch.setattr(server_module, '_pip_command', lambda args: None)
    with patch.object(
        server_module, '_check_working_tree_clean', return_value=(True, []),
    ), patch.object(
        server_module, '_run_git_command', return_value=('', True),
    ) as mock_git:
        response = test_app.post('/api/server/update', json={'confirm': True})

    assert response.status_code == 500
    detail = response.json()['detail']
    assert 'pip' in detail and 'uv' in detail
    assert 'NOT changed' in detail
    mock_git.assert_not_called()


def _run_update(force=True):
    request = server_module.UpdateRequest(confirm=True, force=force)
    user = SimpleNamespace(client_id='test', auth_method='test')
    return asyncio.run(server_module.update_server(request, BackgroundTasks(), user))


def test_pipless_env_without_uv_never_pulls(monkeypatch):
    """#1943: with no pip and no uv, the update stops before git pull."""
    _no_pip(monkeypatch)
    monkeypatch.delenv('UV', raising=False)
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    with (
        patch.object(server_module, '_run_git_command', return_value=('pulled', True)) as mock_git,
        patch.object(server_module, '_run_command', return_value=('No module named pip', False)),
        pytest.raises(HTTPException) as exc,
    ):
        _run_update()

    mock_git.assert_not_called()
    assert exc.value.status_code == 500
    assert 'Neither pip nor uv' in exc.value.detail


def test_pipless_env_with_uv_installs_through_uv(monkeypatch):
    """#1943: a uv-managed env without pip installs with `uv pip` after the pull."""
    _no_pip(monkeypatch)
    monkeypatch.delenv('UV', raising=False)
    monkeypatch.setattr(shutil, 'which', lambda name: 'C:/bin/uv.exe')
    with (
        patch.object(server_module, '_run_git_command', return_value=('pulled', True)),
        patch.object(server_module, '_run_command', return_value=('installed', True)) as mock_run,
    ):
        response = _run_update()

    assert response.status == 'success'
    assert mock_run.call_args.args[0] == [
        'C:/bin/uv.exe', 'pip', 'install', '--python', sys.executable, '-e', '.',
    ]


# --- #2100: pull the configured ref, back up before pulling ---

def _clean_tree():
    return patch.object(server_module, '_check_working_tree_clean', return_value=(True, []))


def _git_recording(events):
    def run_git(args, timeout=server_module.GIT_TIMEOUT_SECONDS):
        events.append(('git', args))
        return 'Updating abc..def', True
    return patch.object(server_module, '_run_git_command', side_effect=run_git)


def _pip_ok():
    return patch.object(server_module, '_run_pip_command', return_value=('Successfully installed', True))


@pytest.mark.integration
@pytest.mark.parametrize('env, expected', [
    ({'MCP_UPDATE_GIT_REMOTE': 'upstream', 'MCP_UPDATE_GIT_BRANCH': 'stable'}, ['pull', 'upstream', 'stable']),
    ({}, ['pull', 'origin', 'main']),
])
def test_update_pulls_the_configured_remote_and_branch(test_app, monkeypatch, env, expected):
    """The update pulls the ref /version/check compares against, not a hardcoded origin/main."""
    monkeypatch.delenv('MCP_UPDATE_GIT_REMOTE', raising=False)
    monkeypatch.delenv('MCP_UPDATE_GIT_BRANCH', raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    events = []
    with _clean_tree(), _git_recording(events), _pip_ok():
        response = test_app.post('/api/server/update', json={'confirm': True})

    assert response.status_code == 200
    assert [args for kind, args in events if kind == 'git'] == [expected]


@pytest.mark.integration
def test_update_backs_up_before_pulling(test_app, monkeypatch):
    """The backup is taken first, and its filename comes back in the response."""
    events = []
    _use_backup_service(monkeypatch, FakeBackupService(events))
    with _clean_tree(), _git_recording(events), _pip_ok():
        response = test_app.post('/api/server/update', json={'confirm': True})

    assert response.status_code == 200
    assert [kind for kind, _ in events] == ['backup', 'git']
    body = response.json()
    assert body['backup_filename'] == 'memory_backup_20261120_120000.db'
    assert 'memory_backup_20261120_120000.db' in body['message']


@pytest.mark.integration
@pytest.mark.parametrize('service', [
    FakeBackupService(result={'success': False, 'error': 'disk full'}),
    FakeBackupService(error=OSError('disk full')),
], ids=['backup-reports-failure', 'backup-raises'])
def test_update_backup_failure_returns_500_and_never_pulls(test_app, monkeypatch, service):
    """A failed backup stops the update before git pull, and says the checkout is unchanged."""
    events = []
    service.events = events
    _use_backup_service(monkeypatch, service)
    with _clean_tree(), _git_recording(events), _pip_ok() as mock_pip:
        response = test_app.post('/api/server/update', json={'confirm': True})

    assert response.status_code == 500
    detail = response.json()['detail']
    assert 'backup' in detail and 'disk full' in detail
    assert 'NOT changed' in detail
    assert [kind for kind, _ in events] == ['backup']
    mock_pip.assert_not_called()


@pytest.mark.integration
def test_update_without_a_local_database_still_updates(test_app, monkeypatch):
    """A backend with no local database file (Cloudflare) has nothing to back up and still updates."""
    events = []
    _use_backup_service(monkeypatch, FakeBackupService(events, db_path=None))
    with _clean_tree(), _git_recording(events), _pip_ok():
        response = test_app.post('/api/server/update', json={'confirm': True})

    assert response.status_code == 200
    assert [kind for kind, _ in events] == ['git']
    body = response.json()
    assert body['backup_filename'] is None
    assert 'no local database file' in body['message']


@pytest.mark.integration
def test_refused_update_leaves_no_backup(test_app, fake_backup):
    """The early checks run before the backup, so a refused request backs nothing up."""
    with patch.object(server_module, '_check_working_tree_clean', return_value=(False, ['M  src/foo.py'])):
        response = test_app.post('/api/server/update', json={'confirm': True})

    assert response.status_code == 409
    assert fake_backup.events == []
