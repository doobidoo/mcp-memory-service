"""Tests for environment variable warning functionality in config/storage.py

Tests the warn_unrecognized_path_var function that warns users about
unrecognized MCP_MEMORY_* environment variables that look like path settings.
"""

import pytest

from mcp_memory_service.config.storage import warn_unrecognized_path_var


class TestWarnUnrecognizedPathVar:
    """Test suite for warn_unrecognized_path_var function."""

    def test_warn_unrecognized_path_var_on_typo_returns_warning_message(self):
        """Should warn on typo like MCP_MEMORY_DB_PATH when no correct path vars are set."""
        environ = {
            'MCP_MEMORY_DB_PATH': '/some/path/db.sqlite',
            'OTHER_VAR': 'value'
        }
        
        result = warn_unrecognized_path_var(environ)
        
        assert result is not None
        assert 'MCP_MEMORY_DB_PATH' in result
        assert 'MCP_MEMORY_SQLITE_PATH' in result
        assert 'unrecognized' in result.lower()

    def test_warn_unrecognized_path_var_no_warning_when_correct_var_set(self):
        """Should not warn when MCP_MEMORY_SQLITE_PATH is correctly set."""
        environ = {
            'MCP_MEMORY_DB_PATH': '/wrong/path.sqlite',
            'MCP_MEMORY_SQLITE_PATH': '/correct/path.sqlite'
        }
        
        result = warn_unrecognized_path_var(environ)
        
        assert result is None

    def test_warn_unrecognized_path_var_no_warning_when_sqlitevec_path_set(self):
        """Should not warn when MCP_MEMORY_SQLITEVEC_PATH is correctly set."""
        environ = {
            'MCP_MEMORY_DATABASE_PATH': '/wrong/path.sqlite',
            'MCP_MEMORY_SQLITEVEC_PATH': '/correct/path.sqlite'
        }
        
        result = warn_unrecognized_path_var(environ)
        
        assert result is None

    def test_warn_unrecognized_path_var_no_warning_on_recognized_non_path_var(self):
        """Should not warn on recognized non-path vars like MCP_MEMORY_USE_ONNX."""
        environ = {
            'MCP_MEMORY_USE_ONNX': 'true',
            'MCP_MEMORY_OFFLINE': 'false'
        }
        
        result = warn_unrecognized_path_var(environ)
        
        assert result is None

    def test_warn_unrecognized_path_var_no_warning_when_both_typo_and_correct_set(self):
        """Should not warn when both typo and correct var are present."""
        environ = {
            'MCP_MEMORY_DB_PATH': '/wrong/path.sqlite',
            'MCP_MEMORY_SQLITE_PATH': '/correct/path.sqlite'
        }
        
        result = warn_unrecognized_path_var(environ)
        
        assert result is None

    def test_warn_unrecognized_path_var_handles_multiple_typos(self):
        """Should warn about multiple unrecognized path-like vars."""
        environ = {
            'MCP_MEMORY_DB_PATH': '/some/db.sqlite',
            'MCP_MEMORY_DATABASE_DIR': '/some/dir/',
            'MCP_MEMORY_USE_ONNX': 'true'  # This should be ignored (recognized)
        }
        
        result = warn_unrecognized_path_var(environ)
        
        assert result is not None
        # Should mention one of the unrecognized vars
        assert ('MCP_MEMORY_DB_PATH' in result or 'MCP_MEMORY_DATABASE_DIR' in result)

    def test_warn_unrecognized_path_var_case_insensitive_path_detection(self):
        """Should detect PATH/Path/db/DB/Dir/DIR in variable names."""
        test_cases = [
            'MCP_MEMORY_DB_PATH',
            'MCP_MEMORY_Database_Path', 
            'MCP_MEMORY_DATA_DIR',
            'MCP_MEMORY_DB_FILE'
        ]
        
        for var_name in test_cases:
            environ = {var_name: '/some/path'}
            result = warn_unrecognized_path_var(environ)
            assert result is not None, f"Should warn for {var_name}"

    def test_warn_unrecognized_path_var_ignores_non_mcp_memory_vars(self):
        """Should ignore variables that don't start with MCP_MEMORY_."""
        environ = {
            'DATABASE_PATH': '/some/path.db',
            'SQLITE_DB_PATH': '/other/path.db',
            'PATH': '/usr/bin'
        }
        
        result = warn_unrecognized_path_var(environ)
        
        assert result is None

    def test_warn_unrecognized_path_var_ignores_non_path_like_mcp_memory_vars(self):
        """Should ignore MCP_MEMORY_ vars that don't look like paths."""
        environ = {
            'MCP_MEMORY_TIMEOUT': '30',
            'MCP_MEMORY_BATCH_SIZE': '100',
            'MCP_MEMORY_ENABLE_FEATURE': 'true'
        }
        
        result = warn_unrecognized_path_var(environ)
        
        assert result is None

    def test_warn_unrecognized_path_var_empty_environ_returns_none(self):
        """Should return None for empty environment."""
        environ = {}
        
        result = warn_unrecognized_path_var(environ)
        
        assert result is None

    def test_warn_unrecognized_path_var_complete_recognized_set_validation(self):
        """Should not warn for any of the 16 recognized MCP_MEMORY_ variables."""
        recognized_vars = [
            'MCP_MEMORY_ALLOW_HASH_EMBEDDINGS',
            'MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS',
            'MCP_MEMORY_ARCHIVE_PATH',
            'MCP_MEMORY_BACKUPS_PATH', 
            'MCP_MEMORY_BASE_DIR',
            'MCP_MEMORY_INCLUDE_HOSTNAME',
            'MCP_MEMORY_INTEGRITY_CHECK_ENABLED',
            'MCP_MEMORY_INTEGRITY_CHECK_INTERVAL',
            'MCP_MEMORY_OFFLINE',
            'MCP_MEMORY_ONNX_ALLOW_DOWNLOAD',
            'MCP_MEMORY_ONNX_PROVIDERS',
            'MCP_MEMORY_SQLITE_PATH',
            'MCP_MEMORY_SQLITE_PRAGMAS',
            'MCP_MEMORY_SQLITEVEC_PATH',
            'MCP_MEMORY_STORAGE_BACKEND',
            'MCP_MEMORY_USE_ONNX'
        ]
        
        for var in recognized_vars:
            environ = {var: 'some_value'}
            result = warn_unrecognized_path_var(environ)
            assert result is None, f"Should not warn for recognized var {var}"

    def test_warn_unrecognized_path_var_sanitizes_env_key_in_message(self):
        """Should escape control characters in the env-var name to prevent log injection.

        The variable name is user-controlled. ``_sanitize_log_value`` neutralizes
        control characters by escaping them (``\\n`` -> literal ``\\n``), so a
        newline or carriage return in the key cannot forge a new log line or
        inject terminal control sequences. The payload text may remain, but inert.
        """
        environ = {
            'MCP_MEMORY_DB_PATH\r\nINJECTED: forged log line': '/some/path.sqlite'
        }

        result = warn_unrecognized_path_var(environ)

        assert result is not None
        # No raw control characters survive (cannot break the log line).
        assert '\n' not in result
        assert '\r' not in result
        assert '\x1b' not in result
        # The control characters are present only in escaped form.
        assert '\\r\\n' in result


class TestWarnUnrecognizedPathVarLoggingPath:
    """Integration tests for the import-time logging branch in storage.py.

    The unit tests above only exercise the pure helper. These reload the config
    module with controlled environment variables so the actual
    ``logger.warning(...)`` branch runs and is asserted, covering the path the
    helper feeds into.
    """

    @pytest.fixture(autouse=True)
    def _restore_storage_module(self):
        """Reload ``config.storage`` under the original environment after each test.

        ``_reload_storage`` calls ``importlib.reload`` so the import-time branch
        runs against controlled env vars. ``monkeypatch`` restores ``os.environ``
        when the test ends, but the reloaded module object left in
        ``sys.modules`` still holds values computed from the test's environment.
        Without this teardown, any later test importing ``config.storage`` would
        see contaminated module-level state — making the suite order-dependent.
        This reloads the module once more, after ``monkeypatch`` has undone the
        env changes, so it reflects the process's real environment again.
        """
        yield
        import importlib
        import mcp_memory_service.config.storage as storage_mod
        importlib.reload(storage_mod)

    def _reload_storage(self, monkeypatch, env):
        """Reload config.storage under a clean, controlled environment."""
        import importlib
        import os

        # Remove every MCP_MEMORY_* var so only `env` is in play, then apply env.
        for key in list(os.environ):
            if key.startswith('MCP_MEMORY_'):
                monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        # sqlite_vec backend is required for the SQLite path branch to execute.
        monkeypatch.setenv('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec')

        import mcp_memory_service.config.storage as storage_mod
        return importlib.reload(storage_mod)

    def test_warning_emitted_on_typo_without_correct_path(self, monkeypatch, caplog):
        """Reloading with a typo'd path var and no correct one should log a warning."""
        import logging

        with caplog.at_level(logging.WARNING, logger='mcp_memory_service.config.storage'):
            self._reload_storage(monkeypatch, {'MCP_MEMORY_DB_PATH': '/tmp/typo.sqlite'})

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any('MCP_MEMORY_DB_PATH' in r.getMessage() for r in warnings), \
            "Expected a warning naming the unrecognized env var"

    def test_no_warning_when_correct_path_set(self, monkeypatch, caplog, tmp_path):
        """Reloading with the correct path var set must not emit the typo warning."""
        import logging

        correct = str(tmp_path / 'correct.sqlite')
        with caplog.at_level(logging.WARNING, logger='mcp_memory_service.config.storage'):
            self._reload_storage(monkeypatch, {
                'MCP_MEMORY_DB_PATH': '/tmp/typo.sqlite',
                'MCP_MEMORY_SQLITE_PATH': correct,
            })

        typo_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'Unrecognized env var' in r.getMessage()
        ]
        assert not typo_warnings, "Should not warn when the correct path var is set"