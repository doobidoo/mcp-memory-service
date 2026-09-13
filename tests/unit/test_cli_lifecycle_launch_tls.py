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
Tests for the TLS configuration `memory launch` hands to the server it spawns.

Regression guard: `memory launch` built its uvicorn command line as a fixed
list -- host, port, log level, nothing else -- and never passed
--ssl-certfile/--ssl-keyfile. A deployment configured for HTTPS in .env came
back up on plain HTTP after `memory restart`, silently, while `memory info`
reported http:// and looked self-consistent. The legacy launcher
(scripts/server/run_http_server.py) did translate the same settings into
uvicorn ssl kwargs, so the two entry points disagreed about the same config.

The scheme the CLI *probes* is a separate decision and stays env-var-only --
see test_cli_lifecycle_tls.py. These tests must not weaken that: the server's
TLS config may come from .env, but resolving it must never mutate os.environ,
because _is_https_enabled() and _cli_allow_self_signed_certs() read it.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from mcp_memory_service.cli import lifecycle

# Captured before the autouse fixture below replaces it, so the structural
# test can still see the real search order.
_REAL_DOTENV_CANDIDATES = lifecycle._dotenv_candidates


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Run every test from a directory with no .env, and with no TLS env vars
    leaking in from the developer's shell.

    Also pins the .env search to that directory. _dotenv_candidates() mirrors
    config.base, which walks up from the source file to the project root -- so
    on a real checkout it finds the developer's own .env and every assertion
    here would depend on their local TLS settings. The candidate list itself is
    covered structurally by TestDotenvCandidates below."""
    monkeypatch.chdir(tmp_path)
    for var in (
        "MCP_HTTPS_ENABLED",
        "MCP_SSL_CERT_FILE",
        "MCP_SSL_KEY_FILE",
        "MCP_SSL_ADDITIONAL_IPS",
        "MCP_SSL_ADDITIONAL_HOSTNAMES",
        "MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(lifecycle, "_dotenv_candidates", lambda: [tmp_path / ".env"])
    return tmp_path


class TestDotenvCandidates:
    """The launcher must search the same places, in the same order, as the
    config module the spawned child imports -- otherwise the two disagree
    about the server's own settings, which is the class of bug being fixed."""

    def test_mirrors_config_base_search_order(self, tmp_path):
        # _REAL_DOTENV_CANDIDATES is captured at import time, before the
        # autouse fixture replaces the module attribute.
        candidates = _REAL_DOTENV_CANDIDATES()
        assert candidates[0] == Path.cwd() / ".env", "cwd must have highest priority"
        # The source-relative project root, how a source install finds its .env.
        repo_root = Path(lifecycle.__file__).parent.parent.parent.parent
        assert repo_root / ".env" in candidates
        assert Path.home() / ".mcp-memory" / ".env" in candidates


@pytest.fixture
def certpair(tmp_path):
    """A cert/key pair that exists on disk. Contents are irrelevant -- the CLI
    only checks presence and readability; uvicorn parses them in the child."""
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("-----BEGIN CERTIFICATE-----\n")
    key.write_text("-----BEGIN PRIVATE KEY-----\n")
    return cert, key


def _write_dotenv(directory, **values):
    directory.joinpath(".env").write_text(
        "".join(f"{k}={v}\n" for k, v in values.items())
    )


class TestResolveServerTls:
    """_resolve_server_tls() decides what the spawned server gets."""

    def test_defaults_to_no_tls(self):
        assert lifecycle._resolve_server_tls() == lifecycle._NO_TLS

    def test_env_vars_enable_tls(self, monkeypatch, certpair):
        cert, key = certpair
        monkeypatch.setenv("MCP_HTTPS_ENABLED", "true")
        monkeypatch.setenv("MCP_SSL_CERT_FILE", str(cert))
        monkeypatch.setenv("MCP_SSL_KEY_FILE", str(key))
        tls = lifecycle._resolve_server_tls()
        assert tls.scheme == "https"
        assert tls.cli_args == ["--ssl-certfile", str(cert), "--ssl-keyfile", str(key)]
        assert tls.uvicorn_kwargs == {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}

    def test_dotenv_enables_tls(self, isolated_cwd, certpair):
        """The whole point of the fix: the server's own config lives in .env,
        and the child's config module loads it, so the launcher must see it
        too. Contrast with _is_https_enabled(), which must NOT."""
        cert, key = certpair
        _write_dotenv(
            isolated_cwd,
            MCP_HTTPS_ENABLED="true",
            MCP_SSL_CERT_FILE=str(cert),
            MCP_SSL_KEY_FILE=str(key),
        )
        tls = lifecycle._resolve_server_tls()
        assert tls.scheme == "https"
        assert tls.cli_args == ["--ssl-certfile", str(cert), "--ssl-keyfile", str(key)]

    def test_env_var_wins_over_dotenv(self, isolated_cwd, monkeypatch, certpair):
        """Mirrors load_dotenv(override=False) in config.base."""
        cert, key = certpair
        _write_dotenv(isolated_cwd, MCP_HTTPS_ENABLED="true")
        monkeypatch.setenv("MCP_HTTPS_ENABLED", "false")
        monkeypatch.setenv("MCP_SSL_CERT_FILE", str(cert))
        monkeypatch.setenv("MCP_SSL_KEY_FILE", str(key))
        assert lifecycle._resolve_server_tls() == lifecycle._NO_TLS

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "enabled"])
    def test_truthy_variants(self, isolated_cwd, certpair, value):
        cert, key = certpair
        _write_dotenv(
            isolated_cwd,
            MCP_HTTPS_ENABLED=value,
            MCP_SSL_CERT_FILE=str(cert),
            MCP_SSL_KEY_FILE=str(key),
        )
        assert lifecycle._resolve_server_tls().scheme == "https"

    def test_does_not_mutate_os_environ(self, isolated_cwd, certpair):
        """The load-bearing invariant. If resolving the server's TLS config
        leaked .env into os.environ, _is_https_enabled() and
        _cli_allow_self_signed_certs() would silently start honouring a stray
        .env -- exactly the hardening those two functions exist to provide."""
        cert, key = certpair
        _write_dotenv(
            isolated_cwd,
            MCP_HTTPS_ENABLED="true",
            MCP_SSL_CERT_FILE=str(cert),
            MCP_SSL_KEY_FILE=str(key),
            MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS="true",
        )
        before = dict(os.environ)
        lifecycle._resolve_server_tls()
        assert dict(os.environ) == before
        assert lifecycle._is_https_enabled() is False
        assert lifecycle._cli_allow_self_signed_certs() is False

    def test_enabled_without_cert_paths_generates_certificate(
        self, isolated_cwd, monkeypatch, certpair
    ):
        cert, key = certpair
        _write_dotenv(isolated_cwd, MCP_HTTPS_ENABLED="true")
        generate = MagicMock(return_value=(str(cert), str(key)))
        monkeypatch.setattr(
            lifecycle, "generate_self_signed_certificate", generate, raising=False
        )

        tls = lifecycle._resolve_server_tls()

        assert tls == _https_tls(cert, key)
        generate.assert_called_once_with(additional_ips=None, additional_hostnames=None)

    def test_generated_certificate_honours_additional_sans(
        self, isolated_cwd, monkeypatch, certpair
    ):
        cert, key = certpair
        _write_dotenv(
            isolated_cwd,
            MCP_HTTPS_ENABLED="true",
            MCP_SSL_ADDITIONAL_IPS="192.0.2.10",
            MCP_SSL_ADDITIONAL_HOSTNAMES="memory.example.test",
        )
        generate = MagicMock(return_value=(str(cert), str(key)))
        monkeypatch.setattr(
            lifecycle, "generate_self_signed_certificate", generate, raising=False
        )

        lifecycle._resolve_server_tls()

        generate.assert_called_once_with(
            additional_ips="192.0.2.10",
            additional_hostnames="memory.example.test",
        )

    def test_enabled_with_only_one_cert_path_is_an_error(self, isolated_cwd, certpair):
        cert, _ = certpair
        _write_dotenv(
            isolated_cwd,
            MCP_HTTPS_ENABLED="true",
            MCP_SSL_CERT_FILE=str(cert),
        )
        with pytest.raises(click.ClickException) as exc:
            lifecycle._resolve_server_tls()
        assert "MCP_SSL_KEY_FILE" in str(exc.value)

    def test_enabled_with_missing_cert_file_is_an_error(self, isolated_cwd, certpair):
        """Never silently downgrade to HTTP -- that is the bug being fixed."""
        cert, key = certpair
        cert.unlink()
        _write_dotenv(
            isolated_cwd,
            MCP_HTTPS_ENABLED="true",
            MCP_SSL_CERT_FILE=str(cert),
            MCP_SSL_KEY_FILE=str(key),
        )
        with pytest.raises(click.ClickException) as exc:
            lifecycle._resolve_server_tls()
        assert str(cert) in str(exc.value)

    def test_disabled_ignores_broken_cert_paths(self, isolated_cwd):
        """Stale cert paths in .env must not block a plain-HTTP launch."""
        _write_dotenv(
            isolated_cwd,
            MCP_HTTPS_ENABLED="false",
            MCP_SSL_CERT_FILE="/nonexistent/cert.pem",
            MCP_SSL_KEY_FILE="/nonexistent/key.pem",
        )
        assert lifecycle._resolve_server_tls() == lifecycle._NO_TLS


def _https_tls(cert, key):
    """What _resolve_server_tls() returns for a valid HTTPS config."""
    return lifecycle._ServerTls(
        "https",
        ["--ssl-certfile", str(cert), "--ssl-keyfile", str(key)],
        {"ssl_certfile": str(cert), "ssl_keyfile": str(key)},
    )


def _run_launch(args, tls=None, captured=None):
    """Invoke launch() with every real-machine side effect mocked out, and
    return the uvicorn argv it would have spawned. Mirrors the isolation in
    test_cli_lifecycle.py: a live PID file, a real port scan or a real open()
    would each make this test depend on the developer's machine."""
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    with (
        patch("subprocess.Popen") as mock_popen,
        patch.object(lifecycle, "_probe_health", return_value=({"status": "healthy"}, False)),
        patch.object(lifecycle, "_write_pid") as mock_write_pid,
        patch.object(lifecycle, "_read_pid", return_value=None),
        patch.object(lifecycle, "_find_process_on_port", return_value=None),
        patch.object(lifecycle, "_ensure_dirs"),
        patch.object(lifecycle, "_log_file", return_value=MagicMock()),
        patch("builtins.open"),
    ):
        mock_popen.return_value = mock_proc
        with patch.object(lifecycle, "_resolve_server_tls",
                          return_value=tls or lifecycle._NO_TLS):
            result = CliRunner().invoke(lifecycle.launch, args)
        if captured is not None:
            captured["write_pid"] = mock_write_pid
            captured["popen"] = mock_popen
        cmd = []
        if mock_popen.called:
            call_args = mock_popen.call_args
            cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
        return result, [str(x) for x in cmd]


class TestLaunchPassesTlsToUvicorn:

    def test_plain_http_passes_no_ssl_args(self):
        result, cmd = _run_launch(["--port", "8000", "--detach"])
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"
        assert "--ssl-certfile" not in cmd
        assert "--ssl-keyfile" not in cmd

    def test_https_passes_cert_and_key(self, certpair):
        cert, key = certpair
        result, cmd = _run_launch(
            ["--port", "8000", "--detach"], tls=_https_tls(cert, key)
        )
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"
        assert "--ssl-certfile" in cmd, f"no TLS in spawned command: {cmd}"
        assert cmd[cmd.index("--ssl-certfile") + 1] == str(cert)
        assert "--ssl-keyfile" in cmd
        assert cmd[cmd.index("--ssl-keyfile") + 1] == str(key)

    def test_certificate_generation_failure_aborts_without_spawning(self, isolated_cwd):
        """Generation failure must stop the launch, not silently serve HTTP."""
        _write_dotenv(isolated_cwd, MCP_HTTPS_ENABLED="true")
        # Deliberately no patch("builtins.open") here, unlike _run_launch:
        # dotenv_values() reads the .env through open(), so patching it would
        # blank the very config this test is about. Nothing opens a log file
        # anyway -- the abort happens before _ensure_dirs().
        with (
            patch("subprocess.Popen") as mock_popen,
            patch.object(lifecycle, "_read_pid", return_value=None),
            patch.object(lifecycle, "_find_process_on_port", return_value=None),
            patch.object(lifecycle, "_ensure_dirs"),
            patch.object(lifecycle, "_write_pid"),
            patch.object(
                lifecycle,
                "generate_self_signed_certificate",
                side_effect=lifecycle.CertificateGenerationError("OpenSSL failed"),
            ),
        ):
            result = CliRunner().invoke(lifecycle.launch, ["--port", "8000", "--detach"])
        assert result.exit_code != 0
        assert not mock_popen.called, "spawned a server despite unusable TLS config"


class TestLaunchRecordsScheme:
    """The launch poll and every later command must probe the scheme that was
    actually launched. Deriving it from os.environ alone means an https server
    configured via .env gets probed over http, the health check fails, and
    launch tears down a child that started perfectly well."""

    def test_pidfile_records_https_when_launched_with_tls(self, certpair):
        cert, key = certpair
        captured = {}
        result, _ = _run_launch(
            ["--port", "8000", "--detach"],
            tls=_https_tls(cert, key),
            captured=captured,
        )
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"
        assert captured["write_pid"].called
        _, kwargs = captured["write_pid"].call_args
        assert kwargs.get("scheme") == "https", f"scheme not recorded: {kwargs}"

    def test_pidfile_records_http_by_default(self):
        captured = {}
        result, _ = _run_launch(["--port", "8000", "--detach"], captured=captured)
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"
        _, kwargs = captured["write_pid"].call_args
        assert kwargs.get("scheme") == "http"

    def test_recorded_scheme_drives_the_base_url(self, tmp_path, monkeypatch):
        """A recorded https server is probed over https even though
        MCP_HTTPS_ENABLED is absent from os.environ."""
        pid_file = tmp_path / "server.pid"
        pid_file.write_text(json.dumps({"pid": 4242, "scheme": "https"}))
        monkeypatch.setattr(lifecycle, "_pid_file", lambda: pid_file)
        assert lifecycle._base_url("127.0.0.1", 8000) == "https://127.0.0.1:8000"

    def test_base_url_falls_back_to_env_without_a_recorded_scheme(self, tmp_path, monkeypatch):
        pid_file = tmp_path / "server.pid"
        pid_file.write_text(json.dumps({"pid": 4242}))
        monkeypatch.setattr(lifecycle, "_pid_file", lambda: pid_file)
        assert lifecycle._base_url("127.0.0.1", 8000) == "http://127.0.0.1:8000"
        monkeypatch.setenv("MCP_HTTPS_ENABLED", "true")
        assert lifecycle._base_url("127.0.0.1", 8000) == "https://127.0.0.1:8000"

    def test_missing_pidfile_falls_back_to_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lifecycle, "_pid_file", lambda: tmp_path / "absent.pid")
        assert lifecycle._base_url("127.0.0.1", 8000) == "http://127.0.0.1:8000"
