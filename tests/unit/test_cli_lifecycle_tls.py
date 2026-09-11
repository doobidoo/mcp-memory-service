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
Tests for TLS verification in the CLI's _probe_health health-check helper.

Regression guard: the health-check helper previously created an
ssl.SSLContext with check_hostname=False and verify_mode=ssl.CERT_NONE
unconditionally, for every HTTPS health check the `memory` CLI made
(launch/stop/restart/info/health), with no way to opt back in to real
verification. Same class of issue as discovery/client.py's
GHSA-x9r8-q2qj-cgvw fix.
"""

import ssl
import urllib.error
from unittest import mock

import pytest
from click.testing import CliRunner

from mcp_memory_service.cli import lifecycle


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Run every test from a directory with no .env, so tests don't depend
    on (or get silently broken by) whatever happens to be in the repo root.
    Also resets the module-level warn-once flag so tests don't leak state
    into each other.

    Also points the PID file at tmp_path. _base_url() consults
    _recorded_scheme() first, which reads the scheme out of the running
    server's PID file at ~/.local/share/mcp-memory/server.pid. On a
    developer machine with an HTTPS server running that file says
    "https", so TestBaseUrl's http assertions fail and, worse, its https
    assertion passes for the wrong reason. CI has no such file, which is
    why this only ever shows up locally.

    The _kill_process real-signal guard and MCP_HTTP_PORT/HOST pinning
    live in conftest.py's autouse _lifecycle_process_safety_net fixture
    and apply here automatically."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(lifecycle, "_pid_file", lambda: tmp_path / "server.pid")
    monkeypatch.delenv("MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS", raising=False)
    monkeypatch.delenv("MCP_HTTPS_ENABLED", raising=False)
    monkeypatch.setattr(lifecycle, "_self_signed_warning_shown", False)
    monkeypatch.setattr(lifecycle, "_ssl_failure_warning_shown", False)
    return tmp_path


class TestCliAllowSelfSignedCerts:
    """Tests for the _cli_allow_self_signed_certs helper.

    Deliberately env-var-only, no .env fallback: a flag that disables
    this CLI's only line of MITM defense shouldn't be settable by a
    stray .env file in whatever directory happens to be cwd. Same
    discipline _is_https_enabled() now follows too -- see
    TestIsHttpsEnabled below."""

    def test_defaults_to_false(self):
        assert lifecycle._cli_allow_self_signed_certs() is False

    def test_explicit_true_enables(self, monkeypatch):
        monkeypatch.setenv("MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS", "true")
        assert lifecycle._cli_allow_self_signed_certs() is True

    def test_explicit_false_stays_disabled(self, monkeypatch):
        monkeypatch.setenv("MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS", "false")
        assert lifecycle._cli_allow_self_signed_certs() is False

    @pytest.mark.parametrize("value", ["1", "yes", "TRUE", "True"])
    def test_truthy_variants_enable(self, monkeypatch, value):
        monkeypatch.setenv("MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS", value)
        assert lifecycle._cli_allow_self_signed_certs() is True

    def test_empty_env_var_is_disabled(self, monkeypatch):
        """`docker run -e NAME` with no value, or a blank systemd
        Environment=, sets the var to "" -- must resolve to the secure
        default, not raise or behave as truthy."""
        monkeypatch.setenv("MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS", "")
        assert lifecycle._cli_allow_self_signed_certs() is False

    def test_dotenv_file_is_ignored(self, isolated_cwd):
        """The opt-in intentionally does not read .env -- confirms a
        stray .env file in cwd cannot silently enable it."""
        (isolated_cwd / ".env").write_text("MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS=true\n")
        assert lifecycle._cli_allow_self_signed_certs() is False


class TestIsHttpsEnabled:
    """Tests for the _is_https_enabled helper.

    Previously fell back to reading MCP_HTTPS_ENABLED from a .env file
    in cwd if unset in the environment. Now env-var-only, same
    discipline as _cli_allow_self_signed_certs() above: this decides
    which scheme the health check probes with, so a stray .env file
    in whatever directory happens to be cwd shouldn't be able to
    silently redirect it.

    Accepted values must match config.base.safe_get_bool_env()'s
    truthy set (true/1/yes/on/enabled) -- see #231."""

    def test_defaults_to_false(self):
        assert lifecycle._is_https_enabled() is False

    def test_explicit_true_enables(self, monkeypatch):
        monkeypatch.setenv("MCP_HTTPS_ENABLED", "true")
        assert lifecycle._is_https_enabled() is True

    def test_explicit_false_stays_disabled(self, monkeypatch):
        monkeypatch.setenv("MCP_HTTPS_ENABLED", "false")
        assert lifecycle._is_https_enabled() is False

    @pytest.mark.parametrize("value", ["1", "yes", "TRUE", "True", "on", "ON", "enabled", "Enabled"])
    def test_truthy_variants_enable(self, monkeypatch, value):
        monkeypatch.setenv("MCP_HTTPS_ENABLED", value)
        assert lifecycle._is_https_enabled() is True

    def test_empty_env_var_is_disabled(self, monkeypatch):
        """`docker run -e NAME` with no value, or a blank systemd
        Environment=, sets the var to "" -- must resolve to the secure
        default, not raise or behave as truthy."""
        monkeypatch.setenv("MCP_HTTPS_ENABLED", "")
        assert lifecycle._is_https_enabled() is False

    def test_dotenv_file_is_ignored(self, isolated_cwd):
        """A stray .env file in cwd can no longer silently switch the
        health-check probe onto (or off) HTTPS."""
        (isolated_cwd / ".env").write_text("MCP_HTTPS_ENABLED=true\n")
        assert lifecycle._is_https_enabled() is False


class TestBaseUrl:
    """End-to-end coverage for _base_url()'s scheme substitution --
    the 5 lifecycle commands (launch/stop/restart/info/health) only
    ever go through this function, never _is_https_enabled() directly,
    so a helper-level-only test suite could pass while the actual
    scheme returned to callers was wrong."""

    def test_defaults_to_http(self):
        assert lifecycle._base_url("127.0.0.1", 8000) == "http://127.0.0.1:8000"

    def test_https_enabled_switches_scheme(self, monkeypatch):
        monkeypatch.setenv("MCP_HTTPS_ENABLED", "true")
        assert lifecycle._base_url("127.0.0.1", 8000) == "https://127.0.0.1:8000"

    def test_dotenv_file_does_not_switch_scheme(self, isolated_cwd):
        """Same guarantee as TestIsHttpsEnabled.test_dotenv_file_is_ignored,
        but confirmed at the level callers actually consume."""
        (isolated_cwd / ".env").write_text("MCP_HTTPS_ENABLED=true\n")
        assert lifecycle._base_url("127.0.0.1", 8000) == "http://127.0.0.1:8000"


class TestHttpGetJsonTlsContext:
    """Tests that _probe_health passes the right ssl context to urlopen."""

    @staticmethod
    def _fake_urlopen(captured):
        def _fake(req, timeout=None, context=None):
            captured["context"] = context
            m = mock.MagicMock()
            m.read.return_value = b'{"status": "healthy"}'
            m.__enter__.return_value = m
            m.__exit__.return_value = False
            return m
        return _fake

    def test_https_default_verifies(self):
        """No explicit opt-in -> no custom (insecure) context is built."""
        captured = {}
        with mock.patch("urllib.request.urlopen", side_effect=self._fake_urlopen(captured)):
            result, _ = lifecycle._probe_health("https://localhost:8443/api/health")

        assert result == {"status": "healthy"}
        assert captured["context"] is None

    def test_https_explicit_opt_in_builds_insecure_context(self, monkeypatch):
        monkeypatch.setenv("MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS", "true")
        captured = {}

        with mock.patch("urllib.request.urlopen", side_effect=self._fake_urlopen(captured)):
            lifecycle._probe_health("https://localhost:8443/api/health")

        assert captured["context"] is not None
        assert captured["context"].verify_mode == ssl.CERT_NONE
        assert captured["context"].check_hostname is False

    def test_https_opt_in_warns_on_stderr(self, monkeypatch, capsys):
        monkeypatch.setenv("MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS", "true")
        captured = {}

        with mock.patch("urllib.request.urlopen", side_effect=self._fake_urlopen(captured)):
            lifecycle._probe_health("https://localhost:8443/api/health")

        err = capsys.readouterr().err
        assert "DISABLED" in err
        assert "MITM" in err

    def test_https_opt_in_warns_only_once_per_process(self, monkeypatch, capsys):
        """Regression guard: `launch` polls this dozens of times per
        invocation; the MITM warning must not scroll by on every call."""
        monkeypatch.setenv("MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS", "true")
        captured = {}

        with mock.patch("urllib.request.urlopen", side_effect=self._fake_urlopen(captured)):
            for _ in range(5):
                lifecycle._probe_health("https://localhost:8443/api/health")

        err = capsys.readouterr().err
        assert err.count("DISABLED") == 1

    def test_https_default_does_not_warn(self, capsys):
        captured = {}
        with mock.patch("urllib.request.urlopen", side_effect=self._fake_urlopen(captured)):
            lifecycle._probe_health("https://localhost:8443/api/health")

        assert capsys.readouterr().err == ""

    def test_ssl_verification_failure_without_opt_in_gets_diagnostic(self, capsys):
        """A real cert failure (e.g. self-signed, hostname mismatch) should
        point at the opt-in flag instead of failing silently like a generic
        connection error -- previously this was indistinguishable from
        'server not running'.

        urllib.request.urlopen never raises ssl.SSLError directly: its
        connection handling wraps every failure -- including SSL errors --
        in a urllib.error.URLError, with the real exception on `.reason`.
        Mocking a bare ssl.SSLCertVerificationError as the side_effect (as
        an earlier version of this test did) exercises a shape urlopen
        never actually produces and would pass even if the production code
        only caught bare ssl.SSLError."""
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(
                ssl.SSLCertVerificationError("certificate verify failed")
            ),
        ):
            result, _ = lifecycle._probe_health("https://localhost:8443/api/health")

        assert result is None
        err = capsys.readouterr().err
        assert "MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS" in err
        assert "certificate verification failed" in err.lower()

    def test_ssl_verification_failure_warns_only_once_per_process(self, capsys):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(
                ssl.SSLCertVerificationError("certificate verify failed")
            ),
        ):
            for _ in range(5):
                lifecycle._probe_health("https://localhost:8443/api/health")

        err = capsys.readouterr().err
        assert err.count("MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS") == 1

    def test_plain_connection_refused_stays_silent(self, capsys):
        """A URLError NOT wrapping an SSL error (e.g. connection refused
        because the server just isn't up yet) must not print the TLS
        diagnostic -- that would be actively misleading during the normal
        `launch` startup-polling window."""
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(ConnectionRefusedError("refused")),
        ):
            result, _ = lifecycle._probe_health("https://localhost:8443/api/health")

        assert result is None
        assert capsys.readouterr().err == ""

    def test_non_cert_ssl_error_gets_generic_message_not_opt_in_advice(self, capsys):
        """Regression guard: a protocol-level SSLError (e.g.
        WRONG_VERSION_NUMBER from MCP_HTTPS_ENABLED=true against a plain
        HTTP server) is not a certificate problem, so telling the user to
        set MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS is actively wrong advice --
        that flag only disables cert verification and cannot fix a
        protocol mismatch. It's also reachable with the opt-in already on,
        since CERT_NONE doesn't affect protocol-level errors, and advising
        a flag that's already set would be visibly broken."""
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(
                ssl.SSLError(1, "[SSL: WRONG_VERSION_NUMBER] wrong version number")
            ),
        ):
            result, _ = lifecycle._probe_health("https://localhost:8443/api/health")

        assert result is None
        err = capsys.readouterr().err
        assert "TLS error" in err
        assert "MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS" not in err

    def test_plain_http_never_builds_a_context(self, monkeypatch):
        monkeypatch.setenv("MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS", "true")
        captured = {}

        with mock.patch("urllib.request.urlopen", side_effect=self._fake_urlopen(captured)):
            lifecycle._probe_health("http://localhost:8000/api/health")

        assert captured["context"] is None

    def test_non_http_scheme_never_builds_a_context(self, monkeypatch):
        monkeypatch.setenv("MCP_MEMORY_ALLOW_SELF_SIGNED_CERTS", "true")
        captured = {}

        with mock.patch("urllib.request.urlopen", side_effect=self._fake_urlopen(captured)):
            result, _ = lifecycle._probe_health("ftp://localhost/api/health")

        # Not an http(s) URL urlopen would actually accept in practice, but
        # this pins that the "https:" prefix check doesn't false-positive.
        assert captured.get("context") is None
        assert result == {"status": "healthy"}

    def test_connection_error_returns_none(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            result, _ = lifecycle._probe_health("https://localhost:8443/api/health")

        assert result is None


class TestProbeHealthTlsBlockedFlag:
    """Tests for _probe_health's second return value -- whether a failure
    was specifically a certificate-verification rejection, as opposed to
    any other reason the health check failed. Callers use this to decide
    whether "no response" means "not running" (safe to kill and restart)
    or "probably running, just unverifiable" (must not restart)."""

    def test_success_is_not_tls_blocked(self):
        captured = {}
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=TestHttpGetJsonTlsContext._fake_urlopen(captured),
        ):
            health, tls_blocked = lifecycle._probe_health("https://localhost:8443/api/health")

        assert health == {"status": "healthy"}
        assert tls_blocked is False

    def test_cert_verification_failure_is_tls_blocked(self):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(
                ssl.SSLCertVerificationError("certificate verify failed")
            ),
        ):
            health, tls_blocked = lifecycle._probe_health("https://localhost:8443/api/health")

        assert health is None
        assert tls_blocked is True

    def test_non_cert_ssl_error_is_not_tls_blocked(self):
        """A protocol-level SSLError doesn't mean "server is alive with an
        untrusted cert" -- it could just as easily mean nothing real is
        listening on that port, so it must not suppress a restart."""
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(
                ssl.SSLError(1, "[SSL: WRONG_VERSION_NUMBER] wrong version number")
            ),
        ):
            health, tls_blocked = lifecycle._probe_health("https://localhost:8443/api/health")

        assert health is None
        assert tls_blocked is False

    def test_connection_refused_is_not_tls_blocked(self):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(ConnectionRefusedError("refused")),
        ):
            health, tls_blocked = lifecycle._probe_health("https://localhost:8443/api/health")

        assert health is None
        assert tls_blocked is False


class TestLaunchDoesNotClobberTlsBlockedServer:
    """Regression guard: before this fix, `launch` treated ANY failed
    health check -- including a certificate-verification failure against
    an otherwise-healthy self-signed-cert server -- as "not running," and
    would kill the recorded PID and start a duplicate. A cert-verification
    failure means something is listening and is probably fine; killing it
    would be the fix causing an outage, not preventing one."""

    def test_does_not_kill_or_relaunch_a_tls_blocked_but_alive_server(self, monkeypatch):
        """existing_pid being truthy already means _read_pid() found the
        process alive, so this guard doesn't need its own liveness recheck
        -- only tls_blocked gates it."""
        monkeypatch.setattr(lifecycle, "_read_pid", lambda: 12345)
        monkeypatch.setattr(lifecycle, "_probe_health", lambda url, timeout=3: (None, True))

        kill_process = mock.Mock()
        monkeypatch.setattr(lifecycle, "_kill_process", kill_process)
        find_process_on_port = mock.Mock()
        monkeypatch.setattr(lifecycle, "_find_process_on_port", find_process_on_port)
        popen = mock.Mock()
        monkeypatch.setattr(lifecycle.subprocess, "Popen", popen)

        runner = CliRunner()
        result = runner.invoke(lifecycle.launch, [])

        assert result.exit_code == 0, result.output
        assert "could not be verified" in result.output
        assert "Not restarting" in result.output
        kill_process.assert_not_called()
        find_process_on_port.assert_not_called()
        popen.assert_not_called()

    def test_poll_loop_stops_early_on_tls_blocked_after_starting(self, monkeypatch):
        """Regression guard: after spawning a fresh child, the startup poll
        loop also needs the tls_blocked signal -- otherwise a self-signed
        user with nothing previously running still burns the full ~30s
        polling window before the same diagnostic finally appears."""
        monkeypatch.setattr(lifecycle, "_read_pid", lambda: None)
        monkeypatch.setattr(lifecycle, "_ensure_dirs", lambda: None)
        monkeypatch.setattr(
            lifecycle, "_log_file", lambda: mock.MagicMock(with_suffix=lambda s: mock.MagicMock())
        )
        monkeypatch.setattr("builtins.open", lambda *a, **k: mock.MagicMock())
        monkeypatch.setattr(lifecycle, "_write_pid", lambda pid, scheme="http": None)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        proc = mock.Mock()
        proc.pid = 99999
        proc.poll.return_value = None  # still alive

        # Tied to whether Popen has actually run yet, not to call
        # ordinality -- robust against any future _find_process_on_port
        # call being added before the spawn (an ordinal counter would
        # silently shift and start lying about which call is which).
        spawned = {"done": False}

        def fake_popen(*args, **kwargs):
            spawned["done"] = True
            return proc

        monkeypatch.setattr(lifecycle.subprocess, "Popen", mock.Mock(side_effect=fake_popen))

        def fake_find_process_on_port(port):
            # Pre-spawn "kill stale process on this port" check: nothing
            # there yet. Post-spawn (inside the poll loop): our own child
            # has finished its startup import and actually bound the
            # port, which is the only state in which the tls_blocked
            # message should be trusted.
            return proc.pid if spawned["done"] else None

        monkeypatch.setattr(lifecycle, "_find_process_on_port", fake_find_process_on_port)

        probe_calls = {"n": 0}

        def fake_probe(url, timeout=3):
            probe_calls["n"] += 1
            return None, True

        monkeypatch.setattr(lifecycle, "_probe_health", fake_probe)

        runner = CliRunner()
        result = runner.invoke(lifecycle.launch, [])

        assert result.exit_code == 0, result.output
        assert "could not be verified" in result.output
        assert "health check timed out" not in result.output
        # Stopped on the first tls_blocked probe rather than looping 60 times.
        assert probe_calls["n"] == 1

    def test_does_not_claim_verification_failure_if_child_already_died(self, monkeypatch):
        """Regression guard: a tls_blocked probe result doesn't guarantee
        the probe reached OUR child -- if something else is already
        listening on the port, our child dies on EADDRINUSE and the probe
        hits that other listener instead. Telling the user to set the
        self-signed opt-in in that case is confident, wrong advice; the
        honest "check logs" timeout message is what should show instead."""
        monkeypatch.setattr(lifecycle, "_read_pid", lambda: None)
        monkeypatch.setattr(lifecycle, "_find_process_on_port", lambda port: None)
        monkeypatch.setattr(lifecycle, "_ensure_dirs", lambda: None)
        monkeypatch.setattr(
            lifecycle, "_log_file", lambda: mock.MagicMock(with_suffix=lambda s: mock.MagicMock())
        )
        monkeypatch.setattr("builtins.open", lambda *a, **k: mock.MagicMock())
        monkeypatch.setattr(lifecycle, "_write_pid", lambda pid, scheme="http": None)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        proc = mock.Mock()
        proc.pid = 99999
        proc.poll.return_value = 1  # already exited (e.g. EADDRINUSE)
        proc.returncode = 1
        monkeypatch.setattr(lifecycle.subprocess, "Popen", mock.Mock(return_value=proc))

        probe_calls = {"n": 0}

        def fake_probe(url, timeout=3):
            probe_calls["n"] += 1
            return None, True

        monkeypatch.setattr(lifecycle, "_probe_health", fake_probe)

        runner = CliRunner()
        result = runner.invoke(lifecycle.launch, [])

        assert result.exit_code == 0, result.output
        assert "could not be verified" not in result.output
        assert "exited with code 1" in result.output
        # Regression guard for the early `break` itself: without it, a
        # confirmed-dead child still polls all 60 times before reporting.
        # Mutation-tested: reverting the break to a no-op leaves this
        # suite green without this assertion.
        assert probe_calls["n"] == 1
        # Regression guard for the saw_tls_block hint being scoped to the
        # genuine-timeout branch only: a dead child's cert failure may
        # belong to a totally different, unrelated process (e.g. one
        # already squatting the port with its own bad cert) and must not
        # be attributed to the user's own server here.
        assert "certificate-verification failure was seen" not in result.output

    def test_does_not_claim_verification_failure_while_child_still_importing(self, monkeypatch):
        """Regression guard: `proc.poll() is None` alone is not enough --
        it only proves the child hasn't exited, not that it has bound the
        port yet. uvicorn does its heavy import (loading
        mcp_memory_service.web.app) before binding the socket, so there is
        a real window where our child is alive but a DIFFERENT
        pre-existing process still owns the port. A probe during that
        window reads as tls_blocked against that other process, and must
        not be attributed to our own child."""
        monkeypatch.setattr(lifecycle, "_read_pid", lambda: None)
        monkeypatch.setattr(lifecycle, "_ensure_dirs", lambda: None)
        monkeypatch.setattr(
            lifecycle, "_log_file", lambda: mock.MagicMock(with_suffix=lambda s: mock.MagicMock())
        )
        monkeypatch.setattr("builtins.open", lambda *a, **k: mock.MagicMock())
        monkeypatch.setattr(lifecycle, "_write_pid", lambda pid, scheme="http": None)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        proc = mock.Mock()
        proc.pid = 99999
        proc.poll.return_value = None  # our child is alive...
        monkeypatch.setattr(lifecycle.subprocess, "Popen", mock.Mock(return_value=proc))
        # ...but a DIFFERENT process (PID 424242) owns the port for the
        # entire polling window -- our child never gets to bind it here.
        # time.sleep is mocked to a no-op, so all 60 iterations run
        # instantly; the loop is left to exhaust naturally.
        monkeypatch.setattr(lifecycle, "_find_process_on_port", lambda port: 424242)
        # _find_process_on_port also runs pre-spawn ("kill stale process on
        # this port"), and with it returning a real, plausibly-live PID
        # here, the un-mocked code would actually SIGTERM/SIGKILL PID
        # 424242 on whatever machine runs this test. Never let a unit test
        # call the real _kill_process with a fabricated PID.
        monkeypatch.setattr(lifecycle, "_kill_process", mock.Mock(return_value=True))
        monkeypatch.setattr(lifecycle, "_stop_process_on_port", mock.Mock(return_value=True))
        monkeypatch.setattr(lifecycle, "_probe_health", lambda url, timeout=3: (None, True))

        runner = CliRunner()
        result = runner.invoke(lifecycle.launch, [])

        assert result.exit_code == 0, result.output
        assert "could not be verified" not in result.output
        assert "health check timed out" in result.output
        # Even though the message never fires, a cert-verification failure
        # WAS observed during polling -- the timeout message should still
        # hint at it rather than giving no TLS-related guidance at all
        # (this is also what happens for real on a host without lsof/
        # netstat, where _find_process_on_port always returns None).
        assert "certificate-verification failure was seen" in result.output

    def test_claims_verification_failure_once_child_actually_binds(self, monkeypatch):
        """The other half of the same story: once our own child finishes
        its startup import and genuinely binds the port, the message must
        start firing -- pinning that the port-ownership check isn't just
        permanently silent, only silent during the window it should be."""
        monkeypatch.setattr(lifecycle, "_read_pid", lambda: None)
        monkeypatch.setattr(lifecycle, "_ensure_dirs", lambda: None)
        monkeypatch.setattr(
            lifecycle, "_log_file", lambda: mock.MagicMock(with_suffix=lambda s: mock.MagicMock())
        )
        monkeypatch.setattr("builtins.open", lambda *a, **k: mock.MagicMock())
        monkeypatch.setattr(lifecycle, "_write_pid", lambda pid, scheme="http": None)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        proc = mock.Mock()
        proc.pid = 99999
        proc.poll.return_value = None
        monkeypatch.setattr(lifecycle, "_kill_process", mock.Mock(return_value=True))

        # Keyed off whether Popen has run yet, like the sibling test --
        # not a raw call counter, so a call added before or after spawn
        # can't silently shift which call means what. Pre-spawn ("kill
        # stale process on this port"): nothing there yet. Post-spawn: a
        # different process (424242) owns the port for the first two
        # poll-loop probes (our child is mid-import), then our own child
        # takes over from the third probe onward.
        spawned = {"done": False}
        post_spawn_calls = {"n": 0}

        def fake_popen(*args, **kwargs):
            spawned["done"] = True
            return proc

        monkeypatch.setattr(lifecycle.subprocess, "Popen", mock.Mock(side_effect=fake_popen))

        def fake_find_process_on_port(port):
            if not spawned["done"]:
                return None
            post_spawn_calls["n"] += 1
            return proc.pid if post_spawn_calls["n"] >= 3 else 424242

        monkeypatch.setattr(lifecycle, "_find_process_on_port", fake_find_process_on_port)
        monkeypatch.setattr(lifecycle, "_probe_health", lambda url, timeout=3: (None, True))

        runner = CliRunner()
        result = runner.invoke(lifecycle.launch, [])

        assert result.exit_code == 0, result.output
        assert "could not be verified" in result.output


class TestRestartWarnsWhenBackendUndetectable:
    """Regression guard: `restart` used to silently fall back to the
    default storage backend whenever the health check failed for any
    reason, including a TLS-verification failure that gives no signal
    about whether the server was actually down. A hybrid/cloudflare
    deployment restarting silently on sqlite_vec is a data-availability
    surprise, not a graceful fallback -- the user needs to know why their
    explicit backend choice didn't carry over."""

    def test_warns_on_stderr_when_backend_cannot_be_read_due_to_tls(self, monkeypatch):
        monkeypatch.setattr(lifecycle, "_probe_health", lambda url, timeout=3: (None, True))
        monkeypatch.setattr(lifecycle, "stop", mock.Mock())
        monkeypatch.setattr(lifecycle, "launch", mock.Mock())
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)

        runner = CliRunner()
        result = runner.invoke(lifecycle.restart, [])

        assert result.exit_code == 0, result.output
        assert "could not be verified" in result.output
        # Regression guard: launch only sets MCP_MEMORY_STORAGE_BACKEND when
        # storage_backend is truthy, so a None here means the server keeps
        # resolving its backend from its own config, not "the default" --
        # the message must not claim otherwise.
        assert "its own configuration" in result.output
        assert "default backend" not in result.output


class TestStatusAndHealthReportTlsBlockedDistinctly:
    """Regression guard: `status` and `health` used to report a cert-
    blocked-but-alive server identically to a genuinely stopped one --
    the same defect class fixed in `launch`/`restart`, just left in two
    more call sites."""

    def test_status_reports_unknown_not_inactive(self, monkeypatch):
        monkeypatch.setattr(lifecycle, "_read_pid", lambda: 12345)
        monkeypatch.setattr(lifecycle, "_probe_health", lambda url, timeout=3: (None, True))
        monkeypatch.setattr(lifecycle, "_log_file", lambda: mock.MagicMock(exists=lambda: False))

        runner = CliRunner()
        result = runner.invoke(lifecycle.status, [])

        assert result.exit_code == 0, result.output
        assert "UNKNOWN" in result.output
        assert "INACTIVE" not in result.output
        assert "could not be verified" in result.output

    def test_health_cmd_reports_tls_block_not_unreachable(self, monkeypatch):
        monkeypatch.setattr(lifecycle, "_probe_health", lambda url, timeout=3: (None, True))

        runner = CliRunner()
        result = runner.invoke(lifecycle.health_cmd, [])

        assert result.exit_code == 1
        assert "could not be verified" in result.output
        assert "not reachable" not in result.output


class TestStopReportsTlsBlockedDistinctly:
    """Same defect class again: `stop`'s no-managed-PID fallback used to
    say "Server is not running" for a cert-blocked-but-alive server."""

    def test_does_not_claim_server_is_not_running(self, monkeypatch):
        monkeypatch.setattr(lifecycle, "_read_pid", lambda: None)
        monkeypatch.setattr(lifecycle, "_find_process_on_port", lambda port: None)
        monkeypatch.setattr(lifecycle, "_probe_health", lambda url, timeout=3: (None, True))

        runner = CliRunner()
        result = runner.invoke(lifecycle.stop, [])

        assert result.exit_code == 0, result.output
        assert "could not be verified" in result.output
        assert "is not running" not in result.output
