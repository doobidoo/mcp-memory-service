"""Regression tests for the local quality-gate LLM helper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HELPER = Path(__file__).parents[2] / "scripts" / "pr" / "lib" / "llm_prompt.py"
GATE = HELPER.parents[1] / "quality_gate.sh"


class _ModelHandler(BaseHTTPRequestHandler):
    """Serve two models: the first fails chat requests and the second works."""

    models = ("broken", "working")

    def do_GET(self) -> None:
        if self.path != "/v1/models":
            self.send_error(404)
            return
        self._send_json(200, {"data": [{"id": model} for model in self.models]})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        if payload["model"] == "broken":
            self._send_json(507, {"error": "insufficient storage"})
            return
        self._send_json(
            200,
            {"choices": [{"message": {"content": "READY"}}]},
        )

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def model_endpoint() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_resolve_model_falls_back_to_next_advertised_model(model_endpoint: str) -> None:
    env = os.environ | {"MCP_QUALITY_LLM_URL": model_endpoint}
    env.pop("MCP_QUALITY_LLM_MODEL", None)

    result = subprocess.run(
        [sys.executable, str(HELPER), "--resolve-model"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "working"


def test_resolve_model_respects_explicit_model(model_endpoint: str) -> None:
    env = os.environ | {
        "MCP_QUALITY_LLM_URL": model_endpoint,
        "MCP_QUALITY_LLM_MODEL": "broken",
    }

    result = subprocess.run(
        [sys.executable, str(HELPER), "--resolve-model"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 3
    assert "broken" in result.stderr


@pytest.mark.parametrize(
    "configured_model", ["broken", None], ids=["explicit-model", "no-usable-models"]
)
def test_quality_gate_preserves_model_failure(
    model_endpoint: str,
    configured_model: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env = os.environ | {
        "MCP_QUALITY_LLM": "local",
        "MCP_QUALITY_LLM_URL": model_endpoint,
        "MCP_QUALITY_LLM_TIMEOUT": "2",
    }
    if configured_model is None:
        env.pop("MCP_QUALITY_LLM_MODEL", None)
        monkeypatch.setattr(_ModelHandler, "models", ("broken",))
    else:
        env["MCP_QUALITY_LLM_MODEL"] = configured_model

    result = subprocess.run(
        ["bash", str(GATE), "--staged"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 3, result.stdout + result.stderr
    assert "broken: HTTP Error 507: Insufficient Storage" in result.stderr
    assert "no usable local analysis model" in result.stdout
    assert "Skipped, NOT passed" in result.stdout
    assert "no local analysis model reachable" not in result.stdout
    assert "Start the local endpoint" not in result.stdout


class _KeepaliveHandler(BaseHTTPRequestHandler):
    """Answer like a busy omlx: headers at once, then a space every 0.2s, no body."""

    def do_POST(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        try:
            for _ in range(200):
                self.wfile.write(b" ")
                self.wfile.flush()
                time.sleep(0.2)
        except OSError:
            pass

    def log_message(self, format: str, *args: object) -> None:
        pass


class _GateHandler(_ModelHandler):
    """One model; answers the readiness probe, then the analysis prompts per `mode`."""

    models = ("m",)
    mode = "clean"
    analysis_calls = 0

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        prompt = json.loads(self.rfile.read(length))["messages"][0]["content"]
        if prompt == "reply with READY":
            self._send_json(200, {"choices": [{"message": {"content": "READY"}}]})
            return
        type(self).analysis_calls += 1
        if self.mode == "fail":
            self._send_json(500, {"error": "busy"})
            return
        self._send_json(200, {"choices": [{"message": {"content": "SECURITY_CLEAN"}}]})


def _serve(handler: type[BaseHTTPRequestHandler]):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


def test_deadline_is_wall_clock_despite_keepalive_bytes() -> None:
    """omlx streams whitespace while a request waits, so a socket read timeout never fires."""
    server, url = _serve(_KeepaliveHandler)
    env = os.environ | {"MCP_QUALITY_LLM_URL": url, "MCP_QUALITY_LLM_MODEL": "m",
                        "MCP_QUALITY_LLM_TIMEOUT": "1"}
    try:
        started = time.monotonic()
        result = subprocess.run([sys.executable, str(HELPER)], input="hi", env=env,
                                capture_output=True, text=True, check=False, timeout=20)
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()
    assert result.returncode == 3, result.stderr
    assert "no reply within 1s" in result.stderr
    assert elapsed < 10


def _staged_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "tests").mkdir()
    (path / "tests" / "test_mod.py").write_text("def test_one():\n    assert 1\n")
    subprocess.run(["git", "-C", str(path), "add", "tests/test_mod.py"], check=True)


def _run_gate(path: Path, url: str) -> subprocess.CompletedProcess:
    env = os.environ | {"MCP_QUALITY_LLM": "local", "MCP_QUALITY_LLM_URL": url,
                        "MCP_QUALITY_LLM_TIMEOUT": "5"}
    env.pop("MCP_QUALITY_LLM_MODEL", None)
    return subprocess.run(["bash", str(GATE), "--staged"], cwd=path, env=env,
                          capture_output=True, text=True, check=False, timeout=60)


def test_failed_analysis_call_is_skipped_not_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_GateHandler, "mode", "fail")
    _staged_repo(tmp_path)
    server, url = _serve(_GateHandler)
    try:
        result = _run_gate(tmp_path, url)
    finally:
        server.shutdown()
        server.server_close()
    assert result.returncode == 3, result.stdout + result.stderr
    assert "Skipped, NOT passed" in result.stdout
    assert "ALL CHECKS PASSED" not in result.stdout


def test_rerun_on_unchanged_files_reuses_cached_replies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_GateHandler, "mode", "clean")
    monkeypatch.setattr(_GateHandler, "analysis_calls", 0)
    _staged_repo(tmp_path)
    server, url = _serve(_GateHandler)
    try:
        first = _run_gate(tmp_path, url)
        calls_after_first = _GateHandler.analysis_calls
        second = _run_gate(tmp_path, url)
    finally:
        server.shutdown()
        server.server_close()
    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert calls_after_first > 0
    assert _GateHandler.analysis_calls == calls_after_first
