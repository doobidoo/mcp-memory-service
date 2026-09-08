"""Regression tests for the local quality-gate LLM helper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HELPER = Path(__file__).parents[2] / "scripts" / "pr" / "lib" / "llm_prompt.py"


class _ModelHandler(BaseHTTPRequestHandler):
    """Serve two models: the first fails chat requests and the second works."""

    def do_GET(self) -> None:
        if self.path != "/v1/models":
            self.send_error(404)
            return
        self._send_json(200, {"data": [{"id": "broken"}, {"id": "working"}]})

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
