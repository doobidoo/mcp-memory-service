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
