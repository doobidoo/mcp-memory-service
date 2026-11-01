"""apiCall() turns a structured error detail into readable text (#1743).

FastAPI answers some failures with an object or a list in ``detail``: the OAuth
scope checks send ``{"error": ..., "error_description": ...}`` and request
validation sends a list of ``{"loc", "msg", "type"}`` entries. ``new Error(detail)``
stringified those to "[object Object]", which every caller then showed in its
toast ("Failed to check for updates: [object Object]"). The real ``apiCall``
runs in Node against a stubbed ``fetch``; the message is what callers display,
and ``err.detail`` must stay the original value for callers that branch on it
(PR #807).
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[2] / "src" / "mcp_memory_service" / "web" / "static" / "app.js"

SCOPE_DETAIL = {"error": "insufficient_scope", "error_description": "Required scope 'admin' not granted"}
VALIDATION_DETAIL = [{"loc": ["body", "query"], "msg": "Field required", "type": "missing"}]

CASES = {
    # status, response body (None = not JSON), expected message
    "oauth_scope_object": (403, {"detail": SCOPE_DETAIL}, "Required scope 'admin' not granted"),
    "error_code_only": (403, {"detail": {"error": "insufficient_scope"}}, "insufficient_scope"),
    "validation_list": (422, {"detail": VALIDATION_DETAIL}, "HTTP 422"),
    "unknown_object": (500, {"detail": {"code": 7}}, "HTTP 500"),
    "nested_description": (403, {"detail": {"error_description": {"lang": "en"}}}, "HTTP 403"),
    "plain_string": (404, {"detail": "Memory not found"}, "Memory not found"),
    "empty_string": (400, {"detail": ""}, "HTTP 400"),
    "no_detail": (502, {}, "HTTP 502"),
    "not_json": (500, None, "HTTP 500"),
}


def _run_api_call(status, body):
    source = APP_JS.read_text(encoding="utf-8")
    # The real class, without bootstrapping the application or its network calls.
    script = source[source.index("class MemoryDashboard {"):source.index("// Initialize the application when DOM is ready")]
    script += f"""
const status = {status};
const body = {json.dumps(body)};
global.fetch = async () => new Response(body === null ? "<html>bad gateway</html>" : JSON.stringify(body),
    {{status, headers: {{"Content-Type": body === null ? "text/html" : "application/json"}}}});
const app = Object.create(MemoryDashboard.prototype);
app.authState = {{}};
app.apiBase = "/api";
app.handleAuthFailure = () => {{ throw new Error("401 path must not run"); }};
app.apiCall("/server/version/check").then(
    () => console.log(JSON.stringify({{resolved: true}})),
    (err) => console.log(JSON.stringify({{message: err.message, status: err.status,
        detail: err.detail === undefined ? "<undefined>" : err.detail}})),
);
"""
    result = subprocess.run(["node", "-"], input=script, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("case", list(CASES))
def test_api_error_message_is_readable_and_detail_is_preserved(case):
    status, body, expected = CASES[case]
    thrown = _run_api_call(status, body)
    assert thrown.get("resolved") is not True, "a failed response must throw"
    assert thrown["message"] == expected
    assert "[object Object]" not in thrown["message"]
    assert thrown["status"] == status
    original = (body or {}).get("detail", "<undefined>")
    assert thrown["detail"] == original, "callers branch on the untouched detail (PR #807)"
