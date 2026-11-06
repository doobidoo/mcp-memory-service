"""Credentials tab must call apiCall(endpoint, method, data), not fetch options (#1839)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[2] / "src" / "mcp_memory_service" / "web" / "static" / "app.js"


def test_credentials_calls_use_positional_method_and_data():
    """A fetch-style options object as the second arg is what broke Test Connection."""
    source = APP_JS.read_text(encoding="utf-8")
    # Narrow to the two credential methods so other apiCall sites stay free to evolve.
    test_fn = source[
        source.index("async testCloudflareCredentials()"):
        source.index("async saveCloudflareCredentials()")
    ]
    save_fn = source[
        source.index("async saveCloudflareCredentials()"):
        source.index("_showCredResult(type, message)")
    ]
    for label, body in (("test", test_fn), ("save", save_fn)):
        assert "apiCall(" in body, label
        # Fail if a call still passes `{ method: ... }` as the second argument.
        bad = re.search(
            r"apiCall\(\s*['\"][^'\"]+['\"]\s*,\s*\{\s*method\s*:",
            body,
        )
        assert bad is None, f"{label} still passes a fetch-style options object to apiCall"
        assert re.search(
            r"apiCall\(\s*['\"]/config/credentials[^'\"]*['\"]\s*,\s*['\"]POST['\"]\s*,",
            body,
        ), f"{label} must call apiCall(path, 'POST', data)"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_test_credentials_posts_json_body_to_fetch():
    """Exercise the real method against a stubbed fetch (regression for #1839)."""
    source = APP_JS.read_text(encoding="utf-8")
    script = source[
        source.index("class MemoryDashboard {"):
        source.index("// Initialize the application when DOM is ready")
    ]
    script += r"""
const calls = [];
global.fetch = async (url, init = {}) => {
  calls.push({ url, method: init.method, body: init.body });
  return new Response(JSON.stringify({ valid: true, expires_on: null }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
};
global.document = {
  getElementById(id) {
    const values = {
      "cred-api-token": { value: "tok" },
      "cred-account-id": { value: "acc" },
      credTestResult: { style: { display: "none" }, className: "", textContent: "" },
      credTestBtn: { disabled: false, textContent: "" },
      credSaveBtn: { disabled: true },
    };
    return values[id] || null;
  },
};
const app = Object.create(MemoryDashboard.prototype);
app.authState = {};
app.apiBase = "/api";
app.t = (_k, fallback) => fallback;
app._showCredResult = () => {};
app._updateCredSaveBtn = () => {};
app.handleAuthFailure = () => { throw new Error("auth"); };
app.testCloudflareCredentials().then(() => {
  console.log(JSON.stringify({ calls, tested: app._credTestedToken }));
}).catch((err) => {
  console.log(JSON.stringify({ error: String(err) }));
});
"""
    result = subprocess.run(["node", "-"], input=script, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "error" not in payload, payload
    assert len(payload["calls"]) == 1
    call = payload["calls"][0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/api/config/credentials/test")
    assert json.loads(call["body"]) == {"api_token": "tok", "account_id": "acc"}
    assert payload["tested"] == "tok"
