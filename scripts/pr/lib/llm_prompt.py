#!/usr/bin/env python3
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

"""Send a prompt to a local OpenAI-compatible endpoint and print the reply.

Reads the prompt from stdin, writes the assistant text to stdout. Used by the
shell quality gates so complexity and security analysis run against a local
model instead of a paid API, and so a missing backend is reported as skipped
rather than passed.

When MCP_QUALITY_LLM_MODEL is not set, the helper tries every model the
endpoint advertises (in listed order) until one succeeds.  This avoids a
single broken model (e.g. oMLX returning 507 Insufficient Storage for a
quantisation that cannot load) silently disabling the entire gate.

Environment:
    MCP_QUALITY_LLM_URL      base URL, default http://127.0.0.1:11437/v1
    MCP_QUALITY_LLM_MODEL    model id; if unset, tries all listed models.
                              Pin this when the endpoint hosts a mix of
                              working and broken models (e.g. oMLX 507s).
    MCP_QUALITY_LLM_API_KEY  optional bearer token
    MCP_QUALITY_LLM_TIMEOUT  seconds per request, default 180

Exit codes:
    0  reply printed on stdout (model name on stderr)
    3  no usable backend (unreachable endpoint, no model, all candidates failed)
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:11437/v1"
EXIT_NO_BACKEND = 3
THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _base_url() -> str:
    return os.environ.get("MCP_QUALITY_LLM_URL", DEFAULT_URL).rstrip("/")


def _headers() -> dict:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("MCP_QUALITY_LLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _request(path: str, payload=None):
    """POST payload (or GET when payload is None) and return the parsed body."""
    data = json.dumps(payload).encode() if payload is not None else None
    timeout = float(os.environ.get("MCP_QUALITY_LLM_TIMEOUT", "180"))
    req = urllib.request.Request(_base_url() + path, data=data, headers=_headers())
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def resolve_model() -> list[str]:
    """Return candidate models to try: configured model first, then all listed."""
    configured = os.environ.get("MCP_QUALITY_LLM_MODEL")
    if configured:
        return [configured]
    listed = _request("/models").get("data") or []
    if not listed:
        raise RuntimeError("endpoint lists no models")
    return [m["id"] for m in listed]


def _chat(payload: dict) -> dict:
    """Chat completion, retried without the thinking hint if the server rejects it."""
    try:
        return _request("/chat/completions", payload)
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise
        payload.pop("chat_template_kwargs", None)
        return _request("/chat/completions", payload)


def complete(prompt: str, model: str) -> str:
    """One chat completion, with thinking disabled where the server accepts it."""
    body = _chat({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        # Qwen3 emits <think> blocks that swamp the parseable markers the gates
        # grep for. Servers that reject the hint get a plain retry.
        "chat_template_kwargs": {"enable_thinking": False},
    })
    text = body["choices"][0]["message"].get("content") or ""
    return THINK_BLOCK.sub("", text).strip()


def main() -> int:
    prompt = sys.stdin.read()
    if not prompt.strip():
        print("llm_prompt: empty prompt on stdin", file=sys.stderr)
        return EXIT_NO_BACKEND
    try:
        candidates = resolve_model()
    except Exception as exc:  # noqa: BLE001 - any failure means "no backend"
        print(f"llm_prompt: {_base_url()} unusable: {exc}", file=sys.stderr)
        return EXIT_NO_BACKEND
    last_exc = None
    for model in candidates:
        try:
            reply = complete(prompt, model)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
        if reply:
            print(f"llm_prompt: using model {model}", file=sys.stderr)
            print(reply)
            return 0
    err = last_exc or "empty reply from all candidates"
    print(f"llm_prompt: {_base_url()} unusable: {err}", file=sys.stderr)
    return EXIT_NO_BACKEND


if __name__ == "__main__":
    sys.exit(main())
