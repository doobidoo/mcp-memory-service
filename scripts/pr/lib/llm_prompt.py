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

Environment:
    MCP_QUALITY_LLM_URL      base URL, default http://127.0.0.1:11437/v1
    MCP_QUALITY_LLM_MODEL    model id; when unset the first advertised model is
                             used, with all remaining models as fallbacks
    MCP_QUALITY_LLM_API_KEY  optional bearer token
    MCP_QUALITY_LLM_TIMEOUT  seconds per request, default 180

Exit codes:
    0  reply printed on stdout
    3  no usable backend (unreachable endpoint, no model, empty reply)
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


def _list_models() -> list[dict]:
    """Return models the endpoint advertises, empty list on failure."""
    try:
        return _request("/models").get("data") or []
    except Exception:  # noqa: BLE001
        return []


def resolve_model() -> tuple[str, list[str]]:
    """Primary model and fallback list.

    When MCP_QUALITY_LLM_MODEL is set, it is the primary with no fallbacks.
    Otherwise the first advertised model is primary and the rest are fallbacks.
    """
    configured = os.environ.get("MCP_QUALITY_LLM_MODEL")
    if configured:
        return configured, []
    listed = _list_models()
    if not listed:
        raise RuntimeError("endpoint lists no models")
    ids = [m["id"] for m in listed]
    return ids[0], ids[1:]


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
        primary, fallbacks = resolve_model()
    except Exception as exc:  # noqa: BLE001
        print(f"llm_prompt: {_base_url()} unusable: {exc}", file=sys.stderr)
        return EXIT_NO_BACKEND

    last_exc: Exception | None = None
    for model in [primary, *fallbacks]:
        try:
            reply = complete(prompt, model)
        except urllib.error.HTTPError:
            raise  # 400 = bad prompt, don't fallback
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
        if reply:
            print(reply)
            return 0

    failed_on = primary if last_exc is None else f"{primary} (+{len(fallbacks)} fallbacks)"
    print(f"llm_prompt: {failed_on} returned empty or failed: {last_exc}", file=sys.stderr)
    return EXIT_NO_BACKEND


if __name__ == "__main__":
    sys.exit(main())
