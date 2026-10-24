# Apple Foundation Models as a local LLM provider

macOS 27 ships Apple's on-device language model together with a command-line tool,
`/usr/bin/fm`. Its `fm serve` subcommand runs an OpenAI-compatible chat completions
server. You do not download anything, and you do not need an account or an API key.
mcp-memory-service can use that server wherever it already accepts an
OpenAI-compatible endpoint:

- the quality scorer,
- the harvest rewriter and classifier,
- the LLM backend of the NLI contradiction check,
- search summarization.

Embeddings are not covered. The framework has no embeddings API, so embeddings stay
on the default ONNX model (CoreML on Apple Silicon) or on an external
`/v1/embeddings` server (see [external-embeddings.md](../deployment/external-embeddings.md)).

## Requirements

- macOS 27 on Apple Silicon.
- Apple Intelligence turned on.
- mcp-memory-service with #1666 or later. Earlier versions omit `"stream": false`, and
  `fm serve` streams by default, so their requests fail and silently fall back to the
  next provider.

Check that the model is available:

```bash
fm available          # "System model available"
```

## Start the server

```bash
fm serve --port 1976  # binds 127.0.0.1; model id is "system"
curl -s http://127.0.0.1:1976/health
```

The server provides `GET /health`, `GET /v1/models` and `POST /v1/chat/completions`.
To start it at login, wrap the same command in a launchd agent with `KeepAlive`.

## Configuration

Add these to `.env` and run `memory restart`.

Quality scoring:

```bash
MCP_QUALITY_AI_PROVIDER=openai-compatible
MCP_QUALITY_AI_BASE_URL=http://127.0.0.1:1976/v1
MCP_QUALITY_AI_MODEL=system
```

Harvest, NLI and search summarization share one provider chain:

```bash
HARVEST_LLM_PROVIDERS=apple
HARVEST_LLM_APPLE_BASE_URL=http://127.0.0.1:1976/v1
HARVEST_LLM_APPLE_MODEL=system
MCP_NLI_BACKEND=llm            # optional: LLM contradiction check
MCP_SEARCH_SUMMARIZE_ENABLED=true  # optional: allow search summaries (off by default)
```

Search summaries need both switches. `MCP_SEARCH_SUMMARIZE_ENABLED=true` permits them,
and each `memory_search` call must also pass `summarize=true`. Without the second one,
search returns ordinary results.

You can list a second provider after `apple`, for example
`HARVEST_LLM_PROVIDERS=apple,groq`. The chain falls through to it when the local
model fails.

## What it is good at

The model is small (about 3B parameters) and its context window is small too. Apple
documents 4096 tokens per session. Every call here is a fresh single-turn request, so
the limit applies per call, but long inputs still do not fit.

The following was measured on an M-series Mac with macOS 27.0.1, against ten sample
memories and five NLI pairs. One server served all calls, sequentially.

| Task | Result | Latency per call | Recommendation |
|------|--------|------------------|----------------|
| NLI contradiction check | 5 of 5 correct | 0.3-0.7 s | Good fit |
| Harvest classifier (keep/drop, strict JSON) | Every response parsed. Dropped all 3 noise fragments, but also 2 of 7 real decisions or bugs | 0.6-2.0 s | Usable; expect some false drops |
| Quality scoring (0-1) | Always returned a parseable number. Scores were weakly differentiated: most samples got 0.2 | about 0.3 s | Keep the default local ONNX scorer |
| Harvest rewriter (1-2 sentence insight) | Often answered in the wrong language: English input came back in Portuguese | 0.7-1.3 s | Not recommended for now |
| Embeddings | Not available | n/a | Use ONNX or an external embeddings server |

These numbers come from a small sample. Treat them as a starting point, not as a
benchmark.

## Limits

- Apple Silicon and macOS 27 only. Linux and Windows users keep Ollama, vLLM or MLX-LM
  as their OpenAI-compatible provider.
- The weights are closed and the model can change with an OS update. Re-check
  behaviour after major macOS updates.
- `response_format: {"type": "json_object"}` returns HTTP 400. The server only accepts
  `json_schema`. mcp-memory-service does not send either one; JSON comes from the
  prompt.
