# xerj agent-memory benchmark adapter

Runs mcp-memory-service through the agent-memory benchmark that
[xerj](https://github.com/xerj-org/xerj) publishes in `benchmarks/agent-memory/`:
85 memories, 75 questions (25 exact, 25 paraphrase, 15 update, 10 distractor), top 5,
no LLM calls. Results and discussion: [docs/BENCHMARKS.md](../../../docs/BENCHMARKS.md#xerj-agent-memory-benchmark).

The dataset, the run loop (`common.py`) and the scorer (`score.py`) stay in the xerj
repo and are not copied here. `run_mms.py` is derived from xerj's own adapter
(Apache-2.0, commit `50e4ffd`); its docstring lists what changed.

## Run

```bash
git clone https://github.com/xerj-org/xerj /tmp/xerj
git -C /tmp/xerj checkout 50e4ffd
H=/tmp/xerj/benchmarks/agent-memory

python -m venv /tmp/mms-bench && /tmp/mms-bench/bin/pip install "mcp-memory-service[sqlite]"
/tmp/mms-bench/bin/python -I scripts/benchmarks/xerj_agent_memory/run_mms.py --harness "$H" --config default
/tmp/mms-bench/bin/python -I scripts/benchmarks/xerj_agent_memory/run_mms.py --harness "$H" --config nodedup
cd "$H" && python3 score.py
```

- `default` keeps every default, including semantic dedup (`MCP_SEMANTIC_DEDUP_ENABLED=true`).
- `nodedup` sets `MCP_SEMANTIC_DEDUP_ENABLED=false` and changes nothing else.

The script stores into a fresh temp directory and exits if production settings
(`MCP_API_KEY`, `MCP_SYNC_*`, `CLOUDFLARE_*`, `MCP_HYBRID_*`, `MCP_SEMANTIC_DEDUP_*`) are
in the environment, before and after the package import, since the import loads an
`.env` file. Use a throwaway venv as above, not a checkout whose `.env` points at a live
database.
