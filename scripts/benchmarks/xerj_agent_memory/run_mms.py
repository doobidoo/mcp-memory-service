"""mcp-memory-service adapter for xerj's agent-memory benchmark.

Derived from benchmarks/agent-memory/run_mms.py in https://github.com/xerj-org/xerj
(Apache-2.0, commit 50e4ffd). Changes from the original:

- runs on any OS: temp dir from tempfile instead of /dev/shm;
- a ``nodedup`` config that sets MCP_SEMANTIC_DEDUP_ENABLED=false;
- refuses to run when production settings leaked in from an .env file;
- writes results/<name>__<config>.json into the harness directory, so the
  harness's own score.py scores it next to the published results.

Usage (see README.md):
    python -I run_mms.py --harness <xerj>/benchmarks/agent-memory --config default
    python -I run_mms.py --harness <xerj>/benchmarks/agent-memory --config nodedup
"""
import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from importlib.metadata import version

LEAK_PREFIXES = ("MCP_SYNC", "CLOUDFLARE_", "MCP_HYBRID", "MCP_SEMANTIC_DEDUP")
LEAK_KEYS = ("MCP_API_KEY", "MCP_HTTPS_ENABLED")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--harness", required=True, help="xerj benchmarks/agent-memory directory")
    ap.add_argument("--config", choices=("default", "nodedup"), default="default")
    ap.add_argument("--name", default="mcp-memory-service", help="system name in the result file")
    return ap.parse_args()


def check_leaks(allowed=()):
    """Exit if production settings are in the environment, e.g. from an .env file."""
    leaked = [k for k in os.environ
              if (k.startswith(LEAK_PREFIXES) or k in LEAK_KEYS) and k not in allowed]
    if leaked:
        sys.exit(f"refusing to run, production settings in the environment: {leaked}")


def isolate(workdir, config):
    """Point every path at workdir."""
    check_leaks()
    os.environ.update(
        HOME=workdir,
        MCP_MEMORY_BASE_DIR=workdir,
        MCP_MEMORY_STORAGE_BACKEND="sqlite_vec",
        MCP_MEMORY_SQLITE_PATH=os.path.join(workdir, "mem.db"),
    )
    if config == "nodedup":
        os.environ["MCP_SEMANTIC_DEDUP_ENABLED"] = "false"
    os.chdir(workdir)


def main():
    args = parse_args()
    harness = os.path.abspath(args.harness)
    workdir = tempfile.mkdtemp(prefix="mms_xerj_")
    isolate(workdir, args.config)
    sys.path.insert(0, harness)
    import common  # inline import: xerj harness module, found via --harness
    from mcp_memory_service.services.memory_service import MemoryService  # inline import: after isolate()
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage  # inline import: after isolate()
    # The package loads an .env on import; check again now that it has.
    check_leaks(allowed=("MCP_SEMANTIC_DEDUP_ENABLED",) if args.config == "nodedup" else ())

    loop = asyncio.new_event_loop()
    storage = SqliteVecMemoryStorage(os.path.join(workdir, "mem.db"))
    loop.run_until_complete(storage.initialize())
    assert storage.semantic_dedup_enabled == (args.config == "default")
    service = MemoryService(storage)
    rejected = []

    def add(mem):
        r = loop.run_until_complete(service.store_memory(content=mem["text"]))
        if not r.get("success", False):
            rejected.append({"id": mem["id"], "resp": json.dumps(r, default=str)[:300]})

    def search(question, k):
        r = loop.run_until_complete(service.retrieve_memories(query=question, n_results=k))
        return [m.get("content", "") for m in r.get("memories", [])]

    dedup = "on" if args.config == "default" else "off"
    res = common.run(args.name, args.config, add, search, {
        "version": "mcp-memory-service " + version("mcp-memory-service"),
        "commit": "installed package",
        "embedder": "all-MiniLM-L6-v2 (default, ONNX)", "llm": None, "llm_calls_per_write": 0,
        "notes": f"in-process MemoryService, semantic dedup {dedup}",
    })
    res["meta"]["rejected_writes"] = rejected
    out = os.path.join(harness, "results", f"{args.name}__{args.config}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {out}; rejected writes: {len(rejected)} {[r['id'] for r in rejected]}")
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
