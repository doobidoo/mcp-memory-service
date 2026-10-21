#!/usr/bin/env python3
"""Mede a latência de `memory_store` numa instância efêmera e isolada.

Sobe um memory-server próprio (porta escolhida por você), apontando para um
snapshot do banco, mede N gravações e mata o processo. A produção nunca é tocada.

Uso típico (medir o efeito do NLI no caminho de escrita):

    # 1. snapshot consistente do acervo real
    python3 scripts/bancada/snapshot_db.py ~/local-data/mcp/sqlite_vec.db /tmp/mms-snap.db

    # 2. com o NLI ligado no store (precisa das credenciais do harvest)
    python3 scripts/bancada/medir_store.py --db /tmp/mms-snap.db --nli on \
        --env-file ~/dtp/ai-configs/services/env/memory-service.env --reps 20

    # 3. com o NLI desligado no store (mesma base, mesma máquina)
    python3 scripts/bancada/medir_store.py --db /tmp/mms-snap.db --nli off --reps 20

Regra de publicação: divulgue agregados (n, p50, p95, máx) e o comando usado;
nunca o conteúdo das memórias do corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_server() -> str:
    """Worktrees não têm venv próprio; usa o do checkout principal como fallback."""
    candidates = [
        os.path.join(REPO, ".venv", "bin", "memory-server"),
        os.path.expanduser("~/git/mms/mms/.venv/bin/memory-server"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise SystemExit(
        f"memory-server não encontrado (procurei: {', '.join(candidates)})"
    )


def free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit("nenhuma porta livre encontrada")


def load_env_file(path: str) -> dict:
    env = {}
    with open(os.path.expanduser(path)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.split("  #")[0].strip()
    return env


def wait_health(port: int, timeout: float = 90.0) -> float:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as r:
                if r.status == 200:
                    return time.monotonic() - started
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise SystemExit(f"serviço não respondeu em {timeout:.0f}s na porta {port}")


TEMAS = [
    "kubernetes",
    "fotossíntese",
    "orçamento público",
    "compilador de C",
    "receita de pão",
    "turbina eólica",
    "protocolo DNS",
    "aquarela",
    "estoque de armazém",
    "sonata para violino",
    "clima de montanha",
    "circuito elétrico",
    "espécie de pássaro",
    "contrato de aluguel",
    "maré alta",
    "tecido de algodão",
    "motor a diesel",
    "jardim vertical",
]


class DuplicateContent(Exception):
    """O servidor recusou por dedup semântica: gere outro texto e tente de novo."""


PALAVRAS = [
    "abacaxi",
    "zebra",
    "cobre",
    "relâmpago",
    "sinfonia",
    "basalto",
    "girassol",
    "xadrez",
    "enxada",
    "neblina",
    "pinguim",
    "carvalho",
    "trombone",
    "vulcão",
    "algodão",
    "esmalte",
    "raposa",
    "cristal",
    "moinho",
    "orquídea",
    "tinteiro",
    "bússola",
    "cacto",
    "fole",
    "marfim",
    "gondola",
    "relva",
    "safira",
    "tambor",
    "veludo",
    "âncora",
    "cipreste",
]


def gerar_texto(i: int, rep: int) -> str:
    """Texto único também do ponto de vista semântico — template fixo cai na dedup.

    A dedup do serviço compara embeddings: um texto com o mesmo formato e um uuid
    diferente ainda é "semanticamente similar" ao da rodada anterior. Cada gravação
    recebe um vocabulário sorteado e um número de série para não colidir.
    """
    tema = TEMAS[i % len(TEMAS)]
    sortidas = ", ".join(random.sample(PALAVRAS, 4))
    return (
        f"Registro {rep} da bancada: {tema}, envolvendo {sortidas}; "
        f"observação {uuid.uuid4().hex[:12]} série {random.randint(10**6, 10**7)}."
    )


def store_once(port: int, text: str) -> float:
    payload = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "memory_store",
                "arguments": {"content": text, "memory_type": "observation"},
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/mcp",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "X-Agent-ID": "bancada",
        },
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = r.read().decode(errors="replace")
    elapsed = time.monotonic() - started
    if "Duplicate content detected" in body:
        raise DuplicateContent(body[:200])
    if '"isError":true' in body.replace(" ", ""):
        raise SystemExit(f"o servidor devolveu erro: {body[:300]}")
    return elapsed


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mede a latência de memory_store numa instância efêmera."
    )
    ap.add_argument(
        "--db", required=True, help="snapshot do banco (ver snapshot_db.py)"
    )
    ap.add_argument(
        "--reps", type=int, default=20, help="gravações medidas (default 20)"
    )
    ap.add_argument(
        "--port", type=int, default=8399, help="porta preferida (default 8399)"
    )
    ap.add_argument(
        "--nli", choices=["on", "off"], default="off", help="MCP_NLI_ON_STORE"
    )
    ap.add_argument(
        "--env-file",
        nargs="+",
        help=(
            "um ou mais arquivos KEY=VALUE aplicados em ordem (ex.: o env base do serviço "
            "e depois o override do host). A bancada força porta, banco e MCP_NLI_ON_STORE."
        ),
    )
    ap.add_argument("--json", help="grava o resultado em JSON neste arquivo")
    ap.add_argument("--label", default="", help="rótulo da medição no relatório")
    args = ap.parse_args()

    db = os.path.expanduser(args.db)
    if not os.path.exists(db):
        raise SystemExit(
            f"snapshot não existe: {db} (rode scripts/bancada/snapshot_db.py)"
        )
    server = find_server()

    port = free_port(args.port)
    # O snapshot é imutável: cada execução grava numa cópia de trabalho e a descarta.
    work_db = f"{db}.work-{port}.db"
    shutil.copy2(db, work_db)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", ""),
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_MEMORY_SQLITE_PATH": work_db,
        "MCP_MEMORY_USE_ONNX": "1",
        "MCP_EMBEDDING_MODEL": "paraphrase-multilingual-MiniLM-L12-v2",
        "MCP_SSE_HOST": "127.0.0.1",
        "MCP_SSE_PORT": str(port),
        "MCP_STREAMABLE_HTTP_MODE": "1",
        "MCP_NLI_ON_STORE": "true" if args.nli == "on" else "false",
    }
    if args.env_file:
        for path in args.env_file:
            env.update(load_env_file(path))
        env["MCP_SSE_PORT"] = str(port)  # a porta é sempre a desta bancada
        env["MCP_MEMORY_SQLITE_PATH"] = work_db
        # O arquivo de env pode trazer a flag com outro valor: a bancada manda.
        env["MCP_NLI_ON_STORE"] = "true" if args.nli == "on" else "false"

    log_path = f"/tmp/bancada-medir-store-{port}.log"
    times: list[float] = []
    startup = 0.0
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [server],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            startup = wait_health(port)
            attempts = 0
            while len(times) < args.reps:
                attempts += 1
                if attempts > args.reps * 4:
                    raise SystemExit(
                        "dedup semântica recusou demais: troque o snapshot ou os textos"
                    )
                try:
                    elapsed = store_once(port, gerar_texto(attempts, len(times) + 1))
                except DuplicateContent:
                    print("  (dedup semântica — gerando outro texto)", flush=True)
                    continue
                times.append(elapsed)
                print(f"  [{len(times):2d}/{args.reps}] {elapsed:7.3f}s", flush=True)
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            proc.wait(timeout=30)
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(work_db + suffix)
                except FileNotFoundError:
                    pass

    result = {
        "label": args.label,
        "db": db,
        "nli_on_store": args.nli,
        "nli_effective": env["MCP_NLI_ON_STORE"],
        "nli_backend": env.get("MCP_NLI_BACKEND", "(default)"),
        "env_files": args.env_file or [],
        "port": port,
        "reps": len(times),
        "startup_s": round(startup, 2),
        "p50_s": round(statistics.median(times), 3),
        "p95_s": round(pct(times, 0.95), 3),
        "max_s": round(max(times), 3),
        "mean_s": round(statistics.fmean(times), 3),
        "log": log_path,
    }
    print(
        f"\n{args.label or 'resultado'}: n={result['reps']} · p50={result['p50_s']}s · "
        f"p95={result['p95_s']}s · máx={result['max_s']}s · servidor no ar em {result['startup_s']}s"
    )
    if args.json:
        with open(os.path.expanduser(args.json), "w") as f:
            json.dump(result, f, indent=2)
        print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
