"""Regressão: o migrador de embeddings não pode perder a partition key do vec0.

Contexto do bug: `scripts/maintenance/migrate_embeddings.py` derruba e recria a tabela
vec0 `memory_embeddings` com DDL fixa, sem a coluna de partição que o schema atual usa
(`store TEXT partition key`). Depois da migração, toda consulta filtrada por store
falha com `no such column: store`, e os modos semantic/ranked/exact devolvem zero —
enquanto o próprio script reporta sucesso (integridade e KNN passam).

Este teste exercita o caminho real: banco temporário com partition key, servidor de
embeddings local de mentira, script executado como subprocesso, e verificação do DDL
resultante e de uma consulta filtrada por store.

Deve FALHAR em `upstream/main` (RED) e passar com a correção (GREEN).
"""

from __future__ import annotations

import json
import sqlite3
import struct
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sqlite_vec = pytest.importorskip("sqlite_vec")

DIM = 384
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "migrate_embeddings.py"


class _Stub(BaseHTTPRequestHandler):
    """API OpenAI-compatible mínima: devolve vetores determinísticos."""

    def log_message(self, *a):  # silencia
        return

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        entradas = req.get("input") or []
        if isinstance(entradas, str):
            entradas = [entradas]
        data = [{"object": "embedding", "index": i,
                 "embedding": [0.01 * (i + 1)] * DIM} for i in range(len(entradas))]
        corpo = json.dumps({"object": "list", "data": data, "model": "stub"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)


@pytest.fixture()
def db_com_partition_key(tmp_path):
    db = tmp_path / "sqlite_vec.db"
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        """
        CREATE TABLE memories (
            rowid INTEGER PRIMARY KEY,
            content_hash TEXT NOT NULL,
            content TEXT NOT NULL,
            store TEXT NOT NULL DEFAULT 'default',
            deleted_at REAL,
            embedding_pending INTEGER DEFAULT 0
        )
        """
    )
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        """
        CREATE VIRTUAL TABLE memory_embeddings USING vec0(
            content_embedding FLOAT[384] distance_metric=cosine,
            store TEXT partition key
        )
        """
    )
    conn.execute(
        "INSERT INTO memories (rowid, content_hash, content, store) VALUES (1, ?, ?, 'default')",
        ("hash-1", "memoria de teste sobre partition key"),
    )
    conn.execute(
        "INSERT INTO memories (rowid, content_hash, content, store) VALUES (2, ?, ?, 'mir')",
        ("hash-2", "memoria em outro store"),
    )
    vetor = struct.pack(f"{DIM}f", *([0.02] * DIM))
    conn.execute("INSERT INTO memory_embeddings (rowid, content_embedding, store) VALUES (1, ?, 'default')", (vetor,))
    conn.execute("INSERT INTO memory_embeddings (rowid, content_embedding, store) VALUES (2, ?, 'mir')", (vetor,))
    conn.commit()
    conn.close()
    return db


def test_migracao_preserva_partition_key(db_com_partition_key):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1/embeddings"
    try:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--url", url, "--model", "stub-model",
             "--db-path", str(db_com_partition_key), "--batch-size", "2"],
            capture_output=True, text=True, timeout=180,
        )
    finally:
        srv.shutdown()

    assert proc.returncode == 0, f"script falhou:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"

    conn = sqlite3.connect(db_com_partition_key)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    try:
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'memory_embeddings'"
        ).fetchone()[0]
        assert "partition key" in ddl.lower(), (
            "o migrador recriou memory_embeddings sem a partition key; "
            f"DDL resultante: {ddl}"
        )

        # A consulta filtrada por store tem de continuar funcionando depois da migração.
        linhas = conn.execute(
            "SELECT rowid FROM memory_embeddings WHERE store = 'mir'"
        ).fetchall()
        assert [r[0] for r in linhas] == [2], f"esperava a memória do store 'mir'; veio {linhas}"
    finally:
        conn.close()
