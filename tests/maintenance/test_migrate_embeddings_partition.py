"""Regressão do migrador de embeddings: partition key e largura do vetor.

Cobre três coisas que o script de manutenção precisa acertar:

1. bancos atuais (``memory_embeddings`` com ``store TEXT partition key``) — a
   migração não pode perder a coluna de partição, senão toda busca filtrada por
   store morre com ``no such column: store`` enquanto o script reporta sucesso;
2. troca de modelo com largura diferente — a DDL recriada tem de refletir a nova
   dimensão, não a antiga;
3. bancos antigos (sem a coluna ``memories.store`` e sem partition key) — a
   migração continua funcionando e não tenta gravar uma coluna que não existe.

Os dois primeiros casos são RED em ``upstream/main``.
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

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "migrate_embeddings.py"
TARGET_DIM = 8  # diferente da origem de propósito: pega falha ao substituir a largura


class _Stub(BaseHTTPRequestHandler):
    """API OpenAI-compatible mínima: devolve vetores determinísticos."""

    dim = TARGET_DIM

    def log_message(self, *args):
        return

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        entradas = req.get("input") or []
        if isinstance(entradas, str):
            entradas = [entradas]
        data = [
            {"object": "embedding", "index": i,
             "embedding": [0.01 * (i + 1)] * self.dim}
            for i in range(len(entradas))
        ]
        corpo = json.dumps({"object": "list", "data": data, "model": "stub"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)


def _criar_db(path: Path, src_dim: int, com_store: bool) -> None:
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    if com_store:
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
    else:
        conn.execute(
            """
            CREATE TABLE memories (
                rowid INTEGER PRIMARY KEY,
                content_hash TEXT NOT NULL,
                content TEXT NOT NULL,
                deleted_at REAL,
                embedding_pending INTEGER DEFAULT 0
            )
            """
        )
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")

    if com_store:
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE memory_embeddings USING vec0(
                content_embedding FLOAT[{src_dim}] distance_metric=cosine,
                store TEXT partition key
            )
            """
        )
    else:
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE memory_embeddings USING vec0(
                content_embedding FLOAT[{src_dim}] distance_metric=cosine
            )
            """
        )

    vetor = struct.pack(f"{src_dim}f", *([0.02] * src_dim))
    if com_store:
        conn.execute(
            "INSERT INTO memories (rowid, content_hash, content, store) VALUES (1, ?, ?, 'default')",
            ("hash-1", "memoria de teste sobre partition key"),
        )
        conn.execute(
            "INSERT INTO memories (rowid, content_hash, content, store) VALUES (2, ?, ?, 'mir')",
            ("hash-2", "memoria em outro store"),
        )
        conn.execute(
            "INSERT INTO memory_embeddings (rowid, content_embedding, store) VALUES (1, ?, 'default')",
            (vetor,),
        )
        conn.execute(
            "INSERT INTO memory_embeddings (rowid, content_embedding, store) VALUES (2, ?, 'mir')",
            (vetor,),
        )
    else:
        conn.execute(
            "INSERT INTO memories (rowid, content_hash, content) VALUES (1, ?, ?)",
            ("hash-1", "memoria de banco antigo"),
        )
        conn.execute(
            "INSERT INTO memory_embeddings (rowid, content_embedding) VALUES (1, ?)",
            (vetor,),
        )
    conn.commit()
    conn.close()


def _rodar_migracao(db: Path) -> subprocess.CompletedProcess:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1/embeddings"
    try:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--url", url, "--model", "stub-model",
             "--db-path", str(db), "--batch-size", "2", "--yes"],
            capture_output=True, text=True, timeout=180,
        )
    finally:
        srv.shutdown()


def _ddl(db: Path) -> str:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'memory_embeddings'"
        ).fetchone()[0]
    finally:
        conn.close()


def test_migracao_preserva_partition_key(tmp_path):
    db = tmp_path / "sqlite_vec.db"
    _criar_db(db, src_dim=4, com_store=True)

    proc = _rodar_migracao(db)
    assert proc.returncode == 0, f"script falhou:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"

    ddl = _ddl(db)
    assert "partition key" in ddl.lower(), (
        "o migrador recriou memory_embeddings sem a partition key; "
        f"DDL resultante: {ddl}"
    )

    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    try:
        linhas = conn.execute(
            "SELECT rowid FROM memory_embeddings WHERE store = 'mir'"
        ).fetchall()
        assert [r[0] for r in linhas] == [2], f"esperava a memória do store 'mir'; veio {linhas}"
    finally:
        conn.close()


def test_migracao_aplica_a_nova_dimensao(tmp_path):
    """Origem com 4 dimensões, alvo com 8: a DDL recriada tem de refletir o alvo."""
    db = tmp_path / "sqlite_vec.db"
    _criar_db(db, src_dim=4, com_store=True)

    proc = _rodar_migracao(db)
    assert proc.returncode == 0, f"script falhou:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"

    ddl = _ddl(db)
    assert f"FLOAT[{TARGET_DIM}]" in ddl.upper(), (
        f"a DDL recriada manteve a dimensão antiga; DDL resultante: {ddl}"
    )


def test_migracao_funciona_em_banco_antigo_sem_store(tmp_path):
    """Banco antigo: sem coluna `store` em memories e sem partition key no vec0."""
    db = tmp_path / "sqlite_vec.db"
    _criar_db(db, src_dim=4, com_store=False)

    proc = _rodar_migracao(db)
    assert proc.returncode == 0, (
        "migração de banco antigo quebrou (provavelmente lendo `store` que não existe):\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    assert "store" not in _ddl(db).lower(), "banco antigo não deve ganhar partition key"


def test_confirm_continue_exige_yes_fora_de_tty():
    """Sem TTY (CI), a checagem de serviço não pode pedir input(): exige --yes."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("migrate_embeddings", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod._confirm_continue(True, False, False) is True      # serviço parado: segue
    assert mod._confirm_continue(False, True, False) is True      # --yes explícito: segue
    assert mod._confirm_continue(False, False, False) is False    # sem TTY e sem --yes: aborta
