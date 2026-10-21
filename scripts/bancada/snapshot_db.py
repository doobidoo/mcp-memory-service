#!/usr/bin/env python3
"""Cópia consistente de um banco SQLite do memory-service (respeita WAL).

`cp` de um banco em modo WAL pode perder transações que ainda estão no -wal.
A API de backup do sqlite3 produz uma cópia consistente com o banco vivo, sem
parar o serviço.

Uso:
    python3 scripts/bancada/snapshot_db.py ~/local-data/mcp/sqlite_vec.db /tmp/mms-snap.db

Verifica ao final: contagem de memórias vivas e integridade (`PRAGMA integrity_check`).
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys


def snapshot(source: str, dest: str) -> dict:
    if not os.path.exists(source):
        raise SystemExit(f"origem não existe: {source}")
    if os.path.exists(dest):
        raise SystemExit(f"destino já existe (apague antes): {dest}")

    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()

    conn = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        memories = conn.execute(
            "SELECT count(*) FROM memories WHERE deleted_at IS NULL"
        ).fetchone()[0]
        beliefs = conn.execute(
            "SELECT count(*) FROM beliefs WHERE status='active' AND confidence >= 0.35"
        ).fetchone()[0]
        events = conn.execute("SELECT count(*) FROM sync_events").fetchone()[0]
    finally:
        conn.close()

    return {
        "dest": dest,
        "size_mb": round(os.path.getsize(dest) / 1_048_576, 1),
        "integrity": integrity,
        "memories": memories,
        "active_beliefs": beliefs,
        "sync_events": events,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cópia consistente de um banco SQLite (respeita WAL)."
    )
    ap.add_argument(
        "source", help="banco de origem (ex.: ~/local-data/mcp/sqlite_vec.db)"
    )
    ap.add_argument("dest", help="arquivo de destino (precisa não existir)")
    args = ap.parse_args()

    info = snapshot(os.path.expanduser(args.source), os.path.expanduser(args.dest))
    print(
        f"snapshot: {info['dest']} ({info['size_mb']} MB) · integridade={info['integrity']}\n"
        f"  memórias vivas={info['memories']} · crenças ativas>=0.35={info['active_beliefs']} · "
        f"eventos de sync={info['sync_events']}"
    )
    if info["integrity"] != "ok":
        print("ERRO: integridade do snapshot não é 'ok'", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
