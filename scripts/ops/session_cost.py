#!/usr/bin/env python3
"""Mede o custo de contexto das sessões do Hermes (tokens estimados).

Motivação: o histórico da sessão é reenviado a cada turno. Sessões longas
queimam tokens de "gestão" — não de trabalho. Este instrumento mede para
permitir decisão (checkpoint, sessão nova) em vez de intuição.

Uso:
    python scripts/ops/session_cost.py                # top 10 sessões
    python scripts/ops/session_cost.py --session ID   # uma sessão
    python scripts/ops/session_cost.py --top 5 --warn 150000

Estimativa: chars/3.5 (misto PT/EN). É aproximação conservadora; serve para
tendência e para disparar política, não para contabilidade exata de fatura.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

CHARS_PER_TOKEN = 3.5  # misto PT/EN com markdown e código
DB_PADRAO = Path.home() / ".hermes" / "state.db"


def medir(db: Path, session_id: str | None, top: int) -> list[tuple[str, int, int]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        if session_id:
            rows = conn.execute(
                "SELECT session_id, COUNT(*), SUM(LENGTH(COALESCE(content,''))) "
                "FROM messages WHERE session_id = ? GROUP BY session_id",
                (session_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT session_id, COUNT(*), SUM(LENGTH(COALESCE(content,''))) "
                "FROM messages GROUP BY session_id "
                "ORDER BY SUM(LENGTH(COALESCE(content,''))) DESC LIMIT ?",
                (top,),
            ).fetchall()
    finally:
        conn.close()
    return [(str(s), int(n), int(c or 0)) for s, n, c in rows]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=str(DB_PADRAO))
    p.add_argument("--session", default=None)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--warn", type=int, default=150_000, help="limiar de alerta (tokens)")
    args = p.parse_args(argv)

    db = Path(args.db)
    if not db.exists():
        print(f"erro: banco nao encontrado: {db}", file=sys.stderr)
        return 2

    rows = medir(db, args.session, args.top)
    if not rows:
        print("nenhuma sessao encontrada")
        return 0

    print(f"{'sessao':<26} {'msgs':>6} {'chars':>10} {'~tokens':>9}  {'nivel'}")
    for sid, n, c in rows:
        tok = int(c / CHARS_PER_TOKEN)
        nivel = "CRITICO" if tok > 2 * args.warn else ("ATENCAO" if tok > args.warn else "ok")
        print(f"{sid:<26} {n:>6} {c:>10} {tok:>9}  {nivel}")

    total = sum(c for _, _, c in rows)
    print(f"\ntotal medido: {int(total / CHARS_PER_TOKEN)} tokens (~{int(total/CHARS_PER_TOKEN*0.7)}-{int(total/CHARS_PER_TOKEN*1.2)} na faixa)")
    print("politica: ATENCAO -> checkpoint no repo/mms; CRITICO -> sessao nova")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
