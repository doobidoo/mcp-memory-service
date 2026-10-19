"""Importador read-only do state.db do Hermes Agent.

Lê sessões de ``~/.hermes/state.db`` (somente-leitura), normaliza os turnos
(descartando o payload bruto de tool — só metadados, D1) e envia ao acervo
do memory-service via tool MCP ``memory_store_session`` (JSON-RPC 2.0).

Spec: docs/rfc/planned/spec-hermes-state-db-importer.md (R1-R8, CA1-CA7).
Testes: tests/test_hermes_state_db_importer.py (contrato no docstring).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB = str(Path.home() / ".hermes" / "state.db")
DEFAULT_STATE = str(Path.home() / ".hermes" / "cache" / "mms-hermes-import.json")
DEFAULT_ENDPOINT = "http://127.0.0.1:3202/mcp"
MAX_TURN_CHARS = 2000
MAX_SESSION_CHARS = 15_000
# Teto do RESUMO de mensagens role='tool' (D1): independente de haver
# quebra de linha — saidas de tool (ex.: JSON de uma linha de read_file)
# nao podem vazar para alem disso.
MAX_TOOL_SUMMARY_CHARS = 200
_HEAD_RATIO = 0.4
MIN_USEFUL = 2

# Marcadores que sinalizam falha no resumo de tool (D1: ok|erro).
_ERROR_MARKERS = (
    "error", "erro", "exception", "traceback", "failed", "failure", "fatal",
)


# ---------------------------------------------------------------------------
# Conexão e leitura (R1: estritamente read-only)
# ---------------------------------------------------------------------------

def open_db(path: str) -> sqlite3.Connection:
    """Abre o state.db em modo somente-leitura e valida o schema mínimo.

    Qualquer escrita pela conexão falha com ``sqlite3.OperationalError`` (R1).
    """
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    _check_schema(conn, path)
    return conn


def _check_schema(conn: sqlite3.Connection, path: str) -> None:
    """Pré-checagem de schema: falha claro listando coluna ausente."""
    required: dict[str, set[str]] = {
        "sessions": {"id", "started_at"},
        "messages": {"session_id", "role", "content"},
    }
    for table, cols_req in required.items():
        cur = conn.execute(f"PRAGMA table_info({table})")
        cols = {row[1] for row in cur.fetchall()}
        if not cols:
            raise SystemExit(f"erro: tabela '{table}' ausente em {path}")
        missing = cols_req - cols
        if missing:
            raise SystemExit(
                f"erro: coluna(s) {', '.join(sorted(missing))} "
                f"ausente(s) em '{table}'"
            )


def _rows_as_dicts(cursor) -> list[dict]:
    """Converte linhas do cursor em dicts simples (JSON-serializável)."""
    cols = [c[0] for c in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def select_sessions(conn: sqlite3.Connection, min_useful: int = MIN_USEFUL) -> list[dict]:
    """Sessões elegíveis: >= ``min_useful`` mensagens úteis (user+assistant).

    Mensagem útil = ``role in ('user','assistant') AND active=1 AND compacted=0``
    (R2). Retorna lista de dicts com, no mínimo, a chave ``id`` (R5).
    """
    cur = conn.execute(
        """
        SELECT s.id, s.profile_name, s.cwd, s.model, s.started_at,
               (SELECT COUNT(*) FROM messages m
                 WHERE m.session_id = s.id
                   AND m.role IN ('user', 'assistant')
                   AND m.active = 1 AND m.compacted = 0) AS useful_msgs
        FROM sessions s
        WHERE (SELECT COUNT(*) FROM messages m
                 WHERE m.session_id = s.id
                   AND m.role IN ('user', 'assistant')
                   AND m.active = 1 AND m.compacted = 0) >= ?
        ORDER BY s.id
        """,
        (min_useful,),
    )
    return _rows_as_dicts(cur)


def _fetch_sessions(conn: sqlite3.Connection) -> list[dict]:
    """Todas as sessões (ordem estável por id)."""
    cur = conn.execute(
        "SELECT id, profile_name, cwd, model, started_at FROM sessions ORDER BY id"
    )
    return _rows_as_dicts(cur)


def _fetch_messages(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """Mensagens agrupadas por session_id, em ordem (timestamp, id)."""
    cur = conn.execute(
        "SELECT session_id, role, content, tool_name, active, compacted, timestamp, id "
        "FROM messages ORDER BY timestamp, id"
    )
    out: dict[str, list[dict]] = {}
    for d in _rows_as_dicts(cur):
        out.setdefault(d["session_id"], []).append(d)
    return out


# ---------------------------------------------------------------------------
# Normalização de turnos (R2, D1)
# ---------------------------------------------------------------------------

def _field(mapping, key: str, default=None):
    """Lê ``key`` de dict ou sqlite3.Row de forma defensiva."""
    try:
        return mapping[key]
    except Exception:
        return default


def _detect_error(text: str) -> str:
    """Heurística de sucesso/falha a partir da primeira linha (D1)."""
    low = (text or "").lower()
    return "erro" if any(m in low for m in _ERROR_MARKERS) else "ok"


def _tool_summary(content: str) -> tuple[str, str]:
    """Resumo de tool (D1): devolve ``(sinal, resumo)``.

    Usa a PRIMEIRA linha da saída, normaliza espaços/quebras múltiplas
    (colapsa para um único espaço) e corta o resumo em
    ``MAX_TOOL_SUMMARY_CHARS`` — teto em caracteres independente de
    haver quebra de linha (saídas de tool como read_file são JSON de
    uma única linha; cortar só pela linha deixaria o payload inteiro).
    O sinal ok/erro é detectado na primeira linha ANTES do corte, para
    que o truncamento do resumo nunca altere a heurística de falha.
    """
    first_line = (content or "").split("\n", 1)[0]
    signal = _detect_error(first_line)
    normalized = " ".join(first_line.split())
    return signal, normalized[:MAX_TOOL_SUMMARY_CHARS]


def build_turns(messages, max_turn_chars: int = MAX_TURN_CHARS) -> list[dict]:
    """Normaliza mensagens em turnos ``{role, content}``.

    - ``user``/``assistant``: conteúdo truncado em ``max_turn_chars``.
    - ``tool``: SOMENTE metadados (nome + ok/erro + resumo da 1ª linha
      com teto de ``MAX_TOOL_SUMMARY_CHARS``, D1); o payload completo
      nunca sai.
    - Descarta ``role='session_meta'``, ``compacted=1`` e ``active=0`` (R2).
    """
    turns: list[dict] = []
    for m in messages:
        role = _field(m, "role")
        if role == "session_meta":
            continue
        if _field(m, "compacted"):
            continue
        if not _field(m, "active"):
            continue
        content = _field(m, "content") or ""
        if role in ("user", "assistant"):
            turns.append({"role": role, "content": content[:max_turn_chars]})
        elif role == "tool":
            name = _field(m, "tool_name") or "?"
            signal, summary = _tool_summary(content)
            meta = f"[tool:{name}] {signal} — {summary}"
            turns.append({"role": "tool", "content": meta[:max_turn_chars]})
        # outros roles: descartados (não fazem parte do acervo)
    return turns


def _delta_turns(turns: list[dict], prev_useful: int) -> list[dict]:
    """Turnos novos após a N-ésima mensagem útil já importada (D2)."""
    seen = 0
    out: list[dict] = []
    for t in turns:
        if seen >= prev_useful:
            out.append(t)
        if t.get("role") in ("user", "assistant"):
            seen += 1
    return out


def _apply_session_budget(
    turns: list[dict],
    max_session_chars: int = MAX_SESSION_CHARS,
    head_ratio: float = _HEAD_RATIO,
) -> tuple[list[dict], int]:
    """Limita o payload da sessão a ``max_session_chars`` (G3.1).

    Distribui o orçamento em head (40%) e tail (60%): acumula turnos do
    início até fechar o head e do fim até fechar o tail, descartando o
    meio. Retorna ``(turnos_kept, turnos_descartados)``.
    """
    total = sum(len(t.get("content", "")) for t in turns)
    if total <= max_session_chars:
        return turns, 0

    head_budget = int(max_session_chars * head_ratio)
    tail_budget = max_session_chars - head_budget

    head: list[dict] = []
    used = 0
    for t in turns:
        c = len(t.get("content", ""))
        if used + c > head_budget:
            break
        head.append(t)
        used += c

    tail: list[dict] = []
    used = 0
    for t in reversed(turns):
        c = len(t.get("content", ""))
        if used + c > tail_budget:
            break
        tail.append(t)
        used += c
    tail.reverse()

    kept = head + tail
    return kept, len(turns) - len(kept)


# ---------------------------------------------------------------------------
# Estado local (R6) e idempotência
# ---------------------------------------------------------------------------

def _load_state(path: str) -> dict:
    """Carrega o estado local; loga WARNING se corrompido e reimporta do zero."""
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "state corrompido em %s: %s — reimportando do zero", path, e
            )
            return {}
    return {}


def _save_state(path: str, state: dict) -> None:
    """Persiste o estado local atomicamente (tmp + os.replace, 0600)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, str(p))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _last_hash(useful: list[dict]) -> str:
    """SHA-256 do conteúdo da última mensagem útil (R6)."""
    if not useful:
        return ""
    last = useful[-1].get("content") or ""
    return hashlib.sha256(last.encode("utf-8", "replace")).hexdigest()


def _update_state(state: dict, record: dict) -> None:
    """Registra a sessão importada para o próximo run (R6)."""
    state[record["id"]] = {
        "useful_msgs": record["useful_msgs"],
        "last_hash": record["last_hash"],
        "imported_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Identidade, tags e payload (R3, R4)
# ---------------------------------------------------------------------------

def _slug(cwd: str | None) -> str:
    """Último componente do path de cwd (R4)."""
    c = (cwd or "").strip().rstrip("/")
    return c.split("/")[-1] if c else ""


def _tags_for(session: dict, action: str) -> list[str]:
    """Tags: source:hermes sempre; project:<slug> se cwd; session-delta se delta."""
    tags = ["source:hermes"]
    slug = _slug(session.get("cwd"))
    if slug:
        tags.append(f"project:{slug}")
    if action in ("delta", "updated"):
        tags.append("session-delta")
    return tags


def _iso8601(epoch: float | None) -> str | None:
    """Converte epoch (REAL) para ISO-8601 UTC."""
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        return None


def _metadata_for(session: dict) -> dict:
    """Metadados: agent_id, model, started_at (ISO-8601 UTC), hermes_session_id."""
    return {
        "agent_id": session.get("profile_name") or "hermes",
        "model": session.get("model"),
        "started_at": _iso8601(session.get("started_at")),
        "hermes_session_id": session.get("id"),
    }


def _payload(record: dict) -> dict:
    """Payload enviado ao acervo por sessão (CA1: único ponto de gravação)."""
    return {
        "turns": record["turns"],
        "session_id": record["id"],
        "tags": record["tags"],
        "metadata": record["metadata"],
        "store": record["store"],
    }


# ---------------------------------------------------------------------------
# Planejamento (R5, R6, R7, R8)
# ---------------------------------------------------------------------------

def _collect(conn: sqlite3.Connection, state: dict, min_useful: int = MIN_USEFUL) -> dict:
    """Planeja o import: registros por sessão, descartes e totais."""
    sessions = _fetch_sessions(conn)
    messages_by_session = _fetch_messages(conn)
    records: list[dict] = []
    discards: dict[str, int] = {}

    for s in sessions:
        sid = s["id"]
        msgs = messages_by_session.get(sid, [])
        useful = [
            m for m in msgs
            if m.get("role") in ("user", "assistant")
            and m.get("active") == 1
            and m.get("compacted") == 0
        ]
        if len(useful) < min_useful:
            discards["insufficient_content"] = (
                discards.get("insufficient_content", 0) + 1
            )
            continue

        turns = build_turns(msgs)
        last_hash = _last_hash(useful)
        prev = state.get(sid)

        if prev is None:
            # Primeira vez: importa o conteúdo completo
            action = "new"
            sent_turns, dropped = _apply_session_budget(turns)
        elif prev.get("useful_msgs") == len(useful) \
                and prev.get("last_hash") == last_hash:
            # Nada mudou: skip
            action, sent_turns, dropped = "skip", [], 0
        elif len(useful) < prev.get("useful_msgs", 0):
            # Sessão encolheu: no-op, não regravar
            action, sent_turns, dropped = "noop", [], 0
        elif prev.get("useful_msgs") == len(useful) \
                and prev.get("last_hash") != last_hash:
            # Última mensagem útil editada (mesma contagem): reimporta
            action = "updated"
            sent_turns, dropped = _apply_session_budget(turns)
        else:
            # len(useful) > prev.useful_msgs: delta normal
            action = "delta"
            sent_turns, dropped = _apply_session_budget(
                _delta_turns(turns, prev.get("useful_msgs", 0))
            )

        records.append({
            "id": sid,
            "action": action,
            "useful_msgs": len(useful),
            "last_hash": last_hash,
            "turns": sent_turns,
            "store": "default",
            "tags": _tags_for(s, action),
            "metadata": _metadata_for(s),
            "truncated": dropped > 0,
            "turns_dropped": dropped,
            "chars_to_send": sum(len(t.get("content", "")) for t in sent_turns),
        })

    totals = {
        "sessions_total": len(sessions),
        "eligible": len(records),
        "to_store": sum(
            1 for r in records if r["action"] in ("new", "delta", "updated")
        ),
        "skipped": sum(1 for r in records if r["action"] == "skip"),
        "discarded": sum(discards.values()),
        "sessions_truncated": sum(1 for r in records if r["truncated"]),
        "chars_to_send": sum(r["chars_to_send"] for r in records),
    }
    return {"records": records, "discards": discards, "totals": totals}


def _to_report(collected: dict, mode: str) -> dict:
    """Relatório auditável (R8): sessões, descartes e totais por destino."""
    sessions = [
        {
            "id": r["id"],
            "action": r["action"],
            "useful_msgs": r["useful_msgs"],
            "store": r["store"],
            "tags": r["tags"],
            "agent_id": r["metadata"]["agent_id"],
            "chars_to_send": r["chars_to_send"],
            "truncated": r["truncated"],
            "turns_dropped": r["turns_dropped"],
        }
        for r in collected["records"]
    ]
    totals = dict(collected["totals"])
    totals["mode"] = mode
    return {"sessions": sessions, "discards": collected["discards"], "totals": totals}


def plan_import(conn: sqlite3.Connection, state: dict, min_useful: int = MIN_USEFUL) -> dict:
    """Relatório de import sem gravar (R5/R7/R8).

    Retorna ``{"sessions": [...], "discards": {motivo: n}, "totals": {...}}``.
    Sessões com <2 mensagens úteis entram em ``discards['insufficient_content']``.
    """
    return _to_report(_collect(conn, state, min_useful=min_useful), mode="plan")


# ---------------------------------------------------------------------------
# Transporte MCP (memory_store_session)
# ---------------------------------------------------------------------------

def _parse_sse(raw: str):
    """Extrai o JSON-RPC relevante de uma resposta SSE.

    Seleciona o evento ``data:`` cujo JSON tem ``result`` (resultado da
    chamada); fallback: primeiro com ``id`` (pode ser erro). Corpo vazio
    ou truncado levanta ``ValueError``.
    """
    candidates: list = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data = line[len("data:"):].strip()
            if data:
                try:
                    candidates.append(json.loads(data))
                except json.JSONDecodeError:
                    continue

    if not candidates:
        stripped = raw.strip()
        if not stripped:
            raise ValueError("corpo de resposta vazio — nada para processar")
        try:
            candidates = [json.loads(stripped)]
        except json.JSONDecodeError:
            raise ValueError(
                f"resposta não é JSON válido nem SSE: {stripped[:200]!r}"
            ) from None

    # Preferir evento com 'result' (sucesso)
    for obj in candidates:
        if isinstance(obj, dict) and "result" in obj:
            return obj
    # Fallback: primeiro com 'id' (pode ser erro JSON-RPC)
    for obj in candidates:
        if isinstance(obj, dict) and "id" in obj:
            return obj
    raise ValueError(
        f"nenhum evento JSON-RPC com 'result' ou 'id' na resposta: {raw[:200]!r}"
    )


def store_session(payload: dict, endpoint: str = DEFAULT_ENDPOINT, timeout: int = 60):
    """Envia o payload ao acervo via tool MCP (JSON-RPC 2.0 sobre SSE).

    Levanta exceção se a resposta contiver ``{"error": {...}}`` ou não
    for JSON/SSE válido. PONTO DE INJEÇÃO: ``main()`` chama apenas esta
    função para gravar (CA1). Autenticação anônima neste ambiente
    (token opcional via ``MCP_API_KEY``).
    """
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "memory_store_session", "arguments": payload},
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json, text/event-stream")
    token = os.environ.get("MCP_API_KEY")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
    result = _parse_sse(raw)
    if isinstance(result, dict) and "error" in result:
        err = result["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise RuntimeError(f"erro JSON-RPC no acervo: {msg}")
    return result


# ---------------------------------------------------------------------------
# CLI (R7: dry-run por padrão)
# ---------------------------------------------------------------------------

def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hermes_source",
        description="Importa sessões do state.db do Hermes para o acervo.",
    )
    p.add_argument("--db", default=DEFAULT_DB, help="caminho do state.db")
    p.add_argument("--state", default=DEFAULT_STATE, help="caminho do estado local")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                   help="endpoint MCP do acervo")
    p.add_argument("--apply", action="store_true", help="grava (padrão é dry-run)")
    p.add_argument("--dry-run", action="store_true",
                   help="apenas relata (comportamento padrão)")
    p.add_argument("--only-session", metavar="ID", default=None,
                   help="restringe a coleta/relatório a uma sessão específica")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv=None) -> int:
    """CLI: planeja o import e (em --apply) grava via ``store_session``.

    Em --apply, persiste o state atomicamente após cada sessão gravada com
    sucesso. Falha no transporte imprime erro legível no stderr e retorna
    código != 0 sem traceback.
    """
    args = _parse_args(argv)
    mode = "apply" if args.apply else "dry-run"
    conn = open_db(args.db)
    exit_code = 0
    try:
        state = _load_state(args.state)
        collected = _collect(conn, state)

        if args.only_session:
            sid = args.only_session
            matching = [r for r in collected["records"] if r["id"] == sid]
            if not matching:
                all_sessions = _fetch_sessions(conn)
                all_ids = {s["id"] for s in all_sessions}
                if sid not in all_ids:
                    print(
                        f"erro: sessao '{sid}' nao encontrada no banco",
                        file=sys.stderr,
                    )
                    return 1
                print(
                    f"erro: sessao '{sid}' nao elegivel "
                    f"(< {MIN_USEFUL} mensagens uteis)",
                    file=sys.stderr,
                )
                return 1
            collected["records"] = matching
            collected["totals"] = {
                "sessions_total": 1,
                "eligible": len(matching),
                "to_store": sum(
                    1 for r in matching
                    if r["action"] in ("new", "delta", "updated")
                ),
                "skipped": sum(1 for r in matching if r["action"] == "skip"),
                "discarded": 0,
                "sessions_truncated": sum(
                    1 for r in matching if r["truncated"]
                ),
                "chars_to_send": sum(
                    r["chars_to_send"] for r in matching
                ),
            }

        report = _to_report(collected, mode=mode)
        if args.apply:
            endpoint_kwargs = (
                {"endpoint": args.endpoint}
                if args.endpoint != DEFAULT_ENDPOINT
                else {}
            )
            for r in collected["records"]:
                if r["action"] in ("new", "delta", "updated"):
                    try:
                        store_session(_payload(r), **endpoint_kwargs)
                    except Exception as e:
                        print(
                            f"erro ao gravar sessao {r['id']}: {e}",
                            file=sys.stderr,
                        )
                        exit_code = 1
                        break
                    _update_state(state, r)
                    _save_state(args.state, state)
        print(json.dumps(report, ensure_ascii=False))
    finally:
        conn.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
