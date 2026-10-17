"""RED tests (TDD, gate G3) do importador do state.db do Hermes.

Spec: docs/rfc/planned/spec-hermes-state-db-importer.md (R1-R8, CA1-CA7, secao 7).
Alvo (sera criado no G4): scripts/harvest/hermes_source.py

CONTRATO ASSUMIDO POR ESTES TESTES
----------------------------------
- hermes_source.open_db(path) -> sqlite3.Connection em modo somente-leitura
  (URI mode=ro). Escrever pela conexao deve falhar (R1).
- hermes_source.select_sessions(conn, min_useful=2) -> lista de sessoes
  elegiveis (>= 2 mensagens uteis user+assistant) (R5).
- hermes_source.build_turns(messages, max_turn_chars=hs.MAX_TURN_CHARS) -> lista de
  {"role": str, "content": str}. `messages` e uma lista de mappings com as
  chaves role, content, tool_name, active, compacted (dict ou sqlite3.Row).
  Mensagens role='tool' saem SOMENTE como metadados (D1): nome da tool +
  indicativo ok/erro + primeira linha truncada; NUNCA o payload completo.
  Linhas compacted=1, active=0 ou role='session_meta' nao saem (R2).
- hermes_source.plan_import(conn, state) -> dict de reporte com no minimo as
  chaves "sessions" (list), "discards" (dict motivo -> contagem) e "totals"
  (dict) (R7/R8).
- hermes_source.main(argv) -> CLI. Flags: --db PATH (default
  ~/.hermes/state.db), --state PATH (default
  ~/.hermes/cache/mms-hermes-import.json), --dry-run (default) / --apply.
  Emite o relatorio como JSON no stdout (R8).
- PONTO DE INJECAO (spy de transporte): funcao de modulo
  hermes_source.store_session(payload), onde payload e
  {"turns": [...], "session_id": str, "tags": [...], "metadata": {...},
  "store": str}. main() deve chamar APENAS essa funcao para gravar cada
  memoria (CA1). Os testes fazem monkeypatch nela.

FIXTURES
--------
Bancos SQLite SINTETICOS em tmp_path com o schema minimo (sessions +
messages). Nenhum teste leia ou escreva o ~/.hermes/state.db real.

Estado esperado agora: 1-12 passam (GREEN de G4 + G3.1) e 13-22 sao RED
(falham agora porque os defeitos T13-T20 existem na implementacao).
Nao usar skip/xfail de proposito: o RED precisa ser falha real.

G3.1 (segunda rodada RED, cirurgica): os testes 10-12 travam o orcamento
de contexto por sessao aprovado em G3 — MAX_SESSION_CHARS = 15.000 por
sessao preservando inicio (head 40%) e fim (tail 60%) e MAX_TURN_CHARS =
2.000 por turno — com o corte registrado no relatorio por sessao (ex.:
'truncated': true e/ou 'turns_dropped': N). Falham agora porque a
implementacao atual (G4) truncava cada turno em 4.000 mas NAO limitava o
numero de turnos por sessao (sessoes com centenas de msgs iam inteiras) e
o relatorio nao marcava o corte.

G3.2 (terceira rodada RED): testes 13-22 travam os defeitos confirmados
por execucao na revisao G5 — T13 (sessao ja importada que ENCOLHEU nao
pode regravar: no-op, sem store_session, sem update de state, marcada
unchanged/noop no relatorio), T14 (edicao da ultima mensagem util com a
mesma contagem deve reimportar como 'updated'), T15 (falha no meio do
lote: persistencia por sessao, re-run sem duplicata, exit != 0 sem
traceback), T16 (200 com erro JSON-RPC: store_session levanta e main nao
marca a sessao como importada, exit != 0), T17 (200 com HTML de proxy:
falha explicita, sem fallback silencioso para texto cru), T18
(_parse_sse escolhe o evento cujo id corresponde a chamada; vazio/
truncado gera erro explicito), T19 (read-only em WAL: le dados do -wal e
o .db fica bit-a-bit identico, CA7 no modo real de producao) e T20
(state corrompido gera WARNING e comportamento explicito — erro ou
reimport declarado — nunca JSONDecodeError engolido em silencio).

G3.3 (quarta rodada RED — bug de producao apos import de sessoes reais):
testes 24-26 travam o vazamento de payload de tool no resumo: o corte
era feito apenas pela PRIMEIRA linha, mas a saida real de ferramentas
como read_file e um JSON de UMA UNICA linha — a 'primeira linha' e o
conteudo inteiro (ate 2.000 chars de payload cru vazavam para o
acervo, violando D1). Corrigido com teto fixo em caracteres no
resumo: MAX_TOOL_SUMMARY_CHARS = 200, independente de haver quebra
de linha, com normalizacao de espacos/quebras multiplas antes do
corte e sinal ok/erro detectado ANTES do truncamento.

Mapeamento teste -> requisito/criterio:
  1 test_tool_payload_excluded        R2, D1
  2 test_compactada_excluida          R2, CA4
  3 test_agent_id_from_profile        R3, CA5
  4 test_idempotencia                 R6, CA2
  5 test_dry_run_nao_grava            R7, R8, CA1
  6 test_readonly_db                  R1, CA7
  7 test_sessao_insuficiente_descartada  R5, CA6
  8 test_store_default_e_tag_project  R3, R4
  9 test_volume_enviado_max_10_percent  CA3 (extra, obrigatoria pela tarefa)
  10 test_orcamento_por_sessao         G3.1, CA3, secao 4.3 (15.000 head+tail)
  11 test_turno_truncado_em_2000       G3.1, CA3, secao 4.3 (2.000 por turno)
  12 test_relatorio_marca_truncamento  G3.1, R8 (corte registrado no relatorio)
  13 test_delta_vazio_vira_noop  G3.2/T13 (sessao encolhida = no-op, sem to_store)
  14 test_edicao_com_mesma_contagem_reimporta  G3.2/T14 (acao 'updated')
  15 test_falha_no_meio_do_lote_nao_duplica  G3.2/T15 (persistencia por sessao)
  16 test_store_session_erro_jsonrpc_levanta  G3.2/T16 (200 + erro JSON-RPC)
  17 test_main_erro_jsonrpc_nao_marca_sessao  G3.2/T16 (main: exit != 0, sem state)
  18 test_store_session_html_nao_json_levanta  G3.2/T17 (200 + HTML = falha)
  19 test_parse_sse_escolhe_resposta_com_id_da_chamada  G3.2/T18
  20 test_parse_sse_vazio_ou_truncado_levanta  G3.2/T18
  21 test_readonly_em_wal_le_dados_e_preserva_db  G3.2/T19, CA7 (WAL)
  22 test_state_corrompido_nao_e_silencioso  G3.2/T20 (WARNING explicito)
  23 test_only_session_*  G3.2 (flags --only-session: restricao/falhas claras)
  24 test_tool_resumo_limitado_sem_quebra_de_linha  G3.3/D1 (payload de 1 linha: teto 200)
  25 test_tool_resumo_com_quebra_de_linha  G3.3/D1 (1a linha + teto 200, sem vazar linhas 2/3)
  26 test_tool_resumo_nao_altera_sinal_erro  G3.3/D1 (corte nao altera o sinal ok/erro)
"""

import hashlib
import http.server
import itertools
import json
import logging
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

# Faz `import hermes_source` resolver para scripts/harvest/hermes_source.py
# assim que o G4 criar o arquivo (agora o import falha: RED esperado).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "harvest"))

# ---------------------------------------------------------------------------
# Schema minimo e builders de fixture sintetica
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    profile_name TEXT,
    cwd TEXT,
    git_repo_root TEXT,
    git_branch TEXT,
    title TEXT,
    started_at REAL,
    message_count INTEGER,
    model TEXT,
    source TEXT
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    role TEXT,
    content TEXT,
    tool_name TEXT,
    active INTEGER,
    compacted INTEGER,
    timestamp REAL
);
"""

_TS = itertools.count(1000.0)


def S(sess_id, profile=None, cwd=None, model="test-model", source="hermes"):
    """Linha de sessions (message_count e preenchido pelo _make_state_db)."""
    return (sess_id, profile, cwd, None, None, f"title-{sess_id}",
            1000.0, None, model, source)


def M(sess_id, role, content, tool_name=None, active=1, compacted=0, ts=None):
    """Linha de messages."""
    return (sess_id, role, content, tool_name, active, compacted,
            ts if ts is not None else next(_TS))


def _make_state_db(tmp_path, sessions, messages):
    """Cria um state.db sintetico em tmp_path e devolve o Path."""
    db = tmp_path / "state.db"
    counts = {}
    for row in messages:
        counts[row[0]] = counts.get(row[0], 0) + 1
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(_SCHEMA)
        conn.executemany(
            "INSERT INTO sessions (id, profile_name, cwd, git_repo_root,"
            " git_branch, title, started_at, message_count, model, source)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            [tuple(list(r[:7]) + [counts.get(r[0], 0)] + list(r[8:]))
             for r in sessions],
        )
        conn.executemany(
            "INSERT INTO messages (session_id, role, content, tool_name,"
            " active, compacted, timestamp) VALUES (?,?,?,?,?,?,?)",
            messages,
        )
        conn.commit()
    finally:
        conn.close()
    return db


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _hs():
    """Importa hermes_source (nao existe no RED; criado no G4).

    O import e feito aqui (lazy) para que cada teste falha individualmente
    em vez de derrubar a coleta inteira do arquivo.
    """
    import hermes_source  # noqa: F401  (modulado por path no topo)
    return hermes_source


class FakeTransport:
    """Espiao do transporte MCP (memory_store_session)."""

    def __init__(self):
        self.payloads = []

    def store(self, payload):
        self.payloads.append(payload)
        return {"ok": True}


def _install_spy(monkeypatch, hs, spy):
    """Ponta de injeção: main() deve gravar via hermes_source.store_session."""
    monkeypatch.setattr(hs, "store_session", spy.store)


def _run_main(hs, argv):
    """Chama main(argv) tolerando SystemExit(0) (CLIs que fazem sys.exit)."""
    try:
        return hs.main(list(argv))
    except SystemExit as e:
        assert e.code in (0, None), f"main() terminou com código {e.code!r}"
        return None


def _extract_json(text):
    """Extrai o primeiro objeto JSON do stdout (R8: relatorio JSON estavel)."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise AssertionError(f"relatorio JSON nao encontrado no stdout: {text[:800]!r}")


def _sess_ids(rows):
    """Extrai ids de sessoes vindas de select_sessions (dict/Row/tuple)."""
    ids = set()
    for r in rows:
        if isinstance(r, dict):
            ids.add(r["id"])
        elif hasattr(r, "keys"):  # sqlite3.Row
            ids.add(r["id"])
        else:
            ids.add(r[0])
    return ids


def _sent_chars(payloads):
    """Total de caracteres efetivamente enviados (turns dos payloads)."""
    return sum(len(t.get("content", "")) for p in payloads for t in p.get("turns", []))


# ---------------------------------------------------------------------------
# 1. R2/D1 — payload de tool nunca sai; so metadados
# ---------------------------------------------------------------------------

def test_tool_payload_excluded():
    """R2/D1: content de tool vira apenas metadados (nome + ok/erro + 1a linha).

    Fixture: mensagem tool com primeira linha curta e payload gigante a
    partir da 2a linha (marcador LEAK_MARKER_PAYLOAD no inicio da 2a linha).
    Comportamento travado: build_turns() NUNCA emite o payload completo —
    qualquer vazamento alem da primeira linha expoe o marcador; o turno de
    tool preserva o nome da tool e fica limitado a max_turn_chars.
    """
    hs = _hs()
    payload = "LEAK_MARKER_PAYLOAD" + "x" * 300_000
    msgs = [
        {"role": "user", "content": "encontre os bugs no parser",
         "tool_name": None, "active": 1, "compacted": 0},
        {"role": "tool", "content": "ok: 3 matches in parser.c\n" + payload,
         "tool_name": "grep_search", "active": 1, "compacted": 0},
        {"role": "assistant", "content": "encontrei os 3 bugs e corrigi",
         "tool_name": None, "active": 1, "compacted": 0},
    ]
    turns = hs.build_turns(msgs)
    assert isinstance(turns, list) and turns, "build_turns deve retornar lista de turnos"
    blob = json.dumps(turns, ensure_ascii=False)
    # Marcador esta no inicio da 2a linha: qualquer vazamento alem da
    # primeira linha o expoe. O payload completo NUNCA pode aparecer.
    assert "LEAK_MARKER_PAYLOAD" not in blob, \
        "payload de tool vazou para os turnos (R2/D1)"
    tool_turns = [t for t in turns if t.get("role") == "tool"]
    assert tool_turns, "turno de tool deve existir como metadado resumido (D1)"
    assert "grep_search" in tool_turns[0].get("content", ""), \
        "metadados da tool devem incluir o nome da ferramenta"
    assert all(len(t.get("content", "")) <= hs.MAX_TURN_CHARS for t in turns), \
        f"turnos acima de MAX_TURN_CHARS ({hs.MAX_TURN_CHARS})"
    assert any(t.get("role") == "user" and "encontre os bugs" in t.get("content", "")
               for t in turns), "conteudo user/assistant ativo deve ser preservado"


# ---------------------------------------------------------------------------
# 2. R2/CA4 — compactada (e inativa / session_meta) nunca sai
# ---------------------------------------------------------------------------

def test_compactada_excluida():
    """R2/CA4: linhas compacted=1, active=0 e role='session_meta' nao aparecem.

    Comportamento travado: mesmo presentes na lista de entrada de
    build_turns(), essas linhas sao descartadas na normalizacao — 0
    memorias de mensagem compactada (CA4). Conteudo unico de cada uma nao
    pode vazar; linhas uteis permanecem.
    """
    hs = _hs()
    msgs = [
        {"role": "user", "content": "pergunta real",
         "tool_name": None, "active": 1, "compacted": 0},
        {"role": "assistant", "content": "COMPACTADA_NAO_APARCE " + "y" * 500,
         "tool_name": None, "active": 1, "compacted": 1},
        {"role": "assistant", "content": "INATIVA_NAO_APARCE " + "z" * 500,
         "tool_name": None, "active": 0, "compacted": 0},
        {"role": "session_meta", "content": "META_NAO_APARCE",
         "tool_name": None, "active": 1, "compacted": 0},
        {"role": "assistant", "content": "resposta util ativa",
         "tool_name": None, "active": 1, "compacted": 0},
    ]
    turns = hs.build_turns(msgs)
    blob = json.dumps(turns, ensure_ascii=False)
    assert "COMPACTADA_NAO_APARCE" not in blob, "linha compacted=1 vazou (R2)"
    assert "INATIVA_NAO_APARCE" not in blob, "linha active=0 vazou (R2)"
    assert "META_NAO_APARCE" not in blob, "session_meta vazou (R2)"
    assert any("pergunta real" in t.get("content", "") for t in turns)
    assert any("resposta util ativa" in t.get("content", "") for t in turns)


# ---------------------------------------------------------------------------
# 3. R3/CA5 — agent_id do profile; fallback 'hermes'; tag source:hermes
# ---------------------------------------------------------------------------

def test_agent_id_from_profile(tmp_path, monkeypatch):
    """R3/CA5: metadata.agent_id = profile_name quando presente, senao 'hermes'.

    Fixture: sessao com profile_name='zero' e sessao sem profile.
    Comportamento travado: todo payload gravado tem metadata.agent_id
    correto (100% das gravadas, CA5) e a tag source:hermes.
    """
    hs = _hs()
    db = _make_state_db(
        tmp_path,
        sessions=[S("s1", profile="zero"), S("s2", profile=None)],
        messages=[M("s1", "user", "ola"), M("s1", "assistant", "oi"),
                  M("s2", "user", "oi"), M("s2", "assistant", "ola")],
    )
    spy = FakeTransport()
    _install_spy(monkeypatch, hs, spy)
    _run_main(hs, ["--apply", "--db", str(db), "--state", str(tmp_path / "state.json")])
    by_session = {p["session_id"]: p for p in spy.payloads}
    assert set(by_session) == {"s1", "s2"}, \
        f"sessoes elegiveis nao gravadas: {sorted(by_session)}"
    assert by_session["s1"]["metadata"]["agent_id"] == "zero", \
        "profile_name='zero' deve virar metadata.agent_id"
    assert by_session["s2"]["metadata"]["agent_id"] == "hermes", \
        "sem profile_name o fallback e 'hermes'"
    for p in spy.payloads:
        assert "source:hermes" in p["tags"], "tag source:hermes obrigatoria (R3)"


# ---------------------------------------------------------------------------
# 4. R6/CA2 — idempotencia: 2o run sem novidade grava 0
# ---------------------------------------------------------------------------

def test_idempotencia(tmp_path, monkeypatch):
    """R6/CA2: reexecutar com o mesmo banco e sem novidade grava ZERO memorias.

    Comportamento travado: run 1 grava a sessao (1 payload); run 2 com o
    mesmo --state e o mesmo banco nao chama store_session (0 novos).
    """
    hs = _hs()
    db = _make_state_db(
        tmp_path,
        sessions=[S("s1")],
        messages=[M("s1", "user", "pergunta"), M("s1", "assistant", "resposta")],
    )
    spy = FakeTransport()
    _install_spy(monkeypatch, hs, spy)
    state = tmp_path / "state.json"
    _run_main(hs, ["--apply", "--db", str(db), "--state", str(state)])
    assert len(spy.payloads) == 1, f"1a execucao deveria gravar 1 memoria, deu {len(spy.payloads)}"
    before = len(spy.payloads)
    _run_main(hs, ["--apply", "--db", str(db), "--state", str(state)])
    assert len(spy.payloads) == before, \
        f"2a execucao sem novidade gravou {len(spy.payloads) - before} memoria(s) (R6)"


# ---------------------------------------------------------------------------
# 5. R7/CA1 — dry-run (default): relatorio sim, gravacao nao
# ---------------------------------------------------------------------------

def test_dry_run_nao_grava(tmp_path, monkeypatch, capsys):
    """R7/CA1: sem --apply o importador nao grava NADA (0 chamadas de store).

    Comportamento travado: main() no modo padrao (dry-run) com spy de
    transporte emite o relatorio JSON no stdout (R8, com a sessao elegivel
    identificavel) e faz 0 chamadas a store_session.
    """
    hs = _hs()
    db = _make_state_db(
        tmp_path,
        sessions=[S("s1")],
        messages=[M("s1", "user", "pergunta"), M("s1", "assistant", "resposta")],
    )
    spy = FakeTransport()
    _install_spy(monkeypatch, hs, spy)
    _run_main(hs, ["--db", str(db), "--state", str(tmp_path / "state.json")])
    assert spy.payloads == [], \
        f"dry-run (sem --apply) gravou {len(spy.payloads)} memoria(s) — CA1 violada"
    report = _extract_json(capsys.readouterr().out)
    blob = json.dumps(report, ensure_ascii=False)
    assert "s1" in blob, "relatorio do dry-run deve listar a sessao elegivel (R7/R8)"


# ---------------------------------------------------------------------------
# 6. R1/CA7 — banco inalterado (sha256) e conexao efetivamente read-only
# ---------------------------------------------------------------------------

def test_readonly_db(tmp_path, monkeypatch):
    """R1/CA7: o state.db fica bit-a-bit identico apos uma execucao --apply.

    Comportamento travado: (a) sha256 do arquivo .db idêntico antes/depois;
    (b) a conexao retornada por open_db() recusa escrita (OperationalError
    de read-only) — o importador abre com URI mode=ro e nunca muta.
    """
    hs = _hs()
    db = _make_state_db(
        tmp_path,
        sessions=[S("s1")],
        messages=[M("s1", "user", "pergunta"), M("s1", "assistant", "resposta")],
    )
    before = _sha256(db)
    spy = FakeTransport()
    _install_spy(monkeypatch, hs, spy)
    _run_main(hs, ["--apply", "--db", str(db), "--state", str(tmp_path / "state.json")])
    assert _sha256(db) == before, \
        "state.db foi modificado durante a execucao (R1/CA7 violada)"
    conn = hs.open_db(str(db))
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("UPDATE sessions SET title='x' WHERE id='s1'")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 7. R5/CA6 — sessao com <2 mensagens uteis: descarte 'insufficient_content'
# ---------------------------------------------------------------------------

def test_sessao_insuficiente_descartada(tmp_path):
    """R5/CA6: sessao com menos de 2 mensagens uteis (user+assistant) e
    descartada com motivo 'insufficient_content' no relatorio.

    Fixture: sessao 's_ok' (user+assistant) e sessao 's_pobrezinha'
    (apenas 1 user + 1 tool — tool nao conta como mensagem util).
    Comportamento travado: plan_import reporta discards['insufficient_content']
    == 1 e select_sessions devolve apenas a sessao elegivel.
    """
    hs = _hs()
    db = _make_state_db(
        tmp_path,
        sessions=[S("s_ok"), S("s_pobrezinha")],
        messages=[M("s_ok", "user", "pergunta"), M("s_ok", "assistant", "resposta"),
                  M("s_pobrezinha", "user", "so uma mensagem"),
                  M("s_pobrezinha", "tool", "resultado da ferramenta",
                    tool_name="grep_search")],
    )
    conn = hs.open_db(str(db))
    try:
        report = hs.plan_import(conn, {})
        eligible = hs.select_sessions(conn)
    finally:
        conn.close()
    assert report["discards"].get("insufficient_content") == 1, \
        f"descarte por 'insufficient_content' nao reportado: {report.get('discards')}"
    ids = _sess_ids(eligible)
    assert "s_pobrezinha" not in ids, "sessao insuficiente foi tratada como elegivel (R5)"
    assert "s_ok" in ids, "sessao com 2+ mensagens uteis deve ser elegivel"


# ---------------------------------------------------------------------------
# 8. R4 — store='default' + tag project:<slug do ultimo componente de cwd>
# ---------------------------------------------------------------------------

def test_store_default_e_tag_project(tmp_path, monkeypatch):
    """R4: memorias gravadas em store='default'; projeto vai por TAG, nao por
    store: project:<slug> derivado do ultimo componente do cwd.

    Fixture: sessao com cwd='/home/dev/projeto/acme-app' (tag esperada
    'project:acme-app') e sessao sem cwd (nao pode ganhar tag project:*).
    """
    hs = _hs()
    db = _make_state_db(
        tmp_path,
        sessions=[S("s1", cwd="/home/dev/projeto/acme-app"), S("s2", cwd=None)],
        messages=[M("s1", "user", "pergunta"), M("s1", "assistant", "resposta"),
                  M("s2", "user", "pergunta2"), M("s2", "assistant", "resposta2")],
    )
    spy = FakeTransport()
    _install_spy(monkeypatch, hs, spy)
    _run_main(hs, ["--apply", "--db", str(db), "--state", str(tmp_path / "state.json")])
    by_session = {p["session_id"]: p for p in spy.payloads}
    assert set(by_session) == {"s1", "s2"}
    assert by_session["s1"]["store"] == "default", "memorias devem ir para store 'default' (R4)"
    assert "project:acme-app" in by_session["s1"]["tags"], \
        f"tag project:<slug do cwd> ausente: {by_session['s1']['tags']}"
    assert not any(t.startswith("project:") for t in by_session["s2"]["tags"]), \
        f"sessao sem cwd nao deve ganhar tag project: {by_session['s2']['tags']}"
    assert "source:hermes" in by_session["s1"]["tags"]


# ---------------------------------------------------------------------------
# 9. CA3 — volume enviado <= 10% do volume lido
# ---------------------------------------------------------------------------

def test_volume_enviado_max_10_percent_do_lido(tmp_path, monkeypatch):
    """CA3: total de caracteres enviados <= 10% do total lido do banco.

    Fixture construida para provar: 1 sessao com tool payload de ~300 KB e
    user/assistant pequenos. O banco tem ~300 KB; o que o transporte recebe
    (turns normalizados com tool resumida a metadados) deve ser <= 10%.
    Comportamento travado: spy do transporte soma os chars dos turns e
    compara com a soma dos content lidos do banco sintetico.
    """
    hs = _hs()
    payload = "LEAK_MARKER_PAYLOAD" + "x" * 300_000
    messages = [
        M("s1", "user", "pergunta curta sobre o parser"),
        M("s1", "tool", "ok: 3 matches in parser.c\n" + payload, tool_name="grep_search"),
        M("s1", "assistant", "resposta curta com a correcao"),
    ]
    total_read = sum(len(m[2] or "") for m in messages)
    db = _make_state_db(tmp_path, sessions=[S("s1")], messages=messages)
    assert total_read > 300_000, "fixture deve ter payload dominante"
    spy = FakeTransport()
    _install_spy(monkeypatch, hs, spy)
    _run_main(hs, ["--apply", "--db", str(db), "--state", str(tmp_path / "state.json")])
    assert len(spy.payloads) == 1, "sessao elegivel deveria ter sido gravada"
    total_sent = _sent_chars(spy.payloads)
    assert total_sent <= 0.10 * total_read, \
        f"volume enviado ({total_sent} chars) > 10% do lido ({total_read}) — CA3 violada"


# ---------------------------------------------------------------------------
# 10. G3.1/CA3/secao 4.3 — orcamento por sessao: 15.000 chars (head+tail)
# ---------------------------------------------------------------------------

def test_orcamento_por_sessao(tmp_path, monkeypatch):
    """G3.1: o payload de uma sessao fica limitado a 15.000 chars de
    conteudo, preservando o inicio (head 40%) e o fim (tail 60%).

    Fixture: 1 sessao com 200 mensagens uteis de exatamente 1.000 chars
    cada (~200.000 chars lidos). Comportamento travado: o payload gravado
    tem no maximo 15.000 chars somados nos turns; o PRIMEIRO e o ULTIMO
    conteudo util aparecem integrais; o MEIO da sessao e cortado.
    """
    hs = _hs()
    n_msgs = 200
    contents = []
    for i in range(n_msgs):
        marker = f"MSG_{i:04d}_"
        contents.append(marker + "x" * (1000 - len(marker)))
    assert all(len(c) == 1000 for c in contents), \
        "fixture: cada mensagem deve ter exatamente 1.000 chars"
    messages = [
        M("s_big", "user" if i % 2 == 0 else "assistant", contents[i])
        for i in range(n_msgs)
    ]
    db = _make_state_db(tmp_path, sessions=[S("s_big")], messages=messages)
    spy = FakeTransport()
    _install_spy(monkeypatch, hs, spy)
    _run_main(hs, ["--apply", "--db", str(db), "--state", str(tmp_path / "state.json")])
    assert len(spy.payloads) == 1, "sessao elegivel deveria ter sido gravada"
    total = _sent_chars(spy.payloads)
    assert total <= 15_000, \
        f"payload da sessao excedeu o orcamento de 15.000 chars: {total} (G3.1/CA3)"
    blob = json.dumps(spy.payloads[0]["turns"], ensure_ascii=False)
    assert contents[0] in blob, \
        "PRIMEIRO conteudo util nao aparece inteiro no payload (head 40%)"
    assert contents[-1] in blob, \
        "ULTIMO conteudo util nao aparece inteiro no payload (tail 60%)"
    mid_marker = f"MSG_{n_msgs // 2:04d}_"
    assert mid_marker not in blob, \
        "meio da sessao deveria ter sido cortado (apenas head+tail devem sair)"


# ---------------------------------------------------------------------------
# 11. G3.1/CA3/secao 4.3 — truncamento por turno: 2.000 chars
# ---------------------------------------------------------------------------

def test_turno_truncado_em_2000(tmp_path, monkeypatch):
    """G3.1: nenhum turno sai com mais de 2.000 chars de conteudo.

    Fixture: sessao com uma mensagem user de 50.000 chars (+ 1 assistant
    curta para garantir o minimo de 2 mensagens uteis). Comportamento
    travado: o turno user no payload gravado tem no maximo 2.000 chars.
    RED: a implementacao atual truncava em 4.000 (MAX_TURN_CHARS).
    """
    hs = _hs()
    messages = [
        M("s1", "user", "B" * 50_000),
        M("s1", "assistant", "resposta curta"),
    ]
    db = _make_state_db(tmp_path, sessions=[S("s1")], messages=messages)
    spy = FakeTransport()
    _install_spy(monkeypatch, hs, spy)
    _run_main(hs, ["--apply", "--db", str(db), "--state", str(tmp_path / "state.json")])
    assert len(spy.payloads) == 1, "sessao elegivel deveria ter sido gravada"
    turns = spy.payloads[0]["turns"]
    user_turns = [t for t in turns if t.get("role") == "user"]
    assert user_turns, \
        "turno da mensagem de 50.000 chars deveria existir (head preservado)"
    lengths = [len(t.get("content", "")) for t in turns]
    assert all(ln <= 2_000 for ln in lengths), \
        f"turno acima do limite de 2.000 chars: max={max(lengths)} (G3.1)"


# ---------------------------------------------------------------------------
# 12. G3.1/R8 — corte registrado no relatorio por sessao
# ---------------------------------------------------------------------------

def _truncation_marks(entry):
    """(truncado?, turnos_descartados) lidos de uma entrada de sessao.

    Aceita a chave booleana de truncamento (ex.: 'truncated': true) e/ou
    a contagem de turnos descartados (ex.: 'turns_dropped': N).
    """
    truncated = False
    dropped = 0
    for k, v in entry.items():
        kl = str(k).lower()
        if "truncat" in kl and v is True:
            truncated = True
        if ("drop" in kl or "discard" in kl) and isinstance(v, int) \
                and not isinstance(v, bool):
            dropped = max(dropped, v)
    return truncated, dropped


def test_relatorio_marca_truncamento(tmp_path):
    """G3.1: quando uma sessao e cortada, o relatorio registra o corte
    para aquela sessao (booleano de truncamento e/ou contagem de turnos
    descartados) — e sessao sem corte nao e marcada.

    Fixture: sessao 's_big' (200 msgs x 1.000 chars — excede 15.000) e
    sessao 's_small' (2 msgs curtas — dentro do orcamento).
    """
    hs = _hs()
    contents = []
    for i in range(200):
        marker = f"MSG_{i:04d}_"
        contents.append(marker + "y" * (1000 - len(marker)))
    big_messages = [
        M("s_big", "user" if i % 2 == 0 else "assistant", contents[i])
        for i in range(200)
    ]
    small_messages = [
        M("s_small", "user", "pergunta curta"),
        M("s_small", "assistant", "resposta curta"),
    ]
    db = _make_state_db(
        tmp_path,
        sessions=[S("s_big"), S("s_small")],
        messages=big_messages + small_messages,
    )
    conn = hs.open_db(str(db))
    try:
        report = hs.plan_import(conn, {})
    finally:
        conn.close()
    entries = {s["id"]: s for s in report["sessions"]}
    assert "s_big" in entries and "s_small" in entries, \
        f"sessoes elegiveis ausentes do relatorio: {sorted(entries)}"
    truncated, dropped = _truncation_marks(entries["s_big"])
    assert truncated or dropped >= 1, \
        f"sessao cortada nao registrada no relatorio: {entries['s_big']}"
    small_truncated, small_dropped = _truncation_marks(entries["s_small"])
    assert not small_truncated and small_dropped == 0, \
        f"sessao dentro do orcamento marcada como truncada: {entries['s_small']}"


# ---------------------------------------------------------------------------
# Helpers G3.2 (captura de exit code sem propagar excecao + HTTP server)
# ---------------------------------------------------------------------------

def _run_main_capture(hs, argv):
    """Chama main(argv) e devolve (return_code, exception).

    Diferente de _run_main, NAO asserciona exit 0: os testes de falha de
    lote (T15/T16) exigem exit != 0 sem excecao propagada nem traceback.
    """
    try:
        return hs.main(list(argv)), None
    except SystemExit as e:
        return (e.code if e.code is not None else 0), None
    except Exception as e:  # noqa: BLE001
        return None, e


def _serve_once(body: bytes, content_type: str = "application/json"):
    """Servidor HTTP local (127.0.0.1, porta aleatoria) que responde
    QUALQUER POST com HTTP 200 e o `body` dado. Devolve (server, url);
    quem usa faz shutdown/server_close no finally."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/mcp"
    return server, url


_PAYLOAD_FAKE = {
    "turns": [{"role": "user", "content": "conteudo de teste"}],
    "session_id": "s1",
    "tags": ["source:hermes"],
    "metadata": {"agent_id": "hermes"},
    "store": "default",
}


# ---------------------------------------------------------------------------
# 13. G3.2/T13 — delta vazio (sessao encolhida) vira no-op
# ---------------------------------------------------------------------------

def test_delta_vazio_vira_noop(tmp_path, monkeypatch, capsys):
    """T13: sessao ja importada que ENCOLHEU (3 -> 2 msgs uteis) nao pode
    ser regravada: store_session nao e chamada para ela, o state local nao
    e atualizado (useful_msgs permanece 3) e o relatorio a marca como
    unchanged/noop (nao contar em to_store)."""
    hs = _hs()
    db = _make_state_db(
        tmp_path,
        sessions=[S("s1")],
        messages=[M("s1", "user", "pergunta"),
                  M("s1", "assistant", "resposta")],
    )
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "s1": {
            "useful_msgs": 3,
            "last_hash": hashlib.sha256(b"terceira mensagem antiga").hexdigest(),
            "imported_at": "2026-01-01T00:00:00+00:00",
        }
    }), encoding="utf-8")
    spy = FakeTransport()
    _install_spy(monkeypatch, hs, spy)
    _run_main(hs, ["--apply", "--db", str(db), "--state", str(state_path)])
    assert not any(p.get("session_id") == "s1" for p in spy.payloads), \
        f"sessao encolhida (3->2) acionou store_session: {spy.payloads}"
    report = _extract_json(capsys.readouterr().out)
    assert report["totals"]["to_store"] == 0, \
        f"sessao encolhida entrou em to_store: {report['totals']}"
    entry = next(s for s in report["sessions"] if s["id"] == "s1")
    assert entry["action"] in ("unchanged", "noop", "skip"), \
        f"relatorio deve marcar sessao encolhida como unchanged/noop: {entry}"
    st = json.loads(state_path.read_text(encoding="utf-8"))
    assert st.get("s1", {}).get("useful_msgs") == 3, \
        f"state da sessao encolhida foi atualizado (deve permanecer 3): {st.get('s1')}"


# ---------------------------------------------------------------------------
# 14. G3.2/T14 — edicao com a mesma contagem deve reimportar ('updated')
# ---------------------------------------------------------------------------

def test_edicao_com_mesma_contagem_reimporta(tmp_path, monkeypatch, capsys):
    """T14: se a ultima mensagem util foi EDITADA (mesma contagem de msgs
    uteis, conteudo diferente), a sessao deve ser reimportada com acao
    'updated' e o conteudo editado deve ser reenviado — hoje o state marca
    a sessao como importada e a edicao se perde em delta vazio."""
    hs = _hs()
    orig_last = "resposta original"
    edited_last = "resposta editada com conteudo novo"
    db = _make_state_db(
        tmp_path,
        sessions=[S("s1")],
        messages=[M("s1", "user", "pergunta"),
                  M("s1", "assistant", edited_last)],
    )
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "s1": {
            "useful_msgs": 2,
            "last_hash": hashlib.sha256(orig_last.encode("utf-8")).hexdigest(),
            "imported_at": "2026-01-01T00:00:00+00:00",
        }
    }), encoding="utf-8")
    spy = FakeTransport()
    _install_spy(monkeypatch, hs, spy)
    _run_main(hs, ["--apply", "--db", str(db), "--state", str(state_path)])
    report = _extract_json(capsys.readouterr().out)
    entry = next(s for s in report["sessions"] if s["id"] == "s1")
    assert entry["action"] == "updated", \
        f"edicao da ultima mensagem (mesma contagem) deve reimportar como 'updated': {entry}"
    stored = [t for p in spy.payloads if p.get("session_id") == "s1"
              for t in p.get("turns", [])]
    blob = json.dumps(stored, ensure_ascii=False)
    assert edited_last in blob, \
        "conteudo editado da ultima mensagem nao foi reenviado (edicao perdida)"


# ---------------------------------------------------------------------------
# 15. G3.2/T15 — falha no meio do lote: persistencia por sessao, sem duplicata
# ---------------------------------------------------------------------------

def test_falha_no_meio_do_lote_nao_duplica(tmp_path, monkeypatch, capsys):
    """T15: com 3 sessoes elegiveis e store_session falhando na 2a:
    (a) a 1a fica registrada no state (persistencia por sessao, atomica);
    (b) o re-run NAO regravar a 1a; (c) main sai com codigo != 0 e sem
    traceback."""
    hs = _hs()
    msgs = []
    for sid in ("s1", "s2", "s3"):
        msgs += [M(sid, "user", f"pergunta {sid}"),
                 M(sid, "assistant", f"resposta {sid}")]
    db = _make_state_db(
        tmp_path,
        sessions=[S("s1"), S("s2"), S("s3")],
        messages=msgs,
    )
    state_path = tmp_path / "state.json"
    seen = []

    def flaky_store(payload):
        seen.append(payload["session_id"])
        if len(seen) == 2:
            raise RuntimeError("falha simulada na 2a sessao do lote")
        return {"ok": True}

    monkeypatch.setattr(hs, "store_session", flaky_store)
    rc, exc = _run_main_capture(hs, ["--apply", "--db", str(db),
                                     "--state", str(state_path)])
    out_err = capsys.readouterr()
    assert exc is None, \
        f"main deve tratar falha do lote sem propagar excecao, mas lancou: {exc!r}"
    assert rc is not None and rc != 0, \
        f"main deve sair com codigo != 0 apos falha no lote, saiu com {rc!r}"
    assert "Traceback" not in out_err.out and "Traceback" not in out_err.err, \
        "main imprimiu traceback ao tratar a falha do lote"
    st = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    assert "s1" in st, \
        "1a sessao do lote nao ficou registrada no state (persistencia por sessao)"
    # re-run com transporte saudavel: a 1a nao pode ser regravada
    spy = FakeTransport()
    _install_spy(monkeypatch, hs, spy)
    rc2, exc2 = _run_main_capture(hs, ["--apply", "--db", str(db),
                                       "--state", str(state_path)])
    assert exc2 is None and rc2 == 0, \
        f"re-execucao deveria ser limpa: rc={rc2!r} exc={exc2!r}"
    sent = [p["session_id"] for p in spy.payloads]
    assert "s1" not in sent, f"re-execucao regravou a 1a sessao: {sent}"
    assert "s2" in sent and "s3" in sent, \
        f"re-execucao deveria gravar s2 e s3: {sent}"


# ---------------------------------------------------------------------------
# 16. G3.2/T16 — 200 com erro JSON-RPC e falha (store_session + main)
# ---------------------------------------------------------------------------

def test_store_session_erro_jsonrpc_levanta():
    """T16: HTTP 200 com resposta SSE contendo {"error": {...}} e FALHA:
    store_session deve levantar excecao (nunca devolver o objeto de erro
    como se fosse resultado)."""
    hs = _hs()
    error_obj = {"jsonrpc": "2.0", "id": 1,
                 "error": {"code": -32601, "message": "Method not found"}}
    body = f"data: {json.dumps(error_obj)}\n\n".encode("utf-8")
    server, url = _serve_once(body, content_type="text/event-stream")
    try:
        with pytest.raises(Exception):  # noqa: B017
            hs.store_session(dict(_PAYLOAD_FAKE), endpoint=url)
    finally:
        server.shutdown()
        server.server_close()


def test_main_erro_jsonrpc_nao_marca_sessao(tmp_path, monkeypatch, capsys):
    """T16 (main): quando store_session falha (erro JSON-RPC), main nao
    pode marcar a sessao como importada (state) e deve sair != 0 sem
    traceback."""
    hs = _hs()
    db = _make_state_db(
        tmp_path,
        sessions=[S("s1")],
        messages=[M("s1", "user", "pergunta"),
                  M("s1", "assistant", "resposta")],
    )
    state_path = tmp_path / "state.json"

    def store_always_fails(payload):
        raise RuntimeError("JSON-RPC error: Method not found")

    monkeypatch.setattr(hs, "store_session", store_always_fails)
    rc, exc = _run_main_capture(hs, ["--apply", "--db", str(db),
                                     "--state", str(state_path)])
    out_err = capsys.readouterr()
    assert exc is None, f"main nao pode propagar excecao do transporte: {exc!r}"
    assert rc is not None and rc != 0, \
        f"main deve sair != 0 quando store_session falha: {rc!r}"
    assert "Traceback" not in out_err.out and "Traceback" not in out_err.err
    st = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    assert "s1" not in st, \
        "sessao com erro JSON-RPC ficou marcada como importada no state"


# ---------------------------------------------------------------------------
# 17. G3.2/T17 — 200 com HTML (proxy) e falha explicita
# ---------------------------------------------------------------------------

def test_store_session_html_nao_json_levanta():
    """T17: HTTP 200 com HTML (pagina de erro de proxy) e falha explicita:
    store_session deve levantar erro claro, sem fallback silencioso para
    texto cru."""
    hs = _hs()
    html = b"<html><body><h1>502 Bad Gateway</h1>proxy error</body></html>"
    server, url = _serve_once(html, content_type="text/html")
    try:
        with pytest.raises(Exception) as ei:  # noqa: B017
            hs.store_session(dict(_PAYLOAD_FAKE), endpoint=url)
        assert str(ei.value), "erro deve ter mensagem clara, nao vazia"
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# 18. G3.2/T18 — _parse_sse: evento certo, e erro em vazio/truncado
# ---------------------------------------------------------------------------

def test_parse_sse_escolhe_resposta_com_id_da_chamada():
    """T18: havendo mais de um evento data: (ex.: uma notification antes do
    result), _parse_sse deve devolver o objeto cujo id corresponde a
    chamada (o result) — nao o primeiro evento."""
    hs = _hs()
    notification = {"jsonrpc": "2.0", "method": "notifications/progress",
                    "params": {"progressToken": "t1", "progress": 1}}
    result = {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
    raw = (
        "event: message\n"
        f"data: {json.dumps(notification)}\n"
        "\n"
        "event: message\n"
        f"data: {json.dumps(result)}\n"
        "\n"
    )
    parsed = hs._parse_sse(raw)
    assert isinstance(parsed, dict), \
        f"_parse_sse deve devolver dict JSON, veio {type(parsed)!r}"
    assert parsed.get("id") == 1 and "result" in parsed, \
        f"_parse_sse escolheu o evento errado (esperado result com id=1): {parsed!r}"


def test_parse_sse_vazio_ou_truncado_levanta():
    """T18: corpo vazio ou truncado deve gerar erro explicito em
    _parse_sse — nunca retorno silencioso do texto cru (ou do vazio)."""
    hs = _hs()
    with pytest.raises(Exception):  # noqa: B017
        hs._parse_sse("")
    with pytest.raises(Exception):  # noqa: B017
        hs._parse_sse('data: {"jsonrpc": "2.0", "id": 1, "resu')


# ---------------------------------------------------------------------------
# 19. G3.2/T19 — read-only em modo WAL (CA7 no modo real de producao)
# ---------------------------------------------------------------------------

def test_readonly_em_wal_le_dados_e_preserva_db(tmp_path):
    """T19/CA7: com PRAGMA journal_mode=WAL e conteudo pendente no -wal,
    open_db deve ler os dados e o sha256 do arquivo .db deve permanecer
    identico apos fechar a conexao (leitura read-only nao muta o db)."""
    hs = _hs()
    db = tmp_path / "state.db"
    conn_a = sqlite3.connect(str(db))
    rows = []
    before = after = None
    try:
        conn_a.execute("PRAGMA journal_mode=WAL")
        conn_a.executescript(_SCHEMA)
        conn_a.execute(
            "INSERT INTO sessions (id, profile_name, cwd, git_repo_root,"
            " git_branch, title, started_at, message_count, model, source)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("s1", None, None, None, None, "title-s1", 1000.0,
             2, "test-model", "hermes"),
        )
        conn_a.executemany(
            "INSERT INTO messages (session_id, role, content, tool_name,"
            " active, compacted, timestamp) VALUES (?,?,?,?,?,?,?)",
            [M("s1", "user", "pergunta"),
             M("s1", "assistant", "resposta inicial")],
        )
        conn_a.commit()
        # 2a conexao insere dados que ficam apenas no -wal (conn_a segue
        # aberta -> nenhum checkpoint derruba o -wal no close)
        conn_b = sqlite3.connect(str(db))
        try:
            conn_b.execute(
                "INSERT INTO messages (session_id, role, content, tool_name,"
                " active, compacted, timestamp) VALUES (?,?,?,?,?,?,?)",
                M("s1", "assistant", "WAL_MARKER_RESPOSTA"),
            )
            conn_b.commit()
        finally:
            conn_b.close()
        wal_path = Path(str(db) + "-wal")
        assert wal_path.exists() and wal_path.stat().st_size > 0, \
            "fixture WAL: arquivo -wal ausente ou vazio (dados nao ficaram no WAL)"
        before = _sha256(db)
        conn = hs.open_db(str(db))
        try:
            rows = conn.execute(
                "SELECT content FROM messages ORDER BY timestamp, id"
            ).fetchall()
        finally:
            conn.close()
        after = _sha256(db)
    finally:
        conn_a.close()
    contents = [r[0] for r in rows]
    assert "WAL_MARKER_RESPOSTA" in contents, \
        "dados gravados no -wal nao foram lidos pela conexao read-only (CA7/WAL)"
    assert after == before, \
        "arquivo .db foi modificado durante leitura read-only em modo WAL (CA7)"


# ---------------------------------------------------------------------------
# 20. G3.2/T20 — state corrompido nao pode ser silencioso
# ---------------------------------------------------------------------------

def test_state_corrompido_nao_e_silencioso(tmp_path, monkeypatch, caplog):
    """T20: arquivo de state invalido deve gerar WARNING (caplog) e
    comportamento EXPLICITO (erro ou reimport declarado) — nunca
    JSONDecodeError engolido em silencio."""
    hs = _hs()
    db = _make_state_db(
        tmp_path,
        sessions=[S("s1")],
        messages=[M("s1", "user", "pergunta"),
                  M("s1", "assistant", "resposta")],
    )
    state_path = tmp_path / "state.json"
    spy1 = FakeTransport()
    _install_spy(monkeypatch, hs, spy1)
    _run_main(hs, ["--apply", "--db", str(db), "--state", str(state_path)])
    assert spy1.payloads, "preliminar: 1a execucao grava a sessao"
    # corrompe o state
    state_path.write_text("{json invalido corrupto!!", encoding="utf-8")
    spy2 = FakeTransport()
    _install_spy(monkeypatch, hs, spy2)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        rc, exc = _run_main_capture(hs, ["--apply", "--db", str(db),
                                         "--state", str(state_path)])
    assert exc is None, \
        f"state corrompido nao pode vazar como excecao nao tratada: {exc!r}"
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, \
        "state corrompido foi engolido em silencio — deve gerar WARNING logado"
    resent = any(p.get("session_id") == "s1" for p in spy2.payloads)
    assert (rc is not None and rc != 0) or resent, \
        "comportamento com state corrompido nao e explicito (nem erro, nem reimport)"


# ---------------------------------------------------------------------------
# 21. --only-session: restringe relatorio a uma unica sessao
# ---------------------------------------------------------------------------

def test_only_session_restringe_ao_id(tmp_path, monkeypatch, capsys):
    """T21: --only-session s2 reporta apenas s2 (sessions_total=1, to_store=1).

    Fixture: 3 sessoes elegiveis (s1, s2, s3). Comportamento travado:
    main(['--dry-run', '--only-session', 's2', ...]) limita o relatorio a
    s2 e os totais refletem apenas essa sessao.
    """
    hs = _hs()
    msgs = []
    for sid in ("s1", "s2", "s3"):
        msgs += [M(sid, "user", f"pergunta {sid}"),
                 M(sid, "assistant", f"resposta {sid}")]
    db = _make_state_db(
        tmp_path,
        sessions=[S("s1"), S("s2"), S("s3")],
        messages=msgs,
    )
    _run_main(hs, [
        "--dry-run", "--only-session", "s2",
        "--db", str(db), "--state", str(tmp_path / "state.json"),
    ])
    report = _extract_json(capsys.readouterr().out)
    session_ids = [s["id"] for s in report["sessions"]]
    assert session_ids == ["s2"], \
        f"--only-session s2 deveria relatar apenas s2, deu: {session_ids}"
    assert report["totals"]["sessions_total"] == 1, \
        f"sessions_total deve ser 1 com --only-session, deu: {report['totals']}"
    assert report["totals"]["eligible"] == 1, \
        f"eligible deve ser 1 com --only-session, deu: {report['totals']}"
    assert report["totals"]["to_store"] == 1, \
        f"to_store deve ser 1 com --only-session, deu: {report['totals']}"


# ---------------------------------------------------------------------------
# 22. --only-session: ID inexistente → falha clara, exit != 0
# ---------------------------------------------------------------------------

def test_only_session_id_inexistente_falha(tmp_path, monkeypatch, capsys):
    """T22: --only-session com ID que nao existe no banco → exit != 0,
    mensagem clara no stderr (menciona o ID), nada gravado, sem traceback."""
    hs = _hs()
    db = _make_state_db(
        tmp_path,
        sessions=[S("s1")],
        messages=[M("s1", "user", "pergunta"),
                  M("s1", "assistant", "resposta")],
    )
    spy = FakeTransport()
    _install_spy(monkeypatch, hs, spy)
    state_path = tmp_path / "state.json"
    rc, exc = _run_main_capture(
        hs, ["--apply", "--only-session", "s_inexistente",
             "--db", str(db), "--state", str(state_path)]
    )
    out_err = capsys.readouterr()
    assert exc is None, \
        f"main nao pode propagar excecao para ID inexistente: {exc!r}"
    assert rc is not None and rc != 0, \
        f"ID inexistente deve dar exit != 0, deu rc={rc!r}"
    assert "s_inexistente" in out_err.err, \
        f"stderr deve mencionar o ID inexistente 's_inexistente': {out_err.err!r}"
    assert "nao encontrada" in out_err.err, \
        f"stderr deve ter mensagem clara de ID nao encontrado: {out_err.err!r}"
    assert "Traceback" not in out_err.err, \
        "falha com ID inexistente nao pode imprimir traceback"
    assert spy.payloads == [], \
        f"nada pode ser gravado quando o ID nao existe: {spy.payloads}"
    if state_path.exists():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        assert "s_inexistente" not in st, \
            f"state nao deve conter entrada para ID inexistente: {st}"


# ---------------------------------------------------------------------------
# 23. --only-session: sessao nao elegivel (<2 msgs uteis) → falha clara
# ---------------------------------------------------------------------------

def test_only_session_nao_elegivel_falha(tmp_path, monkeypatch, capsys):
    """T23: --only-session com sessao que tem <2 mensagens uteis → exit != 0,
    mensagem clara no stderr (menciona o ID), nada gravado, sem traceback."""
    hs = _hs()
    db = _make_state_db(
        tmp_path,
        sessions=[S("s_ok"), S("s_pobre")],
        messages=[M("s_ok", "user", "pergunta"), M("s_ok", "assistant", "resposta"),
                  M("s_pobre", "user", "apenas uma mensagem")],
    )
    spy = FakeTransport()
    _install_spy(monkeypatch, hs, spy)
    state_path = tmp_path / "state.json"
    rc, exc = _run_main_capture(
        hs, ["--apply", "--only-session", "s_pobre",
             "--db", str(db), "--state", str(state_path)]
    )
    out_err = capsys.readouterr()
    assert exc is None, \
        f"main nao pode propagar excecao para sessao nao elegivel: {exc!r}"
    assert rc is not None and rc != 0, \
        f"sessao nao elegivel deve dar exit != 0, deu rc={rc!r}"
    assert "s_pobre" in out_err.err, \
        f"stderr deve mencionar o ID 's_pobre': {out_err.err!r}"
    assert "nao elegivel" in out_err.err, \
        f"stderr deve ter mensagem clara de sessao nao elegivel: {out_err.err!r}"
    assert "Traceback" not in out_err.err, \
        "falha para sessao nao elegivel nao pode imprimir traceback"
    assert spy.payloads == [], \
        f"nada pode ser gravado quando a sessao nao e elegivel: {spy.payloads}"


# ---------------------------------------------------------------------------
# 24. G3.3/D1 — payload de tool em UMA linha (sem \n): teto de 200 chars
# ---------------------------------------------------------------------------

def _tool_turn_content(turns, tool_name):
    """Extrai o content do turno de tool e valida o formato D1.

    Devolve (content, signal, summary) de
    ``'[tool:<nome>] <ok|erro> — <resumo>'``; falha a assercao se o
    formato (prefixo + separador ' — ') estiver quebrado.
    """
    tool_turns = [t for t in turns if t.get("role") == "tool"]
    assert tool_turns, "turno de tool deve existir como metadado resumido (D1)"
    content = tool_turns[0].get("content", "")
    prefix = f"[tool:{tool_name}] "
    assert content.startswith(prefix), \
        f"formato '[tool:<nome>] <ok|erro> — <resumo>' violado: {content[:80]!r}"
    rest = content[len(prefix):]
    signal, sep, summary = rest.partition(" — ")
    assert sep, f"separador ' — ' ausente no turno de tool: {content[:80]!r}"
    assert signal in ("ok", "erro"), f"sinal inesperado: {signal!r}"
    return content, signal, summary


def test_tool_resumo_limitado_sem_quebra_de_linha():
    """T24: payload de tool de 5.000 chars em UMA unica linha (sem \\n —
    o caso real de producao: JSON de uma linha vindo de read_file) nao
    pode vazar inteiro: o RESUMO do turno de tool deve ter no maximo
    200 chars (teto MAX_TOOL_SUMMARY_CHARS), independente de haver
    quebra de linha, e o turno inteiro fica em ~260 chars.

    Bug: o corte era feito pela 'primeira linha'; com payload de uma
    linha, a primeira linha e o conteudo inteiro (2.000 chars de
    payload cru vazavam, violando D1).
    """
    hs = _hs()
    head_s, tail_s = '{"content": "1|', '"}'
    payload = head_s + "A" * (5000 - len(head_s) - len(tail_s)) + tail_s
    assert len(payload) == 5000 and "\n" not in payload, \
        "fixture: payload deve ter exatamente 5.000 chars em uma linha"
    msgs = [
        {"role": "user", "content": "leia o arquivo do parser",
         "tool_name": None, "active": 1, "compacted": 0},
        {"role": "tool", "content": payload,
         "tool_name": "read_file", "active": 1, "compacted": 0},
        {"role": "assistant", "content": "arquivo lido, sem problemas",
         "tool_name": None, "active": 1, "compacted": 0},
    ]
    turns = hs.build_turns(msgs)
    content, signal, summary = _tool_turn_content(turns, "read_file")
    assert len(summary) <= 200, \
        (f"resumo de tool excedeu o teto de 200 chars com payload de uma "
         f"linha (sem quebra): {len(summary)} — D1 violado (G3.3)")
    assert len(content) <= 260, \
        (f"turno de tool excedeu ~260 chars no total: {len(content)} "
         f"— D1 violado (G3.3)")
    assert hs.MAX_TOOL_SUMMARY_CHARS == 200, \
        "constante MAX_TOOL_SUMMARY_CHARS deve existir e valer 200 (G3.3)"


# ---------------------------------------------------------------------------
# 25. G3.3/D1 — saida com 3 linhas: usa a 1a linha e respeita o teto
# ---------------------------------------------------------------------------

def test_tool_resumo_com_quebra_de_linha():
    """T25: saida de tool com 3 linhas usa a PRIMEIRA linha no resumo e
    respeita o teto de 200 chars (1a linha de 300 chars deve ser
    cortada em 200; as linhas 2 e 3 nao podem vazar para o resumo)."""
    hs = _hs()
    first_line = "B" * 300
    content = first_line + "\nSEGUNDA_LINHA_MARKER\nTERCEIRA_LINHA_MARKER"
    msgs = [
        {"role": "user", "content": "pergunta sobre o parser",
         "tool_name": None, "active": 1, "compacted": 0},
        {"role": "tool", "content": content,
         "tool_name": "grep_search", "active": 1, "compacted": 0},
        {"role": "assistant", "content": "resposta com a correcao",
         "tool_name": None, "active": 1, "compacted": 0},
    ]
    turns = hs.build_turns(msgs)
    turn, signal, summary = _tool_turn_content(turns, "grep_search")
    assert summary.startswith("B"), \
        f"resumo deve usar a primeira linha da saida: {summary[:60]!r}"
    assert len(summary) <= 200, \
        (f"resumo de tool com quebras de linha excedeu o teto de 200: "
         f"{len(summary)} (G3.3/D1)")
    assert "SEGUNDA_LINHA_MARKER" not in turn, \
        "2a linha da saida vazou para o resumo (D1)"
    assert "TERCEIRA_LINHA_MARKER" not in turn, \
        "3a linha da saida vazou para o resumo (D1)"


# ---------------------------------------------------------------------------
# 26. G3.3/D1 — o corte do resumo nao pode alterar o sinal ok/erro
# ---------------------------------------------------------------------------

def test_tool_resumo_nao_altera_sinal_erro():
    """T26: saida longa de uma unica linha com 'Traceback' no MEIO
    (alem da posicao do corte em 200 chars) — o sinal do turno deve
    continuar 'erro'. O corte do resumo nao pode afetar a deteccao do
    sinal: a heuristica roda no conteudo ANTES do truncamento (se a
    deteccao fosse feita no resumo ja cortado, o marcador cairia fora
    do corte e o sinal viraria 'ok')."""
    hs = _hs()
    head = "K" * 1500
    marker = "Traceback (most recent call last):"
    content = head + marker + "L" * (5000 - len(head) - len(marker))
    assert len(content) == 5000 and "\n" not in content, \
        "fixture: payload de 5.000 chars em uma linha"
    assert marker not in content[:200], \
        "fixture: 'Traceback' deve estar alem da posicao do corte (200)"
    msgs = [
        {"role": "user", "content": "pergunta sobre o parser",
         "tool_name": None, "active": 1, "compacted": 0},
        {"role": "tool", "content": content,
         "tool_name": "terminal", "active": 1, "compacted": 0},
        {"role": "assistant", "content": "resposta",
         "tool_name": None, "active": 1, "compacted": 0},
    ]
    turns = hs.build_turns(msgs)
    turn, signal, summary = _tool_turn_content(turns, "terminal")
    assert signal == "erro", \
        (f"corte do resumo alterou o sinal: esperado 'erro' (marker "
         f"além da posição do corte), veio {signal!r} (G3.3/D1)")
