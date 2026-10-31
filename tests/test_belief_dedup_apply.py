"""GATE 3 (RED) — arco B.2b: APPLY do dedup semântico de beliefs.

Testes RED para o caminho de APPLY de ``BeliefService.dedup_beliefs(dry_run=False)``.
Hoje (B.2a) o apply levanta ``NotImplementedError`` e o helper de merge
``BeliefService._merge_into_canonical`` NÃO existe. Por isso cada teste que exercita
o apply real deve FALHAR agora (RED pelo motivo certo):
  - os testes que chamam ``dedup_beliefs(dry_run=False)`` falham com NotImplementedError;
  - os testes que chamam ``svc._merge_into_canonical(...)`` falham com AttributeError.
NÃO por import/typo — o símbolo ``BeliefService`` já existe e importa limpo.

Contrato do apply (EARS A1-A6 — B.2b, a implementar):
- A1: dry_run=False + MCP_BELIEF_DEDUP on -> perdedores viram status='superseded'
      (NÃO deletados; proveniência preservada); canônico permanece active/candidate.
- A2: merge une os ``derived_from`` dos perdedores no canônico (união de sets de hashes,
      sem duplicar) e persiste no canônico.
- A3: merge SOMA os feedback counters numéricos do ``metadata`` JSON (ex.: injected_used,
      injected_unused) dos perdedores no canônico.
- A4: apply transacional POR GRUPO — se o merge de um grupo falha, aquele grupo não deixa
      estado parcial (canônico sem metadata pela metade, perdedores não ficam superseded);
      outros grupos continuam (per-group try/except, como sweep_degenerate_beliefs).
- A5: report inclui "merged" (nº de perdedores superseded) além dos campos do dry-run.
- A6: dry_run=True (default) NÃO escreve (R6 do B.2a continua válido).

Padrão imitado de tests/test_belief_dedup.py (fixture belief_db migrações 012+021,
_insert_belief, _vec_blob, vetores _V_A/_V_A_NEAR colineares). Chama a função REAL — sem
mock de _merge_into_canonical nem da dedup (anti-teste-inerte).
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlite_vec import serialize_float32

from mcp_memory_service.consolidation.belief_service import BeliefService


MIGRATIONS = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "storage" / "migrations"


# ── Fixtures (idênticas ao B.2a) ──

@pytest.fixture
def belief_db():
    """In-memory DB com a tabela beliefs (012) + coluna embedding (021)."""
    conn = sqlite3.connect(":memory:")
    conn.executescript((MIGRATIONS / "012_add_belief_store.sql").read_text())
    conn.executescript((MIGRATIONS / "020_add_belief_embedding.sql").read_text())
    return conn


@pytest.fixture
def storage(belief_db):
    """Storage mínimo: só expõe .conn real. embedding_model=None (dedup lê do banco)."""
    class _Storage:
        pass

    s = _Storage()
    s.conn = belief_db
    s.graph = None
    s.embedding_model = None
    return s


def _vec_blob(values):
    return serialize_float32([float(v) for v in values])


def _insert_belief(
    conn,
    belief_hash,
    content,
    *,
    confidence=0.5,
    status="active",
    derived_from=None,
    embedding=None,
    created_at=None,
    updated_at=None,
    metadata=None,
):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO beliefs "
        "(belief_hash, content, confidence, status, created_at, updated_at, "
        " derived_from, contradicted_by, metadata, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            belief_hash,
            content,
            confidence,
            status,
            created_at or now,
            updated_at or now,
            json.dumps(derived_from if derived_from is not None else []),
            "[]",
            json.dumps(metadata if metadata is not None else {}),
            embedding,
        ),
    )
    conn.commit()


def _row(conn, belief_hash):
    """Retorna (status, derived_from(list), metadata(dict)) de um belief."""
    r = conn.execute(
        "SELECT status, derived_from, metadata FROM beliefs WHERE belief_hash = ?",
        (belief_hash,),
    ).fetchone()
    if r is None:
        return None
    return r[0], json.loads(r[1] or "[]"), json.loads(r[2] or "{}")


def _snapshot(conn):
    total = conn.execute("SELECT COUNT(*) FROM beliefs").fetchone()[0]
    per_status = dict(
        conn.execute("SELECT status, COUNT(*) FROM beliefs GROUP BY status").fetchall()
    )
    rows = {
        r[0]: (r[1], r[2], r[3])
        for r in conn.execute(
            "SELECT belief_hash, content, status, metadata FROM beliefs"
        ).fetchall()
    }
    return total, per_status, rows


# Dois vetores quase colineares -> cosine >= 0.90; um ortogonal -> cosine 0.
_V_A = [1.0, 0.0, 0.0]
_V_A_NEAR = [0.9962, 0.0872, 0.0]   # ~5° de A
_V_A_NEAR2 = [0.9848, 0.1736, 0.0]  # ~10° de A -> cosine ~0.985, ainda >= 0.90
_V_B = [0.0, 1.0, 0.0]


def _seed_group_with_metadata(conn):
    """Grupo de 3 duplicatas (b1 canônico por confidence). derived_from e feedback
    counters distintos em cada um, com OVERLAP parcial de hashes em derived_from para
    provar a UNIÃO sem duplicação (A2) e a SOMA de counters (A3).

    Canônico esperado (R4): h1 (confidence 0.90, maior).
    Perdedores: h2, h3 -> devem virar superseded (A1).
    União derived_from: {o1,o2} ∪ {o2,o3} ∪ {o4} = {o1,o2,o3,o4} (o2 não duplica).
    Soma counters: injected_used 5+3+2=10 ; injected_unused 1+0+4=5.
    """
    _insert_belief(
        conn, "h1", "Sempre usar type hints em Python",
        confidence=0.90, status="active",
        derived_from=["o1", "o2"], embedding=_vec_blob(_V_A),
        metadata={"injected_used": 5, "injected_unused": 1},
    )
    _insert_belief(
        conn, "h2", "Use type hints sempre no Python",
        confidence=0.70, status="active",
        derived_from=["o2", "o3"], embedding=_vec_blob(_V_A_NEAR),
        metadata={"injected_used": 3, "injected_unused": 0},
    )
    _insert_belief(
        conn, "h3", "Em Python, use type hints",
        confidence=0.60, status="candidate",
        derived_from=["o4"], embedding=_vec_blob(_V_A_NEAR2),
        metadata={"injected_used": 2, "injected_unused": 4},
    )


# ── A1: perdedores superseded, canônico permanece ──

def test_a1_losers_superseded_canonical_remains(storage, monkeypatch):
    """A1: dry_run=False marca perdedores como superseded e NÃO deleta; canônico fica."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    _seed_group_with_metadata(storage.conn)

    svc = BeliefService(storage)
    svc.dedup_beliefs(dry_run=False)

    # Canônico (h1) permanece active (não foi tocado no status).
    assert _row(storage.conn, "h1")[0] == "active"
    # Perdedores viraram superseded (preservados, não deletados).
    assert _row(storage.conn, "h2")[0] == "superseded"
    assert _row(storage.conn, "h3")[0] == "superseded"
    # Nada foi deletado: 3 linhas continuam na tabela.
    assert storage.conn.execute("SELECT COUNT(*) FROM beliefs").fetchone()[0] == 3


# ── A2: união de derived_from no canônico ──

def test_a2_derived_from_union_merged_into_canonical(storage, monkeypatch):
    """A2: derived_from do canônico = união dos sets dos membros, sem duplicar hashes."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    _seed_group_with_metadata(storage.conn)

    svc = BeliefService(storage)
    svc.dedup_beliefs(dry_run=False)

    canon_df = _row(storage.conn, "h1")[1]
    assert set(canon_df) == {"o1", "o2", "o3", "o4"}
    # União de SET: 'o2' aparecia em h1 e h2, não pode duplicar.
    assert len(canon_df) == len(set(canon_df))


# ── A3: soma de feedback counters no canônico ──

def test_a3_feedback_counters_summed_into_canonical(storage, monkeypatch):
    """A3: counters numéricos do metadata (injected_used/unused) são SOMADOS no canônico."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    _seed_group_with_metadata(storage.conn)

    svc = BeliefService(storage)
    svc.dedup_beliefs(dry_run=False)

    canon_meta = _row(storage.conn, "h1")[2]
    assert canon_meta["injected_used"] == 10   # 5 + 3 + 2
    assert canon_meta["injected_unused"] == 5   # 1 + 0 + 4


# ── A4: transacional por grupo — merge que falha não deixa estado parcial ──

def test_a4_failed_group_is_atomic_and_others_continue(storage, monkeypatch):
    """A4: se o merge de um grupo levanta no meio, aquele grupo fica INTACTO
    (canônico sem metadata pela metade, perdedores NÃO superseded), e outro grupo
    independente é aplicado normalmente.

    Grupo 1 (eixo A): h1 canônico, h2 perdedor -> merge forçado a falhar.
    Grupo 2 (eixo B): g1 canônico, g2 perdedor -> deve ser aplicado (superseded).

    Injetamos a falha monkeypatchando _merge_into_canonical para levantar SÓ quando o
    canônico é 'h1'. Isso exige que o apply chame _merge_into_canonical por grupo.
    """
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    # Grupo 1 (vai falhar)
    _insert_belief(storage.conn, "h1", "A canônico", confidence=0.90,
                   derived_from=["a1"], embedding=_vec_blob(_V_A),
                   metadata={"injected_used": 5})
    _insert_belief(storage.conn, "h2", "A perdedor", confidence=0.70,
                   derived_from=["a2"], embedding=_vec_blob(_V_A_NEAR),
                   metadata={"injected_used": 3})
    # Grupo 2 (deve aplicar)
    _insert_belief(storage.conn, "g1", "B canônico", confidence=0.90,
                   derived_from=["b1"], embedding=_vec_blob(_V_B),
                   metadata={"injected_used": 7})
    _insert_belief(storage.conn, "g2", "B perdedor", confidence=0.70,
                   derived_from=["b2"], embedding=_vec_blob([0.0872, 0.9962, 0.0]),
                   metadata={"injected_used": 2})

    svc = BeliefService(storage)

    real_merge = svc._merge_into_canonical  # existe só após B.2b

    def _boom_on_group1(canonical, losers, *args, **kwargs):
        if canonical == "h1":
            raise RuntimeError("merge falhou de propósito no grupo 1")
        return real_merge(canonical, losers, *args, **kwargs)

    monkeypatch.setattr(svc, "_merge_into_canonical", _boom_on_group1)

    svc.dedup_beliefs(dry_run=False)

    # Grupo 1 INTACTO: perdedor NÃO superseded, canônico com metadata original (sem soma).
    assert _row(storage.conn, "h2")[0] == "active"
    assert _row(storage.conn, "h1")[2].get("injected_used") == 5
    assert set(_row(storage.conn, "h1")[1]) == {"a1"}
    # Grupo 2 aplicado: perdedor superseded, canônico com união + soma.
    assert _row(storage.conn, "g2")[0] == "superseded"
    assert _row(storage.conn, "g1")[2].get("injected_used") == 9  # 7 + 2
    assert set(_row(storage.conn, "g1")[1]) == {"b1", "b2"}


# ── A5: report inclui "merged" ──

def test_a5_report_includes_merged_count(storage, monkeypatch):
    """A5: apply report traz 'merged' (nº de perdedores superseded) além de groups/evaluated."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    _seed_group_with_metadata(storage.conn)

    svc = BeliefService(storage)
    report = svc.dedup_beliefs(dry_run=False)

    # Um grupo de 3 membros -> 2 perdedores merged. Campos do dry-run preservados.
    assert report["merged"] == 2
    assert report["groups"] == 1
    assert report["evaluated"] == 3


# ── A6: dry_run=True não escreve ──

def test_a6_dry_run_true_does_not_write(storage, monkeypatch):
    """A6: dry_run=True (default) continua sem mutar o corpus (R6 do B.2a)."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    _seed_group_with_metadata(storage.conn)

    before = _snapshot(storage.conn)

    svc = BeliefService(storage)
    svc.dedup_beliefs(dry_run=True)

    after = _snapshot(storage.conn)
    assert after == before, f"dry-run mutou o corpus: before={before} after={after}"


# ── _merge_into_canonical chamado diretamente (contrato do helper B.2b) ──

def test_merge_into_canonical_unites_and_sums(storage, monkeypatch):
    """O helper _merge_into_canonical (B.2b) une derived_from + soma counters + supersede
    os perdedores, chamado diretamente. FALHA agora com AttributeError (helper não existe)."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    _seed_group_with_metadata(storage.conn)

    svc = BeliefService(storage)
    now = datetime.now(timezone.utc)
    svc._merge_into_canonical("h1", ["h2", "h3"], now)

    assert set(_row(storage.conn, "h1")[1]) == {"o1", "o2", "o3", "o4"}
    assert _row(storage.conn, "h1")[2]["injected_used"] == 10
    assert _row(storage.conn, "h2")[0] == "superseded"
    assert _row(storage.conn, "h3")[0] == "superseded"


# ── A4 (rollback INTERNO da transação real) — tuvok gap de cobertura ──

def test_a4_internal_rollback_leaves_canonical_untouched(storage, monkeypatch):
    """A4 (interno): se o supersede de um loser falha DENTRO do BEGIN, o rollback
    reverte o UPDATE já feito no canônico — canônico não fica com metadata/derived_from
    pela metade. Exercita o rollback real (não o isolamento por grupo do chamador)."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    _seed_group_with_metadata(storage.conn)  # h1 canônico; h2,h3 perdedores

    svc = BeliefService(storage)
    before = _row(storage.conn, "h1")  # (status, derived_from, metadata) originais

    # Faz o 2º UPDATE (primeiro supersede) estourar DEPOIS do UPDATE do canônico,
    # já dentro da transação. sqlite3.Connection.execute é read-only, então
    # envolvemos a conn num proxy que delega tudo e injeta a falha no supersede.
    real_conn = storage.conn

    class _FailingConn:
        def __init__(self, inner):
            self._inner = inner
        def execute(self, sql, *args, **kwargs):
            if sql.strip().upper().startswith("UPDATE BELIEFS SET STATUS"):
                raise sqlite3.OperationalError("boom no supersede (dentro da transação)")
            return self._inner.execute(sql, *args, **kwargs)
        def __getattr__(self, name):
            return getattr(self._inner, name)

    storage.conn = _FailingConn(real_conn)

    with pytest.raises(sqlite3.OperationalError):
        svc._merge_into_canonical("h1", ["h2", "h3"], datetime.now(timezone.utc))

    storage.conn = real_conn  # restaura p/ ler o estado

    after = _row(storage.conn, "h1")
    assert after == before, f"canônico mutado apesar do rollback: before={before} after={after}"
    # Perdedores seguem com o status ORIGINAL (supersede revertido): h2 active, h3 candidate.
    assert _row(storage.conn, "h2")[0] == "active"
    assert _row(storage.conn, "h3")[0] == "candidate"
