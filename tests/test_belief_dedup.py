"""GATE 3 (RED) — arco B.2a: dry-run de dedup semântico de beliefs.

Testes RED para ``BeliefService.dedup_beliefs(dry_run: bool) -> dict`` — função que
AINDA NÃO existe em ``belief_service.py``. Por isso cada teste que chama a função real
deve FALHAR com ``AttributeError: 'BeliefService' object has no attribute 'dedup_beliefs'``
(RED pelo motivo certo), NÃO por import/typo.

Contrato (EARS R1-R6 de docs/rfc/planned/rfc-belief-dedup-semantic.md):
- R1: flag MCP_BELIEF_DEDUP OFF -> {"evaluated":0,"groups":0} sem ler o corpus.
- R2: flag ON + dry_run=True -> agrupa active/candidate com cosine>=THRESHOLD (default 0.90), SEM escrever.
- R3: beliefs com embedding NULL são pulados e contados em "skipped".
- R4: canônico = confidence desc -> |derived_from| desc -> updated_at desc -> belief_hash (NÃO usa utility_prior).
- R5: respeita MCP_BELIEF_DEDUP_CAP (default 1000), sinaliza capped:true.
- R6: dry-run NÃO altera status/content/metadata nem contagem (total e por-status idênticas antes/depois).

Padrão imitado de tests/test_belief_store.py (fixture belief_db + mock_storage). Os embeddings
são BLOBs reais (serialize_float32) de vetores controlados para forçar cosine alto entre dois
beliefs e baixo com um terceiro — SEM mockar a dedup nem seus internos (anti-teste-inerte).
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlite_vec import serialize_float32

from mcp_memory_service.consolidation.belief_service import BeliefService


MIGRATIONS = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "storage" / "migrations"


# ── Fixtures ──

@pytest.fixture
def belief_db():
    """In-memory DB com a tabela beliefs (012) + coluna embedding (021)."""
    conn = sqlite3.connect(":memory:")
    conn.executescript((MIGRATIONS / "012_add_belief_store.sql").read_text())
    conn.executescript((MIGRATIONS / "020_add_belief_embedding.sql").read_text())
    return conn


@pytest.fixture
def storage(belief_db):
    """Storage mínimo: só expõe .conn real. embedding_model=None (dedup lê do banco, não re-embeda)."""
    class _Storage:
        pass

    s = _Storage()
    s.conn = belief_db
    s.graph = None
    s.embedding_model = None
    return s


def _vec_blob(values):
    """Serializa um vetor float32 do mesmo jeito que _compute_belief_embedding."""
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


def _snapshot(conn):
    """Estado observável do corpus: contagem total + por-status + (content,status,metadata) por hash."""
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


# Dois vetores quase colineares -> cosine ~= 1.0 (>= 0.90); um ortogonal -> cosine 0.
_V_A = [1.0, 0.0, 0.0]
_V_A_NEAR = [0.9962, 0.0872, 0.0]   # ~5° de A -> cosine ~= 0.996
_V_B = [0.0, 1.0, 0.0]             # ortogonal a A -> cosine 0


def _seed_two_dupes_one_distinct(conn):
    """Grupo: b1 ~ b2 (duplicatas). b3 distinto. Confidences controladas para R4."""
    _insert_belief(conn, "h_dup_1", "Sempre usar type hints em Python",
                   confidence=0.80, embedding=_vec_blob(_V_A), derived_from=["o1", "o2"])
    _insert_belief(conn, "h_dup_2", "Use type hints sempre no Python",
                   confidence=0.70, embedding=_vec_blob(_V_A_NEAR), derived_from=["o3"])
    _insert_belief(conn, "h_distinct", "Prefira HTTPS a SSH para git",
                   confidence=0.60, embedding=_vec_blob(_V_B), derived_from=["o4"])


# ── R1: flag OFF -> no-op, sem ler corpus ──

def test_r1_flag_off_returns_zero_without_reading_corpus(storage, monkeypatch):
    """R1: MCP_BELIEF_DEDUP ausente/off -> {"evaluated":0,"groups":0}, nada tocado."""
    monkeypatch.delenv("MCP_BELIEF_DEDUP", raising=False)
    _seed_two_dupes_one_distinct(storage.conn)

    svc = BeliefService(storage)
    report = svc.dedup_beliefs(dry_run=True)

    assert report["evaluated"] == 0
    assert report["groups"] == 0


# ── R2: flag ON + dry_run -> agrupa por cosine >= threshold, sem escrever ──

def test_r2_flag_on_groups_by_cosine_threshold(storage, monkeypatch):
    """R2: um grupo detectado (b1~b2), o distinto fica de fora."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    monkeypatch.delenv("MCP_BELIEF_DEDUP_THRESHOLD", raising=False)  # default 0.90
    _seed_two_dupes_one_distinct(storage.conn)

    svc = BeliefService(storage)
    report = svc.dedup_beliefs(dry_run=True)

    # 3 beliefs avaliados, 1 grupo de duplicatas (2 membros), o terceiro é singleton.
    assert report["evaluated"] == 3
    assert report["groups"] == 1


def test_r2_distinct_beliefs_not_grouped(storage, monkeypatch):
    """R2: dois beliefs ortogonais (cosine 0) NÃO formam grupo."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    _insert_belief(storage.conn, "h_x", "Alfa", confidence=0.5, embedding=_vec_blob(_V_A))
    _insert_belief(storage.conn, "h_y", "Beta", confidence=0.5, embedding=_vec_blob(_V_B))

    svc = BeliefService(storage)
    report = svc.dedup_beliefs(dry_run=True)

    assert report["evaluated"] == 2
    assert report["groups"] == 0


# ── R3: embedding NULL -> pulado e contado em skipped ──

def test_r3_null_embedding_skipped(storage, monkeypatch):
    """R3: belief com embedding NULL não é deduplicável; entra em 'skipped'."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    _seed_two_dupes_one_distinct(storage.conn)
    _insert_belief(storage.conn, "h_null", "Sem embedding ainda",
                   confidence=0.5, embedding=None)

    svc = BeliefService(storage)
    report = svc.dedup_beliefs(dry_run=True)

    assert report["skipped"] == 1
    # o grupo de duplicatas continua detectado apesar do NULL
    assert report["groups"] == 1


# ── R4: eleição determinística do canônico (NÃO usa utility_prior) ──

def test_r4_canonical_elected_by_confidence_then_provenance(storage, monkeypatch):
    """R4: maior confidence vence. b1 (0.80) é canônico sobre b2 (0.70)."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    _seed_two_dupes_one_distinct(storage.conn)

    svc = BeliefService(storage)
    report = svc.dedup_beliefs(dry_run=True)

    groups = report["group_details"]
    assert len(groups) == 1
    assert groups[0]["canonical"] == "h_dup_1"
    assert set(groups[0]["superseded"]) == {"h_dup_2"}


def test_r4_tiebreak_by_derived_from_size(storage, monkeypatch):
    """R4: confidence empatado -> maior |derived_from| vence (não utility_prior)."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    _insert_belief(storage.conn, "h_few", "Lição A", confidence=0.75,
                   derived_from=["o1"], embedding=_vec_blob(_V_A))
    _insert_belief(storage.conn, "h_many", "Lição A bis", confidence=0.75,
                   derived_from=["o1", "o2", "o3"], embedding=_vec_blob(_V_A_NEAR))

    svc = BeliefService(storage)
    report = svc.dedup_beliefs(dry_run=True)

    groups = report["group_details"]
    assert len(groups) == 1
    assert groups[0]["canonical"] == "h_many"


# ── R5: respeita CAP e sinaliza capped ──

def test_r5_respects_cap_and_flags_capped(storage, monkeypatch):
    """R5: com CAP=1 e dois grupos distintos, para após 1 grupo e marca capped:true."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    monkeypatch.setenv("MCP_BELIEF_DEDUP_CAP", "1")
    # Grupo 1: eixo A
    _insert_belief(storage.conn, "g1_a", "A um", confidence=0.8, embedding=_vec_blob(_V_A))
    _insert_belief(storage.conn, "g1_b", "A dois", confidence=0.7, embedding=_vec_blob(_V_A_NEAR))
    # Grupo 2: eixo B
    _insert_belief(storage.conn, "g2_a", "B um", confidence=0.8, embedding=_vec_blob(_V_B))
    _insert_belief(storage.conn, "g2_b", "B dois", confidence=0.7,
                   embedding=_vec_blob([0.0872, 0.9962, 0.0]))

    svc = BeliefService(storage)
    report = svc.dedup_beliefs(dry_run=True)

    assert report["capped"] is True
    assert report["groups"] == 1


# ── R6: dry-run não altera NADA no corpus ──

def test_r6_dry_run_does_not_mutate_corpus(storage, monkeypatch):
    """R6: estado observável (total, por-status, content/status/metadata) idêntico antes/depois."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    _seed_two_dupes_one_distinct(storage.conn)
    _insert_belief(storage.conn, "h_cand", "Candidato qualquer",
                   confidence=0.55, status="candidate", embedding=_vec_blob([0.0, 0.0, 1.0]))

    before = _snapshot(storage.conn)

    svc = BeliefService(storage)
    svc.dedup_beliefs(dry_run=True)

    after = _snapshot(storage.conn)
    assert after == before, f"dry-run mutou o corpus: before={before} after={after}"


# ── G5 (tuvok) P3: apply path (B.2b) not implemented -> fail loud, not silent no-op ──

def test_dry_run_false_applies_merge(storage, monkeypatch):
    """B.2b implementado: dedup_beliefs(dry_run=False) agora APLICA (supersede dos
    perdedores + report['merged']), não levanta mais NotImplementedError.
    (Cobertura detalhada do apply em tests/test_belief_dedup_apply.py.)"""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    _seed_two_dupes_one_distinct(storage.conn)
    svc = BeliefService(storage)
    report = svc.dedup_beliefs(dry_run=False)
    assert report["merged"] == 1  # h_dup_2 superseded sob h_dup_1
    assert storage.conn.execute(
        "SELECT status FROM beliefs WHERE belief_hash='h_dup_2'"
    ).fetchone()[0] == "superseded"


def test_grouping_is_stable_across_insertion_order(monkeypatch):
    """P2 regression: ORDER BY belief_hash makes group count independent of
    physical insertion order (same corpus -> same groups)."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")

    def _count_groups(order):
        conn = sqlite3.connect(":memory:")
        conn.executescript((MIGRATIONS / "012_add_belief_store.sql").read_text())
        conn.executescript((MIGRATIONS / "020_add_belief_embedding.sql").read_text())
        for h in order:
            v = {"a": _V_A, "b": _V_A_NEAR, "c": _V_B}[h]
            _insert_belief(conn, h, f"belief {h}", confidence=0.5, embedding=_vec_blob(v))

        class _S:
            pass
        s = _S(); s.conn = conn; s.graph = None; s.embedding_model = None
        return BeliefService(s).dedup_beliefs(dry_run=True)["groups"]

    assert _count_groups(["a", "b", "c"]) == _count_groups(["c", "b", "a"])
