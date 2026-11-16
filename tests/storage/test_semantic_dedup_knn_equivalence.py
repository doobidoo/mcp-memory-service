"""The KNN pushdown in `_check_semantic_duplicate` answers what the scan answered (#2032).

Three layers, all against a real `SqliteVecMemoryStorage` with deterministic
embeddings:

- equivalence: for a battery of probes the function's answer equals the old
  full-window scan's answer — same match, same hash, same similarity to
  1e-6 — across every filter variant (default, ``current_only``,
  ``exclude_original_hash``);
- the contract this PR exists for: the read goes through
  ``content_embedding MATCH``/``k`` and no longer issues a
  ``vec_distance_cosine`` per window row (that query shape is what made a
  busy 24h window cost ~4x more per store());
- behavior pins: stale/deleted/other-store/superseded/chunk rows stay
  excluded, and the threshold edge (0.9 in, 0.7 out) holds.
"""

import math
import time

import pytest
from sqlite_vec import serialize_float32

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

# The query main runs, verbatim — the equivalence oracle
OLD_SCAN = (
    "SELECT m.content_hash, m.content, "
    "vec_distance_cosine(me.content_embedding, ?) as similarity "
    "FROM memories m "
    "JOIN memory_embeddings me ON m.rowid = me.rowid "
    "WHERE m.created_at > ? AND m.deleted_at IS NULL AND m.store = ?"
    "{extra} ORDER BY similarity IS NULL, similarity ASC LIMIT 1"
)
CURRENT_ONLY_SQL = " AND (m.superseded_by IS NULL OR m.superseded_by = '')"
EXCLUDE_HASH_SQL = (
    " AND (json_extract(m.metadata, '$.original_hash') IS NULL"
    " OR json_extract(m.metadata, '$.original_hash') != ?)"
)


def _unit_cos(dim, cos_value):
    """[c, s, 0...] — cosine to [1, 0...] is exactly cos_value."""
    s = math.sqrt(max(0.0, 1.0 - cos_value * cos_value))
    return [cos_value, s] + [0.0] * (dim - 2)


@pytest.fixture
async def storage(temp_db_path, monkeypatch):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/knn_equivalence.db")
    await storage.initialize()
    dim = storage.embedding_dimension

    vectors = {
        "PROBE sentinel": [1.0] + [0.0] * (dim - 1),
        "twin of the probe": [1.0] + [0.0] * (dim - 1),      # cos 1.0
        "near miss": _unit_cos(dim, 0.90),                   # dup at 0.85
        "just inside": _unit_cos(dim, 0.86),                 # dup, edge side
        "just outside": _unit_cos(dim, 0.84),                # no dup, edge side
        "clearly different": _unit_cos(dim, 0.70),           # no dup
        "orthogonal note": _unit_cos(dim, 0.0),
    }
    monkeypatch.setattr(
        storage, "_generate_embedding",
        lambda content: vectors.get(content, _unit_cos(dim, 0.1)))

    # seeding must not be refused by the dedup it exists to test
    storage.semantic_dedup_enabled = False
    for text in vectors:
        if text != "PROBE sentinel":
            m = Memory(content=text, content_hash=generate_content_hash(text))
            stored, _ = await storage.store(m)
            assert stored, f"seeding failed for {text!r}"
    yield storage
    await storage.close()


def _old_scan_row(storage, content, store_name, current_only, exclude_hash):
    extra, params = "", []
    if current_only:
        extra += CURRENT_ONLY_SQL
        if exclude_hash:
            extra += EXCLUDE_HASH_SQL
            params.append(exclude_hash)
    blob = serialize_float32(storage._generate_embedding(content))
    row = storage.conn.execute(
        OLD_SCAN.format(extra=extra),
        (blob, time.time() - 86400.0, store_name, *params),
    ).fetchone()
    return row


def _old_answer(row, threshold):
    """What main's `_check_semantic_duplicate` returns for this scan row."""
    if row and row[2] is not None and row[2] <= (1.0 - threshold):
        return (True, row[0], 1.0 - float(row[2]), row[1])
    return (False, None, None, None)


async def test_same_answer_as_the_scan_for_every_variant(storage):
    for current_only, exclude in ((False, None), (True, None), (True, "deadbeef")):
        got = await storage._check_semantic_duplicate(
            "PROBE sentinel", current_only=current_only,
            exclude_original_hash=exclude)
        row = _old_scan_row(storage, "PROBE sentinel", "default",
                            current_only, exclude)
        matched, hash_, similarity, content = _old_answer(row, 0.85)
        assert got[0] is matched
        assert got[1] == hash_
        if matched:
            assert abs(got.similarity - similarity) < 1e-6
            assert got.matched_content == content


async def test_threshold_edge_pins(storage):
    near = await storage._check_semantic_duplicate("PROBE sentinel")
    assert near[0] is True
    # the twin at cosine 1.0 wins over 'near miss' at 0.90 and 'just inside'
    # at 0.86 — both dup-class, ordering is part of the answer
    assert abs(near.similarity - 1.0) < 1e-6
    assert near.matched_content == "twin of the probe"


async def test_stale_deleted_otherstore_rows_stay_excluded(storage):
    now = time.time()
    conn = storage.conn
    # push the twin out of the window: the nearest live row becomes
    # 'near miss' (cos 0.90 beats 'just inside' at 0.86), still a dup
    conn.execute("UPDATE memories SET created_at = ? WHERE content = ?",
                 (now - 48 * 3600, "twin of the probe"))
    got = await storage._check_semantic_duplicate("PROBE sentinel")
    assert got[0] is True and got.matched_content == "near miss"

    # tombstone it, and move the next candidates off the axes
    conn.execute("UPDATE memories SET created_at = ?, deleted_at = ? "
                 "WHERE content = ?",
                 (now - 60, now - 30, "just inside"))
    conn.execute("UPDATE memories SET store = ? WHERE content = ?",
                 ("elsewhere", "near miss"))
    got = await storage._check_semantic_duplicate("PROBE sentinel")
    # 'just outside' at 0.84 is the nearest survivor — below the threshold,
    # and a non-match carries no similarity by design
    assert got[0] is False and got[1] is None


async def test_current_only_and_chunk_exclusion(storage):
    conn = storage.conn
    twin_hash = generate_content_hash("twin of the probe")
    conn.execute("UPDATE memories SET superseded_by = ? WHERE content_hash = ?",
                 ("replacement-hash", twin_hash))
    got = await storage._check_semantic_duplicate("PROBE sentinel")
    assert got.matched_content == "twin of the probe"  # default query ignores lineage
    got = await storage._check_semantic_duplicate("PROBE sentinel", current_only=True)
    assert got.matched_content == "near miss"          # superseded twin is out

    # a chunk of the same split write must not be its own supersede target:
    # tag the CURRENT nearest ('near miss', 0.90) so skipping it is visible —
    # the answer drops to 'just inside' (0.86), still a dup
    conn.execute(
        "UPDATE memories SET metadata = ? WHERE content = ?",
        ('{"original_hash": "chunk-family"}', "near miss"))
    got = await storage._check_semantic_duplicate(
        "PROBE sentinel", current_only=True, exclude_original_hash="chunk-family")
    assert got.matched_content == "just inside"
    assert got[0] is True


async def test_empty_window_returns_no_match(storage):
    storage.conn.execute("UPDATE memories SET created_at = 1")
    got = await storage._check_semantic_duplicate("PROBE sentinel")
    assert got[0] is False


async def test_window_past_the_k_cap_matches_the_scan(storage):
    """>4096 live candidates: the k cap engages and the answer must not change.

    The discriminator matters more than the size: 5,000 STALE rows sit closer
    to the probe than the target live row. If sqlite-vec filtered after
    taking the top-k, those stale rows would fill every k slot and the read
    would come back empty; the rowid filter running inside the KNN scan is
    what makes the cap exact for this top-1 read. The scan oracle over the
    same rows agrees.
    """
    import random

    dim = storage.embedding_dimension
    rng = random.Random(2032)
    conn = storage.conn
    now = time.time()

    def _cos_row(cos_value):
        s = math.sqrt(max(0.0, 1.0 - cos_value * cos_value))
        return [cos_value, s] + [0.0] * (dim - 2)

    def _bulk_insert(rows, prefix):
        mem_sql = ("INSERT INTO memories(content_hash, content, created_at, "
                   "deleted_at, store) VALUES (?,?,?,?,?)")
        emb_sql = ("INSERT INTO memory_embeddings(rowid, content_embedding, "
                   "store) VALUES (?,?,?)")
        rowids = []
        for i, (vec, created_at, store_name) in enumerate(rows):
            cur = conn.execute(mem_sql, (f"{prefix}-{i}", f"{prefix} note {i}",
                                         created_at, None, store_name))
            rowids.append(cur.lastrowid)
            conn.execute(emb_sql, (cur.lastrowid,
                                   serialize_float32(vec), store_name))
        return rowids

    filler = [([rng.gauss(0, 1) for _ in range(dim)], now - 3600, "default")
              for _ in range(4200)]
    # closer to the probe than anything live, but outside the window
    stale_near = [(_cos_row(0.99), now - 48 * 3600, "default")
                  for _ in range(5000)]
    _bulk_insert(filler + stale_near, "bulk")

    live = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE created_at > ? AND deleted_at "
        "IS NULL AND store = 'default'", (now - 86400,)).fetchone()[0]
    assert live > 4096  # the cap is what this run exercises

    # the fixture's own twin (cos 1.0) is the expected match; the 5,000
    # stale rows at 0.99 sit between it and everything else
    got = await storage._check_semantic_duplicate("PROBE sentinel")
    row = _old_scan_row(storage, "PROBE sentinel", "default", False, None)
    matched, hash_, similarity, content = _old_answer(row, 0.85)
    assert matched is True and hash_ is not None
    assert got[0] is True
    assert got[1] == hash_
    assert got.matched_content == content == "twin of the probe"
    assert abs(got.similarity - similarity) < 1e-6
    assert abs(got.similarity - 1.0) < 1e-6


async def test_read_goes_through_the_knn_pushdown(storage):
    """The contract of #2032: no per-row vec_distance_cosine over the window."""

    class _ConnSpy:
        def __init__(self, conn, captured):
            self._conn, self._captured = conn, captured

        def execute(self, sql, params=()):
            self._captured.append(sql)
            return self._conn.execute(sql, params)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    captured = []
    real_conn = storage.conn
    storage.conn = _ConnSpy(real_conn, captured)
    try:
        await storage._check_semantic_duplicate("PROBE sentinel")
    finally:
        storage.conn = real_conn

    dedup_reads = [s for s in captured if "memory_embeddings" in s]
    assert any("MATCH" in s and "k = ?" in s for s in dedup_reads), captured
    assert not any("vec_distance_cosine" in s for s in dedup_reads)
