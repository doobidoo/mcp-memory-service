"""Unit tests for ONNX Hub tokenizer truncation and embedding regeneration (#2073)."""

import asyncio
import importlib
import json
import sqlite3
import struct
import sys
import types
from pathlib import Path


def _import_onnx_embeddings_with_stubs(monkeypatch):
    """Import onnx_embeddings even in minimal unit-test environments without numpy."""
    if "numpy" not in sys.modules:
        try:
            import numpy  # noqa: F401
        except ImportError:
            fake_np = types.ModuleType("numpy")
            fake_np.ndarray = object
            monkeypatch.setitem(sys.modules, "numpy", fake_np)
    mod_name = "mcp_memory_service.embeddings.onnx_embeddings"
    if mod_name in sys.modules:
        return importlib.reload(sys.modules[mod_name])
    return importlib.import_module(mod_name)


def _load_regenerate_module(monkeypatch):
    """Import scripts/maintenance/regenerate_embeddings.py with lightweight storage/config stubs."""
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        fake_sqlite_vec = types.ModuleType("sqlite_vec")
        fake_sqlite_vec.serialize_float32 = lambda vec: struct.pack(f"{len(vec)}f", *vec)
        monkeypatch.setitem(sys.modules, "sqlite_vec", fake_sqlite_vec)

    fake_factory = types.ModuleType("mcp_memory_service.storage.factory")
    fake_factory.create_storage_instance = None
    monkeypatch.setitem(sys.modules, "mcp_memory_service.storage.factory", fake_factory)

    fake_cfg = types.ModuleType("mcp_memory_service.config")
    fake_cfg.SQLITE_VEC_PATH = ":memory:"
    monkeypatch.setitem(sys.modules, "mcp_memory_service.config", fake_cfg)

    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root / "scripts" / "maintenance"))
    if "regenerate_embeddings" in sys.modules:
        return importlib.reload(sys.modules["regenerate_embeddings"])
    return importlib.import_module("regenerate_embeddings")


def test_resolve_max_length_handles_offset_and_sentinel(monkeypatch, tmp_path: Path):
    mod = _import_onnx_embeddings_with_stubs(monkeypatch)
    cls = mod.ONNXEmbeddingModel

    # Default when neither config.json nor tokenizer_config.json exists.
    assert cls._resolve_max_length(tmp_path) == 512

    # XLM-RoBERTa (intfloat/multilingual-e5-small): 514 - (pad_token_id=1 + 1) = 512,
    # ignoring Hugging Face's VERY_LARGE_INTEGER sentinel in tokenizer_config.json.
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "xlm-roberta", "max_position_embeddings": 514, "pad_token_id": 1}),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 1000000000000000019884624838656}),
        encoding="utf-8",
    )
    assert cls._resolve_max_length(tmp_path) == 512

    # MPNet (all-mpnet-base-v2): 514 - 2 = 512.
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "mpnet", "max_position_embeddings": 514, "pad_token_id": 1}),
        encoding="utf-8",
    )
    assert cls._resolve_max_length(tmp_path) == 512

    # BGE-M3 long-context XLM-RoBERTa: 8194 - 2 = 8192.
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "xlm-roberta", "max_position_embeddings": 8194, "pad_token_id": 1}),
        encoding="utf-8",
    )
    assert cls._resolve_max_length(tmp_path) == 8192

    # BERT with explicit stricter tokenizer_config.json limit (256 < 512).
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "bert", "max_position_embeddings": 512, "pad_token_id": 0}),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 256}),
        encoding="utf-8",
    )
    assert cls._resolve_max_length(tmp_path) == 256


def test_init_model_enables_truncation_only_when_null(monkeypatch, tmp_path: Path):
    mod = _import_onnx_embeddings_with_stubs(monkeypatch)

    model_dir = tmp_path / "e5"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"stub")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "xlm-roberta", "max_position_embeddings": 514, "pad_token_id": 1}),
        encoding="utf-8",
    )

    class _StubTokenizer:
        def __init__(self, initial_truncation):
            self.truncation = initial_truncation
            self.enabled_with = None

        def enable_truncation(self, max_length):
            self.enabled_with = max_length
            self.truncation = {"max_length": max_length}

    class _StubIO:
        def __init__(self, name, shape=None):
            self.name = name
            self.shape = shape or [1, 384]

    class _StubSession:
        def __init__(self, *_a, **_kw):
            pass

        def get_outputs(self):
            return [_StubIO("last_hidden_state", [1, 512, 384])]

        def get_inputs(self):
            return [_StubIO("input_ids"), _StubIO("attention_mask")]

    fake_ort = types.SimpleNamespace(InferenceSession=_StubSession)
    monkeypatch.setattr(mod, "ONNX_AVAILABLE", True)
    monkeypatch.setattr(mod, "TOKENIZERS_AVAILABLE", True)
    monkeypatch.setattr(mod, "ort", fake_ort, raising=False)
    monkeypatch.setattr(mod.ONNXEmbeddingModel, "_download_model_if_needed", lambda self: None)

    # Case 1: Hub tokenizer has truncation=None -> enable_truncation(512) called.
    tok_null = _StubTokenizer(initial_truncation=None)
    monkeypatch.setattr(mod, "Tokenizer", types.SimpleNamespace(from_file=lambda _p: tok_null), raising=False)

    orig_init = mod.ONNXEmbeddingModel._init_model
    monkeypatch.setattr(
        mod.ONNXEmbeddingModel,
        "_init_model",
        lambda self: (setattr(self, "_model_dir", model_dir), orig_init(self)),
    )
    m1 = mod.ONNXEmbeddingModel(model_name="multilingual-e5-small")
    assert tok_null.enabled_with == 512
    assert m1._tokenizer.truncation == {"max_length": 512}

    # Case 2: Bundled tokenizer already defines truncation=256 -> preserved untouched.
    tok_preset = _StubTokenizer(initial_truncation={"max_length": 256})
    monkeypatch.setattr(mod, "Tokenizer", types.SimpleNamespace(from_file=lambda _p: tok_preset), raising=False)
    m2 = mod.ONNXEmbeddingModel(model_name="multilingual-e5-small")
    assert tok_preset.enabled_with is None
    assert m2._tokenizer.truncation == {"max_length": 256}


def test_regenerate_all_batches_commits_and_rolls_back_failed_insert(monkeypatch):
    """Verify 10-row batch transactions and savepoint rollback when INSERT fails after DELETE (#2073)."""
    regen = _load_regenerate_module(monkeypatch)

    raw_conn = sqlite3.connect(":memory:")
    raw_conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content_hash TEXT)")
    raw_conn.execute(
        """
        CREATE TABLE memory_embeddings (
            rowid INTEGER PRIMARY KEY,
            content_embedding BLOB NOT NULL CHECK(length(content_embedding) = 8),
            store TEXT NOT NULL
        )
        """
    )
    memories = []
    orig_blob = struct.pack("2f", 0.1, 0.9)
    for idx in range(1, 13):
        h = f"hash_{idx:02d}"
        raw_conn.execute("INSERT INTO memories (id, content_hash) VALUES (?, ?)", (idx, h))
        raw_conn.execute(
            "INSERT INTO memory_embeddings (rowid, content_embedding, store) VALUES (?, ?, ?)",
            (idx, orig_blob, "orig_store"),
        )
        memories.append({"content_hash": h, "content": f"text {idx}", "store": "updated_store"})
    raw_conn.commit()

    class _TrackingVec0LikeConn:
        """Rejects INSERT OR REPLACE (matching vec0) and counts explicit batch commits."""

        def __init__(self, inner: sqlite3.Connection):
            self._inner = inner
            self.commit_calls = 0

        @property
        def in_transaction(self) -> bool:
            return self._inner.in_transaction

        def execute(self, sql: str, params=()):
            if "INSERT OR REPLACE" in sql.upper():
                raise sqlite3.IntegrityError(
                    "UNIQUE constraint failed on memory_embeddings primary key"
                )
            return self._inner.execute(sql, params)

        def commit(self):
            self.commit_calls += 1
            return self._inner.commit()

    class _StubStorage:
        def __init__(self, conn):
            self.conn = conn

        def _generate_embedding(self, text: str):
            # Force row 2 insertion to fail CHECK(length == 8) AFTER DELETE succeeded!
            if text == "text 2":
                return [0.1, 0.2, 0.3]
            return [0.25, 0.75]

    tracked = _TrackingVec0LikeConn(raw_conn)
    ok, errs = asyncio.run(regen._regenerate_all(_StubStorage(tracked), memories))
    assert (ok, errs) == (11, 1)
    # 11 successful rows -> batch commit fired once at i=10, and row 12 remains in open
    # outer transaction until regenerate_embeddings() performs its final commit.
    assert tracked.commit_calls == 1
    assert raw_conn.in_transaction is True
    raw_conn.commit()

    # Row 2 failed during INSERT -> rolled back to savepoint so old vector and store survived!
    row2 = raw_conn.execute(
        "SELECT content_embedding, store FROM memory_embeddings WHERE rowid = 2"
    ).fetchone()
    assert row2 == (orig_blob, "orig_store")

    # Subsequent rows in the same batch (e.g. row 3 and row 12) succeeded cleanly!
    new_blob = struct.pack("2f", 0.25, 0.75)
    for ok_id in (1, 3, 10, 12):
        row_ok = raw_conn.execute(
            "SELECT content_embedding, store FROM memory_embeddings WHERE rowid = ?",
            (ok_id,),
        ).fetchone()
        assert row_ok == (new_blob, "updated_store")


def test_regenerate_all_real_vec0_savepoint_rollback_when_available(monkeypatch):
    """Exercise actual sqlite-vec vec0 virtual table rollback when sqlite_vec is installed."""
    try:
        import sqlite_vec
    except ImportError:
        return

    regen = _load_regenerate_module(monkeypatch)
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content_hash TEXT)")
    conn.execute(
        """
        CREATE VIRTUAL TABLE memory_embeddings USING vec0(
            content_embedding FLOAT[2] distance_metric=cosine,
            store TEXT partition key
        )
        """
    )
    old_vec = sqlite_vec.serialize_float32([1.0, 0.0])
    conn.execute("INSERT INTO memories (id, content_hash) VALUES (1, 'h1'), (2, 'h2')")
    conn.execute(
        "INSERT INTO memory_embeddings (rowid, content_embedding, store) VALUES (1, ?, 'keep_me'), (2, ?, 'old_s')",
        (old_vec, old_vec),
    )
    conn.commit()

    class _Vec0Storage:
        def __init__(self, c):
            self.conn = c

        def _generate_embedding(self, text: str):
            # 3-float vector fails vec0 FLOAT[2] dimension check on INSERT after DELETE!
            return [0.2, 0.3, 0.4] if text == "fail_insert" else [0.0, 1.0]

    ok, errs = asyncio.run(
        regen._regenerate_all(
            _Vec0Storage(conn),
            [
                {"content_hash": "h1", "content": "fail_insert", "store": "mutated"},
                {"content_hash": "h2", "content": "ok_insert", "store": "new_s"},
            ],
        )
    )
    conn.commit()
    assert (ok, errs) == (1, 1)

    r1 = conn.execute(
        "SELECT content_embedding, store FROM memory_embeddings WHERE rowid = 1"
    ).fetchone()
    assert r1 == (old_vec, "keep_me")

    r2 = conn.execute(
        "SELECT content_embedding, store FROM memory_embeddings WHERE rowid = 2"
    ).fetchone()
    assert r2 == (sqlite_vec.serialize_float32([0.0, 1.0]), "new_s")
