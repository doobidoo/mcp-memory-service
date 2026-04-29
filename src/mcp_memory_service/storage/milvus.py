# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Milvus storage backend for MCP Memory Service.

Uses the pymilvus MilvusClient API. Supports three deployment targets that
share the same code path:

  * Milvus Lite (default):       uri="./milvus.db"
  * Self-hosted Milvus server:    uri="http://localhost:19530"
  * Zilliz Cloud:                 uri="https://xxx.zillizcloud.com", token="..."

Design notes:
  * AUTOINDEX + COSINE metric, so the same schema works on Lite, server and Cloud.
  * Primary key is the memory's content_hash. This gives O(1) lookup and makes
    cleanup_duplicates a no-op (the PK itself enforces uniqueness).
  * Tags are stored as a comma-delimited string with leading and trailing
    commas (",python,web,"). Exact tag match is done with
    ``tags like "%,<tag>,%"`` which is supported by Milvus 2.4+ and Milvus Lite.
  * Metadata is stored as a JSON string in a VARCHAR field.
  * Deletion is a hard delete — Milvus does not provide efficient filtering of
    tombstones during ANN search, so there is no tombstone/soft-delete here.
"""

import asyncio
import json
import logging
import math
import os
import threading
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone, timedelta, date
from typing import Any, Dict, List, Optional, Tuple

# Disable wandb BEFORE importing sentence-transformers — same rationale as
# sqlite_vec.py (Issue #311). Safe to set even when transformers is unused.
os.environ.setdefault('WANDB_DISABLED', 'true')
os.environ.setdefault('WANDB_MODE', 'disabled')

try:
    from pymilvus import MilvusClient, DataType
    PYMILVUS_AVAILABLE = True
except ImportError:
    PYMILVUS_AVAILABLE = False
    MilvusClient = None  # type: ignore
    DataType = None  # type: ignore
    logging.getLogger(__name__).warning(
        "pymilvus not available. Install with: pip install pymilvus milvus-lite"
    )

# BM25 full-text search support (Milvus 2.5+).
# Function / FunctionType were added in pymilvus ≥ 2.5; older versions lack them.
try:
    from pymilvus import Function, FunctionType  # type: ignore[attr-defined]
    _BM25_IMPORTS_AVAILABLE = True
except ImportError:
    _BM25_IMPORTS_AVAILABLE = False
    Function = None  # type: ignore
    FunctionType = None  # type: ignore

# Hybrid search helpers (also Milvus 2.5+).
try:
    from pymilvus import AnnSearchRequest, RRFRanker  # type: ignore[attr-defined]
    _HYBRID_SEARCH_AVAILABLE = True
except ImportError:
    _HYBRID_SEARCH_AVAILABLE = False
    AnnSearchRequest = None  # type: ignore
    RRFRanker = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore

from .base import MemoryStorage
from ..models.memory import Memory, MemoryQueryResult

logger = logging.getLogger(__name__)


# -- Module-level caches / constants -----------------------------------------

# Embedding model cache, keyed by model name. Shared across MilvusMemoryStorage
# instances in the same process so that repeated factory creations (e.g., tests)
# don't reload a multi-hundred-MB model.
_MODEL_CACHE: Dict[str, Any] = {}
_DIMENSION_CACHE: Dict[str, int] = {}

# Bounded LRU embedding cache. Keyed by ``f"{model_name}::{text}"`` so different
# models don't collide, and the full text is stored rather than ``hash(text)``
# (Python's 64-bit hash can collide and silently return a wrong embedding,
# corrupting retrieval). The cache is capped to prevent unbounded growth in
# long-lived processes. asyncio is single-threaded, but the underlying
# embedding call runs via ``asyncio.to_thread`` so we guard with a Lock.
_EMBEDDING_CACHE_MAX = 1024
_EMBEDDING_CACHE: "OrderedDict[str, List[float]]" = OrderedDict()
_EMBEDDING_CACHE_LOCK = threading.Lock()


def _embedding_cache_get(key: str) -> Optional[List[float]]:
    with _EMBEDDING_CACHE_LOCK:
        value = _EMBEDDING_CACHE.get(key)
        if value is None:
            return None
        _EMBEDDING_CACHE.move_to_end(key)
        return value


def _embedding_cache_put(key: str, value: List[float]) -> None:
    with _EMBEDDING_CACHE_LOCK:
        if key in _EMBEDDING_CACHE:
            _EMBEDDING_CACHE.move_to_end(key)
            _EMBEDDING_CACHE[key] = value
            return
        _EMBEDDING_CACHE[key] = value
        while len(_EMBEDDING_CACHE) > _EMBEDDING_CACHE_MAX:
            _EMBEDDING_CACHE.popitem(last=False)


def _embedding_cache_size() -> int:
    with _EMBEDDING_CACHE_LOCK:
        return len(_EMBEDDING_CACHE)

# Milvus VARCHAR hard cap. We leave a small safety margin so overhead fields
# (metadata JSON keys etc.) don't push past the Milvus-side limit.
_MILVUS_VARCHAR_MAX = 65535
_CONTENT_MAX_LEN = _MILVUS_VARCHAR_MAX - 256
_TAGS_MAX_LEN = 8192
_ISO_MAX_LEN = 64
_MEMORY_TYPE_MAX_LEN = 128
_ID_MAX_LEN = 128

# Milvus per-call limit ceiling.
_MILVUS_MAX_LIMIT = 16384

# Reciprocal Rank Fusion smoothing constant for hybrid search.
# k=60 is the standard default from the RRF paper (Cormack et al., 2009).
RRF_RANKER_K = 60

# Defensive caps — mirror sqlite_vec semantics to prevent DoS-style large requests.
_MAX_TAGS_FOR_SEARCH = 100


def _sanitize_log_value(value: object) -> str:
    """Sanitize a user-provided value for safe inclusion in log messages."""
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")


def _escape_like(value: str) -> str:
    """Strip Milvus ``like`` wildcard characters from a user-supplied tag.

    Milvus uses ``%`` and ``_`` as wildcards in ``like`` expressions and does
    not support escape characters. We drop these so that a malicious or noisy
    tag like ``"a%b"`` cannot cause unintended cross-matches. Tag names in
    practice don't contain these characters.
    """
    return value.replace("%", "").replace("_", "")


def _tags_to_string(tags: Optional[List[str]]) -> str:
    """Encode a tag list as a comma-delimited string with leading/trailing commas.

    Leading/trailing commas let us match an exact tag with
    ``tags like "%,<tag>,%"`` rather than a substring match.
    """
    if not tags:
        return ""
    clean = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
    if not clean:
        return ""
    return "," + ",".join(clean) + ","


def _string_to_tags(raw: Optional[str]) -> List[str]:
    """Decode the comma-delimited tag string back into a list."""
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _safe_json_loads(text: str, context: str = "") -> Dict[str, Any]:
    """Parse JSON with defensive fallbacks; always return a dict."""
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("JSON decode error in %s: %s", context, exc)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("Non-dict JSON in %s: %s", context, type(parsed).__name__)
        return {}
    return parsed


# -- Storage implementation --------------------------------------------------


class MilvusMemoryStorage(MemoryStorage):
    """Milvus-backed storage implementation.

    Parameters
    ----------
    uri : str
        Milvus endpoint. ``./milvus.db`` (default) uses Milvus Lite (single
        local file, no external dependencies). An HTTP(S) URL points at a
        self-hosted Milvus or a Zilliz Cloud endpoint.
    token : Optional[str]
        Authentication token. Required for Zilliz Cloud; optional for
        self-hosted Milvus with auth enabled; ignored by Milvus Lite.
    collection_name : str
        Name of the Milvus collection that holds the memories. A new
        collection is created on ``initialize()`` if it does not already exist.
    embedding_model : str
        SentenceTransformer model name. ``all-MiniLM-L6-v2`` (384-dim) by
        default — matches the rest of the project.
    """

    @property
    def max_content_length(self) -> Optional[int]:
        return _CONTENT_MAX_LEN

    @property
    def supports_chunking(self) -> bool:
        return True

    def __init__(
        self,
        uri: str = "./milvus.db",
        token: Optional[str] = None,
        collection_name: str = "mcp_memory",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        if not PYMILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is required for MilvusMemoryStorage. "
                "Install with: pip install 'mcp-memory-service[milvus]'"
            )

        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        self.client: Optional[MilvusClient] = None
        self.embedding_model = None
        self.embedding_dimension = 384  # default for all-MiniLM-L6-v2
        self._initialized = False

        # Whether the live collection has a ``content_lower`` field. New
        # collections always do; pre-existing collections from older versions
        # may not. ``get_by_exact_content`` falls back to a client-side scan
        # when this is False. Set in ``_ensure_collection``.
        self._has_content_lower = False

        # Whether the collection has a BM25 function index on ``content``.
        # New collections get one automatically; pre-existing collections
        # without the ``sparse_vector`` field fall back to vector-only search.
        self._has_bm25 = False

        # Whether this storage is backed by Milvus Lite (embedded daemon) as
        # opposed to a remote Milvus server or Zilliz Cloud endpoint. We
        # mirror pymilvus's own heuristic (``uri.endswith('.db')``) — see
        # pymilvus/client/connection_manager.py. Only the Lite code path
        # reconnects on a dead channel, because Milvus Lite's daemon
        # subprocess becomes unreachable after roughly 60s of idle
        # (upstream issue https://github.com/milvus-io/milvus-lite/issues/334).
        # Remote backends don't have that problem and must fail fast instead.
        self._is_lite = bool(self.uri and self.uri.endswith(".db"))

        # Single lock per storage instance. Every CRUD call acquires this
        # once in :meth:`_call_client`. pymilvus's sync gRPC channel is not
        # safe under concurrent access from multiple worker threads against
        # Milvus Lite, and every write_lock holder also holds the sole right
        # to invoke the client — one lock is both enough and necessary.
        self._write_lock = asyncio.Lock()

        # Semantic deduplication — mirrors sqlite_vec knobs/env vars so
        # backends behave identically from the service layer's perspective.
        self.semantic_dedup_enabled = (
            os.getenv("MCP_SEMANTIC_DEDUP_ENABLED", "true").lower() == "true"
        )
        self.semantic_dedup_time_window = int(
            os.getenv("MCP_SEMANTIC_DEDUP_TIME_WINDOW_HOURS", "24")
        )
        self.semantic_dedup_threshold = float(
            os.getenv("MCP_SEMANTIC_DEDUP_THRESHOLD", "0.85")
        )

        # For a Lite-style file URI, make sure the parent directory exists.
        # For HTTP(S) URIs we skip this — the path component is not a filesystem path.
        if not self.uri.startswith(("http://", "https://")):
            parent = os.path.dirname(self.uri)
            if parent:
                os.makedirs(parent, exist_ok=True)

        logger.info("Initialized MilvusMemoryStorage (uri=%s, collection=%s)",
                    self.uri, self.collection_name)

    # -- Initialization ------------------------------------------------------

    async def initialize(self) -> None:
        """Connect to Milvus, load the embedding model, and ensure the collection exists."""
        if self._initialized:
            return

        await self._initialize_embedding_model()
        await asyncio.to_thread(self._connect_client)
        await asyncio.to_thread(self._ensure_collection)

        self._initialized = True
        logger.info(
            "MilvusMemoryStorage ready (collection=%s, dim=%s)",
            self.collection_name, self.embedding_dimension,
        )

    def _connect_client(self) -> None:
        kwargs: Dict[str, Any] = {"uri": self.uri}
        if self.token:
            kwargs["token"] = self.token
        self.client = MilvusClient(**kwargs)

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist.

        If a collection with the same name exists but has a different vector
        dimension, we log a warning rather than mutating it — the caller must
        decide whether to drop and rebuild.
        """
        assert self.client is not None

        if self.client.has_collection(collection_name=self.collection_name):
            self._validate_existing_collection()
            return

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=_ID_MAX_LEN,
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedding_dimension,
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=_MILVUS_VARCHAR_MAX,
            enable_analyzer=True,
        )
        # Lower-cased mirror of ``content``. Populated on every insert/upsert
        # so that ``get_by_exact_content`` can push a case-insensitive
        # substring filter down to Milvus (its ``like`` operator is
        # case-sensitive and has no escape syntax, so we match against the
        # pre-lowered mirror instead of scanning rows in Python).
        schema.add_field(
            field_name="content_lower",
            datatype=DataType.VARCHAR,
            max_length=_MILVUS_VARCHAR_MAX,
        )
        schema.add_field(
            field_name="tags",
            datatype=DataType.VARCHAR,
            max_length=_TAGS_MAX_LEN,
        )
        schema.add_field(
            field_name="memory_type",
            datatype=DataType.VARCHAR,
            max_length=_MEMORY_TYPE_MAX_LEN,
        )
        schema.add_field(
            field_name="metadata",
            datatype=DataType.VARCHAR,
            max_length=_MILVUS_VARCHAR_MAX,
        )
        schema.add_field(field_name="created_at", datatype=DataType.DOUBLE)
        schema.add_field(field_name="updated_at", datatype=DataType.DOUBLE)
        schema.add_field(
            field_name="created_at_iso",
            datatype=DataType.VARCHAR,
            max_length=_ISO_MAX_LEN,
        )
        schema.add_field(
            field_name="updated_at_iso",
            datatype=DataType.VARCHAR,
            max_length=_ISO_MAX_LEN,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        # BM25 full-text search: add sparse vector field + BM25 function
        # for new collections when pymilvus supports it (≥ 2.5).
        if _BM25_IMPORTS_AVAILABLE:
            try:
                schema.add_field(
                    field_name="sparse_vector",
                    datatype=DataType.SPARSE_FLOAT_VECTOR,
                )
                bm25_function = Function(
                    name="bm25_fn",
                    input_field_names=["content"],
                    output_field_names=["sparse_vector"],
                    function_type=FunctionType.BM25,
                )
                schema.add_function(bm25_function)
                index_params.add_index(
                    field_name="sparse_vector",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="BM25",
                )
                self._has_bm25 = True
                logger.info("BM25 full-text search enabled for new collection '%s'", self.collection_name)
            except Exception as exc:
                logger.warning(
                    "Failed to add BM25 function to collection '%s' — "
                    "using vector-only search: %s",
                    self.collection_name, exc,
                )
                self._has_bm25 = False

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        self._has_content_lower = True
        logger.info(
            "Created Milvus collection '%s' (dim=%s)",
            self.collection_name, self.embedding_dimension,
        )

    @staticmethod
    def _extract_vector_dim(info: Dict[str, Any]) -> Optional[int]:
        """Return the vector field dimension from a ``describe_collection`` payload."""
        for field in info.get("fields", []):
            if field.get("name") != "vector":
                continue
            params = field.get("params") or {}
            raw = params.get("dim") if params.get("dim") is not None else field.get("dim")
            try:
                return int(raw) if raw is not None else None
            except (TypeError, ValueError):
                return None
        return None

    def _validate_existing_collection(self) -> None:
        """Warn if the on-disk collection's vector dim disagrees with our model.

        Also detects whether the collection has the ``content_lower`` field
        that ``get_by_exact_content`` relies on for server-side filtering.
        Collections created before this field was introduced fall back to a
        client-side scan; a warning is logged once on connection.
        """
        assert self.client is not None
        try:
            info = self.client.describe_collection(collection_name=self.collection_name)
        except Exception as exc:  # describe may fail on very old servers
            logger.debug("describe_collection failed (ignored): %s", exc)
            return

        dim = self._extract_vector_dim(info)
        if dim is not None and dim != self.embedding_dimension:
            logger.warning(
                "Existing Milvus collection '%s' uses dim=%s but the current "
                "embedding model produces dim=%s. Retrieval will fail until the "
                "dimensions match. Drop the collection or switch embedding models.",
                self.collection_name, dim, self.embedding_dimension,
            )

        field_names = {f.get("name") for f in info.get("fields", [])}
        self._has_content_lower = "content_lower" in field_names
        if not self._has_content_lower:
            logger.warning(
                "Collection '%s' lacks the 'content_lower' field — "
                "get_by_exact_content will fall back to a slower client-side "
                "scan. Recreate the collection to pick up server-side filtering.",
                self.collection_name,
            )

        # Detect BM25 capability by checking for sparse_vector field
        self._has_bm25 = "sparse_vector" in field_names
        if not self._has_bm25:
            logger.warning(
                "BM25 full-text search unavailable for collection '%s' — "
                "using vector-only search",
                self.collection_name,
            )

    async def _initialize_embedding_model(self) -> None:
        """Load the sentence-transformers model (or cached instance)."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is required for MilvusMemoryStorage. "
                "Install with: pip install 'mcp-memory-service[milvus]'"
            )

        cache_key = f"st_{self.embedding_model_name}"
        if cache_key in _MODEL_CACHE:
            self.embedding_model = _MODEL_CACHE[cache_key]
            self.embedding_dimension = _DIMENSION_CACHE.get(cache_key, self.embedding_dimension)
            return

        def _load_model():
            device = self._resolve_device()
            # If the model is already cached locally, load from the concrete
            # snapshot path and turn on offline mode. That avoids the HEAD
            # request huggingface_hub makes when you pass a model name, which
            # retries for ~30s on networks that can't reach huggingface.co
            # even though the model is fully present on disk. Mirrors the
            # approach used in sqlite_vec._initialize_embedding_model.
            hf_home = os.environ.get('HF_HOME', os.path.expanduser("~/.cache/huggingface"))
            safe_name = self.embedding_model_name.replace('/', '--')
            cache_path = os.path.join(
                hf_home, "hub", f"models--sentence-transformers--{safe_name}"
            )
            local_snapshot = None
            if os.path.isdir(cache_path):
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                snapshots_dir = os.path.join(cache_path, "snapshots")
                if os.path.isdir(snapshots_dir):
                    children = [
                        os.path.join(snapshots_dir, d)
                        for d in os.listdir(snapshots_dir)
                        if os.path.isdir(os.path.join(snapshots_dir, d))
                    ]
                    if children:
                        local_snapshot = children[0]

            if local_snapshot:
                logger.info("Loading cached model from %s on device=%s", local_snapshot, device)
                return SentenceTransformer(local_snapshot, device=device)

            logger.info(
                "Loading embedding model '%s' on device=%s (may download)",
                self.embedding_model_name, device,
            )
            return SentenceTransformer(self.embedding_model_name, device=device)

        self.embedding_model = await asyncio.to_thread(_load_model)

        probe = await asyncio.to_thread(
            self.embedding_model.encode, ["__dimension_probe__"]
        )
        try:
            self.embedding_dimension = int(probe.shape[1])
        except AttributeError:
            self.embedding_dimension = len(probe[0])

        _MODEL_CACHE[cache_key] = self.embedding_model
        _DIMENSION_CACHE[cache_key] = self.embedding_dimension
        logger.info("Embedding model loaded (dim=%s)", self.embedding_dimension)

    def _resolve_device(self) -> str:
        """Pick the best available torch device without forcing torch as a hard dep."""
        try:
            from ..utils.system_detection import get_torch_device
            return get_torch_device()
        except Exception:  # noqa: BLE001 — fall back silently
            return "cpu"

    def _validate_embedding(self, embedding: List[float]) -> None:
        """Raise ValueError if ``embedding`` has wrong dim or contains bad values."""
        if len(embedding) != self.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(embedding)}"
            )
        if not all(
            isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x)
            for x in embedding
        ):
            raise ValueError("Embedding contains NaN or infinity")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single piece of text (sync).

        Results are memoized in a bounded LRU keyed by ``(model_name, text)``
        — keying by the full string avoids the silent-wrong-embedding risk of
        ``hash(text)`` collisions.
        """
        if not self.embedding_model:
            raise RuntimeError("Embedding model not loaded. Call initialize() first.")

        cache_key = f"{self.embedding_model_name}::{text}"
        cached = _embedding_cache_get(cache_key)
        if cached is not None:
            return cached

        raw = self.embedding_model.encode([text], convert_to_numpy=True)[0]
        embedding = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        self._validate_embedding(embedding)
        _embedding_cache_put(cache_key, embedding)
        return embedding

    # -- Helpers -------------------------------------------------------------

    def _ensure_initialized(self) -> bool:
        if not self._initialized or self.client is None:
            logger.error("MilvusMemoryStorage used before initialize()")
            return False
        return True

    # Error substrings that indicate the underlying gRPC channel to the
    # local Milvus Lite daemon is gone (upstream milvus-lite issue #334 —
    # the daemon subprocess becomes unreachable after ~60s of idle). We
    # intentionally match on substrings rather than exception types because
    # grpc raises a plain ``ValueError`` for the closed-channel case while
    # pymilvus raises ``MilvusException`` for the server-unavailable case.
    _LITE_DEAD_CHANNEL_MARKERS = (
        "Cannot invoke RPC on closed channel",
        "server unavailable",
        "Fail connecting to server",
        "illegal connection params",
    )

    @classmethod
    def _is_lite_dead_channel_error(cls, exc: BaseException) -> bool:
        """Match the exact two error signatures from upstream issue #334."""
        text = str(exc)
        return any(m in text for m in cls._LITE_DEAD_CHANNEL_MARKERS)

    def _reconnect_lite_client(self) -> None:
        """Replace ``self.client`` with a fresh ``MilvusClient`` for the same
        Lite database file.

        Must be called while ``self._write_lock`` is held so concurrent
        coroutines can't both observe the dead client and start reconnecting
        at the same time. Does NOT call ``close()`` on the old client — that
        call would block or raise on a dead Lite daemon. We just drop the
        reference and let GC collect it.
        """
        kwargs: Dict[str, Any] = {"uri": self.uri}
        if self.token:
            kwargs["token"] = self.token
        self.client = MilvusClient(**kwargs)

    async def _call_client(self, method_name: str, *args, **kwargs):
        """Invoke ``self.client.<method_name>(*args, **kwargs)`` safely.

        * Every invocation holds ``self._write_lock`` (one lock per storage
          instance) — pymilvus's sync gRPC channel is not safe under
          concurrent access from multiple worker threads against Milvus Lite.
        * The sync pymilvus method runs via ``asyncio.to_thread`` so it
          doesn't block the event loop.
        * **Remote Milvus / Zilliz Cloud:** any RPC failure is logged with
          full traceback and re-raised. No reconnect. The caller translates
          the exception into its contract return value ``(False, <message>)``.
        * **Milvus Lite only:** if the RPC fails with one of the two exact
          error signatures from upstream issue #334 (closed channel /
          server unavailable), we replace ``self.client`` with a fresh one
          (still holding the same lock) and retry the call exactly once.
          A failure on the retry propagates unchanged.
        """
        async with self._write_lock:
            if self.client is None:
                raise RuntimeError("MilvusMemoryStorage was not initialized")
            return await self._invoke_locked(method_name, args, kwargs)

    async def _invoke_locked(
        self, method_name: str, args: tuple, kwargs: dict,
    ):
        """Run the RPC under the already-acquired write lock.

        Separated from :meth:`_call_client` so both halves stay well under
        the complexity-≤8 budget.
        """
        try:
            return await asyncio.to_thread(
                getattr(self.client, method_name), *args, **kwargs
            )
        except Exception as exc:  # noqa: BLE001 — always log + re-raise
            if self._is_lite and self._is_lite_dead_channel_error(exc):
                logger.warning(
                    "Milvus Lite channel died on %s (%s) — reconnecting and retrying once "
                    "(upstream milvus-lite issue #334)",
                    method_name, exc,
                )
                self._reconnect_lite_client()
                result = await asyncio.to_thread(
                    getattr(self.client, method_name), *args, **kwargs
                )
                logger.info(
                    "Milvus Lite reconnect succeeded; %s retried OK", method_name,
                )
                return result
            logger.exception("Milvus RPC failed on %s: %s", method_name, exc)
            raise

    @staticmethod
    def _iso_from_epoch(epoch: float) -> str:
        """Render an epoch second value as an ISO-8601 UTC string."""
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

    def _resolve_timestamps(
        self, memory: Memory
    ) -> Tuple[float, str, float, str]:
        """Return ``(created_at, created_at_iso, updated_at, updated_at_iso)``.

        Derives both epoch and ISO values from the same ``now`` when a field
        is missing, guaranteeing the two representations refer to the same
        moment (previously ``created_at`` and ``created_at_iso`` could be
        produced from different clock reads).
        """
        now = time.time()
        created_at = float(memory.created_at) if memory.created_at is not None else now
        created_iso = memory.created_at_iso or self._iso_from_epoch(created_at)
        updated_at = float(memory.updated_at) if memory.updated_at is not None else now
        updated_iso = memory.updated_at_iso or self._iso_from_epoch(updated_at)
        return created_at, created_iso, updated_at, updated_iso

    def _memory_to_entity(self, memory: Memory, embedding: List[float]) -> Dict[str, Any]:
        created_at, created_iso, updated_at, updated_iso = self._resolve_timestamps(memory)
        content = memory.content or ""
        entity: Dict[str, Any] = {
            "id": memory.content_hash,
            "vector": embedding,
            "content": content,
            "tags": _tags_to_string(memory.tags),
            "memory_type": memory.memory_type or "",
            "metadata": json.dumps(memory.metadata) if memory.metadata else "{}",
            "created_at": created_at,
            "updated_at": updated_at,
            "created_at_iso": created_iso,
            "updated_at_iso": updated_iso,
        }
        if self._has_content_lower:
            entity["content_lower"] = content.lower()
        return entity

    def _entity_to_memory(self, row: Dict[str, Any]) -> Optional[Memory]:
        try:
            return Memory(
                content=row.get("content", "") or "",
                content_hash=row.get("id", "") or "",
                tags=_string_to_tags(row.get("tags")),
                memory_type=row.get("memory_type") or None,
                metadata=_safe_json_loads(row.get("metadata", ""), "milvus_entity"),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
                created_at_iso=row.get("created_at_iso") or None,
                updated_at_iso=row.get("updated_at_iso") or None,
            )
        except Exception as exc:  # noqa: BLE001 — never kill a batch on one bad row
            logger.warning("Failed to convert Milvus entity to Memory: %s", exc)
            return None

    @staticmethod
    def _tag_like_clauses(tags: List[str], joiner: str) -> Tuple[str, bool]:
        """Build a ``tags like "%,...,%"`` clause for the given tags.

        Returns ``(clause, matched_any)``. ``matched_any`` is False iff every
        input tag was rejected (e.g. non-string); callers use this to decide
        whether to return empty results rather than ignore the filter.
        """
        tag_clauses = []
        for tag in tags:
            if not isinstance(tag, str):
                logger.warning("Skipping non-string tag of type %s", type(tag).__name__)
                continue
            stripped = _escape_like(tag.strip())
            if not stripped:
                continue
            safe = stripped.replace('"', '\\"')
            tag_clauses.append(f'tags like "%,{safe},%"')

        if not tag_clauses:
            return "", False
        if len(tag_clauses) == 1:
            return tag_clauses[0], True
        return "(" + f" {joiner} ".join(tag_clauses) + ")", True

    @staticmethod
    def _combine_filter(*parts: Optional[str]) -> str:
        """AND-combine non-empty filter fragments into a single filter string."""
        keep = [p for p in parts if p]
        if not keep:
            return ""
        if len(keep) == 1:
            return keep[0]
        return " and ".join(f"({p})" for p in keep)

    _OUTPUT_FIELDS = (
        "id", "content", "tags", "memory_type", "metadata",
        "created_at", "updated_at", "created_at_iso", "updated_at_iso",
    )

    # -- Semantic dedup ------------------------------------------------------

    async def _check_semantic_duplicate(
        self,
        content: str,
        time_window_hours: int,
        similarity_threshold: float,
    ) -> Tuple[bool, Optional[str]]:
        """Look for a recently stored memory that is semantically similar.

        Returns ``(is_duplicate, existing_hash)``. Mirrors the sqlite_vec
        implementation: search the top-1 neighbour inside the time window and
        compare its cosine similarity against the threshold.
        """
        if not self._ensure_initialized():
            return False, None

        cutoff = time.time() - time_window_hours * 3600
        try:
            embedding = self._generate_embedding(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Semantic dedup skipped — embedding failed: %s", exc)
            return False, None

        try:
            results = await self._call_client(
                "search",
                collection_name=self.collection_name,
                data=[embedding],
                anns_field="vector",
                filter=f"created_at > {cutoff}",
                limit=1,
                output_fields=["id"],
                search_params={"metric_type": "COSINE"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Semantic dedup search failed: %s", exc)
            return False, None

        if not results or not results[0]:
            return False, None
        hit = results[0][0]
        similarity = float(hit.get("distance", 0.0))
        if similarity >= similarity_threshold:
            return True, hit.get("id") or hit.get("entity", {}).get("id")
        return False, None

    # -- Store ---------------------------------------------------------------

    async def store(self, memory: Memory, skip_semantic_dedup: bool = False) -> Tuple[bool, str]:
        # Explicit invariant log: if this ever reports True, something
        # outside ``close()`` has nulled out the client and that is a bug.
        logger.debug("store() entry — self.client is None: %r", self.client is None)
        if not self._ensure_initialized():
            return False, "Milvus storage not initialized"

        try:
            existing = await self.get_by_hash(memory.content_hash)
            if existing is not None:
                return False, "Duplicate content detected (exact match)"

            if self.semantic_dedup_enabled and not skip_semantic_dedup:
                is_dup, hit_hash = await self._check_semantic_duplicate(
                    memory.content,
                    time_window_hours=self.semantic_dedup_time_window,
                    similarity_threshold=self.semantic_dedup_threshold,
                )
                if is_dup:
                    return False, f"Duplicate content detected (semantically similar to {hit_hash})"

            try:
                embedding = self._generate_embedding(memory.content)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to embed memory %s: %s", memory.content_hash, exc)
                return False, f"Failed to generate embedding: {exc}"

            entity = self._memory_to_entity(memory, embedding)

            await self._call_client(
                "insert",
                collection_name=self.collection_name,
                data=[entity],
            )

            logger.info("Stored memory %s", memory.content_hash)
            return True, "Memory stored successfully"

        except Exception as exc:  # noqa: BLE001 — contract requires (bool, str) not a raise
            logger.error("Failed to store memory: %s\n%s", exc, traceback.format_exc())
            return False, f"Failed to store memory: {exc}"

    async def _prepare_batch_entity(
        self, memory: Memory
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Build a single batch entity or return a failure message for it."""
        try:
            existing = await self.get_by_hash(memory.content_hash)
            if existing is not None:
                return None, "Duplicate content detected (exact match)"
            embedding = self._generate_embedding(memory.content)
        except Exception as exc:  # noqa: BLE001 — report per-item, don't abort the batch
            return None, f"Failed to prepare memory: {exc}"
        return self._memory_to_entity(memory, embedding), None

    async def _flush_batch_insert(
        self,
        results: List[Tuple[bool, str]],
        to_insert: List[Dict[str, Any]],
        insert_indices: List[int],
    ) -> None:
        """Commit a prepared batch and update ``results`` in place."""
        if not to_insert:
            return
        try:
            await self._call_client(
                "insert",
                collection_name=self.collection_name,
                data=to_insert,
            )
            outcome = (True, "Memory stored successfully")
        except Exception as exc:  # noqa: BLE001 — whole batch failed
            logger.error("Milvus batch insert failed: %s", exc)
            outcome = (False, f"Failed to store memory: {exc}")
        for idx in insert_indices:
            results[idx] = outcome

    async def store_batch(self, memories: List[Memory]) -> List[Tuple[bool, str]]:
        if not memories:
            return []
        if not self._ensure_initialized():
            return [(False, "Milvus storage not initialized")] * len(memories)

        results: List[Tuple[bool, str]] = [(False, "not processed")] * len(memories)
        to_insert: List[Dict[str, Any]] = []
        insert_indices: List[int] = []

        for idx, memory in enumerate(memories):
            entity, err = await self._prepare_batch_entity(memory)
            if entity is None:
                results[idx] = (False, err or "Failed to prepare memory")
            else:
                to_insert.append(entity)
                insert_indices.append(idx)

        await self._flush_batch_insert(results, to_insert, insert_indices)
        return results

    # -- Retrieve ------------------------------------------------------------

    def _build_tag_filter(
        self, tags: Optional[List[str]]
    ) -> Tuple[str, bool]:
        """Prepare a tag-based filter for ``retrieve``.

        Returns ``(filter_expression, ok)``. ``ok=False`` means the caller
        must short-circuit to an empty result list (tags were supplied but
        all rejected).
        """
        if not tags:
            return "", True
        if len(tags) > _MAX_TAGS_FOR_SEARCH:
            logger.warning(
                "Too many tags (%s), truncating to %s",
                len(tags), _MAX_TAGS_FOR_SEARCH,
            )
            tags = tags[:_MAX_TAGS_FOR_SEARCH]
        tag_filter, matched = self._tag_like_clauses(tags, joiner="or")
        if not matched:
            logger.warning("Tag filter had no valid tags; returning empty.")
            return "", False
        return tag_filter, True

    def _hit_to_row(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a MilvusClient search hit into a plain row dict.

        Different pymilvus versions return output_fields inlined on the hit,
        or nested under an ``entity`` key. Handle both consistently.
        """
        entity = hit.get("entity") if isinstance(hit, dict) else None
        row = dict(entity) if entity else {}
        for field in self._OUTPUT_FIELDS:
            if field in hit and field not in row:
                row[field] = hit[field]
        if "id" not in row and "id" in hit:
            row["id"] = hit["id"]
        return row

    def _hit_to_result(
        self, hit: Dict[str, Any], query: str
    ) -> Optional[MemoryQueryResult]:
        row = self._hit_to_row(hit)
        memory = self._entity_to_memory(row)
        if memory is None:
            return None
        similarity = float(hit.get("distance", 0.0))
        relevance = max(0.0, min(1.0, similarity))
        memory.record_access(query)
        return MemoryQueryResult(
            memory=memory,
            relevance_score=relevance,
            debug_info={"distance": similarity, "backend": "milvus"},
        )

    async def _run_search(
        self, query_embedding: List[float], tag_filter: str, fetch_n: int
    ) -> List[Dict[str, Any]]:
        """Execute a vector search with a tag filter, returning the raw hit list."""
        try:
            search_results = await self._call_client(
                "search",
                collection_name=self.collection_name,
                data=[query_embedding],
                anns_field="vector",
                filter=tag_filter,
                limit=fetch_n,
                output_fields=list(self._OUTPUT_FIELDS),
                search_params={"metric_type": "COSINE"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Milvus search failed: %s", exc)
            return []
        if not search_results or not search_results[0]:
            return []
        return list(search_results[0])

    async def _run_hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        tag_filter: str,
        fetch_n: int,
    ) -> List[Dict[str, Any]]:
        """Execute hybrid vector + BM25 search with RRF merging.

        Falls back to vector-only search if hybrid_search fails.
        """
        if not _HYBRID_SEARCH_AVAILABLE:
            return await self._run_search(query_embedding, tag_filter, fetch_n)

        try:
            # Dense vector ANN request
            vector_req = AnnSearchRequest(
                data=[query_embedding],
                anns_field="vector",
                param={"metric_type": "COSINE"},
                limit=fetch_n,
                expr=tag_filter if tag_filter else None,
            )

            # BM25 sparse vector request
            bm25_req = AnnSearchRequest(
                data=[query],
                anns_field="sparse_vector",
                param={"metric_type": "BM25"},
                limit=fetch_n,
                expr=tag_filter if tag_filter else None,
            )

            search_results = await self._call_client(
                "hybrid_search",
                collection_name=self.collection_name,
                reqs=[vector_req, bm25_req],
                ranker=RRFRanker(k=RRF_RANKER_K),
                limit=fetch_n,
                output_fields=list(self._OUTPUT_FIELDS),
            )

            if not search_results or not search_results[0]:
                return []

            # Deduplicate by content_hash (id field), keeping higher-ranked entry
            seen: set = set()
            deduped: List[Dict[str, Any]] = []
            for hit in search_results[0]:
                hit_id = hit.get("id") or hit.get("entity", {}).get("id")
                if hit_id and hit_id in seen:
                    continue
                if hit_id:
                    seen.add(hit_id)
                deduped.append(hit)
            return deduped

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Hybrid search failed, falling back to vector-only: %s", exc,
            )
            return await self._run_search(query_embedding, tag_filter, fetch_n)

    def _embed_query(self, query: str) -> Optional[List[float]]:
        """Return an embedding for ``query`` or ``None`` on failure (logged)."""
        try:
            return self._generate_embedding(query)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to generate query embedding: %s", exc)
            return None

    @staticmethod
    def _retrieve_fetch_limit(n_results: int, tag_filter: str) -> int:
        # Filtered ANN under HNSW can return fewer than `limit` results when
        # the selectivity filter eliminates candidates; over-fetch 3× to
        # compensate so the caller still gets the requested number of hits.
        base = n_results * 3 if tag_filter else n_results
        return max(1, min(base, _MILVUS_MAX_LIMIT))

    def _rank_and_trim(
        self,
        hits: List[Dict[str, Any]],
        query: str,
        n_results: int,
        min_confidence: float,
    ) -> List[MemoryQueryResult]:
        results: List[MemoryQueryResult] = []
        for hit in hits:
            result = self._hit_to_result(hit, query)
            if result is not None:
                results.append(result)
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        results = results[:n_results]
        if min_confidence > 0.0:
            results = [r for r in results if r.relevance_score >= min_confidence]
        return results

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> List[MemoryQueryResult]:
        logger.debug("retrieve() entry — self.client is None: %r", self.client is None)
        if not self._ensure_initialized():
            return []

        query_embedding = self._embed_query(query)
        if query_embedding is None:
            return []

        tag_filter, ok = self._build_tag_filter(tags)
        if not ok:
            return []

        fetch_n = self._retrieve_fetch_limit(n_results, tag_filter)
        if self._has_bm25:
            hits = await self._run_hybrid_search(query, query_embedding, tag_filter, fetch_n)
        else:
            hits = await self._run_search(query_embedding, tag_filter, fetch_n)
        return self._rank_and_trim(hits, query, n_results, min_confidence)

    # -- Tag-based search ----------------------------------------------------

    async def search_by_tag(
        self,
        tags: List[str],
        time_start: Optional[float] = None,
    ) -> List[Memory]:
        if not tags or not self._ensure_initialized():
            return []

        tag_filter, matched = self._tag_like_clauses(tags, joiner="or")
        if not matched:
            return []
        time_filter = f"created_at >= {time_start}" if time_start is not None else None
        filter_expr = self._combine_filter(tag_filter, time_filter)

        return await self._query_memories(
            filter_expr=filter_expr,
            limit=_MILVUS_MAX_LIMIT,
            sort_desc_key="created_at",
        )

    @staticmethod
    def _normalize_tag_operation(operation: str) -> str:
        op = (operation or "AND").strip().upper()
        if op not in ("AND", "OR"):
            logger.warning("Unsupported tag operation '%s'; defaulting to AND", operation)
            return "AND"
        return op

    @staticmethod
    def _time_range_filters(
        time_start: Optional[float], time_end: Optional[float]
    ) -> List[str]:
        parts: List[str] = []
        if time_start is not None:
            parts.append(f"created_at >= {time_start}")
        if time_end is not None:
            parts.append(f"created_at <= {time_end}")
        return parts

    async def search_by_tags(
        self,
        tags: List[str],
        operation: str = "AND",
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
    ) -> List[Memory]:
        if not tags or not self._ensure_initialized():
            return []

        op = self._normalize_tag_operation(operation)
        tag_filter, matched = self._tag_like_clauses(
            tags, joiner="and" if op == "AND" else "or"
        )
        if not matched:
            return []

        time_parts = self._time_range_filters(time_start, time_end)
        filter_expr = self._combine_filter(tag_filter, *time_parts)

        return await self._query_memories(
            filter_expr=filter_expr,
            limit=_MILVUS_MAX_LIMIT,
            sort_desc_key="updated_at",
        )

    # -- Delete --------------------------------------------------------------

    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        if not self._ensure_initialized():
            return False, "Milvus storage not initialized"

        existing = await self.get_by_hash(content_hash)
        if existing is None:
            return False, f"Memory with hash {content_hash} not found"

        try:
            await self._call_client(
                "delete",
                collection_name=self.collection_name,
                ids=[content_hash],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to delete memory %s: %s", content_hash, exc)
            return False, f"Failed to delete memory: {exc}"

        logger.info("Deleted memory %s", content_hash)
        return True, f"Successfully deleted memory {content_hash}"

    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        if not self._ensure_initialized():
            return 0, "Milvus storage not initialized"

        tag_filter, matched = self._tag_like_clauses([tag], joiner="or")
        if not matched:
            return 0, f"No memories found with tag '{tag}'"
        return await self._delete_matching(tag_filter, f"Successfully deleted {{count}} memories with tag '{tag}'")

    async def delete_by_tags(self, tags: List[str]) -> Tuple[int, str, List[str]]:
        if not self._ensure_initialized():
            return 0, "Milvus storage not initialized", []
        if not tags:
            return 0, "No tags provided", []

        tag_filter, matched = self._tag_like_clauses(tags, joiner="or")
        if not matched:
            return 0, f"No memories found matching any of the {len(tags)} tags", []

        hashes = await self._collect_hashes(tag_filter)
        if not hashes:
            return 0, f"No memories found matching any of the {len(tags)} tags", []

        await self._call_client(
            "delete",
            collection_name=self.collection_name,
            ids=hashes,
        )
        return (
            len(hashes),
            f"Successfully deleted {len(hashes)} memories matching {len(tags)} tag(s)",
            hashes,
        )

    async def delete_by_timeframe(
        self,
        start_date: date,
        end_date: date,
        tag: Optional[str] = None,
    ) -> Tuple[int, str]:
        if not self._ensure_initialized():
            return 0, "Milvus storage not initialized"

        start_ts = datetime.combine(start_date, datetime.min.time()).timestamp()
        end_ts = datetime.combine(end_date, datetime.max.time()).timestamp()
        time_filter = f"created_at >= {start_ts} and created_at <= {end_ts}"

        tag_filter = ""
        if tag:
            tag_filter, matched = self._tag_like_clauses([tag], joiner="or")
            if not matched:
                return 0, f"Deleted 0 memories from {start_date} to {end_date}" + (
                    f" with tag '{tag}'" if tag else ""
                )

        filter_expr = self._combine_filter(tag_filter, time_filter)
        suffix = f" with tag '{tag}'" if tag else ""
        count, _ = await self._delete_matching_parts(
            filter_expr,
            f"Deleted {{count}} memories from {start_date} to {end_date}{suffix}",
        )
        return count, f"Deleted {count} memories from {start_date} to {end_date}{suffix}"

    async def delete_before_date(
        self,
        before_date: date,
        tag: Optional[str] = None,
    ) -> Tuple[int, str]:
        if not self._ensure_initialized():
            return 0, "Milvus storage not initialized"

        before_ts = datetime.combine(before_date, datetime.min.time()).timestamp()
        time_filter = f"created_at < {before_ts}"

        tag_filter = ""
        if tag:
            tag_filter, matched = self._tag_like_clauses([tag], joiner="or")
            if not matched:
                return 0, f"Deleted 0 memories before {before_date}" + (
                    f" with tag '{tag}'" if tag else ""
                )

        filter_expr = self._combine_filter(tag_filter, time_filter)
        suffix = f" with tag '{tag}'" if tag else ""
        count, _ = await self._delete_matching_parts(
            filter_expr,
            f"Deleted {{count}} memories before {before_date}{suffix}",
        )
        return count, f"Deleted {count} memories before {before_date}{suffix}"

    async def _delete_matching(self, filter_expr: str, success_template: str) -> Tuple[int, str]:
        count, hashes = await self._delete_matching_parts(filter_expr, success_template)
        if count > 0:
            return count, success_template.format(count=count)
        return 0, "No memories found"

    async def _delete_matching_parts(
        self, filter_expr: str, _success_template: str
    ) -> Tuple[int, List[str]]:
        hashes = await self._collect_hashes(filter_expr)
        if not hashes:
            return 0, []
        await self._call_client(
            "delete",
            collection_name=self.collection_name,
            ids=hashes,
        )
        return len(hashes), hashes

    async def _collect_hashes(self, filter_expr: str) -> List[str]:
        rows = await self._call_client(
            "query",
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=["id"],
            limit=_MILVUS_MAX_LIMIT,
        )
        return [row["id"] for row in rows if row.get("id")]

    # -- Reads ---------------------------------------------------------------

    async def get_by_hash(self, content_hash: str) -> Optional[Memory]:
        if not self._ensure_initialized():
            return None

        try:
            rows = await self._call_client(
                "get",
                collection_name=self.collection_name,
                ids=[content_hash],
                output_fields=list(self._OUTPUT_FIELDS),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to get memory %s: %s", content_hash, exc)
            return None

        if not rows:
            return None
        return self._entity_to_memory(rows[0])

    async def get_by_exact_content(self, content: str) -> List[Memory]:
        """Case-insensitive substring match on content.

        Matches sqlite_vec's semantics (``LIKE '%...%' COLLATE NOCASE``
        ordered by ``created_at DESC``). The filter is pushed down to Milvus
        via the mirrored ``content_lower`` field so we don't fetch every row
        into Python. On legacy collections that predate ``content_lower``,
        falls back to a bounded client-side scan.
        """
        if not self._ensure_initialized():
            return []

        if not self._has_content_lower:
            return await self._get_by_exact_content_fallback(content)

        needle = _escape_like(content.lower())
        if not needle:
            return []
        safe = needle.replace('"', '\\"')
        filter_expr = f'content_lower like "%{safe}%"'
        return await self._query_memories(
            filter_expr=filter_expr,
            limit=_MILVUS_MAX_LIMIT,
            sort_desc_key="created_at",
        )

    async def _get_by_exact_content_fallback(self, content: str) -> List[Memory]:
        """Client-side scan for collections without ``content_lower`` field."""
        memories = await self.get_all_memories()
        needle = content.lower()
        return [m for m in memories if needle in (m.content or "").lower()]

    async def cleanup_duplicates(self) -> Tuple[int, str]:
        # With content_hash as the primary key, Milvus rejects duplicate PKs
        # at insert time (upsert replaces in place), so there is nothing to
        # clean up. Keep the method for contract compliance.
        return 0, "No duplicate memories found"

    _PROTECTED_UPDATE_KEYS = frozenset({
        "content", "content_hash", "tags", "memory_type", "metadata",
        "embedding", "created_at", "created_at_iso", "updated_at", "updated_at_iso",
    })

    def _resolve_updated_tags(
        self, existing: Memory, updates: Dict[str, Any]
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        if "tags" not in updates:
            return list(existing.tags), None
        if not isinstance(updates["tags"], list):
            return None, "Tags must be provided as a list of strings"
        return list(updates["tags"]), None

    def _resolve_updated_metadata(
        self, existing: Memory, updates: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        new_metadata = dict(existing.metadata or {})
        if "metadata" in updates:
            if not isinstance(updates["metadata"], dict):
                return None, "Metadata must be provided as a dictionary"
            new_metadata.update(updates["metadata"])
        for key, value in updates.items():
            if key not in self._PROTECTED_UPDATE_KEYS:
                new_metadata[key] = value
        return new_metadata, None

    def _merge_updates(
        self, existing: Memory, updates: Dict[str, Any]
    ) -> Tuple[Optional[Tuple[List[str], Optional[str], Dict[str, Any]]], Optional[str]]:
        """Validate ``updates`` and merge them into a (tags, type, metadata) tuple.

        Returns ``(merged, None)`` on success, or ``(None, error_message)`` on
        a validation failure.
        """
        new_tags, err = self._resolve_updated_tags(existing, updates)
        if err is not None:
            return None, err
        new_type = updates.get("memory_type", existing.memory_type)
        new_metadata, err = self._resolve_updated_metadata(existing, updates)
        if err is not None:
            return None, err
        return (new_tags, new_type, new_metadata), None

    def _compute_update_timestamps(
        self, existing: Memory, updates: Dict[str, Any], preserve_timestamps: bool
    ) -> Tuple[Optional[float], Optional[str], float, str]:
        """Pick the (created_at, created_at_iso, updated_at, updated_at_iso) tuple for an upsert.

        Both ``now`` and its ISO rendering come from the same ``time.time()``
        reading, so the two representations never refer to different moments.
        """
        now = time.time()
        now_iso = self._iso_from_epoch(now)
        structural = any(k in updates for k in ("tags", "memory_type", "content"))

        if preserve_timestamps and not structural:
            updated_at = existing.updated_at if existing.updated_at is not None else now
            updated_at_iso = existing.updated_at_iso or now_iso
            return existing.created_at, existing.created_at_iso, updated_at, updated_at_iso

        if not preserve_timestamps:
            created_at = updates.get("created_at", existing.created_at)
            created_at_iso = updates.get("created_at_iso", existing.created_at_iso)
            return (
                created_at, created_at_iso,
                updates.get("updated_at", now), updates.get("updated_at_iso", now_iso),
            )

        # preserve_timestamps=True with a structural change — bump updated_at.
        return existing.created_at, existing.created_at_iso, now, now_iso

    @staticmethod
    def _summarize_updated_fields(
        updates: Dict[str, Any], protected: frozenset
    ) -> List[str]:
        summary: List[str] = []
        for key in ("tags", "memory_type"):
            if key in updates:
                summary.append(key)
        if "metadata" in updates:
            summary.append("custom_metadata")
        for key in updates:
            if key not in protected and key not in ("tags", "memory_type", "metadata"):
                summary.append(key)
        summary.append("updated_at")
        return summary

    def _build_update_entity(
        self,
        existing: Memory,
        merged: Tuple[List[str], Optional[str], Dict[str, Any]],
        timestamps: Tuple[Optional[float], Optional[str], float, str],
        embedding: List[float],
    ) -> Dict[str, Any]:
        new_tags, new_type, new_metadata = merged
        created_at, created_at_iso, updated_at, updated_at_iso = timestamps
        # Derive a single fallback instant so missing created_at and missing
        # created_at_iso both reference the same moment.
        now = time.time()
        if created_at is None:
            created_at = now
            created_at_iso = created_at_iso or self._iso_from_epoch(now)
        else:
            created_at = float(created_at)
            created_at_iso = created_at_iso or self._iso_from_epoch(created_at)
        content = existing.content or ""
        entity: Dict[str, Any] = {
            "id": existing.content_hash,
            "vector": embedding,
            "content": content,
            "tags": _tags_to_string(new_tags),
            "memory_type": new_type or "",
            "metadata": json.dumps(new_metadata),
            "created_at": created_at,
            "updated_at": float(updated_at),
            "created_at_iso": created_at_iso,
            "updated_at_iso": updated_at_iso,
        }
        if self._has_content_lower:
            entity["content_lower"] = content.lower()
        return entity

    async def update_memory_metadata(
        self,
        content_hash: str,
        updates: Dict[str, Any],
        preserve_timestamps: bool = True,
    ) -> Tuple[bool, str]:
        if not self._ensure_initialized():
            return False, "Milvus storage not initialized"

        existing = await self.get_by_hash(content_hash)
        if existing is None:
            return False, f"Memory with hash {content_hash} not found"

        merged, err = self._merge_updates(existing, updates)
        if merged is None:
            return False, err  # type: ignore[return-value]

        timestamps = self._compute_update_timestamps(existing, updates, preserve_timestamps)

        try:
            embedding = self._generate_embedding(existing.content)
        except Exception as exc:  # noqa: BLE001
            return False, f"Failed to generate embedding for upsert: {exc}"

        entity = self._build_update_entity(existing, merged, timestamps, embedding)

        try:
            await self._call_client(
                "upsert",
                collection_name=self.collection_name,
                data=[entity],
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"Error updating memory metadata: {exc}"

        summary = self._summarize_updated_fields(updates, self._PROTECTED_UPDATE_KEYS)
        return True, f"Updated fields: {', '.join(summary)}"

    # -- Stats / misc --------------------------------------------------------

    async def get_stats(self) -> Dict[str, Any]:
        if not self._ensure_initialized():
            return {"error": "Milvus storage not initialized"}

        week_ago = time.time() - 7 * 24 * 60 * 60

        try:
            total_rows = await self._call_client(
                "query",
                collection_name=self.collection_name,
                filter="",
                output_fields=["count(*)"],
            )
            recent_rows = await self._call_client(
                "query",
                collection_name=self.collection_name,
                filter=f"created_at >= {week_ago}",
                output_fields=["count(*)"],
            )
            total = self._extract_count(total_rows)
            recent = self._extract_count(recent_rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Milvus stats query failed: %s", exc)
            total = recent = 0

        unique_tags = len(await self.get_all_tags())

        return {
            "backend": "milvus",
            "uri": self.uri,
            "collection": self.collection_name,
            "total_memories": total,
            "unique_tags": unique_tags,
            "memories_this_week": recent,
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
        }

    @staticmethod
    def _extract_count(rows: Any) -> int:
        if not rows:
            return 0
        first = rows[0]
        # Different client versions have used 'count(*)' or 'count'.
        for key in ("count(*)", "count", "total"):
            if isinstance(first, dict) and key in first:
                try:
                    return int(first[key])
                except (TypeError, ValueError):
                    return 0
        return 0

    async def get_all_tags(self) -> List[str]:
        if not self._ensure_initialized():
            return []

        try:
            rows = await self._call_client(
                "query",
                collection_name=self.collection_name,
                filter="",
                output_fields=["tags"],
                limit=_MILVUS_MAX_LIMIT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_all_tags failed: %s", exc)
            return []

        seen: set[str] = set()
        for row in rows:
            for tag in _string_to_tags(row.get("tags")):
                seen.add(tag)
        return sorted(seen)

    async def get_recent_memories(self, n: int = 10) -> List[Memory]:
        return await self.get_all_memories(limit=n, offset=0)

    async def get_all_memories(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Memory]:
        if not self._ensure_initialized():
            return []

        filters: List[Optional[str]] = []
        if memory_type is not None:
            safe_type = memory_type.replace('"', '\\"')
            filters.append(f'memory_type == "{safe_type}"')
        if tags:
            tag_filter, matched = self._tag_like_clauses(tags, joiner="or")
            if matched:
                filters.append(tag_filter)
            else:
                return []

        return await self._query_memories(
            filter_expr=self._combine_filter(*filters),
            limit=limit if limit is not None else _MILVUS_MAX_LIMIT,
            offset=offset,
            sort_desc_key="created_at",
        )

    async def count_all_memories(
        self,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        if not self._ensure_initialized():
            return 0

        filters: List[Optional[str]] = []
        if memory_type is not None:
            safe_type = memory_type.replace('"', '\\"')
            filters.append(f'memory_type == "{safe_type}"')
        if tags:
            tag_filter, matched = self._tag_like_clauses(tags, joiner="or")
            if matched:
                filters.append(tag_filter)
            else:
                return 0

        filter_expr = self._combine_filter(*filters)

        try:
            rows = await self._call_client(
                "query",
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["count(*)"],
            )
            return self._extract_count(rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("count_all_memories failed: %s", exc)
            return 0

    async def get_memories_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[Memory]:
        if not self._ensure_initialized():
            return []
        filter_expr = f"created_at >= {start_time} and created_at <= {end_time}"
        return await self._query_memories(
            filter_expr=filter_expr,
            limit=_MILVUS_MAX_LIMIT,
            sort_desc_key="created_at",
        )

    async def get_memory_timestamps(self, days: Optional[int] = None) -> List[float]:
        if not self._ensure_initialized():
            return []

        filter_expr = ""
        if days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
            filter_expr = f"created_at >= {cutoff}"

        try:
            rows = await self._call_client(
                "query",
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["created_at"],
                limit=_MILVUS_MAX_LIMIT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_memory_timestamps failed: %s", exc)
            return []

        timestamps = [row["created_at"] for row in rows if row.get("created_at") is not None]
        return sorted(timestamps, reverse=True)

    _QUERY_ITER_BATCH = 1000

    async def _query_memories(
        self,
        filter_expr: str,
        limit: int,
        offset: int = 0,
        sort_desc_key: Optional[str] = None,
    ) -> List[Memory]:
        """Stream rows via ``QueryIterator`` and return sorted, sliced ``Memory``
        objects.

        Iterating avoids the silent 16384-row truncation that a plain
        ``client.query(limit=...)`` imposes: the cap only applies per RPC, not
        per scan, so a filter that selects more than 16384 matching rows used
        to return an arbitrary window. Sorting is still done client-side
        (Milvus Lite has no server-side ``order_by``), but it is now applied
        to the full matching set rather than to a random subset.
        """
        if limit <= 0:
            return []
        rows = await self._iterate_all_rows(filter_expr)
        memories: List[Memory] = []
        for row in rows:
            mem = self._entity_to_memory(row)
            if mem is not None:
                memories.append(mem)

        if sort_desc_key:
            memories.sort(key=lambda m: getattr(m, sort_desc_key) or 0.0, reverse=True)

        if offset:
            memories = memories[offset:]
        return memories[:limit]

    async def _iterate_all_rows(self, filter_expr: str) -> List[Dict[str, Any]]:
        """Collect every row matching ``filter_expr`` using ``QueryIterator``.

        One lock-acquisition covers the full iteration so other coroutines
        can't interleave RPCs on the shared pymilvus channel.
        """
        async with self._write_lock:
            if self.client is None:
                raise RuntimeError("MilvusMemoryStorage was not initialized")
            return await asyncio.to_thread(self._drain_query_iterator, filter_expr)

    def _drain_query_iterator(self, filter_expr: str) -> List[Dict[str, Any]]:
        """Sync helper that drains a ``QueryIterator`` into a plain list."""
        assert self.client is not None
        iterator = self.client.query_iterator(
            collection_name=self.collection_name,
            filter=filter_expr or "",
            output_fields=list(self._OUTPUT_FIELDS),
            batch_size=self._QUERY_ITER_BATCH,
        )
        rows: List[Dict[str, Any]] = []
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                rows.extend(batch)
        finally:
            try:
                iterator.close()
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass
        return rows

    # -- Teardown ------------------------------------------------------------

    async def close(self) -> None:
        if self.client is None:
            return
        try:
            close = getattr(self.client, "close", None)
            if callable(close):
                await asyncio.to_thread(close)
        except Exception as exc:  # noqa: BLE001 — teardown must never raise
            logger.debug("MilvusClient.close failed (ignored): %s", exc)
        finally:
            self.client = None
            self._initialized = False
