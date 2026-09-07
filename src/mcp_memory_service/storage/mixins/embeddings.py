"""EmbeddingsMixin: embedding model initialization, generation, and fallback."""

import hashlib
import logging
import math
import os
import struct
import sys
import traceback
from typing import List

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ...utils.system_detection import get_torch_device

logger = logging.getLogger(__name__)

# Module-level caches
_MODEL_CACHE = {}
_DIMENSION_CACHE = {}
_EMBEDDING_CACHE = {}

# Module-level flag: emit hash-fallback warning only once per process
_HASH_FALLBACK_WARNED = False


def clear_model_caches() -> dict:
    """Clear embedding model caches to free memory."""
    import gc  # inline import: only needed by this cache-clearing helper

    global _MODEL_CACHE, _EMBEDDING_CACHE, _DIMENSION_CACHE

    model_count = len(_MODEL_CACHE)
    embedding_count = len(_EMBEDDING_CACHE)

    _MODEL_CACHE.clear()
    _DIMENSION_CACHE.clear()
    _EMBEDDING_CACHE.clear()

    collected = gc.collect()

    logger.info(
        f"Model caches cleared - "
        f"Models: {model_count}, Embeddings: {embedding_count}, "
        f"GC collected: {collected} objects"
    )

    return {
        "models_cleared": model_count,
        "embeddings_cleared": embedding_count,
        "gc_collected": collected
    }


def get_model_cache_stats() -> dict:
    """Get statistics about the model cache."""
    return {
        "model_count": len(_MODEL_CACHE),
        "model_keys": list(_MODEL_CACHE.keys()),
        "embedding_count": len(_EMBEDDING_CACHE)
    }


import re


class _HashEmbeddingModel:
    """Deterministic, pure-Python embedding fallback."""

    def __init__(self, embedding_dimension: int):
        self.embedding_dimension = int(embedding_dimension)

    def encode(self, texts: List[str], convert_to_numpy: bool = False):
        vectors = [self._embed_one(text) for text in texts]
        if convert_to_numpy:
            try:
                import numpy as np
                return np.asarray(vectors, dtype=np.float32)
            except Exception:
                return vectors
        return vectors

    def _embed_one(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.embedding_dimension

        floats: List[float] = []
        counter = 0
        needed = self.embedding_dimension
        text_bytes = text.encode("utf-8", errors="ignore")

        while len(floats) < needed:
            digest = hashlib.sha256(text_bytes + b"\x1f" + struct.pack("<I", counter)).digest()
            counter += 1
            for i in range(0, len(digest) - 3, 4):
                (val,) = struct.unpack("<i", digest[i : i + 4])
                floats.append(val / 2147483648.0)
                if len(floats) >= needed:
                    break

        return floats


class EmbeddingsMixin:
    """Mixin providing embedding model initialization and generation."""

    def _get_existing_db_embedding_dimension(self) -> int | None:
        """Read the embedding dimension from an existing vec0 table, or None if not present."""
        try:
            if not self.conn:
                return None
            cursor = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_embeddings'"
            )
            row = cursor.fetchone()
            if not row:
                return None
            match = re.search(r'FLOAT\[(\d+)\]', row[0])
            if match:
                return int(match.group(1))
        except Exception as exc:
            logger.debug("Could not read embedding dimension from sqlite_master: %s", exc)
        return None

    async def _initialize_embedding_model(self):
        """Initialize the embedding model (ONNX or SentenceTransformer based on configuration)."""
        global _MODEL_CACHE, _HASH_FALLBACK_WARNED

        is_docker = getattr(self, '_is_docker_environment', lambda: False)()
        if is_docker:
            logger.info("🐳 Docker environment detected - adjusting model loading strategy")

        try:
            external_api_url = os.environ.get('MCP_EXTERNAL_EMBEDDING_URL')
            if external_api_url:
                storage_backend = os.environ.get('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec')
                if storage_backend in ('hybrid', 'cloudflare'):
                    logger.warning(
                        f"⚠️  External embedding API not supported with '{storage_backend}' backend. "
                        "External APIs only work with 'sqlite_vec' backend. "
                        f"The '{storage_backend}' backend will use its default embedding method. "
                        "Falling back to local models (ONNX/SentenceTransformer) for SQLite-vec component."
                    )
                    external_api_url = None

            if external_api_url:
                logger.info(f"Using external embedding API: {external_api_url}")
                try:
                    from ...embeddings.external_api import get_external_embedding_model

                    DEFAULT_EXTERNAL_EMBEDDING_MODEL = 'nomic-embed-text'
                    external_model_name = os.environ.get('MCP_EXTERNAL_EMBEDDING_MODEL', DEFAULT_EXTERNAL_EMBEDDING_MODEL)
                    external_api_key = os.environ.get('MCP_EXTERNAL_EMBEDDING_API_KEY')
                    cache_key = f"external_{external_api_url}_{external_model_name}_{external_api_key}"

                    if cache_key in _MODEL_CACHE:
                        self.embedding_model = _MODEL_CACHE[cache_key]
                        # Also on the cache-hit path, not just the fresh connect
                        # below: the global model cache means this is the normal
                        # path for every process after the first, so setting the
                        # name only on a cold start would leave health reporting
                        # wrong almost everywhere.
                        self.embedding_model_name = external_model_name
                        if cache_key in _DIMENSION_CACHE:
                            self.embedding_dimension = _DIMENSION_CACHE[cache_key]
                        elif hasattr(self.embedding_model, 'embedding_dimension'):
                            self.embedding_dimension = self.embedding_model.embedding_dimension
                            _DIMENSION_CACHE[cache_key] = self.embedding_dimension
                        logger.info("Using cached external embedding model")
                        return

                    ext_model = get_external_embedding_model(external_api_url, external_model_name)
                    self.embedding_model = ext_model
                    # Without this the attribute keeps the constructor's local
                    # default, and every health surface that reads it reports a
                    # model that is not the one producing the vectors (#254).
                    self.embedding_model_name = external_model_name
                    self.embedding_dimension = ext_model.embedding_dimension
                    _MODEL_CACHE[cache_key] = ext_model
                    _DIMENSION_CACHE[cache_key] = self.embedding_dimension

                    if self.embedding_dimension != 384:
                        logger.warning(
                            f"⚠️  External embedding dimension ({self.embedding_dimension}) differs from "
                            f"default ONNX dimension (384). Ensure this matches your database schema "
                            f"or you may encounter errors. To fix: delete your database or use a "
                            f"compatible model."
                        )

                    logger.info(f"External embedding API connected. Dimension: {self.embedding_dimension}")
                    return
                except (ConnectionError, RuntimeError, ImportError) as e:
                    existing_dim = await self._run_in_thread(self._get_existing_db_embedding_dimension)
                    dim_detail = (
                        f" The existing database uses dimension {existing_dim}."
                        f" Falling back to a local model would cause a dimension mismatch and"
                        f" corrupt all store/search operations."
                        if existing_dim is not None else ""
                    )
                    raise RuntimeError(
                        f"External embedding API at {external_api_url} is unreachable: {e}."
                        f"{dim_detail}"
                        f" Ensure your embedding service is running before starting mcp-memory-service."
                    ) from e

            use_onnx = os.environ.get('MCP_MEMORY_USE_ONNX', '').lower() not in ('0', 'false', 'no')

            if use_onnx:
                logger.info("Attempting to use ONNX embeddings (PyTorch-free)")
                try:
                    from ...embeddings import get_onnx_embedding_model
                    from ...embeddings.onnx_embeddings import ONNX_AVAILABLE as _onnx_ok

                    cache_key = f"onnx_{self.embedding_model_name}"
                    if _onnx_ok and cache_key in _MODEL_CACHE:
                        self.embedding_model = _MODEL_CACHE[cache_key]
                        if cache_key in _DIMENSION_CACHE:
                            self.embedding_dimension = _DIMENSION_CACHE[cache_key]
                        elif hasattr(self.embedding_model, 'embedding_dimension'):
                            self.embedding_dimension = self.embedding_model.embedding_dimension
                            _DIMENSION_CACHE[cache_key] = self.embedding_dimension
                        logger.info("Using cached ONNX embedding model")
                        return
                    onnx_model = get_onnx_embedding_model(self.embedding_model_name)
                    if onnx_model:
                        # The ONNX path currently serves only all-MiniLM-L6-v2 (384-dim).
                        # If a different model was requested, say so loudly instead of
                        # logging plain success — otherwise the requested model is silently
                        # ignored and the user is unaware they are getting MiniLM-384 (#143).
                        requested_base = (self.embedding_model_name or '').split('/')[-1]
                        if requested_base != 'all-MiniLM-L6-v2':
                            logger.warning(
                                "ONNX backend does not honor the requested embedding model '%s'; "
                                "it only serves all-MiniLM-L6-v2 (%s-dim). The requested model is "
                                "being ignored. Set MCP_MEMORY_USE_ONNX=0 to load it via "
                                "SentenceTransformers, or configure an external embedding API "
                                "(MCP_EXTERNAL_EMBEDDING_URL).",
                                self.embedding_model_name, onnx_model.embedding_dimension,
                            )
                        self.embedding_model = onnx_model
                        self.embedding_dimension = onnx_model.embedding_dimension
                        _MODEL_CACHE[cache_key] = onnx_model
                        _DIMENSION_CACHE[cache_key] = self.embedding_dimension
                        logger.info(f"ONNX embedding model loaded successfully. Dimension: {self.embedding_dimension}")
                        return
                    else:
                        logger.warning("ONNX model creation failed, falling back to SentenceTransformer")
                except ImportError as e:
                    logger.warning(f"ONNX dependencies not available: {e}")
                except Exception as e:
                    logger.warning(f"Failed to initialize ONNX embeddings: {e}")

            # Check SentenceTransformer availability
            # Use sys.modules introspection to allow test patches on SENTENCE_TRANSFORMERS_AVAILABLE
            _st_mod = sys.modules.get(__name__, None)
            _st_flag = getattr(_st_mod, 'SENTENCE_TRANSFORMERS_AVAILABLE', False) if _st_mod else SENTENCE_TRANSFORMERS_AVAILABLE
            _st_available = _st_flag or SentenceTransformer is not None
            if not _st_available:
                if not getattr(self, '_hash_fallback_warned', False):
                    logger.warning(
                        "No embedding backend available; using hash embeddings (reduced quality). "
                        "Install ML dependencies for semantic search: pip install mcp-memory-service[ml]. "
                        "The hash fallback is refused on databases that already contain memories; "
                        "set MCP_MEMORY_ALLOW_HASH_EMBEDDINGS=1 to override for testing."
                    )
                    _HASH_FALLBACK_WARNED = True
                    self._hash_fallback_warned = True
                await self._initialize_hash_embedding_fallback()
                return

            cache_key = self.embedding_model_name
            if cache_key in _MODEL_CACHE:
                self.embedding_model = _MODEL_CACHE[cache_key]
                if cache_key in _DIMENSION_CACHE:
                    self.embedding_dimension = _DIMENSION_CACHE[cache_key]
                elif hasattr(self.embedding_model, 'get_sentence_embedding_dimension'):
                    dim = self.embedding_model.get_sentence_embedding_dimension()
                    if dim:
                        self.embedding_dimension = dim
                        _DIMENSION_CACHE[cache_key] = dim
                elif hasattr(self.embedding_model, 'embedding_dimension'):
                    self.embedding_dimension = self.embedding_model.embedding_dimension
                    _DIMENSION_CACHE[cache_key] = self.embedding_dimension
                logger.info(f"Using cached embedding model: {self.embedding_model_name}")
                return

            device = get_torch_device()

            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            logger.info(f"Using device: {device}")

            hf_home = os.environ.get('HF_HOME', os.path.expanduser("~/.cache/huggingface"))
            model_cache_path = os.path.join(hf_home, "hub", f"models--sentence-transformers--{self.embedding_model_name.replace('/', '--')}")
            if os.path.exists(model_cache_path):
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                logger.info("📦 Found cached model - enabling offline mode")

            try:
                hf_home = os.environ.get('HF_HOME', os.path.expanduser("~/.cache/huggingface"))
                cache_path = os.path.join(hf_home, "hub", f"models--sentence-transformers--{self.embedding_model_name.replace('/', '--')}")
                if os.path.exists(cache_path):
                    snapshots_path = os.path.join(cache_path, "snapshots")
                    if os.path.exists(snapshots_path):
                        snapshot_dirs = [d for d in os.listdir(snapshots_path) if os.path.isdir(os.path.join(snapshots_path, d))]
                        if snapshot_dirs:
                            model_path = os.path.join(snapshots_path, snapshot_dirs[0])
                            logger.info(f"Loading model from cache: {model_path}")
                            self.embedding_model = SentenceTransformer(model_path, device=device)
                        else:
                            raise FileNotFoundError("No snapshot found")
                    else:
                        raise FileNotFoundError("No snapshots directory")
                else:
                    raise FileNotFoundError("No cache found")
            except FileNotFoundError as cache_error:
                logger.warning(f"Model not in cache: {cache_error}")
                try:
                    logger.info("Attempting to download model from Hugging Face...")
                    self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
                except OSError as download_error:
                    error_msg = str(download_error)
                    if any(phrase in error_msg.lower() for phrase in ['connection', 'network', 'couldn\'t connect', 'huggingface.co']):
                        docker_help = self._get_docker_network_help() if is_docker else ""
                        raise RuntimeError(
                            f"🔌 Model Download Error: Cannot connect to huggingface.co\n"
                            f"{'='*60}\n"
                            f"The model '{self.embedding_model_name}' needs to be downloaded but the connection failed.\n"
                            f"{docker_help}"
                            f"\n💡 Solutions:\n"
                            f"1. Mount pre-downloaded models as a volume:\n"
                            f"   # On host machine, download the model first:\n"
                            f"   python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{self.embedding_model_name}')\"\n"
                            f"   \n"
                            f"   # Then run container with cache mount:\n"
                            f"   docker run -v ~/.cache/huggingface:/root/.cache/huggingface ...\n"
                            f"\n"
                            f"2. Configure Docker network (if behind proxy):\n"
                            f"   docker run -e HTTPS_PROXY=your-proxy -e HTTP_PROXY=your-proxy ...\n"
                            f"\n"
                            f"3. Use offline mode with pre-cached models:\n"
                            f"   docker run -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 ...\n"
                            f"\n"
                            f"4. Use host network mode (if appropriate for your setup):\n"
                            f"   docker run --network host ...\n"
                            f"\n"
                            f"📚 See docs: https://github.com/doobidoo/mcp-memory-service/blob/main/docs/deployment/docker.md#model-download-issues\n"
                            f"{'='*60}"
                        ) from download_error
                    else:
                        raise
            except Exception as cache_error:
                logger.warning(f"Failed to load from cache: {cache_error}")
                logger.info("Attempting normal model loading...")
                self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)

            test_embedding = self.embedding_model.encode(["test"], convert_to_numpy=True)
            self.embedding_dimension = test_embedding.shape[1]

            _MODEL_CACHE[cache_key] = self.embedding_model
            _DIMENSION_CACHE[cache_key] = self.embedding_dimension

            logger.info(f"✅ Embedding model loaded successfully. Dimension: {self.embedding_dimension}")

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            logger.error(traceback.format_exc())
            if not getattr(self, '_hash_fallback_warned', False):
                logger.warning(
                    "No embedding backend available; using hash embeddings (reduced quality). "
                    "Install ML dependencies for semantic search: pip install mcp-memory-service[ml]. "
                    "The hash fallback is refused on databases that already contain memories; "
                    "set MCP_MEMORY_ALLOW_HASH_EMBEDDINGS=1 to override for testing."
                )
                _HASH_FALLBACK_WARNED = True
                self._hash_fallback_warned = True
            await self._initialize_hash_embedding_fallback()

    def _count_existing_rows(self) -> tuple:
        """Return (memories, embedding_rows), treating missing tables as empty.

        Both counts matter, for different reasons. The emptiness decision needs
        their sum, so a database holding real embeddings is never treated as
        empty. The error message needs them apart: they are separate tables of
        roughly equal size on a normally-populated database, so reporting the
        sum as a memory count overstated it by about a factor of two (#228).
        """
        counts = []
        for table in ("memories", "memory_embeddings"):
            count = 0
            try:
                cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
                row = cursor.fetchone()
                if row and row[0]:
                    count = int(row[0])
            except Exception as exc:
                # Missing table (fresh DB) or unreadable — treat as empty for this table.
                logger.debug("Could not count rows in %s: %s", table, exc)
            counts.append(count)
        return counts[0], counts[1]

    def _describe_existing_rows(self) -> str:
        """Phrase the row counts for an error message, without inventing a total."""
        memories, embedding_rows = self._count_existing_rows() if self.conn else (0, 0)
        return f"{memories} memories and {embedding_rows} embedding rows"

    def _hash_fallback_allowed(self) -> bool:
        """Decide whether writing hash pseudo-vectors into this database is safe.

        Policy:
          - MCP_MEMORY_ALLOW_HASH_EMBEDDINGS=1/true/yes force-allows the fallback.
          - MCP_MEMORY_ALLOW_HASH_EMBEDDINGS=0/false/no force-refuses it.
          - Unset: allow only when the database is effectively empty (no existing
            ``memories`` or ``memory_embeddings`` rows). A database that already
            contains real memories is protected, because hash pseudo-vectors would
            silently poison semantic search with no per-row marker to find them later.

        Raises RuntimeError with an actionable message when the fallback is refused.
        """
        override = os.environ.get('MCP_MEMORY_ALLOW_HASH_EMBEDDINGS', '').strip().lower()
        if override in ('1', 'true', 'yes'):
            return True
        if override in ('0', 'false', 'no'):
            raise RuntimeError(
                "The configured embedding backend is unavailable and "
                "MCP_MEMORY_ALLOW_HASH_EMBEDDINGS is set to refuse the hash fallback. "
                "Refusing to write hash pseudo-vectors into a database holding "
                f"{self._describe_existing_rows()}. Install a real embedding backend "
                "(pip install 'mcp-memory-service[sqlite-ml]' or 'mcp-memory-service[ml]') "
                "or set MCP_MEMORY_ALLOW_HASH_EMBEDDINGS=1 to override for testing."
            )

        # Default policy: allow only on an effectively-empty database.
        memories, embedding_rows = self._count_existing_rows() if self.conn else (0, 0)
        if memories + embedding_rows == 0:
            return True
        raise RuntimeError(
            "The configured embedding backend is unavailable; refusing to write hash "
            f"pseudo-vectors into a database holding {memories} memories and "
            f"{embedding_rows} embedding rows "
            "(this would permanently poison semantic search with no per-row marker to "
            "find affected rows later). Install a real embedding backend "
            "(pip install 'mcp-memory-service[sqlite-ml]' or 'mcp-memory-service[ml]') "
            "or set MCP_MEMORY_ALLOW_HASH_EMBEDDINGS=1 to override for testing."
        )

    async def _initialize_hash_embedding_fallback(self):
        """Initialize hash embedding model, matching existing DB dimension if possible."""
        # Guard: never silently poison a database that already holds real memories.
        # Called synchronously (a quick COUNT read) — initialization is single-threaded,
        # and this raises RuntimeError when the fallback is refused.
        self._hash_fallback_allowed()

        existing_dim = await self._run_in_thread(self._get_existing_db_embedding_dimension)
        if existing_dim and existing_dim != self.embedding_dimension:
            logger.warning(
                f"Adjusting hash embedding dimension from {self.embedding_dimension} to "
                f"{existing_dim} to match existing database schema."
            )
            self.embedding_dimension = existing_dim
        self.embedding_model = _HashEmbeddingModel(self.embedding_dimension)
        self.embedding_backend_degraded = True

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not self.embedding_model:
            raise RuntimeError("No embedding model available. Ensure sentence-transformers is installed and model is loaded.")

        try:
            if self.enable_cache:
                cache_key = hash(text)
                if cache_key in _EMBEDDING_CACHE:
                    return _EMBEDDING_CACHE[cache_key]

            embedding = self.embedding_model.encode([text], convert_to_numpy=True)[0]
            if hasattr(embedding, "tolist"):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)

            if not embedding_list:
                raise ValueError("Generated embedding is empty")

            if len(embedding_list) != self.embedding_dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding_list)}")

            if not all(isinstance(x, (int, float)) and not math.isnan(x) and x != float('inf') and x != float('-inf') for x in embedding_list):
                raise ValueError("Embedding contains invalid values (NaN or infinity)")

            if self.enable_cache:
                _EMBEDDING_CACHE[cache_key] = embedding_list

            return embedding_list

        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise RuntimeError(f"Failed to generate embedding: {str(e)}") from e
