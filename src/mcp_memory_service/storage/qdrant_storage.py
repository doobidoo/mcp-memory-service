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
        Qdrant storage backend for MCP Memory Service.
Provides embedded vector storage using Qdrant with circuit breaker and model tracking.
"""

import asyncio
import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import exceptions as qdrant_exceptions
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    OrderBy,
    PayloadSchemaType,
    PointStruct,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# Import sentence transformers with fallback
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import dateutil with fallback
try:
    from dateutil import parser as dateutil_parser

    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

from ..config import QdrantSettings
from ..models.memory import Memory, MemoryQueryResult
from ..utils.system_detection import get_torch_device
from .base import MemoryStorage

logger = logging.getLogger(__name__)

# Log warning after logger is defined
if not SENTENCE_TRANSFORMERS_AVAILABLE:
    logger.warning("sentence_transformers not available. Install for embedding support.")


class StorageError(Exception):
    """Storage-related errors."""

    pass


def is_retryable_error(exception: Exception) -> bool:
    """
    Determine if an exception is retryable (transient 5xx server errors only).

    Args:
        exception: Exception to check

    Returns:
        True if exception is a retryable 5xx server error, False otherwise

    Notes:
        - 5xx server errors are transient and retryable
        - 4xx client errors are permanent (configuration/validation) and NOT retryable
        - Dimension mismatches are NOT retryable (configuration error)
        - Validation errors are NOT retryable (data error)
    """
    if isinstance(exception, qdrant_exceptions.UnexpectedResponse):
        # Only retry 5xx server errors (transient failures)
        return hasattr(exception, "status_code") and 500 <= exception.status_code < 600
    return False


class QdrantStorage(MemoryStorage):
    """
    Qdrant embedded mode storage implementation with model change detection.

    Provides local-first vector storage with circuit breaker fault tolerance
    and automatic embedding model change detection.
    """

    # Special point ID for metadata (reserved, won't conflict with hash-based memory IDs)
    METADATA_POINT_ID = 1

    def __init__(
        self,
        embedding_model: str,
        collection_name: str = "memories",
        quantization_enabled: bool = False,
        distance_metric: str = "Cosine",
        storage_path: str | None = None,
        url: str | None = None,
    ):
        """
        Initialize Qdrant storage backend in embedded or server mode.

        Args:
            embedding_model: Embedding model name (e.g., "all-MiniLM-L6-v2")
            collection_name: Qdrant collection name (default: "memories")
            quantization_enabled: Enable binary quantization for memory savings
            distance_metric: Vector distance metric (default: "Cosine")
            storage_path: Path to Qdrant storage directory (embedded mode)
            url: Qdrant server URL (server mode, e.g., "http://localhost:6333")

        Note:
            If url is provided, uses network client mode (multi-process safe).
            Otherwise uses embedded mode (single-process only, file locking).
        """
        # Validate mode selection
        if url and storage_path:
            raise ValueError("Cannot specify both url and storage_path. Choose embedded OR server mode.")
        if not url and not storage_path:
            raise ValueError("Must specify either url (server mode) or storage_path (embedded mode).")

        self.url = url
        self.storage_path = storage_path
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.quantization_enabled = quantization_enabled
        self.distance_metric = distance_metric

        # Circuit breaker state
        self._failure_count = 0
        self._circuit_open_until: datetime | None = None
        self._failure_threshold = 5  # Open circuit after 5 consecutive failures
        self._circuit_timeout = 60  # Reclose circuit after 60 seconds

        # Model tracking
        self._vector_size: int | None = None  # Detected from embedding model

        # Qdrant client (initialized in initialize())
        self.client = None
        self._initialized = False

        # Load Qdrant settings
        self.config = QdrantSettings()

        # Embedding service (will be initialized later)
        self.embedding_service = None

        # Thread-safe model loading (prevents race condition)

        self._model_lock = threading.Lock()

        mode = "server" if self.url else "embedded"
        location = self.url if self.url else self.storage_path
        logger.info(
            f"Initializing QdrantStorage: mode={mode}, location={location}, "
            f"model={embedding_model}, collection={collection_name}"
        )

    @property
    def max_content_length(self) -> int | None:
        """
        Maximum content length supported by this storage backend.

        Returns:
            None for unlimited (Qdrant has no inherent content length limit)
        """
        return None  # Qdrant has no inherent content length limit

    @property
    def supports_chunking(self) -> bool:
        """
        Whether this backend supports automatic content chunking.

        Returns:
            True - Qdrant supports chunking via metadata linking
        """
        return True  # Qdrant supports chunking via metadata

    async def initialize(self) -> None:
        """
        Initialize Qdrant storage with model change detection.

        Detects if embedding model changed (different dimensions) and fails
        with clear migration instructions. Prevents silent corruption from
        dimension mismatches.
        """
        if self._initialized:
            logger.debug("QdrantStorage already initialized")
            return

        mode = "server" if self.url else "embedded"
        location = self.url if self.url else self.storage_path
        logger.info(f"Initializing Qdrant storage in {mode} mode: {location}")

        # Create Qdrant client in either embedded or server mode
        loop = asyncio.get_event_loop()
        if self.url:
            # Server mode - network client (multi-process safe)
            self.client = await loop.run_in_executor(None, lambda: QdrantClient(url=self.url))
            logger.info(f"Connected to Qdrant server at {self.url}")
        else:
            # Embedded mode - file-based (single-process only, exclusive lock)
            self.client = await loop.run_in_executor(None, lambda: QdrantClient(path=self.storage_path))
            logger.info(f"Initialized Qdrant embedded storage at {self.storage_path}")

        # Detect vector dimensions from embedding model
        self._vector_size = await self._detect_vector_dimensions()
        logger.info(f"Detected vector dimensions: {self._vector_size}")

        # Check if collection exists
        if await self._collection_exists():
            logger.info(f"Collection '{self.collection_name}' exists, verifying model compatibility")
            # Verify model hasn't changed
            await self._verify_model_compatibility()
        else:
            logger.info(f"Creating new collection '{self.collection_name}' with model metadata")
            # Create new collection with model metadata
            await self._create_collection_with_metadata()

        self._initialized = True
        logger.info("QdrantStorage initialization complete")

    async def _detect_vector_dimensions(self) -> int:
        """
        Detect vector dimensions from embedding model.

        Returns:
            The dimension of vectors produced by the embedding model
        """
        # For now, we'll use a placeholder that matches the embedding service pattern
        # The actual embedding service will be injected from the MemoryService layer
        # This matches the pattern used in sqlite_vec and cloudflare

        # Common embedding model dimensions (will be detected dynamically in production)
        model_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-MiniLM-L12-v2": 384,
            "all-mpnet-base-v2": 768,
            "text-embedding-ada-002": 1536,
            "@cf/baai/bge-base-en-v1.5": 768,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "intfloat/e5-small": 384,
            "intfloat/e5-base": 768,
            "intfloat/e5-large": 1024,
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "infgrad/stella-base-en-v2": 768,
            "nomic-ai/nomic-embed-text-v1": 768,  # Assuming 768 based on common practice
        }

        # Try to get dimensions from known models first
        for known_model, dims in model_dimensions.items():
            if known_model in self.embedding_model:
                logger.info(f"Using known dimensions for model {self.embedding_model}: {dims}")
                return dims

        # Default to 384 (all-MiniLM-L6-v2 dimensions) as fallback
        # In production, this will generate a test embedding to detect actual size
        logger.warning(f"Unknown model {self.embedding_model}, defaulting to 384 dimensions")
        return 384

    async def _verify_model_compatibility(self) -> None:
        """
        Verify existing collection uses same embedding model.

        Raises StorageError with migration instructions if model changed.
        """
        loop = asyncio.get_event_loop()

        # Get collection info
        collection_info = await loop.run_in_executor(None, self.client.get_collection, self.collection_name)

        # Try to retrieve metadata from the special __metadata__ point
        try:
            metadata_points = await loop.run_in_executor(
                None, lambda: self.client.retrieve(collection_name=self.collection_name, ids=[self.METADATA_POINT_ID])
            )

            if metadata_points and len(metadata_points) > 0:
                metadata = metadata_points[0].payload
                stored_model = metadata.get("embedding_model")
                stored_dimensions = metadata.get("vector_size", metadata.get("dimensions"))

                # Check for model change
                if stored_model and stored_model != self.embedding_model:
                    raise StorageError(
                        f"Embedding model changed from {stored_model} to {self.embedding_model}.\n"
                        f"Run migration: python scripts/migration/migrate_to_new_model.py "
                        f"--old-model {stored_model} --new-model {self.embedding_model}"
                    )

                # Check dimension mismatch
                if stored_dimensions and stored_dimensions != self._vector_size:
                    raise StorageError(
                        f"Vector dimension mismatch: stored={stored_dimensions}, current={self._vector_size}\n"
                        f"Model: {self.embedding_model}\n\n"
                        f"This indicates a configuration error or model version change.\n"
                        f"Run migration: python scripts/migration/migrate_to_new_model.py "
                        f"--old-model {stored_model or 'unknown'} --new-model {self.embedding_model}"
                    )
        except Exception as e:
            if "not found" not in str(e).lower():
                logger.warning(f"Could not retrieve metadata point: {e}")

        # Also check the collection's vector configuration
        if hasattr(collection_info.config, "params") and hasattr(collection_info.config.params, "vectors"):
            collection_vector_size = collection_info.config.params.vectors.size
            if collection_vector_size != self._vector_size:
                raise StorageError(
                    f"Collection vector size ({collection_vector_size}) doesn't match "
                    f"current model dimensions ({self._vector_size}).\n"
                    f"Run migration: python scripts/migration/migrate_to_new_model.py "
                    f"--old-model unknown --new-model {self.embedding_model}"
                )

        logger.info("Model compatibility verified")

    async def _create_collection_with_metadata(self) -> None:
        """
        Create collection and store model metadata.
        """
        loop = asyncio.get_event_loop()

        # Prepare HNSW config
        hnsw_config = HnswConfigDiff(
            m=self.config.HNSW_M,
            ef_construct=self.config.HNSW_EF_CONSTRUCT,
            full_scan_threshold=self.config.HNSW_FULL_SCAN_THRESHOLD,
        )

        # Prepare quantization config if enabled
        quantization_config = None
        if self.quantization_enabled:
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8, quantile=0.99, always_ram=self.config.QUANTIZATION_ALWAYS_RAM
                )
            )

        # Create collection with vector configuration
        await loop.run_in_executor(
            None,
            lambda: self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
                hnsw_config=hnsw_config,
                quantization_config=quantization_config,
                on_disk_payload=self.config.ON_DISK_PAYLOAD,
            ),
        )

        logger.info(f"Created collection '{self.collection_name}' with vector size {self._vector_size}")

        # Store metadata as a special point
        metadata_point = PointStruct(
            id=self.METADATA_POINT_ID,
            vector=[0.0] * self._vector_size,  # Dummy vector
            payload={
                "embedding_model": self.embedding_model,
                "vector_size": self._vector_size,
                "dimensions": self._vector_size,  # Store both for compatibility
                "created_at": datetime.now().timestamp(),  # Use float for FLOAT payload index
                "distance_metric": "Cosine",
                "quantization_enabled": self.quantization_enabled,
            },
        )

        await loop.run_in_executor(
            None, lambda: self.client.upsert(collection_name=self.collection_name, points=[metadata_point])
        )

        logger.info("Stored model metadata in __metadata__ point")

        # Create payload index on created_at for efficient server-side sorting
        await loop.run_in_executor(
            None,
            lambda: self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="created_at",
                field_schema=PayloadSchemaType.FLOAT,
            ),
        )
        logger.info("Created payload index on 'created_at' for server-side sorting")

    async def _collection_exists(self) -> bool:
        """
        Check if the collection exists in Qdrant.

        Returns:
            True if collection exists, False otherwise
        """
        loop = asyncio.get_event_loop()

        try:
            collections = await loop.run_in_executor(None, self.client.get_collections)

            collection_names = [col.name for col in collections.collections]
            exists = self.collection_name in collection_names

            logger.debug(f"Collection '{self.collection_name}' exists: {exists}")
            return exists

        except Exception as e:
            logger.error(f"Error checking if collection exists: {e}")
            return False

    def _check_circuit_breaker(self) -> None:
        """
        Check if circuit breaker is open and fail fast if so.

        Raises:
            StorageError: If circuit breaker is open with retry timestamp
        """
        if self._circuit_open_until is not None:
            # Check if circuit is still open
            if datetime.now() < self._circuit_open_until:
                retry_time = self._circuit_open_until.strftime("%Y-%m-%d %H:%M:%S")
                raise StorageError(f"Circuit breaker is open until {retry_time}. Service temporarily unavailable.")
            else:
                # Circuit timeout expired, reset to closed state
                logger.info("Circuit breaker timeout expired, resetting to closed state")
                self._circuit_open_until = None
                self._failure_count = 0

    def _record_failure(self) -> None:
        """
        Record a failure and open circuit breaker if threshold reached.

        Opens circuit after 5 consecutive failures with 60 second timeout.
        """
        self._failure_count += 1
        logger.warning(f"Recorded failure #{self._failure_count}")

        if self._failure_count >= self._failure_threshold:
            self._circuit_open_until = datetime.now() + timedelta(seconds=self._circuit_timeout)
            logger.error(
                f"Circuit breaker opened after {self._failure_count} consecutive failures. "
                f"Will retry at {self._circuit_open_until.strftime('%Y-%m-%d %H:%M:%S')}"
            )

    def _record_success(self) -> None:
        """
        Record a successful operation and reset circuit breaker state.

        Resets failure count and closes circuit if it was in failure state.
        """
        if self._failure_count > 0:
            logger.info(f"Operation successful, resetting circuit breaker (was at {self._failure_count} failures)")
            self._failure_count = 0
            self._circuit_open_until = None
        else:
            logger.debug("Operation successful, circuit breaker already in healthy state")

    def _hash_to_uuid(self, hash_string: str) -> str | uuid.UUID:
        """
        Convert a SHA256 hash to a UUID format for Qdrant compatibility.

        Qdrant's embedded mode requires UUID format for point IDs.
        We convert the first 32 chars of the hash to UUID format.

        Args:
            hash_string: SHA256 hash string (64 chars)

        Returns:
            UUID derived from the hash
        """
        # Take first 32 chars of the hash (UUID needs 32 hex chars)
        # SHA256 gives us 64 chars, so we have plenty
        uuid_hex = hash_string[:32]
        # Convert to UUID format
        return str(uuid.UUID(uuid_hex))

    def _generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for text using configured model.

        Thread-safe lazy loading with lock to prevent race conditions.

        Args:
            text: Text to generate embedding for

        Returns:
            List of float values representing the embedding vector

        Raises:
            RuntimeError: If sentence transformers not available
            Exception: If embedding generation fails
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence_transformers not installed. Install with: pip install sentence-transformers")

        try:
            # Thread-safe lazy loading - prevents concurrent requests from loading model twice
            if not hasattr(self, "_embedding_model_instance"):
                with self._model_lock:
                    # Double-check after acquiring lock (another thread may have loaded it)
                    if not hasattr(self, "_embedding_model_instance"):
                        logger.info(f"Loading embedding model: {self.embedding_model}")
                        device = get_torch_device()
                        self._embedding_model_instance = SentenceTransformer(self.embedding_model, device=device)
                        logger.info(f"Loaded model: {self.embedding_model} on device: {device}")

            # Generate embedding (outside lock - model is thread-safe for inference)
            embeddings = self._embedding_model_instance.encode(text, convert_to_tensor=False)
            embedding_list = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

            # Validate embedding
            if not embedding_list:
                raise ValueError("Generated embedding is empty")

            # Update vector size if not set (first embedding)
            if self._vector_size is None:
                self._vector_size = len(embedding_list)
                logger.info(f"Detected vector size from actual embedding: {self._vector_size}")

            return embedding_list

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e.__class__.__name__}: {str(e)}")
            raise

    def _normalize_tags(self, tags_raw: Any) -> list[str]:
        """
        Normalize tags to list format, handling both string (legacy) and list formats.

        Args:
            tags_raw: Tags in any format (string, list, or None)

        Returns:
            List of trimmed, non-empty tag strings
        """
        if isinstance(tags_raw, str):
            # Legacy format: comma-separated string
            tags = tags_raw.split(",") if tags_raw else []
        elif isinstance(tags_raw, list):
            tags = tags_raw
        else:
            tags = []

        # Filter out empty strings and trim whitespace
        return [t.strip() for t in tags if t and t.strip()]

    def _normalize_timestamp(self, ts: Any) -> float:
        """
        Normalize timestamp to float, handling both ISO strings and Unix timestamps.

        Args:
            ts: Timestamp in any format (ISO string, float, int, or None)

        Returns:
            Unix timestamp as float, or 0.0 if invalid/missing
        """
        if ts is None:
            return 0.0
        if isinstance(ts, int | float):
            return float(ts)
        if isinstance(ts, str):
            if DATEUTIL_AVAILABLE:
                try:
                    parsed_dt = dateutil_parser.isoparse(ts)
                    return parsed_dt.timestamp()
                except (ValueError, TypeError):
                    pass
            else:
                # Fallback without dateutil
                try:
                    # Replace 'Z' with '+00:00' for UTC timezone before parsing
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    return dt.timestamp()
                except (ValueError, TypeError):
                    pass
        return 0.0

    @retry(
        retry=retry_if_exception(is_retryable_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    async def store(self, memory: Memory) -> tuple[bool, str]:
        """
        Store a memory in Qdrant with circuit breaker protection and retry logic.

        Retries up to 3 times for transient 5xx server errors with exponential backoff (1-5s).
        Does NOT retry for 4xx client errors or dimension mismatches (configuration errors).

        Args:
            memory: Memory object to store

        Returns:
            Tuple of (success, message)
        """
        # Check circuit breaker first
        self._check_circuit_breaker()

        try:
            # Generate embedding if not provided
            if memory.embedding is None:
                try:
                    loop = asyncio.get_event_loop()
                    memory.embedding = await loop.run_in_executor(None, self._generate_embedding, memory.content)
                    logger.debug(f"Generated embedding for memory {memory.content_hash[:8]}...")
                except Exception as e:
                    logger.error(f"Failed to generate embedding for memory: {e}")
                    return False, f"Failed to generate embedding: {str(e)}"

            # Validate embedding dimensions
            if len(memory.embedding) != self._vector_size:
                error_msg = (
                    f"Embedding dimension mismatch: expected {self._vector_size}, "
                    f"got {len(memory.embedding)}. "
                    f"This indicates a configuration error or model version change."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Create Qdrant point structure
            # Convert hash to UUID for Qdrant compatibility
            point_id = self._hash_to_uuid(memory.content_hash)
            point = PointStruct(
                id=point_id,
                vector=memory.embedding,
                payload={
                    "content": memory.content,
                    "tags": memory.tags,
                    "metadata": memory.metadata or {},
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                    "memory_type": memory.memory_type or "",
                    "content_hash": memory.content_hash,
                },
            )

            # Upsert to Qdrant (idempotent operation)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.client.upsert(collection_name=self.collection_name, points=[point]))

            # Record success for circuit breaker
            self._record_success()

            logger.debug(f"Stored memory {memory.content_hash[:8]}... in Qdrant")
            return True, "Memory stored successfully"

        except ValueError as e:
            # Dimension mismatch - don't retry, don't record failure (configuration error)
            logger.error(f"Validation error storing memory: {e}")
            raise

        except Exception as e:
            # Record failure for circuit breaker
            self._record_failure()
            logger.error(f"Failed to store memory in Qdrant: {e}")
            return False, f"Failed to store memory: {str(e)}"

    @retry(
        retry=retry_if_exception(is_retryable_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        min_similarity: float | None = None,
        offset: int = 0,
    ) -> list[MemoryQueryResult]:
        """
        Retrieve memories by semantic search with optional filtering and retry logic.

        Retries up to 3 times for transient 5xx server errors with exponential backoff (1-5s).
        Does NOT retry for 4xx client errors (configuration errors).

        Args:
            query: Search query text
            n_results: Maximum number of results to return
            tags: Optional list of tags to filter by (matches ANY tag - OR logic)
            memory_type: Optional memory type filter (AND logic)
            min_similarity: Optional minimum similarity threshold
            offset: Number of results to skip for pagination (default: 0)

        Returns:
            List of MemoryQueryResult objects, filtered and sorted by relevance
        """
        # Check circuit breaker first
        self._check_circuit_breaker()

        try:
            # Generate query embedding using the same method as store
            try:
                # Use _generate_embedding which handles lazy model loading
                loop = asyncio.get_event_loop()
                query_embedding = await loop.run_in_executor(None, lambda: self._generate_embedding(query))
            except Exception as e:
                logger.error(f"Failed to generate query embedding: {e}")
                self._record_failure()
                return []

            # Build Qdrant Filter for tags and memory_type
            query_filter = None
            must_conditions = []
            should_conditions = []

            # Tags filter: OR logic (match ANY tag)
            if tags:
                # should[] creates OR logic - matches if ANY condition is true
                # Use MatchAny for array fields
                should_conditions.append(FieldCondition(key="tags", match=MatchAny(any=tags)))

            # Memory type filter: AND logic (must match exactly)
            if memory_type:
                # must[] creates AND logic - all conditions must be true
                must_conditions.append(FieldCondition(key="memory_type", match=MatchValue(value=memory_type)))

            # Create Filter object only if we have conditions
            if must_conditions or should_conditions:
                query_filter = Filter(
                    must=must_conditions if must_conditions else None, should=should_conditions if should_conditions else None
                )

            # Execute Qdrant search with filters and offset
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,  # Changed from query_vector
                    query_filter=query_filter,
                    limit=n_results,
                    offset=offset,
                    score_threshold=min_similarity if min_similarity else None,
                    with_payload=True,
                    with_vectors=False,  # Exclude vectors to reduce response size
                ),
            )

            # Convert ScoredPoints to MemoryQueryResult objects
            # query_points() returns QueryResponse with .points attribute
            results = []
            for scored_point in search_results.points:
                try:
                    # Skip metadata point
                    if scored_point.id == self.METADATA_POINT_ID:
                        continue

                    # Extract payload data
                    payload = scored_point.payload

                    # Create Memory object from payload
                    memory = Memory(
                        content=payload.get("content", ""),
                        content_hash=payload.get("content_hash", str(scored_point.id)),
                        tags=self._normalize_tags(payload.get("tags", [])),
                        memory_type=payload.get("memory_type"),
                        metadata=payload.get("metadata", {}),
                        created_at=payload.get("created_at"),
                        updated_at=payload.get("updated_at"),
                    )

                    # Qdrant score is already a similarity score (1.0 = perfect match for cosine)
                    # No conversion needed unlike sqlite-vec distance
                    relevance_score = float(scored_point.score)

                    # Apply minimum similarity filter if specified
                    # (score_threshold in search should handle this, but double-check)
                    if min_similarity is not None and relevance_score < min_similarity:
                        continue

                    results.append(
                        MemoryQueryResult(
                            memory=memory,
                            relevance_score=relevance_score,
                            debug_info={"score": scored_point.score, "backend": "qdrant"},
                        )
                    )

                except Exception as parse_error:
                    logger.warning(f"Failed to parse search result: {parse_error}")
                    continue

            # Record success for circuit breaker
            self._record_success()

            logger.debug(f"Retrieved {len(results)} memories from Qdrant for query")
            return results

        except Exception as e:
            # Record failure for circuit breaker
            self._record_failure()
            logger.error(f"Failed to retrieve memories from Qdrant: {e}")
            return []

    async def _generate_query_embedding(self, query: str) -> list[float]:
        """
        Generate embedding for query text.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            RuntimeError: If embedding generation fails
        """
        # This will be called by MemoryService which injects the embedding service
        # For now, we use the same pattern as sqlite_vec
        if not self.embedding_service:
            raise RuntimeError("No embedding service available")

        # The embedding service is expected to have an encode() method
        # This matches the sentence-transformers and ONNX embedding interfaces
        try:
            embedding = self.embedding_service.encode([query], convert_to_numpy=True)[0]
            embedding_list = embedding.tolist()

            # Validate embedding dimensions
            if len(embedding_list) != self._vector_size:
                raise ValueError(f"Embedding dimension mismatch: expected {self._vector_size}, got {len(embedding_list)}")

            return embedding_list

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}") from e

    @retry(
        retry=retry_if_exception(is_retryable_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    async def search_by_tag(
        self,
        tags: list[str],
        limit: int = 10,
        offset: int = 0,
        match_all: bool = False,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
    ) -> list[Memory]:
        """
        Search memories by tags with optional date filtering and retry logic.

        Retries up to 3 times for transient 5xx server errors with exponential backoff (1-5s).
        Does NOT retry for 4xx client errors (configuration errors).

        Args:
            tags: List of tags to search for
            limit: Maximum number of results to return (default: 10)
            offset: Number of results to skip for pagination (default: 0)
            match_all: If True, memory must have ALL tags; if False, ANY tag (default: False)
            start_timestamp: Filter memories from this timestamp (inclusive)
            end_timestamp: Filter memories until this timestamp (inclusive)

        Returns:
            List of Memory objects ordered by created_at DESC
        """
        # Check circuit breaker
        self._check_circuit_breaker()

        try:
            loop = asyncio.get_event_loop()

            # Build filter conditions
            filter_conditions = []

            # Tag filter (OR logic for ANY, AND logic for ALL)
            if tags:
                if match_all:
                    # AND logic - memory must have ALL tags
                    tag_filter = Filter(must=[FieldCondition(key="tags", match=MatchValue(value=tag)) for tag in tags])
                else:
                    # OR logic - memory must have ANY tag
                    tag_filter = Filter(should=[FieldCondition(key="tags", match=MatchValue(value=tag)) for tag in tags])
                filter_conditions.append(tag_filter)

            # Timestamp range filter (AND logic if provided)
            if start_timestamp is not None or end_timestamp is not None:
                timestamp_conditions = []

                if start_timestamp is not None:
                    timestamp_conditions.append(FieldCondition(key="created_at", range=Range(gte=start_timestamp)))

                if end_timestamp is not None:
                    timestamp_conditions.append(FieldCondition(key="created_at", range=Range(lte=end_timestamp)))

                if timestamp_conditions:
                    timestamp_filter = Filter(must=timestamp_conditions)
                    filter_conditions.append(timestamp_filter)

            # Combine filters (AND between tag and timestamp filters)
            combined_filter = None
            if len(filter_conditions) == 1:
                combined_filter = filter_conditions[0]
            elif len(filter_conditions) > 1:
                # Merge filters
                combined_filter = Filter(must=filter_conditions)

            # Execute scroll with pagination
            scroll_result = await loop.run_in_executor(
                None,
                lambda: self.client.scroll(
                    collection_name=self.collection_name, scroll_filter=combined_filter, limit=limit, offset=offset
                ),
            )

            # Convert Points to Memory objects
            memories = []
            points = scroll_result[0] if scroll_result else []

            for point in points:
                # Skip metadata point
                if point.id == self.METADATA_POINT_ID:
                    continue

                payload = point.payload
                memory = Memory(
                    content=payload.get("content", ""),
                    content_hash=str(point.id),
                    tags=payload.get("tags", []),
                    memory_type=payload.get("memory_type"),
                    metadata=payload.get("metadata", {}),
                    created_at=payload.get("created_at", 0.0),
                    updated_at=payload.get("updated_at", 0.0),
                )
                memories.append(memory)

            # Record success
            self._record_success()

            logger.debug(f"search_by_tag returned {len(memories)} results (tags={tags}, limit={limit}, offset={offset})")

            return memories

        except Exception as e:
            logger.error(f"search_by_tag failed: {e}")
            self._record_failure()
            raise StorageError(f"Tag search failed: {e}") from e

    async def get_memory_by_hash(self, content_hash: str) -> Memory | None:
        """
        Retrieve a specific memory by its content hash.

        Args:
            content_hash: The content hash of the memory

        Returns:
            Memory object if found, None otherwise
        """
        self._check_circuit_breaker()

        try:
            loop = asyncio.get_event_loop()
            # Convert hash to UUID format (must match how points are stored)
            point_id = self._hash_to_uuid(content_hash)
            points = await loop.run_in_executor(
                None, lambda: self.client.retrieve(collection_name=self.collection_name, ids=[point_id])
            )

            if not points or len(points) == 0:
                self._record_success()
                return None

            point = points[0]
            payload = point.payload

            memory = Memory(
                content=payload.get("content", ""),
                content_hash=content_hash,
                tags=payload.get("tags", []),
                memory_type=payload.get("memory_type"),
                metadata=payload.get("metadata", {}),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                embedding=point.vector if hasattr(point, "vector") else None,
            )

            self._record_success()
            return memory

        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to retrieve memory by hash {content_hash}: {e}")
            return None

    @retry(
        retry=retry_if_exception(is_retryable_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    async def delete(self, content_hash: str) -> tuple[bool, str]:
        """
        Delete a memory by its hash with retry logic.

        Retries up to 3 times for transient 5xx server errors with exponential backoff (1-5s).
        Does NOT retry for 4xx client errors (configuration errors).

        Args:
            content_hash: Hash of the memory to delete

        Returns:
            Tuple of (success, message)
        """
        self._check_circuit_breaker()

        try:
            # Convert hash to UUID for Qdrant compatibility
            point_id = self._hash_to_uuid(content_hash)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: self.client.delete(collection_name=self.collection_name, points_selector=[point_id])
            )

            self._record_success()
            logger.debug(f"Deleted memory {content_hash[:8]}...")
            return True, "Memory deleted successfully"

        except Exception as e:
            self._record_failure()
            error_msg = f"Failed to delete memory: {e}"
            logger.error(error_msg)
            return False, error_msg

    async def delete_by_tag(self, tag: str) -> tuple[int, str]:
        """
        Delete memories by tag.

        Args:
            tag: Tag to filter by

        Returns:
            Tuple of (count_deleted, message)
        """
        self._check_circuit_breaker()

        try:
            # Build filter for tag
            tag_filter = Filter(must=[FieldCondition(key="tags", match=MatchValue(value=tag))])

            # Get count before deletion for reporting
            loop = asyncio.get_event_loop()
            scroll_result = await loop.run_in_executor(
                None, lambda: self.client.scroll(collection_name=self.collection_name, scroll_filter=tag_filter, limit=10000)
            )

            points, _ = scroll_result
            count = len(points)

            if count == 0:
                self._record_success()
                return 0, f"No memories found with tag '{tag}'"

            # Delete all points with this tag
            await loop.run_in_executor(
                None, lambda: self.client.delete(collection_name=self.collection_name, points_selector=tag_filter)
            )

            self._record_success()
            logger.debug(f"Deleted {count} memories with tag '{tag}'")
            return count, f"Deleted {count} memories with tag '{tag}'"

        except Exception as e:
            self._record_failure()
            error_msg = f"Failed to delete memories by tag: {e}"
            logger.error(error_msg)
            return 0, error_msg

    async def delete_by_all_tags(self, tags: list[str]) -> tuple[int, str]:
        """
        Delete memories matching ALL of the given tags (AND logic).

        Args:
            tags: List of tags - only memories containing ALL tags will be deleted

        Returns:
            Tuple of (count_deleted, message)
        """
        self._check_circuit_breaker()

        try:
            # Build filter with AND logic
            tag_conditions = [FieldCondition(key="tags", match=MatchValue(value=tag)) for tag in tags]

            all_tags_filter = Filter(must=tag_conditions)

            # Get count before deletion
            loop = asyncio.get_event_loop()
            scroll_result = await loop.run_in_executor(
                None,
                lambda: self.client.scroll(collection_name=self.collection_name, scroll_filter=all_tags_filter, limit=10000),
            )

            points, _ = scroll_result
            count = len(points)

            if count == 0:
                self._record_success()
                return 0, f"No memories found with ALL tags {tags}"

            # Delete all points matching ALL tags
            await loop.run_in_executor(
                None, lambda: self.client.delete(collection_name=self.collection_name, points_selector=all_tags_filter)
            )

            self._record_success()
            logger.debug(f"Deleted {count} memories with ALL tags {tags}")
            return count, f"Deleted {count} memories with ALL tags"

        except Exception as e:
            self._record_failure()
            error_msg = f"Failed to delete memories by all tags: {e}"
            logger.error(error_msg)
            return 0, error_msg

    async def cleanup_duplicates(self) -> tuple[int, str]:
        """
        Remove duplicate memories.

        Returns:
            Tuple of (count_removed, message)
        """
        self._check_circuit_breaker()

        try:
            # Qdrant prevents duplicates automatically via upsert with content_hash as ID
            self._record_success()
            return 0, "No duplicates found (Qdrant prevents duplicates via content_hash ID)"

        except Exception as e:
            self._record_failure()
            error_msg = f"Failed to cleanup duplicates: {e}"
            logger.error(error_msg)
            return 0, error_msg

    async def update_memory_metadata(
        self, content_hash: str, updates: dict[str, Any], preserve_timestamps: bool = True
    ) -> tuple[bool, str]:
        """
        Update memory metadata without recreating the entire memory entry.

        Args:
            content_hash: Hash of the memory to update
            updates: Dictionary of metadata fields to update
            preserve_timestamps: Whether to preserve original created_at timestamp

        Returns:
            Tuple of (success, message)

        Note:
            - Only metadata, tags, and memory_type can be updated
            - Content and content_hash cannot be modified
            - updated_at timestamp is always refreshed
            - created_at is preserved unless preserve_timestamps=False
        """
        self._check_circuit_breaker()

        try:
            # First retrieve the existing memory
            memory = await self.get_memory_by_hash(content_hash)
            if not memory:
                return False, f"Memory not found: {content_hash}"

            # Build updated payload preserving existing fields
            updated_payload = {
                "content": memory.content,
                "content_hash": content_hash,
                "tags": updates.get("tags", memory.tags),
                "memory_type": updates.get("memory_type", memory.memory_type),
                "metadata": updates.get("metadata", memory.metadata),
                "created_at": memory.created_at if preserve_timestamps else datetime.now().timestamp(),
                "updated_at": datetime.now().timestamp(),
            }

            # Update the point payload in Qdrant
            # Convert hash to UUID format (must match how points are stored)
            point_id = self._hash_to_uuid(content_hash)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.set_payload(
                    collection_name=self.collection_name, payload=updated_payload, points=[point_id]
                ),
            )

            self._record_success()
            logger.debug(f"Updated metadata for memory {content_hash[:8]}...")
            return True, "Memory metadata updated successfully"

        except Exception as e:
            self._record_failure()
            error_msg = f"Failed to update memory metadata: {e}"
            logger.error(error_msg)
            return False, error_msg

    async def get_stats(self) -> dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary containing stats like total_memories, storage_backend, etc.
        """
        self._check_circuit_breaker()

        try:
            loop = asyncio.get_event_loop()
            collection_info = await loop.run_in_executor(None, lambda: self.client.get_collection(self.collection_name))

            # Exclude __metadata__ point from count
            total_count = collection_info.points_count
            actual_memories = max(0, total_count - 1)

            self._record_success()

            return {
                "total_memories": actual_memories,
                "storage_backend": "qdrant",
                "status": "operational",
                "collection_name": self.collection_name,
                "vector_size": self._vector_size,
                "embedding_model": self.embedding_model,
                "quantization_enabled": self.quantization_enabled,
                "circuit_breaker": {
                    "status": "open" if self._circuit_open_until else "closed",
                    "failure_count": self._failure_count,
                },
            }

        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to get stats: {e}")
            return {"total_memories": 0, "storage_backend": "qdrant", "status": "error", "error": str(e)}

    async def get_all_tags(self) -> list[str]:
        """
        Get all unique tags in the storage.

        Returns:
            List of unique tag strings
        """
        self._check_circuit_breaker()

        try:
            # Scroll through all points and collect unique tags
            loop = asyncio.get_event_loop()
            all_tags = set()
            offset = None

            while True:
                scroll_result = await loop.run_in_executor(
                    None,
                    lambda offset=offset: self.client.scroll(
                        collection_name=self.collection_name, limit=100, offset=offset, with_payload=True, with_vectors=False
                    ),
                )

                points, next_offset = scroll_result

                for point in points:
                    if point.id == self.METADATA_POINT_ID:
                        continue

                    tags = self._normalize_tags(point.payload.get("tags", []))
                    if tags:
                        all_tags.update(tags)

                if next_offset is None:
                    break

                offset = next_offset

            self._record_success()
            return sorted(all_tags)

        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to get all tags: {e}")
            return []

    async def get_recent_memories(self, n: int = 10) -> list[Memory]:
        """
        Get n most recent memories.

        Uses server-side sorting via Qdrant's order_by for efficiency.

        Args:
            n: Number of recent memories to retrieve

        Returns:
            List of Memory objects ordered by created_at DESC
        """
        self._check_circuit_breaker()

        try:
            loop = asyncio.get_event_loop()

            # Use server-side sorting - requires payload index on created_at
            # Request n+1 to account for potential metadata point
            scroll_result = await loop.run_in_executor(
                None,
                lambda: self.client.scroll(
                    collection_name=self.collection_name,
                    limit=n + 1,
                    with_payload=True,
                    with_vectors=False,
                    order_by=OrderBy(key="created_at", direction="desc"),
                ),
            )

            points, _ = scroll_result
            memories = []

            for point in points:
                if point.id == self.METADATA_POINT_ID:
                    continue

                payload = point.payload
                memory = Memory(
                    content=payload.get("content", ""),
                    content_hash=payload.get("content_hash", str(point.id)),
                    tags=self._normalize_tags(payload.get("tags", [])),
                    memory_type=payload.get("memory_type"),
                    metadata=payload.get("metadata", {}),
                    created_at=payload.get("created_at"),
                    updated_at=payload.get("updated_at"),
                )
                memories.append(memory)

                if len(memories) >= n:
                    break

            self._record_success()
            return memories

        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to get recent memories: {e}")
            return []

    async def recall_memory(
        self,
        query: str,
        n_results: int = 5,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        min_similarity: float | None = None,
    ) -> list[Memory]:
        """
        Recall memories based on natural language time expression.

        Args:
            query: Natural language query
            n_results: Maximum number of results
            tags: Optional tag filter
            memory_type: Optional memory type filter
            min_similarity: Optional minimum similarity threshold

        Returns:
            List of Memory objects
        """
        results = await self.retrieve(query, n_results, tags, memory_type, min_similarity)
        return [r.memory for r in results]

    async def get_all_memories(
        self, limit: int = None, offset: int = 0, memory_type: str | None = None, tags: list[str] | None = None
    ) -> list[Memory]:
        """
        Get all memories in storage ordered by creation time (newest first).

        Uses server-side sorting via Qdrant's order_by for efficiency.

        Args:
            limit: Maximum number of memories to return (None for all)
            offset: Number of memories to skip (for pagination)
            memory_type: Optional filter by memory type
            tags: Optional filter by tags (matches ANY of the provided tags)

        Returns:
            List of Memory objects ordered by created_at DESC
        """
        self._check_circuit_breaker()

        try:
            # Build filter
            conditions = []
            if memory_type:
                conditions.append(FieldCondition(key="memory_type", match=MatchValue(value=memory_type)))
            if tags:
                # Add tag filter with OR logic (match ANY tag)
                conditions.append(FieldCondition(key="tags", match=MatchAny(any=tags)))

            scroll_filter = Filter(must=conditions) if conditions else None

            loop = asyncio.get_event_loop()
            memories = []
            skipped = 0
            start_from = None
            target_count = limit if limit else 10000  # Reasonable upper bound if no limit

            # Server-side sorted scroll with pagination via start_from
            while len(memories) < target_count:
                batch_size = min(100, target_count - len(memories) + (offset - skipped if skipped < offset else 0) + 1)

                scroll_result = await loop.run_in_executor(
                    None,
                    lambda sf=start_from, bs=batch_size: self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=scroll_filter,
                        limit=bs,
                        with_payload=True,
                        with_vectors=False,
                        order_by=OrderBy(key="created_at", direction="desc", start_from=sf),
                    ),
                )

                points, _ = scroll_result

                if not points:
                    break

                for point in points:
                    if point.id == self.METADATA_POINT_ID:
                        continue

                    # Handle offset by skipping
                    if skipped < offset:
                        skipped += 1
                        continue

                    payload = point.payload
                    memory = Memory(
                        content=payload.get("content", ""),
                        content_hash=payload.get("content_hash", str(point.id)),
                        tags=self._normalize_tags(payload.get("tags", [])),
                        memory_type=payload.get("memory_type"),
                        metadata=payload.get("metadata", {}),
                        created_at=payload.get("created_at"),
                        updated_at=payload.get("updated_at"),
                    )
                    memories.append(memory)

                    if limit and len(memories) >= limit:
                        break

                # Get the last created_at for start_from pagination
                if points:
                    last_created_at = points[-1].payload.get("created_at")
                    if last_created_at is not None:
                        start_from = last_created_at
                    else:
                        break  # Can't paginate without timestamps

                if limit and len(memories) >= limit:
                    break

            self._record_success()
            return memories

        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to get all memories: {e}")
            return []

    async def get_by_hash(self, content_hash: str) -> Memory | None:
        """Get a memory by its content hash."""
        self._check_circuit_breaker()

        try:
            # Convert hash to UUID
            point_id = self._hash_to_uuid(content_hash)

            # Retrieve point by ID
            loop = asyncio.get_event_loop()
            points = await loop.run_in_executor(
                None,
                lambda: self.client.retrieve(
                    collection_name=self.collection_name, ids=[point_id], with_payload=True, with_vectors=False
                ),
            )

            if not points:
                return None

            point = points[0]
            payload = point.payload

            memory = Memory(
                content=payload.get("content", ""),
                content_hash=payload.get("content_hash", content_hash),
                tags=payload.get("tags", []),
                memory_type=payload.get("memory_type"),
                metadata=payload.get("metadata", {}),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
            )

            self._record_success()
            return memory

        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to get memory by hash {content_hash}: {e}")
            return None

    async def count_all_memories(self, memory_type: str | None = None, tags: list[str] | None = None) -> int:
        """
        Get total count of memories in storage.

        Args:
            memory_type: Optional filter by memory type
            tags: Optional filter by tags (memories matching ANY of the tags)

        Returns:
            Total number of memories
        """
        self._check_circuit_breaker()

        try:
            if not memory_type and not tags:
                # Fast path: just get collection count
                loop = asyncio.get_event_loop()
                collection_info = await loop.run_in_executor(None, lambda: self.client.get_collection(self.collection_name))
                # Exclude __metadata__ point
                count = max(0, collection_info.points_count - 1)
                self._record_success()
                return count

            # Slow path: filter required
            memories = await self.get_all_memories(memory_type=memory_type, tags=tags)
            self._record_success()
            return len(memories)

        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to count memories: {e}")
            return 0

    async def get_memories_by_time_range(self, start_time: float, end_time: float) -> list[Memory]:
        """
        Get memories within a time range.

        Args:
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            List of Memory objects within the time range
        """
        self._check_circuit_breaker()

        try:
            # Build timestamp range filter
            time_filter = Filter(must=[FieldCondition(key="created_at", range=Range(gte=start_time, lte=end_time))])

            # Scroll through matching points
            loop = asyncio.get_event_loop()
            memories = []
            offset = None

            while True:
                scroll_result = await loop.run_in_executor(
                    None,
                    lambda offset=offset: self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=time_filter,
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    ),
                )

                points, next_offset = scroll_result

                for point in points:
                    if point.id == self.METADATA_POINT_ID:
                        continue

                    payload = point.payload
                    memory = Memory(
                        content=payload.get("content", ""),
                        content_hash=payload.get("content_hash", str(point.id)),
                        tags=self._normalize_tags(payload.get("tags", [])),
                        memory_type=payload.get("memory_type"),
                        metadata=payload.get("metadata", {}),
                        created_at=payload.get("created_at"),
                        updated_at=payload.get("updated_at"),
                    )
                    memories.append(memory)

                if next_offset is None:
                    break

                offset = next_offset

            # Sort by created_at
            memories.sort(key=lambda m: self._normalize_timestamp(m.created_at))

            self._record_success()
            return memories

        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to get memories by time range: {e}")
            return []

    async def get_memory_connections(self) -> dict[str, int]:
        """
        Get memory connection statistics.

        Returns:
            Dictionary of connection statistics
        """
        # Not implemented for Qdrant
        return {}

    async def get_access_patterns(self) -> dict[str, datetime]:
        """
        Get memory access pattern statistics.

        Returns:
            Dictionary of access pattern statistics
        """
        # Not implemented for Qdrant
        return {}

    async def count_semantic_search(
        self,
        query: str,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        min_similarity: float | None = None,
    ) -> int:
        """
        Count memories matching semantic search criteria.

        Note: This performs the full semantic search and counts results.
        This is expensive but accurate (as per user requirement for count after semantic filtering).

        Args:
            query: Search query text
            tags: Optional list of tags to filter by (matches ANY tag)
            memory_type: Optional memory type filter
            min_similarity: Optional minimum similarity threshold

        Returns:
            Total number of memories matching the criteria
        """
        try:
            # Perform full semantic search to get accurate count
            results = await self.retrieve(
                query=query,
                n_results=10000,  # High limit to get all matching results
                tags=tags,
                memory_type=memory_type,
                min_similarity=min_similarity,
                offset=0,
            )
            return len(results)
        except Exception as e:
            logger.error(f"Error counting semantic search results: {str(e)}")
            return 0

    async def count_tag_search(
        self,
        tags: list[str],
        match_all: bool = False,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
    ) -> int:
        """
        Count memories matching tag search with optional date filtering.

        Args:
            tags: List of tags to search for
            match_all: If True, memory must have ALL tags; if False, ANY tag (default)
            start_timestamp: Filter memories from this timestamp (inclusive)
            end_timestamp: Filter memories until this timestamp (inclusive)

        Returns:
            Total number of memories matching the criteria
        """
        try:
            if not tags:
                return 0

            # Check circuit breaker
            self._check_circuit_breaker()

            # Build filter conditions matching search_by_tag logic
            filter_conditions = []

            # Tag filter (OR logic for ANY, AND logic for ALL)
            if match_all:
                # AND logic - memory must have ALL tags
                tag_filter = Filter(must=[FieldCondition(key="tags", match=MatchValue(value=tag)) for tag in tags])
            else:
                # OR logic - memory must have ANY tag
                tag_filter = Filter(should=[FieldCondition(key="tags", match=MatchValue(value=tag)) for tag in tags])
            filter_conditions.append(tag_filter)

            # Timestamp range filter
            if start_timestamp is not None or end_timestamp is not None:
                timestamp_conditions = []

                if start_timestamp is not None:
                    timestamp_conditions.append(FieldCondition(key="created_at", range=Range(gte=start_timestamp)))

                if end_timestamp is not None:
                    timestamp_conditions.append(FieldCondition(key="created_at", range=Range(lte=end_timestamp)))

                if timestamp_conditions:
                    timestamp_filter = Filter(must=timestamp_conditions)
                    filter_conditions.append(timestamp_filter)

            # Combine filters
            combined_filter = None
            if len(filter_conditions) == 1:
                combined_filter = filter_conditions[0]
            elif len(filter_conditions) > 1:
                combined_filter = Filter(must=filter_conditions)

            # Use Qdrant's count API
            loop = asyncio.get_event_loop()
            count_result = await loop.run_in_executor(
                None,
                lambda: self.client.count(
                    collection_name=self.collection_name,
                    count_filter=combined_filter,
                    exact=True,  # Get exact count
                ),
            )

            # Only subtract 1 for metadata point if no filter applied
            # (metadata point won't match tag/time filters anyway)
            count = count_result.count
            if count > 0 and not combined_filter:
                count = max(0, count - 1)

            self._record_success()
            return count

        except Exception as e:
            logger.error(f"Error counting tag search results: {str(e)}")
            self._record_failure()
            return 0

    async def count_time_range(
        self,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
        tags: list[str] | None = None,
        memory_type: str | None = None,
    ) -> int:
        """
        Count memories within time range with optional filters.

        Args:
            start_timestamp: Filter from this time (inclusive)
            end_timestamp: Filter until this time (inclusive)
            tags: Optional tag filter (ANY match)
            memory_type: Optional type filter

        Returns:
            Total number of memories matching the criteria
        """
        try:
            # Check circuit breaker
            self._check_circuit_breaker()

            # Build filter conditions
            conditions = []

            # Time range filters
            if start_timestamp is not None:
                conditions.append(FieldCondition(key="created_at", range=Range(gte=start_timestamp)))

            if end_timestamp is not None:
                conditions.append(FieldCondition(key="created_at", range=Range(lte=end_timestamp)))

            # Memory type filter
            if memory_type is not None:
                conditions.append(FieldCondition(key="memory_type", match=MatchValue(value=memory_type)))

            # Tags filter (OR logic - match ANY tag)
            if tags:
                tag_conditions = [FieldCondition(key="tags", match=MatchValue(value=tag)) for tag in tags]
                # Use should for OR logic
                conditions.append(Filter(should=tag_conditions))

            # Combine all conditions with AND logic
            combined_filter = Filter(must=conditions) if conditions else None

            # Use Qdrant's count API
            loop = asyncio.get_event_loop()
            count_result = await loop.run_in_executor(
                None, lambda: self.client.count(collection_name=self.collection_name, count_filter=combined_filter, exact=True)
            )

            # Only subtract 1 for metadata point if no filter applied
            # (metadata point won't match tag/time filters anyway)
            count = count_result.count
            if count > 0 and not combined_filter:
                count = max(0, count - 1)

            self._record_success()
            return count

        except Exception as e:
            logger.error(f"Error counting time range results: {str(e)}")
            self._record_failure()
            return 0

    async def close(self) -> None:
        """
        Close the Qdrant client connection.

        Safe to call multiple times. Idempotent operation.
        """
        if self.client is not None:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.client.close)
                logger.info("Qdrant client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Qdrant client: {e}")
            finally:
                self.client = None
