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
SQLite-vec storage backend for MCP Memory Service.
Provides a lightweight alternative to ChromaDB using sqlite-vec extension.
"""

import sqlite3
import json
import logging
import traceback
import time
import os
import sys
import platform
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional, Set, Callable
from datetime import datetime
import asyncio
import random

# Import sqlite-vec with fallback
try:
    import sqlite_vec
    from sqlite_vec import serialize_float32
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False
    print("WARNING: sqlite-vec not available. Install with: pip install sqlite-vec")

# Import sentence transformers with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("WARNING: sentence_transformers not available. Install for embedding support.")

from .base import MemoryStorage
from ..models.memory import Memory, MemoryQueryResult
from ..utils.hashing import generate_content_hash
from ..utils.system_detection import (
    get_system_info,
    get_optimal_embedding_settings,
    get_torch_device,
    AcceleratorType
)

logger = logging.getLogger(__name__)

# Global model cache for performance optimization
_MODEL_CACHE = {}
_EMBEDDING_CACHE = {}


class SqliteVecMemoryStorage(MemoryStorage):
    """
    SQLite-vec based memory storage implementation.
    
    This backend provides a lightweight alternative to ChromaDB using sqlite-vec
    for vector similarity search while maintaining the same interface.
    """
    
    def __init__(self, db_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize SQLite-vec storage.
        
        Args:
            db_path: Path to SQLite database file
            embedding_model: Name of sentence transformer model to use
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.conn = None
        self.embedding_model = None
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        
        # Performance settings
        self.enable_cache = True
        self.batch_size = 32
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        
        logger.info(f"Initialized SQLite-vec storage at: {self.db_path}")

    def _safe_json_loads(self, json_str: str, context: str = "") -> dict:
        """Safely parse JSON with comprehensive error handling and logging."""
        if not json_str:
            return {}
        try:
            result = json.loads(json_str)
            if not isinstance(result, dict):
                logger.warning(f"Non-dict JSON in {context}: {type(result)}")
                return {}
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {context}: {e}, data: {json_str[:100]}...")
            return {}
        except TypeError as e:
            logger.error(f"JSON type error in {context}: {e}")
            return {}

    async def _execute_with_retry(self, operation: Callable, max_retries: int = 3, initial_delay: float = 0.1):
        """
        Execute a database operation with exponential backoff retry logic.
        
        Args:
            operation: The database operation to execute
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            
        Returns:
            The result of the operation
            
        Raises:
            The last exception if all retries fail
        """
        last_exception = None
        delay = initial_delay
        
        for attempt in range(max_retries + 1):
            try:
                return operation()
            except sqlite3.OperationalError as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Check if error is related to database locking
                if "locked" in error_msg or "busy" in error_msg:
                    if attempt < max_retries:
                        # Add jitter to prevent thundering herd
                        jittered_delay = delay * (1 + random.uniform(-0.1, 0.1))
                        logger.warning(f"Database locked, retrying in {jittered_delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(jittered_delay)
                        # Exponential backoff
                        delay *= 2
                        continue
                    else:
                        logger.error(f"Database locked after {max_retries} retries")
                else:
                    # Non-retryable error
                    raise
            except Exception as e:
                # Non-SQLite errors are not retried
                raise
        
        # If we get here, all retries failed
        raise last_exception
    
    def _check_extension_support(self):
        """Check if Python's sqlite3 supports loading extensions."""
        test_conn = None
        try:
            test_conn = sqlite3.connect(":memory:")
            if not hasattr(test_conn, 'enable_load_extension'):
                return False, "Python sqlite3 module not compiled with extension support"
            
            # Test if we can actually enable extension loading
            test_conn.enable_load_extension(True)
            test_conn.enable_load_extension(False)
            return True, "Extension loading supported"
            
        except AttributeError as e:
            return False, f"enable_load_extension not available: {e}"
        except sqlite3.OperationalError as e:
            return False, f"Extension loading disabled: {e}"
        except Exception as e:
            return False, f"Extension support check failed: {e}"
        finally:
            if test_conn:
                test_conn.close()

    async def initialize(self):
        """Initialize the SQLite database with vec0 extension."""
        try:
            if not SQLITE_VEC_AVAILABLE:
                raise ImportError("sqlite-vec is not available. Install with: pip install sqlite-vec")
            
            # Check if ONNX embeddings are enabled (preferred for Docker)
            from ..config import USE_ONNX
            if USE_ONNX:
                logger.info("ONNX embeddings enabled - skipping sentence-transformers installation")
                # ONNX embeddings don't require sentence-transformers, but we still need to initialize the database
                # Continue with database initialization below
                
            # Check sentence-transformers availability (only if ONNX disabled)
            if not USE_ONNX:
                global SENTENCE_TRANSFORMERS_AVAILABLE
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    raise ImportError("sentence-transformers is not available. Install with: pip install sentence-transformers torch")
            
            # Check if extension loading is supported
            extension_supported, support_message = self._check_extension_support()
            if not extension_supported:
                error_msg = f"SQLite extension loading not supported: {support_message}"
                logger.error(error_msg)
                
                # Provide detailed error message with solutions
                platform_info = f"{platform.system()} {platform.release()}"
                solutions = []
                
                if platform.system().lower() == "darwin":  # macOS
                    solutions.extend([
                        "Install Python via Homebrew: brew install python",
                        "Use pyenv with extension support: PYTHON_CONFIGURE_OPTS='--enable-loadable-sqlite-extensions' pyenv install 3.12.0",
                        "Switch to ChromaDB backend: export MCP_MEMORY_STORAGE_BACKEND=chromadb"
                    ])
                elif platform.system().lower() == "linux":
                    solutions.extend([
                        "Install Python with extension support: apt install python3-dev sqlite3",
                        "Rebuild Python with: ./configure --enable-loadable-sqlite-extensions",
                        "Switch to ChromaDB backend: export MCP_MEMORY_STORAGE_BACKEND=chromadb"
                    ])
                else:  # Windows and others
                    solutions.extend([
                        "Use official Python installer from python.org",
                        "Install Python with conda: conda install python",
                        "Switch to ChromaDB backend: export MCP_MEMORY_STORAGE_BACKEND=chromadb"
                    ])
                
                detailed_error = f"""
{error_msg}

Platform: {platform_info}
Python Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}

SOLUTIONS:
{chr(10).join(f"  • {solution}" for solution in solutions)}

The sqlite-vec backend requires Python compiled with --enable-loadable-sqlite-extensions.
Consider using the ChromaDB backend as an alternative: it provides the same functionality
without requiring SQLite extensions.

To switch backends permanently, set: MCP_MEMORY_STORAGE_BACKEND=chromadb
"""
                raise RuntimeError(detailed_error.strip())
            
            # Connect to database
            self.conn = sqlite3.connect(self.db_path)
            
            # Load sqlite-vec extension with proper error handling
            try:
                self.conn.enable_load_extension(True)
                sqlite_vec.load(self.conn)
                self.conn.enable_load_extension(False)
                logger.info("sqlite-vec extension loaded successfully")
            except Exception as e:
                error_msg = f"Failed to load sqlite-vec extension: {e}"
                logger.error(error_msg)
                if self.conn:
                    self.conn.close()
                    self.conn = None
                
                # Provide specific guidance based on the error
                if "enable_load_extension" in str(e):
                    detailed_error = f"""
{error_msg}

This error occurs when Python's sqlite3 module is not compiled with extension support.
This is common on macOS with the system Python installation.

RECOMMENDED SOLUTIONS:
  • Use Homebrew Python: brew install python && rehash
  • Use pyenv with extensions: PYTHON_CONFIGURE_OPTS='--enable-loadable-sqlite-extensions' pyenv install 3.12.0
  • Switch to ChromaDB backend: export MCP_MEMORY_STORAGE_BACKEND=chromadb

The ChromaDB backend provides the same vector search functionality without requiring SQLite extensions.
"""
                else:
                    detailed_error = f"""
{error_msg}

Failed to load the sqlite-vec extension. This could be due to:
  • Incompatible sqlite-vec version
  • Missing system dependencies
  • SQLite version incompatibility

SOLUTIONS:
  • Reinstall sqlite-vec: pip install --force-reinstall sqlite-vec
  • Switch to ChromaDB backend: export MCP_MEMORY_STORAGE_BACKEND=chromadb
  • Check SQLite version: python -c "import sqlite3; print(sqlite3.sqlite_version)"
"""
                raise RuntimeError(detailed_error.strip())
            
            # Apply default pragmas for concurrent access
            default_pragmas = {
                "journal_mode": "WAL",  # Enable WAL mode for concurrent access
                "busy_timeout": "5000",  # 5 second timeout for locked database
                "synchronous": "NORMAL",  # Balanced performance/safety
                "cache_size": "10000",  # Increase cache size
                "temp_store": "MEMORY"  # Use memory for temp tables
            }
            
            # Check for custom pragmas from environment variable
            custom_pragmas = os.environ.get("MCP_MEMORY_SQLITE_PRAGMAS", "")
            if custom_pragmas:
                # Parse custom pragmas (format: "pragma1=value1,pragma2=value2")
                for pragma_pair in custom_pragmas.split(","):
                    pragma_pair = pragma_pair.strip()
                    if "=" in pragma_pair:
                        pragma_name, pragma_value = pragma_pair.split("=", 1)
                        default_pragmas[pragma_name.strip()] = pragma_value.strip()
                        logger.info(f"Custom pragma from env: {pragma_name}={pragma_value}")
            
            # Apply all pragmas
            applied_pragmas = []
            for pragma_name, pragma_value in default_pragmas.items():
                try:
                    self.conn.execute(f"PRAGMA {pragma_name}={pragma_value}")
                    applied_pragmas.append(f"{pragma_name}={pragma_value}")
                except sqlite3.Error as e:
                    logger.warning(f"Failed to set pragma {pragma_name}={pragma_value}: {e}")
            
            logger.info(f"SQLite pragmas applied: {', '.join(applied_pragmas)}")
            
            # Create regular table for memory data
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT,
                    memory_type TEXT,
                    metadata TEXT,
                    created_at REAL,
                    updated_at REAL,
                    created_at_iso TEXT,
                    updated_at_iso TEXT
                )
            ''')
            
            # Initialize embedding model BEFORE creating vector table
            await self._initialize_embedding_model()
            
            # Now create virtual table with correct dimensions
            self.conn.execute(f'''
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
                    content_embedding FLOAT[{self.embedding_dimension}]
                )
            ''')
            
            # Create indexes for better performance
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)')
            
            logger.info(f"SQLite-vec storage initialized successfully with embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            error_msg = f"Failed to initialize SQLite-vec storage: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def _is_docker_environment(self) -> bool:
        """Detect if running inside a Docker container."""
        # Check for Docker-specific files/environment
        if os.path.exists('/.dockerenv'):
            return True
        if os.environ.get('DOCKER_CONTAINER'):
            return True
        # Check if running in common container environments
        if any(os.environ.get(var) for var in ['KUBERNETES_SERVICE_HOST', 'MESOS_SANDBOX']):
            return True
        # Check cgroup for docker/containerd/podman
        try:
            with open('/proc/self/cgroup', 'r') as f:
                return any('docker' in line or 'containerd' in line for line in f)
        except (IOError, FileNotFoundError):
            pass
        return False

    async def _initialize_embedding_model(self):
        """Initialize the embedding model (ONNX or SentenceTransformer based on configuration)."""
        global _MODEL_CACHE

        # Detect if we're in Docker
        is_docker = self._is_docker_environment()
        if is_docker:
            logger.info("🐳 Docker environment detected - adjusting model loading strategy")

        try:
            # Check if we should use ONNX
            use_onnx = os.environ.get('MCP_MEMORY_USE_ONNX', '').lower() in ('1', 'true', 'yes')

            if use_onnx:
                # Try to use ONNX embeddings
                logger.info("Attempting to use ONNX embeddings (PyTorch-free)")
                try:
                    from ..embeddings import get_onnx_embedding_model

                    # Check cache first
                    cache_key = f"onnx_{self.embedding_model_name}"
                    if cache_key in _MODEL_CACHE:
                        self.embedding_model = _MODEL_CACHE[cache_key]
                        logger.info(f"Using cached ONNX embedding model: {self.embedding_model_name}")
                        return

                    # Create ONNX model
                    onnx_model = get_onnx_embedding_model(self.embedding_model_name)
                    if onnx_model:
                        self.embedding_model = onnx_model
                        self.embedding_dimension = onnx_model.embedding_dimension
                        _MODEL_CACHE[cache_key] = onnx_model
                        logger.info(f"ONNX embedding model loaded successfully. Dimension: {self.embedding_dimension}")
                        return
                    else:
                        logger.warning("ONNX model creation failed, falling back to SentenceTransformer")
                except ImportError as e:
                    logger.warning(f"ONNX dependencies not available: {e}")
                except Exception as e:
                    logger.warning(f"Failed to initialize ONNX embeddings: {e}")

            # Fall back to SentenceTransformer
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Neither ONNX nor sentence-transformers available. Install one: pip install onnxruntime tokenizers OR pip install sentence-transformers torch")

            # Check cache first
            cache_key = self.embedding_model_name
            if cache_key in _MODEL_CACHE:
                self.embedding_model = _MODEL_CACHE[cache_key]
                logger.info(f"Using cached embedding model: {self.embedding_model_name}")
                return

            # Get system info for optimal settings
            system_info = get_system_info()
            device = get_torch_device()

            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            logger.info(f"Using device: {device}")

            # Configure for offline mode if models are cached
            # Only set offline mode if we detect cached models to prevent initial downloads
            hf_home = os.environ.get('HF_HOME', os.path.expanduser("~/.cache/huggingface"))
            model_cache_path = os.path.join(hf_home, "hub", f"models--sentence-transformers--{self.embedding_model_name.replace('/', '--')}")
            if os.path.exists(model_cache_path):
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                logger.info("📦 Found cached model - enabling offline mode")

            # Try to load from cache first, fallback to direct model name
            try:
                # First try loading from Hugging Face cache
                hf_home = os.environ.get('HF_HOME', os.path.expanduser("~/.cache/huggingface"))
                cache_path = os.path.join(hf_home, "hub", f"models--sentence-transformers--{self.embedding_model_name.replace('/', '--')}")
                if os.path.exists(cache_path):
                    # Find the snapshot directory
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
                # Try to download the model (may fail in Docker without network)
                try:
                    logger.info("Attempting to download model from Hugging Face...")
                    self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
                except OSError as download_error:
                    # Check if this is a network connectivity issue
                    error_msg = str(download_error)
                    if any(phrase in error_msg.lower() for phrase in ['connection', 'network', 'couldn\'t connect', 'huggingface.co']):
                        # Provide Docker-specific help
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
                        # Re-raise if not a network issue
                        raise
            except Exception as cache_error:
                logger.warning(f"Failed to load from cache: {cache_error}")
                # Fallback to normal loading (may fail if offline)
                logger.info("Attempting normal model loading...")
                self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)

            # Update embedding dimension based on actual model
            test_embedding = self.embedding_model.encode(["test"], convert_to_numpy=True)
            self.embedding_dimension = test_embedding.shape[1]

            # Cache the model
            _MODEL_CACHE[cache_key] = self.embedding_model

            logger.info(f"✅ Embedding model loaded successfully. Dimension: {self.embedding_dimension}")

        except RuntimeError:
            # Re-raise our custom errors with helpful messages
            raise
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue without embeddings - some operations may still work
            logger.warning("⚠️ Continuing without embedding support - search functionality will be limited")

    def _get_docker_network_help(self) -> str:
        """Get Docker-specific network troubleshooting help."""
        # Try to detect the Docker platform
        docker_platform = "Docker"
        if os.environ.get('DOCKER_DESKTOP_VERSION'):
            docker_platform = "Docker Desktop"
        elif os.path.exists('/proc/version'):
            try:
                with open('/proc/version', 'r') as f:
                    version = f.read().lower()
                    if 'microsoft' in version:
                        docker_platform = "Docker Desktop for Windows"
            except (IOError, FileNotFoundError):
                pass

        return (
            f"\n🐳 Docker Environment Detected ({docker_platform})\n"
            f"This appears to be a network connectivity issue common in Docker containers.\n"
        )
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not self.embedding_model:
            raise RuntimeError("No embedding model available. Ensure sentence-transformers is installed and model is loaded.")
        
        try:
            # Check cache first
            if self.enable_cache:
                cache_key = hash(text)
                if cache_key in _EMBEDDING_CACHE:
                    return _EMBEDDING_CACHE[cache_key]
            
            # Generate embedding
            embedding = self.embedding_model.encode([text], convert_to_numpy=True)[0]
            embedding_list = embedding.tolist()
            
            # Validate embedding
            if not embedding_list:
                raise ValueError("Generated embedding is empty")
            
            if len(embedding_list) != self.embedding_dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding_list)}")
            
            # Validate values are finite
            if not all(isinstance(x, (int, float)) and not (x != x) and x != float('inf') and x != float('-inf') for x in embedding_list):
                raise ValueError("Embedding contains invalid values (NaN or infinity)")
            
            # Cache the result
            if self.enable_cache:
                _EMBEDDING_CACHE[cache_key] = embedding_list
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise RuntimeError(f"Failed to generate embedding: {str(e)}") from e
    
    async def store(self, memory: Memory) -> Tuple[bool, str]:
        """Store a memory in the SQLite-vec database."""
        try:
            if not self.conn:
                return False, "Database not initialized"
            
            # Check for duplicates
            cursor = self.conn.execute(
                'SELECT content_hash FROM memories WHERE content_hash = ?',
                (memory.content_hash,)
            )
            if cursor.fetchone():
                return False, "Duplicate content detected"
            
            # Generate and validate embedding
            try:
                embedding = self._generate_embedding(memory.content)
            except Exception as e:
                logger.error(f"Failed to generate embedding for memory {memory.content_hash}: {str(e)}")
                return False, f"Failed to generate embedding: {str(e)}"
            
            # Prepare metadata
            tags_str = ",".join(memory.tags) if memory.tags else ""
            metadata_str = json.dumps(memory.metadata) if memory.metadata else "{}"
            
            # Insert into memories table (metadata) with retry logic
            def insert_memory():
                cursor = self.conn.execute('''
                    INSERT INTO memories (
                        content_hash, content, tags, memory_type,
                        metadata, created_at, updated_at, created_at_iso, updated_at_iso
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    memory.content_hash,
                    memory.content,
                    tags_str,
                    memory.memory_type,
                    metadata_str,
                    memory.created_at,
                    memory.updated_at,
                    memory.created_at_iso,
                    memory.updated_at_iso
                ))
                return cursor.lastrowid
            
            memory_rowid = await self._execute_with_retry(insert_memory)
            
            # Insert into embeddings table with retry logic
            def insert_embedding():
                # Check if we can insert with specific rowid
                try:
                    self.conn.execute('''
                        INSERT INTO memory_embeddings (rowid, content_embedding)
                        VALUES (?, ?)
                    ''', (
                        memory_rowid,
                        serialize_float32(embedding)
                    ))
                except sqlite3.Error as e:
                    # If rowid insert fails, try without specifying rowid
                    logger.warning(f"Failed to insert with rowid {memory_rowid}: {e}. Trying without rowid.")
                    self.conn.execute('''
                        INSERT INTO memory_embeddings (content_embedding)
                        VALUES (?)
                    ''', (
                        serialize_float32(embedding),
                    ))
            
            await self._execute_with_retry(insert_embedding)
            
            # Commit with retry logic
            await self._execute_with_retry(self.conn.commit)
            
            logger.info(f"Successfully stored memory: {memory.content_hash}")
            return True, "Memory stored successfully"
            
        except Exception as e:
            error_msg = f"Failed to store memory: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, error_msg
    
    async def retrieve(self, query: str, n_results: int = 5) -> List[MemoryQueryResult]:
        """Retrieve memories using semantic search."""
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return []
            
            if not self.embedding_model:
                logger.warning("No embedding model available, cannot perform semantic search")
                return []
            
            # Generate query embedding
            try:
                query_embedding = self._generate_embedding(query)
            except Exception as e:
                logger.error(f"Failed to generate query embedding: {str(e)}")
                return []
            
            # First, check if embeddings table has data
            cursor = self.conn.execute('SELECT COUNT(*) FROM memory_embeddings')
            embedding_count = cursor.fetchone()[0]
            
            if embedding_count == 0:
                logger.warning("No embeddings found in database. Memories may have been stored without embeddings.")
                return []
            
            # Perform vector similarity search using JOIN with retry logic
            def search_memories():
                # Try direct rowid join first
                cursor = self.conn.execute('''
                    SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                           m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso, 
                           e.distance
                    FROM memories m
                    INNER JOIN (
                        SELECT rowid, distance 
                        FROM memory_embeddings 
                        WHERE content_embedding MATCH ?
                        ORDER BY distance
                        LIMIT ?
                    ) e ON m.id = e.rowid
                    ORDER BY e.distance
                ''', (serialize_float32(query_embedding), n_results))
                
                # Check if we got results
                results = cursor.fetchall()
                if not results:
                    # Log debug info
                    logger.debug("No results from vector search. Checking database state...")
                    mem_count = self.conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0]
                    logger.debug(f"Memories table has {mem_count} rows, embeddings table has {embedding_count} rows")
                
                return results
            
            search_results = await self._execute_with_retry(search_memories)
            
            results = []
            for row in search_results:
                try:
                    # Parse row data
                    content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                    created_at, updated_at, created_at_iso, updated_at_iso, distance = row[5:]
                    
                    # Parse tags and metadata
                    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")
                    
                    # Create Memory object
                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )
                    
                    # Calculate relevance score (lower distance = higher relevance)
                    relevance_score = max(0.0, 1.0 - distance)
                    
                    results.append(MemoryQueryResult(
                        memory=memory,
                        relevance_score=relevance_score,
                        debug_info={"distance": distance, "backend": "sqlite-vec"}
                    ))
                    
                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue
            
            logger.info(f"Retrieved {len(results)} memories for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    async def search_by_tag(self, tags: List[str]) -> List[Memory]:
        """Search memories by tags."""
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return []
            
            if not tags:
                return []
            
            # Build query for tag search (OR logic)
            tag_conditions = " OR ".join(["tags LIKE ?" for _ in tags])
            tag_params = [f"%{tag}%" for tag in tags]
            
            cursor = self.conn.execute(f'''
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
                WHERE {tag_conditions}
                ORDER BY created_at DESC
            ''', tag_params)
            
            results = []
            for row in cursor.fetchall():
                try:
                    content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                    created_at, updated_at, created_at_iso, updated_at_iso = row[5:]
                    
                    # Parse tags and metadata
                    memory_tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")
                    
                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=memory_tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )
                    
                    results.append(memory)
                    
                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue
            
            logger.info(f"Found {len(results)} memories with tags: {tags}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search by tags: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    async def search_by_tags(self, tags: List[str], operation: str = "AND") -> List[Memory]:
        """Search memories by tags with AND/OR operation support."""
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return []
            
            if not tags:
                return []
            
            # Build query based on operation
            if operation.upper() == "AND":
                # All tags must be present (each tag must appear in the tags field)
                tag_conditions = " AND ".join(["tags LIKE ?" for _ in tags])
            else:  # OR operation (default for backward compatibility)
                tag_conditions = " OR ".join(["tags LIKE ?" for _ in tags])
            
            tag_params = [f"%{tag}%" for tag in tags]
            
            cursor = self.conn.execute(f'''
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories 
                WHERE {tag_conditions}
                ORDER BY updated_at DESC
            ''', tag_params)
            
            results = []
            for row in cursor.fetchall():
                try:
                    content_hash, content, tags_str, memory_type, metadata_str, created_at, updated_at, created_at_iso, updated_at_iso = row
                    
                    # Parse tags and metadata
                    memory_tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")
                    
                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=memory_tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )
                    
                    results.append(memory)
                    
                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue
            
            logger.info(f"Found {len(results)} memories with tags: {tags} (operation: {operation})")
            return results

        except Exception as e:
            logger.error(f"Failed to search by tags with operation {operation}: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    async def search_by_tag_chronological(self, tags: List[str], limit: int = None, offset: int = 0) -> List[Memory]:
        """
        Search memories by tags with chronological ordering and database-level pagination.

        This method addresses Gemini Code Assist's performance concern by pushing
        ordering and pagination to the database level instead of doing it in Python.

        Args:
            tags: List of tags to search for
            limit: Maximum number of memories to return (None for all)
            offset: Number of memories to skip (for pagination)

        Returns:
            List of Memory objects ordered by created_at DESC
        """
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return []

            if not tags:
                return []

            # Build query for tag search (OR logic) with database-level ordering and pagination
            tag_conditions = " OR ".join(["tags LIKE ?" for _ in tags])
            tag_params = [f"%{tag}%" for tag in tags]

            # Build pagination clauses
            limit_clause = f"LIMIT {limit}" if limit is not None else ""
            offset_clause = f"OFFSET {offset}" if offset > 0 else ""

            query = f'''
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
                WHERE {tag_conditions}
                ORDER BY created_at DESC
                {limit_clause} {offset_clause}
            '''

            cursor = self.conn.execute(query, tag_params)
            results = []

            for row in cursor.fetchall():
                try:
                    content_hash, content, tags_str, memory_type, metadata_str, created_at, updated_at, created_at_iso, updated_at_iso = row

                    # Parse tags and metadata
                    memory_tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")

                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=memory_tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )

                    results.append(memory)

                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue

            logger.info(f"Found {len(results)} memories with tags: {tags} using database-level pagination (limit={limit}, offset={offset})")
            return results

        except Exception as e:
            logger.error(f"Failed to search by tags chronologically: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        """Delete a memory by its content hash."""
        try:
            if not self.conn:
                return False, "Database not initialized"
            
            # Get the id first to delete corresponding embedding
            cursor = self.conn.execute('SELECT id FROM memories WHERE content_hash = ?', (content_hash,))
            row = cursor.fetchone()
            
            if row:
                memory_id = row[0]
                # Delete from both tables
                self.conn.execute('DELETE FROM memory_embeddings WHERE rowid = ?', (memory_id,))
                cursor = self.conn.execute('DELETE FROM memories WHERE content_hash = ?', (content_hash,))
                self.conn.commit()
            else:
                return False, f"Memory with hash {content_hash} not found"
            
            if cursor.rowcount > 0:
                logger.info(f"Deleted memory: {content_hash}")
                return True, f"Successfully deleted memory {content_hash}"
            else:
                return False, f"Memory with hash {content_hash} not found"
                
        except Exception as e:
            error_msg = f"Failed to delete memory: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    async def get_by_hash(self, content_hash: str) -> Optional[Memory]:
        """Get a memory by its content hash."""
        try:
            if not self.conn:
                return None
            
            cursor = self.conn.execute('''
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories WHERE content_hash = ?
            ''', (content_hash,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            content_hash, content, tags_str, memory_type, metadata_str = row[:5]
            created_at, updated_at, created_at_iso, updated_at_iso = row[5:]
            
            # Parse tags and metadata
            tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
            metadata = self._safe_json_loads(metadata_str, "memory_retrieval")
            
            memory = Memory(
                content=content,
                content_hash=content_hash,
                tags=tags,
                memory_type=memory_type,
                metadata=metadata,
                created_at=created_at,
                updated_at=updated_at,
                created_at_iso=created_at_iso,
                updated_at_iso=updated_at_iso
            )
            
            return memory
            
        except Exception as e:
            logger.error(f"Failed to get memory by hash {content_hash}: {str(e)}")
            return None
    
    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        """Delete memories by tag."""
        try:
            if not self.conn:
                return 0, "Database not initialized"
            
            # Get the ids first to delete corresponding embeddings
            cursor = self.conn.execute('SELECT id FROM memories WHERE tags LIKE ?', (f"%{tag}%",))
            memory_ids = [row[0] for row in cursor.fetchall()]
            
            # Delete from both tables
            for memory_id in memory_ids:
                self.conn.execute('DELETE FROM memory_embeddings WHERE rowid = ?', (memory_id,))
            
            cursor = self.conn.execute('DELETE FROM memories WHERE tags LIKE ?', (f"%{tag}%",))
            self.conn.commit()
            
            count = cursor.rowcount
            logger.info(f"Deleted {count} memories with tag: {tag}")
            
            if count > 0:
                return count, f"Successfully deleted {count} memories with tag '{tag}'"
            else:
                return 0, f"No memories found with tag '{tag}'"
                
        except Exception as e:
            error_msg = f"Failed to delete by tag: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg
    
    async def cleanup_duplicates(self) -> Tuple[int, str]:
        """Remove duplicate memories based on content hash."""
        try:
            if not self.conn:
                return 0, "Database not initialized"
            
            # Find duplicates (keep the first occurrence)
            cursor = self.conn.execute('''
                DELETE FROM memories 
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) 
                    FROM memories 
                    GROUP BY content_hash
                )
            ''')
            self.conn.commit()
            
            count = cursor.rowcount
            logger.info(f"Cleaned up {count} duplicate memories")
            
            if count > 0:
                return count, f"Successfully removed {count} duplicate memories"
            else:
                return 0, "No duplicate memories found"
                
        except Exception as e:
            error_msg = f"Failed to cleanup duplicates: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg
    
    async def update_memory_metadata(self, content_hash: str, updates: Dict[str, Any], preserve_timestamps: bool = True) -> Tuple[bool, str]:
        """Update memory metadata without recreating the entire memory entry."""
        try:
            if not self.conn:
                return False, "Database not initialized"
            
            # Get current memory
            cursor = self.conn.execute('''
                SELECT content, tags, memory_type, metadata, created_at, created_at_iso
                FROM memories WHERE content_hash = ?
            ''', (content_hash,))
            
            row = cursor.fetchone()
            if not row:
                return False, f"Memory with hash {content_hash} not found"
            
            content, current_tags, current_type, current_metadata_str, created_at, created_at_iso = row
            
            # Parse current metadata
            current_metadata = self._safe_json_loads(current_metadata_str, "update_memory_metadata")
            
            # Apply updates
            new_tags = current_tags
            new_type = current_type
            new_metadata = current_metadata.copy()
            
            # Handle tag updates
            if "tags" in updates:
                if isinstance(updates["tags"], list):
                    new_tags = ",".join(updates["tags"])
                else:
                    return False, "Tags must be provided as a list of strings"
            
            # Handle memory type updates
            if "memory_type" in updates:
                new_type = updates["memory_type"]
            
            # Handle metadata updates
            if "metadata" in updates:
                if isinstance(updates["metadata"], dict):
                    new_metadata.update(updates["metadata"])
                else:
                    return False, "Metadata must be provided as a dictionary"
            
            # Handle other custom fields
            protected_fields = {
                "content", "content_hash", "tags", "memory_type", "metadata",
                "embedding", "created_at", "created_at_iso", "updated_at", "updated_at_iso"
            }
            
            for key, value in updates.items():
                if key not in protected_fields:
                    new_metadata[key] = value
            
            # Update timestamps
            now = time.time()
            now_iso = datetime.utcfromtimestamp(now).isoformat() + "Z"
            
            if not preserve_timestamps:
                created_at = now
                created_at_iso = now_iso
            
            # Update the memory
            self.conn.execute('''
                UPDATE memories SET
                    tags = ?, memory_type = ?, metadata = ?,
                    updated_at = ?, updated_at_iso = ?,
                    created_at = ?, created_at_iso = ?
                WHERE content_hash = ?
            ''', (
                new_tags, new_type, json.dumps(new_metadata),
                now, now_iso, created_at, created_at_iso, content_hash
            ))
            
            self.conn.commit()
            
            # Create summary of updated fields
            updated_fields = []
            if "tags" in updates:
                updated_fields.append("tags")
            if "memory_type" in updates:
                updated_fields.append("memory_type")
            if "metadata" in updates:
                updated_fields.append("custom_metadata")
            
            for key in updates.keys():
                if key not in protected_fields and key not in ["tags", "memory_type", "metadata"]:
                    updated_fields.append(key)
            
            updated_fields.append("updated_at")
            
            summary = f"Updated fields: {', '.join(updated_fields)}"
            logger.info(f"Successfully updated metadata for memory {content_hash}")
            return True, summary
            
        except Exception as e:
            error_msg = f"Error updating memory metadata: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, error_msg
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            if not self.conn:
                return {"error": "Database not initialized"}
            
            cursor = self.conn.execute('SELECT COUNT(*) FROM memories')
            total_memories = cursor.fetchone()[0]
            
            # Count unique individual tags (not tag sets)
            cursor = self.conn.execute('SELECT tags FROM memories WHERE tags IS NOT NULL AND tags != ""')
            unique_tags = len(set(
                tag.strip()
                for (tag_string,) in cursor
                if tag_string
                for tag in tag_string.split(",")
                if tag.strip()
            ))
            
            # Count memories from this week (last 7 days)
            import time
            week_ago = time.time() - (7 * 24 * 60 * 60)
            cursor = self.conn.execute('SELECT COUNT(*) FROM memories WHERE created_at >= ?', (week_ago,))
            memories_this_week = cursor.fetchone()[0]

            # Get database file size
            file_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

            return {
                "backend": "sqlite-vec",
                "total_memories": total_memories,
                "unique_tags": unique_tags,
                "memories_this_week": memories_this_week,
                "database_size_bytes": file_size,
                "database_size_mb": round(file_size / (1024 * 1024), 2),
                "embedding_model": self.embedding_model_name,
                "embedding_dimension": self.embedding_dimension
            }
            
        except sqlite3.Error as e:
            logger.error(f"Database error getting stats: {str(e)}")
            return {"error": f"Database error: {str(e)}"}
        except OSError as e:
            logger.error(f"File system error getting stats: {str(e)}")
            return {"error": f"File system error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error getting stats: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}
    
    def sanitized(self, tags):
        """Sanitize and normalize tags to a JSON string.
        
        This method provides compatibility with the ChromaMemoryStorage interface.
        """
        if tags is None:
            return json.dumps([])
        
        # If we get a string, split it into an array
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        # If we get an array, use it directly
        elif isinstance(tags, list):
            tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        else:
            return json.dumps([])
                
        # Return JSON string representation of the array
        return json.dumps(tags)
    
    async def recall(self, query: Optional[str] = None, n_results: int = 5, start_timestamp: Optional[float] = None, end_timestamp: Optional[float] = None) -> List[MemoryQueryResult]:
        """
        Retrieve memories with combined time filtering and optional semantic search.
        
        Args:
            query: Optional semantic search query. If None, only time filtering is applied.
            n_results: Maximum number of results to return.
            start_timestamp: Optional start time for filtering.
            end_timestamp: Optional end time for filtering.
            
        Returns:
            List of MemoryQueryResult objects.
        """
        try:
            if not self.conn:
                logger.error("Database not initialized, cannot retrieve memories")
                return []
            
            # Build time filtering WHERE clause
            time_conditions = []
            params = []
            
            if start_timestamp is not None:
                time_conditions.append("created_at >= ?")
                params.append(float(start_timestamp))
            
            if end_timestamp is not None:
                time_conditions.append("created_at <= ?")
                params.append(float(end_timestamp))
            
            time_where = " AND ".join(time_conditions) if time_conditions else ""
            
            logger.info(f"Time filtering conditions: {time_where}, params: {params}")
            
            # Determine whether to use semantic search or just time-based filtering
            if query and self.embedding_model:
                # Combined semantic search with time filtering
                try:
                    # Generate query embedding
                    query_embedding = self._generate_embedding(query)
                    
                    # Build SQL query with time filtering
                    base_query = '''
                        SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                               m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso, 
                               e.distance
                        FROM memories m
                        JOIN (
                            SELECT rowid, distance 
                            FROM memory_embeddings 
                            WHERE content_embedding MATCH ?
                            ORDER BY distance
                            LIMIT ?
                        ) e ON m.id = e.rowid
                    '''
                    
                    if time_where:
                        base_query += f" WHERE {time_where}"
                    
                    base_query += " ORDER BY e.distance"
                    
                    # Prepare parameters: embedding, limit, then time filter params
                    query_params = [serialize_float32(query_embedding), n_results] + params
                    
                    cursor = self.conn.execute(base_query, query_params)
                    
                    results = []
                    for row in cursor.fetchall():
                        try:
                            # Parse row data
                            content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                            created_at, updated_at, created_at_iso, updated_at_iso, distance = row[5:]
                            
                            # Parse tags and metadata
                            tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                            metadata = self._safe_json_loads(metadata_str, "memory_metadata")
                            
                            # Create Memory object
                            memory = Memory(
                                content=content,
                                content_hash=content_hash,
                                tags=tags,
                                memory_type=memory_type,
                                metadata=metadata,
                                created_at=created_at,
                                updated_at=updated_at,
                                created_at_iso=created_at_iso,
                                updated_at_iso=updated_at_iso
                            )
                            
                            # Calculate relevance score (lower distance = higher relevance)
                            relevance_score = max(0.0, 1.0 - distance)
                            
                            results.append(MemoryQueryResult(
                                memory=memory,
                                relevance_score=relevance_score,
                                debug_info={"distance": distance, "backend": "sqlite-vec", "time_filtered": bool(time_where)}
                            ))
                            
                        except Exception as parse_error:
                            logger.warning(f"Failed to parse memory result: {parse_error}")
                            continue
                    
                    logger.info(f"Retrieved {len(results)} memories for semantic query with time filter")
                    return results
                    
                except Exception as query_error:
                    logger.error(f"Error in semantic search with time filter: {str(query_error)}")
                    # Fall back to time-based retrieval on error
                    logger.info("Falling back to time-based retrieval")
            
            # Time-based filtering only (or fallback from failed semantic search)
            base_query = '''
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
            '''
            
            if time_where:
                base_query += f" WHERE {time_where}"
            
            base_query += " ORDER BY created_at DESC LIMIT ?"
            
            # Add limit parameter
            params.append(n_results)
            
            cursor = self.conn.execute(base_query, params)
            
            results = []
            for row in cursor.fetchall():
                try:
                    content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                    created_at, updated_at, created_at_iso, updated_at_iso = row[5:]
                    
                    # Parse tags and metadata
                    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")
                    
                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )
                    
                    # For time-based retrieval, we don't have a relevance score
                    results.append(MemoryQueryResult(
                        memory=memory,
                        relevance_score=None,
                        debug_info={"backend": "sqlite-vec", "time_filtered": bool(time_where), "query_type": "time_based"}
                    ))
                    
                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue
            
            logger.info(f"Retrieved {len(results)} memories for time-based query")
            return results
            
        except Exception as e:
            logger.error(f"Error in recall: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    async def get_all_memories(self) -> List[Memory]:
        """
        Get all memories from the database.
        
        Returns:
            List of all Memory objects in the database.
        """
        try:
            if not self.conn:
                logger.error("Database not initialized, cannot retrieve memories")
                return []
            
            cursor = self.conn.execute('''
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
                ORDER BY created_at DESC
            ''')
            
            results = []
            for row in cursor.fetchall():
                try:
                    content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                    created_at, updated_at, created_at_iso, updated_at_iso = row[5:]
                    
                    # Parse tags and metadata
                    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")
                    
                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )
                    
                    results.append(memory)
                    
                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue
            
            logger.info(f"Retrieved {len(results)} total memories")
            return results
            
        except Exception as e:
            logger.error(f"Error getting all memories: {str(e)}")
            return []

    async def get_memories_by_time_range(self, start_time: float, end_time: float) -> List[Memory]:
        """Get memories within a specific time range."""
        try:
            await self.initialize()
            cursor = self.conn.execute('''
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
                WHERE created_at BETWEEN ? AND ?
                ORDER BY created_at DESC
            ''', (start_time, end_time))
            
            results = []
            for row in cursor.fetchall():
                try:
                    content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                    created_at, updated_at, created_at_iso, updated_at_iso = row[5:]
                    
                    # Parse tags and metadata
                    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")
                    
                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )
                    
                    results.append(memory)
                    
                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue
            
            logger.info(f"Retrieved {len(results)} memories in time range {start_time}-{end_time}")
            return results
            
        except Exception as e:
            logger.error(f"Error getting memories by time range: {str(e)}")
            return []

    async def get_memory_connections(self) -> Dict[str, int]:
        """Get memory connection statistics."""
        try:
            await self.initialize()
            # For now, return basic statistics based on tags and content similarity
            cursor = self.conn.execute('''
                SELECT tags, COUNT(*) as count
                FROM memories
                WHERE tags IS NOT NULL AND tags != ''
                GROUP BY tags
            ''')
            
            connections = {}
            for row in cursor.fetchall():
                tags_str, count = row
                if tags_str:
                    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
                    for tag in tags:
                        connections[f"tag:{tag}"] = connections.get(f"tag:{tag}", 0) + count
            
            return connections
            
        except Exception as e:
            logger.error(f"Error getting memory connections: {str(e)}")
            return {}

    async def get_access_patterns(self) -> Dict[str, datetime]:
        """Get memory access pattern statistics."""
        try:
            await self.initialize()
            # Return recent access patterns based on updated_at timestamps
            cursor = self.conn.execute('''
                SELECT content_hash, updated_at_iso
                FROM memories
                WHERE updated_at_iso IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT 100
            ''')
            
            patterns = {}
            for row in cursor.fetchall():
                content_hash, updated_at_iso = row
                try:
                    patterns[content_hash] = datetime.fromisoformat(updated_at_iso.replace('Z', '+00:00'))
                except Exception:
                    # Fallback for timestamp parsing issues
                    patterns[content_hash] = datetime.now()
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting access patterns: {str(e)}")
            return {}

    def _row_to_memory(self, row) -> Optional[Memory]:
        """Convert database row to Memory object."""
        try:
            content_hash, content, tags_str, memory_type, metadata_str, created_at, updated_at, created_at_iso, updated_at_iso = row

            # Parse tags (comma-separated format)
            tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
            
            # Parse metadata
            metadata = self._safe_json_loads(metadata_str, "get_by_hash")
            
            return Memory(
                content=content,
                content_hash=content_hash,
                tags=tags,
                memory_type=memory_type,
                metadata=metadata,
                created_at=created_at,
                updated_at=updated_at,
                created_at_iso=created_at_iso,
                updated_at_iso=updated_at_iso
            )
            
        except Exception as e:
            logger.error(f"Error converting row to memory: {str(e)}")
            return None

    async def get_all_memories(self, limit: int = None, offset: int = 0, memory_type: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Memory]:
        """
        Get all memories in storage ordered by creation time (newest first).

        Args:
            limit: Maximum number of memories to return (None for all)
            offset: Number of memories to skip (for pagination)
            memory_type: Optional filter by memory type
            tags: Optional filter by tags (matches ANY of the provided tags)

        Returns:
            List of Memory objects ordered by created_at DESC, optionally filtered by type and tags
        """
        try:
            await self.initialize()

            # Build query with optional memory_type and tags filters
            query = '''
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
            '''

            params = []
            where_conditions = []

            # Add memory_type filter if specified
            if memory_type is not None:
                where_conditions.append('memory_type = ?')
                params.append(memory_type)

            # Add tags filter if specified (using database-level filtering like search_by_tag_chronological)
            if tags and len(tags) > 0:
                tag_conditions = " OR ".join(["tags LIKE ?" for _ in tags])
                where_conditions.append(f"({tag_conditions})")
                params.extend([f"%{tag}%" for tag in tags])

            # Apply WHERE clause if we have any conditions
            if where_conditions:
                query += ' WHERE ' + ' AND '.join(where_conditions)

            query += ' ORDER BY created_at DESC'

            if limit is not None:
                query += ' LIMIT ?'
                params.append(limit)

            if offset > 0:
                query += ' OFFSET ?'
                params.append(offset)
            
            cursor = self.conn.execute(query, params)
            memories = []
            
            for row in cursor.fetchall():
                memory = self._row_to_memory(row)
                if memory:
                    memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Error getting all memories: {str(e)}")
            return []

    async def get_recent_memories(self, n: int = 10) -> List[Memory]:
        """
        Get n most recent memories.
        
        Args:
            n: Number of recent memories to return
            
        Returns:
            List of the n most recent Memory objects
        """
        return await self.get_all_memories(limit=n, offset=0)

    async def count_all_memories(self, memory_type: Optional[str] = None) -> int:
        """
        Get total count of memories in storage.

        Args:
            memory_type: Optional filter by memory type

        Returns:
            Total number of memories, optionally filtered by type
        """
        try:
            await self.initialize()

            if memory_type is not None:
                cursor = self.conn.execute('SELECT COUNT(*) FROM memories WHERE memory_type = ?', (memory_type,))
            else:
                cursor = self.conn.execute('SELECT COUNT(*) FROM memories')

            result = cursor.fetchone()
            return result[0] if result else 0

        except Exception as e:
            logger.error(f"Error counting memories: {str(e)}")
            return 0

    async def get_all_tags_with_counts(self) -> List[Dict[str, Any]]:
        """
        Get all tags with their usage counts.

        Returns:
            List of dictionaries with 'tag' and 'count' keys, sorted by count descending
        """
        try:
            await self.initialize()

            # Get all tags from the database
            cursor = self.conn.execute('''
                SELECT tags
                FROM memories
                WHERE tags IS NOT NULL AND tags != ''
            ''')

            # Use Counter with generator expression for memory efficiency
            tag_counter = Counter(
                tag.strip()
                for (tag_string,) in cursor
                if tag_string
                for tag in tag_string.split(",")
                if tag.strip()
            )

            # Return as list of dicts sorted by count descending
            return [{"tag": tag, "count": count} for tag, count in tag_counter.most_common()]

        except sqlite3.Error as e:
            logger.error(f"Database error getting tags with counts: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting tags with counts: {str(e)}")
            raise

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("SQLite-vec storage connection closed")