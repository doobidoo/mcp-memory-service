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
from typing import List, Dict, Any, Tuple, Optional, Set, Callable
from datetime import datetime, timezone
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
    
    async def initialize(self):
        """Initialize the SQLite database with vec0 extension."""
        try:
            if not SQLITE_VEC_AVAILABLE:
                raise ImportError("sqlite-vec is not available. Install with: pip install sqlite-vec")
            
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers is not available. Install with: pip install sentence-transformers torch")
            
            # Connect to database
            self.conn = sqlite3.connect(self.db_path)
            self.conn.enable_load_extension(True)
            
            # Load sqlite-vec extension
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            
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
    
    async def _initialize_embedding_model(self):
        """Initialize the sentence transformer model for embeddings."""
        global _MODEL_CACHE
        
        try:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Sentence transformers not available. Install with: pip install sentence-transformers torch")
            
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
            
            # Load model with optimal settings
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
            
            # Update embedding dimension based on actual model
            test_embedding = self.embedding_model.encode(["test"], convert_to_numpy=True)
            self.embedding_dimension = test_embedding.shape[1]
            
            # Cache the model
            _MODEL_CACHE[cache_key] = self.embedding_model
            
            logger.info(f"Embedding model loaded successfully. Dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue without embeddings - some operations may still work
    
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
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    
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
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    
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
            current_metadata = json.loads(current_metadata_str) if current_metadata_str else {}
            
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
            now_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            
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
            
            cursor = self.conn.execute('SELECT COUNT(DISTINCT tags) FROM memories WHERE tags != ""')
            unique_tags = cursor.fetchone()[0]
            
            # Get database file size
            file_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                "backend": "sqlite-vec",
                "total_memories": total_memories,
                "unique_tags": unique_tags,
                "database_size_bytes": file_size,
                "database_size_mb": round(file_size / (1024 * 1024), 2),
                "embedding_model": self.embedding_model_name,
                "embedding_dimension": self.embedding_dimension
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {"error": str(e)}
    
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
                            metadata = json.loads(metadata_str) if metadata_str else {}
                            
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
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    
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
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("SQLite-vec storage connection closed")