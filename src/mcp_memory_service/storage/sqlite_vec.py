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
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime
import asyncio

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
    
    async def initialize(self):
        """Initialize the SQLite database with vec0 extension."""
        try:
            if not SQLITE_VEC_AVAILABLE:
                raise ImportError("sqlite-vec is not available. Install with: pip install sqlite-vec")
            
            # Connect to database
            self.conn = sqlite3.connect(self.db_path)
            self.conn.enable_load_extension(True)
            
            # Load sqlite-vec extension
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            
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
            
            # Create virtual table for vector embeddings
            self.conn.execute(f'''
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
                    content_embedding FLOAT[{self.embedding_dimension}]
                )
            ''')
            
            # Create indexes for better performance
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)')
            
            # Initialize embedding model
            await self._initialize_embedding_model()
            
            logger.info("SQLite-vec storage initialized successfully")
            
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
                logger.warning("Sentence transformers not available. Embeddings will not work.")
                return
            
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
            logger.warning("No embedding model available")
            return [0.0] * self.embedding_dimension
        
        try:
            # Check cache first
            if self.enable_cache:
                cache_key = hash(text)
                if cache_key in _EMBEDDING_CACHE:
                    return _EMBEDDING_CACHE[cache_key]
            
            # Generate embedding
            embedding = self.embedding_model.encode([text], convert_to_numpy=True)[0]
            embedding_list = embedding.tolist()
            
            # Cache the result
            if self.enable_cache:
                _EMBEDDING_CACHE[cache_key] = embedding_list
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return [0.0] * self.embedding_dimension
    
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
            
            # Generate embedding
            embedding = self._generate_embedding(memory.content)
            
            # Prepare metadata
            tags_str = ",".join(memory.tags) if memory.tags else ""
            metadata_str = json.dumps(memory.metadata) if memory.metadata else "{}"
            
            # Ensure timestamps are set
            if not memory.created_at:
                memory.touch()
            
            # Insert into memories table (metadata)
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
            
            # Get the rowid to use as reference for the embedding
            memory_rowid = cursor.lastrowid
            
            # Insert into embeddings table
            self.conn.execute('''
                INSERT INTO memory_embeddings (rowid, content_embedding)
                VALUES (?, ?)
            ''', (
                memory_rowid,
                serialize_float32(embedding)
            ))
            
            self.conn.commit()
            
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
            query_embedding = self._generate_embedding(query)
            
            # Perform vector similarity search using JOIN
            cursor = self.conn.execute('''
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
                ORDER BY e.distance
            ''', (serialize_float32(query_embedding), n_results))
            
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
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("SQLite-vec storage connection closed")