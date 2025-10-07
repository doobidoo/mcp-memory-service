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

"""Utilities for database validation and health checks."""
from typing import Dict, Any, Tuple
import logging
import os
import json
from datetime import datetime
import importlib

logger = logging.getLogger(__name__)

async def validate_database(storage) -> Tuple[bool, str]:
    """Validate database health and configuration."""
    try:
        # Check if storage is properly initialized
        if storage is None:
            return False, "Storage is not initialized"
        
        # Special case for direct access without checking for attribute 'collection'
        # This fixes compatibility issues with SQLite-vec and other storage backends
        storage_type = storage.__class__.__name__
        
        # First, use the 'is_initialized' method if available (preferred)
        if hasattr(storage, 'is_initialized') and callable(storage.is_initialized):
            try:
                init_status = storage.is_initialized()
                if not init_status:
                    # Get detailed status for debugging
                    if hasattr(storage, 'get_initialization_status') and callable(storage.get_initialization_status):
                        status = storage.get_initialization_status()
                        return False, f"Storage not fully initialized: {status}"
                    else:
                        return False, "Storage initialization incomplete"
            except Exception as init_error:
                logger.warning(f"Error checking initialization status: {init_error}")
                # Continue with alternative checks
        
        # SQLite-vec backend validation
        if storage_type == "SqliteVecMemoryStorage":
            if not hasattr(storage, 'conn') or storage.conn is None:
                return False, "SQLite database connection is not initialized"
            
            # Check for database health
            try:
                # Make sure the tables exist
                try:
                    cursor = storage.conn.execute('SELECT name FROM sqlite_master WHERE type="table" AND name="memories"')
                    if not cursor.fetchone():
                        return False, "SQLite database is missing required tables"
                except Exception as table_error:
                    return False, f"Failed to check for tables: {str(table_error)}"
                
                # Try a simple query to verify database connection
                cursor = storage.conn.execute('SELECT COUNT(*) FROM memories')
                memory_count = cursor.fetchone()[0]
                logger.info(f"SQLite-vec database contains {memory_count} memories")
                
                # Test if embedding generation works (if model is available)
                if hasattr(storage, 'embedding_model') and storage.embedding_model:
                    test_text = "Database validation test"
                    embedding = storage._generate_embedding(test_text)
                    if not embedding or len(embedding) != storage.embedding_dimension:
                        logger.warning("Embedding generation may not be working properly")
                else:
                    logger.warning("No embedding model available, some functionality may be limited")
                
                return True, "SQLite-vec database validation successful"
                
            except Exception as e:
                return False, f"SQLite database access error: {str(e)}"

        # Cloudflare storage validation
        elif storage_type == "CloudflareStorage":
            try:
                # Check if storage is properly initialized
                if not hasattr(storage, 'client') or storage.client is None:
                    return False, "Cloudflare storage client is not initialized"

                # Check basic connectivity by getting stats
                stats = await storage.get_stats()
                memory_count = stats.get("total_memories", 0)
                logger.info(f"Cloudflare storage contains {memory_count} memories")

                # Test embedding generation if available
                test_text = "Database validation test"
                try:
                    embedding = await storage._generate_embedding(test_text)
                    if not embedding or not isinstance(embedding, list):
                        logger.warning("Embedding generation may not be working properly")
                except Exception as embed_error:
                    logger.warning(f"Embedding test failed: {str(embed_error)}")

                return True, "Cloudflare storage validation successful"

            except Exception as e:
                return False, f"Cloudflare storage access error: {str(e)}"

        else:
            return False, f"Unknown storage type: {storage_type}"
            
    except Exception as e:
        logger.error(f"Database validation failed: {str(e)}")
        return False, f"Database validation failed: {str(e)}"

async def get_database_stats(storage) -> Dict[str, Any]:
    """Get detailed database statistics with proper error handling."""
    try:
        # Check if storage is properly initialized
        if storage is None:
            return {
                "status": "error",
                "error": "Storage is not initialized"
            }
        
        # Determine storage type
        storage_type = storage.__class__.__name__
        
        # SQLite-vec backend stats
        if storage_type == "SqliteVecMemoryStorage":
            # Use the storage's own stats method if available
            if hasattr(storage, 'get_stats') and callable(storage.get_stats):
                try:
                    stats = storage.get_stats()
                    stats["status"] = "healthy"
                    return stats
                except Exception as stats_error:
                    logger.warning(f"Error calling get_stats method: {stats_error}")
                    # Fall back to our implementation
            
            # Otherwise, gather basic stats
            if not hasattr(storage, 'conn') or storage.conn is None:
                return {
                    "status": "error",
                    "error": "SQLite database connection is not initialized"
                }
            
            try:
                # Check if tables exist
                cursor = storage.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # Count memories if the table exists
                memory_count = 0
                if 'memories' in tables:
                    cursor = storage.conn.execute('SELECT COUNT(*) FROM memories')
                    memory_count = cursor.fetchone()[0]
                
                # Get unique tags if the table exists
                unique_tags = 0
                if 'memories' in tables:
                    cursor = storage.conn.execute('SELECT COUNT(DISTINCT tags) FROM memories WHERE tags != ""')
                    unique_tags = cursor.fetchone()[0]
                
                # Get database file size
                db_path = storage.db_path if hasattr(storage, 'db_path') else "unknown"
                file_size = os.path.getsize(db_path) if isinstance(db_path, str) and os.path.exists(db_path) else 0
                
                # Get embedding model info
                embedding_model = "unknown"
                embedding_dimension = 0
                
                if hasattr(storage, 'embedding_model_name'):
                    embedding_model = storage.embedding_model_name
                
                if hasattr(storage, 'embedding_dimension'):
                    embedding_dimension = storage.embedding_dimension
                
                # Gather tables information
                tables_info = {}
                for table in tables:
                    try:
                        cursor = storage.conn.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        tables_info[table] = {"count": count}
                    except Exception:
                        tables_info[table] = {"count": "unknown"}
                
                return {
                    "backend": "sqlite-vec",
                    "status": "healthy",
                    "total_memories": memory_count,
                    "unique_tags": unique_tags,
                    "database_size_bytes": file_size,
                    "database_size_mb": round(file_size / (1024 * 1024), 2) if file_size > 0 else 0,
                    "embedding_model": embedding_model,
                    "embedding_dimension": embedding_dimension,
                    "tables": tables,
                    "tables_info": tables_info
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Error getting SQLite-vec stats: {str(e)}"
                }
        
        # Cloudflare storage stats
        elif storage_type == "CloudflareStorage":
            try:
                # Get storage stats from the Cloudflare storage implementation
                storage_stats = await storage.get_stats()

                # Add cloudflare-specific info
                cloudflare_info = {
                    "vectorize_index": storage.vectorize_index,
                    "d1_database_id": storage.d1_database_id,
                    "r2_bucket": storage.r2_bucket,
                    "embedding_model": storage.embedding_model,
                    "large_content_threshold": storage.large_content_threshold
                }

                return {
                    **storage_stats,
                    "cloudflare": cloudflare_info,
                    "backend": "cloudflare",
                    "status": "healthy"
                }

            except Exception as stats_error:
                return {
                    "status": "error",
                    "error": f"Error getting Cloudflare stats: {str(stats_error)}",
                    "backend": "cloudflare"
                }

        else:
            return {
                "status": "error",
                "error": f"Unknown storage type: {storage_type}"
            }
            
    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

async def repair_database(storage) -> Tuple[bool, str]:
    """Attempt to repair database issues."""
    try:
        # Determine storage type
        storage_type = storage.__class__.__name__
        
        # SQLite-vec backend repair
        if storage_type == "SqliteVecMemoryStorage":
            # For SQLite, we'll try to check and recreate the tables if needed
            if not hasattr(storage, 'conn') or storage.conn is None:
                # Try to reconnect
                try:
                    storage.conn = storage.conn or __import__('sqlite3').connect(storage.db_path)
                    
                    # Try to reload the extension
                    if importlib.util.find_spec('sqlite_vec'):
                        import sqlite_vec
                        storage.conn.enable_load_extension(True)
                        sqlite_vec.load(storage.conn)
                        storage.conn.enable_load_extension(False)
                    
                    # Recreate tables if needed
                    storage.conn.execute('''
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
                    embedding_dimension = getattr(storage, 'embedding_dimension', 384)
                    storage.conn.execute(f'''
                        CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
                            content_embedding FLOAT[{embedding_dimension}]
                        )
                    ''')
                    
                    # Create indexes for better performance
                    storage.conn.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash)')
                    storage.conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)')
                    storage.conn.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)')
                    
                    storage.conn.commit()
                    return True, "SQLite-vec database repaired"
                    
                except Exception as e:
                    return False, f"SQLite-vec repair failed: {str(e)}"
        
        # Cloudflare storage repair
        elif storage_type == "CloudflareStorage":
            # For Cloudflare storage, we can't repair infrastructure (Vectorize, D1, R2)
            # but we can validate the connection and re-initialize if needed
            try:
                # Validate current state
                is_valid, message = await validate_database(storage)
                if is_valid:
                    return True, "Cloudflare storage is already healthy"

                # Try to re-initialize the storage connection
                await storage.initialize()

                # Validate repair
                is_valid, message = await validate_database(storage)
                if is_valid:
                    return True, "Cloudflare storage connection successfully repaired"
                else:
                    return False, f"Cloudflare storage repair failed: {message}"

            except Exception as repair_error:
                return False, f"Cloudflare storage repair failed: {str(repair_error)}"

        else:
            return False, f"Unknown storage type: {storage_type}, cannot repair"
                
    except Exception as e:
        logger.error(f"Error repairing database: {str(e)}")
        return False, f"Error repairing database: {str(e)}"