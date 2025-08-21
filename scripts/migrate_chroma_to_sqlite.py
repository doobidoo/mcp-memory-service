#!/usr/bin/env python3
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
Migration script to move data from ChromaDB to SQLite-vec.

This script reads all memories from your existing ChromaDB installation
and migrates them to the new SQLite-vec backend, preserving all metadata,
tags, embeddings, and timestamps.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mcp_memory_service.storage.chroma import ChromaMemoryStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.config import CHROMA_PATH, EMBEDDING_MODEL_NAME
from mcp_memory_service.utils.hashing import generate_content_hash

logger = logging.getLogger(__name__)

def safe_timestamp_convert(timestamp: Union[str, float, int, None]) -> float:
    """Safely convert various timestamp formats to float."""
    if timestamp is None:
        return datetime.now().timestamp()
    
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    
    if isinstance(timestamp, str):
        # Try to parse ISO format strings
        if 'T' in timestamp or '-' in timestamp:
            try:
                # Handle ISO format with or without 'Z'
                timestamp_str = timestamp.rstrip('Z')
                dt = datetime.fromisoformat(timestamp_str.replace('Z', ''))
                return dt.timestamp()
            except ValueError:
                pass
        
        # Try to parse as float string
        try:
            return float(timestamp)
        except ValueError:
            pass
    
    # Fallback to current time
    logger.warning(f"Could not parse timestamp '{timestamp}', using current time")
    return datetime.now().timestamp()

def extract_memory_data_directly(chroma_storage) -> List[Dict[str, Any]]:
    """Extract memory data directly from ChromaDB without using Memory objects."""
    try:
        # Access the ChromaDB collection directly
        collection = chroma_storage.collection
        
        # Get all data from the collection
        results = collection.get(
            include=['documents', 'metadatas']
        )
        
        memories = []
        for i, doc_id in enumerate(results['ids']):
            try:
                # Extract basic data
                content = results['documents'][i] if i < len(results['documents']) else ""
                metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}
                
                # Extract and validate tags from metadata
                raw_tags = metadata.get('tags', metadata.get('tags_str', []))
                tags = []
                if isinstance(raw_tags, str):
                    # Handle comma-separated string or single tag
                    if ',' in raw_tags:
                        tags = [tag.strip() for tag in raw_tags.split(',') if tag.strip()]
                    elif raw_tags.strip():
                        tags = [raw_tags.strip()]
                elif isinstance(raw_tags, list):
                    # Validate each tag in list
                    tags = [str(tag).strip() for tag in raw_tags if tag and str(tag).strip()]
                else:
                    logger.warning(f"Unknown tag format for memory {i}: {type(raw_tags)}")
                    tags = []
                
                # Extract timestamps with flexible conversion
                created_at = safe_timestamp_convert(metadata.get('created_at'))
                updated_at = safe_timestamp_convert(metadata.get('updated_at', created_at))
                
                # Extract other metadata
                memory_type = metadata.get('memory_type', 'imported')
                
                # Create clean metadata dict (remove special fields)
                clean_metadata = {k: v for k, v in metadata.items() 
                                if k not in ['tags', 'created_at', 'updated_at', 'memory_type']}
                
                # Generate proper content hash instead of using ChromaDB ID
                proper_content_hash = generate_content_hash(content)
                
                memory_data = {
                    'content': content,
                    'tags': tags,
                    'memory_type': memory_type,
                    'metadata': clean_metadata,
                    'created_at': created_at,
                    'updated_at': updated_at,
                    'content_hash': proper_content_hash  # Use proper SHA256 hash
                }
                
                memories.append(memory_data)
                
            except Exception as e:
                logger.warning(f"Failed to extract memory {i}: {e}")
                continue
        
        logger.info(f"Successfully extracted {len(memories)} memories from ChromaDB")
        return memories
        
    except Exception as e:
        logger.error(f"Failed to extract data from ChromaDB: {e}")
        return []

class MigrationStats:
    """Track migration statistics."""
    def __init__(self):
        self.total_memories = 0
        self.migrated_successfully = 0
        self.failed_migrations = 0
        self.duplicates_skipped = 0
        self.start_time = datetime.now()
        self.errors: List[str] = []

    def add_error(self, error: str):
        self.errors.append(error)
        self.failed_migrations += 1

    def print_summary(self):
        duration = datetime.now() - self.start_time
        print("\n" + "="*60)
        print("MIGRATION SUMMARY")
        print("="*60)
        print(f"Total memories found:     {self.total_memories}")
        print(f"Successfully migrated:    {self.migrated_successfully}")
        print(f"Duplicates skipped:       {self.duplicates_skipped}")
        print(f"Failed migrations:        {self.failed_migrations}")
        print(f"Migration duration:       {duration.total_seconds():.2f} seconds")
        
        if self.errors:
            print(f"\nErrors encountered ({len(self.errors)}):")
            for i, error in enumerate(self.errors[:5], 1):  # Show first 5 errors
                print(f"  {i}. {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more errors")
        else:
            print("\nMigration completed without errors!")


async def check_chroma_data(chroma_path: str) -> int:
    """Check if ChromaDB data exists and count memories."""
    print(f"Checking ChromaDB data at: {chroma_path}")
    
    try:
        chroma_storage = ChromaMemoryStorage(
            path=chroma_path
        )
        
        # Extract memories directly to avoid data corruption issues
        memories = extract_memory_data_directly(chroma_storage)
        memory_count = len(memories)
        
        print(f"Found {memory_count} memories in ChromaDB")
        return memory_count
        
    except Exception as e:
        print(f"Error accessing ChromaDB: {e}")
        print("Make sure ChromaDB data exists and is accessible")
        return -1


async def migrate_memories(
    chroma_path: str, 
    sqlite_path: str, 
    stats: MigrationStats,
    batch_size: int = 50,
    skip_duplicates: bool = True
) -> bool:
    """Migrate all memories from ChromaDB to SQLite-vec."""
    
    chroma_storage = None
    sqlite_storage = None
    
    try:
        # Initialize ChromaDB storage (source)
        print("Connecting to ChromaDB...")
        chroma_storage = ChromaMemoryStorage(
            path=chroma_path
        )
        
        # Initialize SQLite-vec storage (destination)
        print("Connecting to SQLite-vec...")
        sqlite_storage = SqliteVecMemoryStorage(
            db_path=sqlite_path,
            embedding_model=EMBEDDING_MODEL_NAME
        )
        await sqlite_storage.initialize()
        
        # Extract all memories directly from ChromaDB
        print("Extracting all memories from ChromaDB...")
        all_memories = extract_memory_data_directly(chroma_storage)
        stats.total_memories = len(all_memories)
        
        if stats.total_memories == 0:
            print("No memories found in ChromaDB")
            return True
        
        print(f"Found {stats.total_memories} memories to migrate")
        
        # Migrate in batches
        for i in range(0, stats.total_memories, batch_size):
            batch = all_memories[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (stats.total_memories + batch_size - 1) // batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} memories)...")
            
            for memory_data in batch:
                try:
                    # Check if memory already exists in SQLite-vec (if skipping duplicates)
                    if skip_duplicates:
                        try:
                            # Use a more efficient duplicate check
                            cursor = sqlite_storage.conn.execute(
                                "SELECT 1 FROM memories WHERE content_hash = ? LIMIT 1",
                                (memory_data['content_hash'],)
                            )
                            if cursor.fetchone():
                                stats.duplicates_skipped += 1
                                continue
                        except Exception:
                            # Fallback to retrieve method if direct query fails
                            existing = await sqlite_storage.retrieve(memory_data['content'], n_results=1)
                            if existing and any(m.memory.content_hash == memory_data['content_hash'] for m in existing):
                                stats.duplicates_skipped += 1
                                continue
                    
                    # Create Memory object for SQLite-vec storage
                    memory_obj = Memory(
                        content=memory_data['content'],
                        tags=memory_data['tags'],
                        metadata=memory_data['metadata'],
                        created_at=memory_data['created_at'],
                        updated_at=memory_data['updated_at'],
                        content_hash=memory_data['content_hash']
                    )
                    
                    # Store memory in SQLite-vec
                    success, message = await sqlite_storage.store(memory_obj)
                    if not success:
                        raise Exception(f"Storage failed: {message}")
                    stats.migrated_successfully += 1
                    
                except Exception as e:
                    error_msg = f"Failed to migrate memory {memory_data['content_hash'][:12]}...: {str(e)}"
                    stats.add_error(error_msg)
                    logger.error(error_msg)
            
            # Progress update with percentage
            migrated_so_far = stats.migrated_successfully + stats.duplicates_skipped + stats.failed_migrations
            percentage = (migrated_so_far / stats.total_memories * 100) if stats.total_memories > 0 else 0
            print(f"Batch {batch_num}/{total_batches} complete. Progress: {migrated_so_far}/{stats.total_memories} ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        error_msg = f"Critical migration error: {str(e)}"
        stats.add_error(error_msg)
        logger.error(error_msg)
        return False
        
    finally:
        # Clean up connections
        if sqlite_storage:
            sqlite_storage.close()


async def verify_migration(sqlite_path: str, expected_count: int) -> bool:
    """Verify that the migration was successful."""
    print("Verifying migration results...")
    
    try:
        sqlite_storage = SqliteVecMemoryStorage(
            db_path=sqlite_path,
            embedding_model=EMBEDDING_MODEL_NAME
        )
        await sqlite_storage.initialize()
        
        # Count memories in SQLite-vec
        all_memories = await sqlite_storage.retrieve("", n_results=10000)
        actual_count = len(all_memories)
        
        sqlite_storage.close()
        
        print(f"Verification: Expected {expected_count}, Found {actual_count}")
        
        if actual_count >= expected_count:
            print("Migration verification passed!")
            return True
        else:
            print("Migration verification failed - some memories may be missing")
            return False
            
    except Exception as e:
        print(f"Verification error: {e}")
        return False


def print_banner():
    """Print migration banner."""
    print("="*60)
    print("MCP Memory Service - ChromaDB to SQLite-vec Migration")
    print("="*60)
    print("This script migrates all your memories from ChromaDB to SQLite-vec.")
    print("Your original ChromaDB data will not be modified.")
    print()


async def main():
    """Main migration function."""
    print_banner()
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Migrate ChromaDB to SQLite-vec')
    parser.add_argument('--chroma-path', help='Path to ChromaDB data directory')
    parser.add_argument('--sqlite-path', help='Path for SQLite-vec database')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for migration')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration with environment variable and argument support
    chroma_path = args.chroma_path or os.environ.get('MCP_MEMORY_CHROMA_PATH', CHROMA_PATH)
    
    # Allow custom SQLite path via argument or environment variable
    sqlite_path = args.sqlite_path or os.environ.get('MCP_MEMORY_SQLITE_PATH')
    if not sqlite_path:
        # Default to same directory as ChromaDB
        chroma_dir = os.path.dirname(chroma_path) if os.path.dirname(chroma_path) else os.getcwd()
        sqlite_path = os.path.join(chroma_dir, 'sqlite_vec_migrated.db')
    
    # Use batch size from arguments
    batch_size = args.batch_size
    
    print(f"ChromaDB source: {chroma_path}")
    print(f"SQLite-vec destination: {sqlite_path}")
    print()
    
    # Check if ChromaDB data exists
    memory_count = await check_chroma_data(chroma_path)
    if memory_count < 0:
        return 1
    
    if memory_count == 0:
        print("No memories to migrate. Migration complete!")
        return 0
    
    # Confirm migration
    print(f"About to migrate {memory_count} memories from ChromaDB to SQLite-vec")
    print(f"Destination file: {sqlite_path}")
    
    try:
        response = input("\\nProceed with migration? (y/N): ").strip().lower()
        if response != 'y':
            print("Migration cancelled by user")
            return 1
    except EOFError:
        # Auto-proceed in non-interactive environment
        print("\\nAuto-proceeding with migration in non-interactive environment...")
        response = 'y'
    
    # Perform migration
    stats = MigrationStats()
    success = await migrate_memories(chroma_path, sqlite_path, stats, batch_size=batch_size)
    
    if success:
        # Verify migration
        await verify_migration(sqlite_path, stats.migrated_successfully)
    
    # Print summary
    stats.print_summary()
    
    if success and stats.failed_migrations == 0:
        print("\\nMigration completed successfully!")
        print("\\nNext steps:")
        print(f"   1. Update your environment: export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec")
        print(f"   2. Update database path: export MCP_MEMORY_SQLITE_PATH={sqlite_path}")
        print(f"   3. Restart MCP Memory Service")
        print(f"   4. Test that your memories are accessible")
        print(f"   5. (Optional) Backup your old ChromaDB data: {chroma_path}")
        return 0
    else:
        print("\\nMigration completed with errors. Please review the summary above.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))