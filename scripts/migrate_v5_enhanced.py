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
Enhanced ChromaDB to SQLite-vec Migration Script for v5.0.0+

This script provides a robust migration path from ChromaDB to SQLite-vec with:
- Custom data path support
- Proper content hash generation
- Tag format validation and correction
- Progress indicators
- Transaction-based migration with rollback
- Dry-run mode for testing
- Comprehensive error handling
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

# Try importing with progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install 'tqdm' for progress bars: pip install tqdm")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import storage modules
try:
    from mcp_memory_service.storage.chroma import ChromaMemoryStorage
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.models.memory import Memory
    from mcp_memory_service.utils.hashing import generate_content_hash
except ImportError as e:
    print(f"Error importing MCP modules: {e}")
    print("Make sure you're running this from the MCP Memory Service directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MigrationConfig:
    """Configuration for migration process."""
    
    def __init__(self):
        self.chroma_path: Optional[str] = None
        self.sqlite_path: Optional[str] = None
        self.batch_size: int = 50
        self.dry_run: bool = False
        self.skip_duplicates: bool = True
        self.backup_path: Optional[str] = None
        self.verbose: bool = False
        self.validate_only: bool = False
        self.force: bool = False
    
    @classmethod
    def from_args(cls, args) -> 'MigrationConfig':
        """Create config from command line arguments."""
        config = cls()
        config.chroma_path = args.chroma_path
        config.sqlite_path = args.sqlite_path
        config.batch_size = args.batch_size
        config.dry_run = args.dry_run
        config.skip_duplicates = not args.no_skip_duplicates
        config.backup_path = args.backup
        config.verbose = args.verbose
        config.validate_only = args.validate_only
        config.force = args.force
        return config
    
    def resolve_paths(self):
        """Resolve and validate data paths."""
        # Resolve ChromaDB path
        if not self.chroma_path:
            # Check environment variable first
            self.chroma_path = os.environ.get('MCP_MEMORY_CHROMA_PATH')
            
            if not self.chroma_path:
                # Use default locations based on platform
                home = Path.home()
                if sys.platform == 'darwin':  # macOS
                    default_base = home / 'Library' / 'Application Support' / 'mcp-memory'
                elif sys.platform == 'win32':  # Windows
                    default_base = Path(os.getenv('LOCALAPPDATA', '')) / 'mcp-memory'
                else:  # Linux
                    default_base = home / '.local' / 'share' / 'mcp-memory'
                
                # Try multiple possible locations
                possible_paths = [
                    home / '.mcp_memory_chroma',  # Legacy location
                    default_base / 'chroma_db',    # New standard location
                    Path.cwd() / 'chroma_db',      # Current directory
                ]
                
                for path in possible_paths:
                    if path.exists():
                        self.chroma_path = str(path)
                        logger.info(f"Found ChromaDB at: {path}")
                        break
                
                if not self.chroma_path:
                    raise ValueError(
                        "Could not find ChromaDB data. Please specify --chroma-path or "
                        "set MCP_MEMORY_CHROMA_PATH environment variable"
                    )
        
        # Resolve SQLite path
        if not self.sqlite_path:
            # Check environment variable first
            self.sqlite_path = os.environ.get('MCP_MEMORY_SQLITE_PATH')
            
            if not self.sqlite_path:
                # Default to same directory as ChromaDB with different name
                chroma_dir = Path(self.chroma_path).parent
                self.sqlite_path = str(chroma_dir / 'sqlite_vec.db')
                logger.info(f"Using default SQLite path: {self.sqlite_path}")
        
        # Resolve backup path if needed
        if self.backup_path is None and not self.dry_run and not self.validate_only:
            backup_dir = Path(self.sqlite_path).parent / 'backups'
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.backup_path = str(backup_dir / f'migration_backup_{timestamp}.json')


class EnhancedMigrationTool:
    """Enhanced migration tool with proper error handling and progress tracking."""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.stats = {
            'total': 0,
            'migrated': 0,
            'skipped': 0,
            'failed': 0,
            'errors': []
        }
        self.chroma_storage = None
        self.sqlite_storage = None
    
    def generate_proper_content_hash(self, content: str) -> str:
        """Generate a proper SHA256 content hash."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def validate_and_fix_tags(self, tags: Any) -> List[str]:
        """Validate and fix tag format."""
        if not tags:
            return []
        
        if isinstance(tags, str):
            # Handle comma-separated string
            if ',' in tags:
                return [tag.strip() for tag in tags.split(',') if tag.strip()]
            # Handle single tag
            return [tags.strip()] if tags.strip() else []
        
        if isinstance(tags, list):
            # Clean and validate list tags
            clean_tags = []
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    clean_tags.append(tag.strip())
            return clean_tags
        
        # Unknown format - log warning and return empty
        logger.warning(f"Unknown tag format: {type(tags)} - {tags}")
        return []
    
    def safe_timestamp_convert(self, timestamp: Any) -> float:
        """Safely convert various timestamp formats to float."""
        if timestamp is None:
            return datetime.now().timestamp()
        
        # Handle numeric timestamps
        if isinstance(timestamp, (int, float)):
            # Check if timestamp is reasonable (between 2000 and 2100)
            if 946684800 <= float(timestamp) <= 4102444800:
                return float(timestamp)
            else:
                logger.warning(f"Timestamp {timestamp} out of reasonable range, using current time")
                return datetime.now().timestamp()
        
        # Handle string timestamps
        if isinstance(timestamp, str):
            # Try ISO format
            for fmt in [
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S',
            ]:
                try:
                    dt = datetime.strptime(timestamp.rstrip('Z'), fmt)
                    return dt.timestamp()
                except ValueError:
                    continue
            
            # Try parsing as float string
            try:
                ts = float(timestamp)
                if 946684800 <= ts <= 4102444800:
                    return ts
            except ValueError:
                pass
        
        # Fallback to current time
        logger.warning(f"Could not parse timestamp '{timestamp}', using current time")
        return datetime.now().timestamp()
    
    async def extract_memories_from_chroma(self) -> List[Dict[str, Any]]:
        """Extract all memories from ChromaDB with proper error handling."""
        memories = []
        
        try:
            # Initialize ChromaDB storage
            logger.info("Connecting to ChromaDB...")
            self.chroma_storage = ChromaMemoryStorage(path=self.config.chroma_path)
            
            # Access the collection directly
            collection = self.chroma_storage.collection
            if not collection:
                raise ValueError("ChromaDB collection not initialized")
            
            # Get all data from collection
            logger.info("Extracting memories from ChromaDB...")
            results = collection.get(include=['documents', 'metadatas', 'embeddings'])
            
            if not results or not results.get('ids'):
                logger.warning("No memories found in ChromaDB")
                return memories
            
            total = len(results['ids'])
            logger.info(f"Found {total} memories to process")
            
            # Process each memory
            for i in range(total):
                try:
                    # Extract data with defaults
                    doc_id = results['ids'][i]
                    content = results['documents'][i] if i < len(results.get('documents', [])) else ""
                    metadata = results['metadatas'][i] if i < len(results.get('metadatas', [])) else {}
                    embedding = results['embeddings'][i] if i < len(results.get('embeddings', [])) else None
                    
                    if not content:
                        logger.warning(f"Skipping memory {doc_id}: empty content")
                        continue
                    
                    # Generate proper content hash
                    content_hash = self.generate_proper_content_hash(content)
                    
                    # Extract and validate tags
                    raw_tags = metadata.get('tags', metadata.get('tags_str', []))
                    tags = self.validate_and_fix_tags(raw_tags)
                    
                    # Extract timestamps
                    created_at = self.safe_timestamp_convert(
                        metadata.get('created_at', metadata.get('timestamp'))
                    )
                    updated_at = self.safe_timestamp_convert(
                        metadata.get('updated_at', created_at)
                    )
                    
                    # Extract memory type
                    memory_type = metadata.get('memory_type', metadata.get('type', 'imported'))
                    
                    # Clean metadata (remove special fields)
                    clean_metadata = {}
                    exclude_keys = {
                        'tags', 'tags_str', 'created_at', 'updated_at', 
                        'timestamp', 'timestamp_float', 'timestamp_str',
                        'memory_type', 'type', 'content_hash',
                        'created_at_iso', 'updated_at_iso'
                    }
                    
                    for key, value in metadata.items():
                        if key not in exclude_keys and value is not None:
                            clean_metadata[key] = value
                    
                    # Create memory record
                    memory_data = {
                        'content': content,
                        'content_hash': content_hash,
                        'tags': tags,
                        'memory_type': memory_type,
                        'metadata': clean_metadata,
                        'embedding': embedding,
                        'created_at': created_at,
                        'updated_at': updated_at,
                        'original_id': doc_id  # Keep for reference
                    }
                    
                    memories.append(memory_data)
                    
                    if self.config.verbose and (i + 1) % 100 == 0:
                        logger.info(f"Processed {i + 1}/{total} memories")
                
                except Exception as e:
                    logger.error(f"Failed to extract memory {i}: {e}")
                    self.stats['errors'].append(f"Extract error at index {i}: {str(e)}")
                    continue
            
            logger.info(f"Successfully extracted {len(memories)} memories")
            return memories
            
        except Exception as e:
            logger.error(f"Critical error extracting from ChromaDB: {e}")
            raise
    
    async def migrate_to_sqlite(self, memories: List[Dict[str, Any]]) -> bool:
        """Migrate memories to SQLite-vec with transaction support."""
        if not memories:
            logger.warning("No memories to migrate")
            return True
        
        try:
            # Initialize SQLite-vec storage
            logger.info(f"Initializing SQLite-vec at {self.config.sqlite_path}")
            self.sqlite_storage = SqliteVecMemoryStorage(
                db_path=self.config.sqlite_path,
                embedding_model=os.environ.get('MCP_MEMORY_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            )
            await self.sqlite_storage.initialize()
            
            # Start transaction
            conn = self.sqlite_storage.conn
            conn.execute("BEGIN TRANSACTION")
            
            try:
                # Migrate memories in batches
                total = len(memories)
                batch_size = self.config.batch_size
                
                # Use progress bar if available
                if TQDM_AVAILABLE and not self.config.dry_run:
                    progress_bar = tqdm(total=total, desc="Migrating memories")
                else:
                    progress_bar = None
                
                for i in range(0, total, batch_size):
                    batch = memories[i:i + batch_size]
                    
                    if not self.config.dry_run:
                        for memory_data in batch:
                            try:
                                # Check for duplicates
                                if self.config.skip_duplicates:
                                    existing = conn.execute(
                                        "SELECT 1 FROM memories WHERE content_hash = ? LIMIT 1",
                                        (memory_data['content_hash'],)
                                    ).fetchone()
                                    
                                    if existing:
                                        self.stats['skipped'] += 1
                                        if progress_bar:
                                            progress_bar.update(1)
                                        continue
                                
                                # Create Memory object
                                memory = Memory(
                                    content=memory_data['content'],
                                    content_hash=memory_data['content_hash'],
                                    tags=memory_data['tags'],
                                    memory_type=memory_data.get('memory_type'),
                                    metadata=memory_data.get('metadata', {}),
                                    created_at=memory_data['created_at'],
                                    updated_at=memory_data['updated_at']
                                )
                                
                                # Store memory
                                success, message = await self.sqlite_storage.store(memory)
                                
                                if success:
                                    self.stats['migrated'] += 1
                                else:
                                    raise Exception(f"Failed to store: {message}")
                                
                                if progress_bar:
                                    progress_bar.update(1)
                            
                            except Exception as e:
                                self.stats['failed'] += 1
                                self.stats['errors'].append(
                                    f"Migration error for {memory_data['content_hash'][:8]}: {str(e)}"
                                )
                                if progress_bar:
                                    progress_bar.update(1)
                    
                    else:
                        # Dry run - just count
                        self.stats['migrated'] += len(batch)
                        if progress_bar:
                            progress_bar.update(len(batch))
                
                if progress_bar:
                    progress_bar.close()
                
                # Commit transaction
                if not self.config.dry_run:
                    conn.execute("COMMIT")
                    logger.info("Transaction committed successfully")
                else:
                    conn.execute("ROLLBACK")
                    logger.info("Dry run - transaction rolled back")
                
                return True
                
            except Exception as e:
                # Rollback on error
                conn.execute("ROLLBACK")
                logger.error(f"Migration failed, transaction rolled back: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Critical error during migration: {e}")
            return False
        
        finally:
            # Clean up
            if self.sqlite_storage:
                self.sqlite_storage.close()
    
    async def create_backup(self, memories: List[Dict[str, Any]]):
        """Create a JSON backup of memories."""
        if not self.config.backup_path or self.config.dry_run:
            return
        
        logger.info(f"Creating backup at {self.config.backup_path}")
        
        backup_data = {
            'version': '2.0',
            'created_at': datetime.now().isoformat(),
            'source': self.config.chroma_path,
            'total_memories': len(memories),
            'memories': memories
        }
        
        # Remove embeddings from backup to reduce size
        for memory in backup_data['memories']:
            memory.pop('embedding', None)
        
        with open(self.config.backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        logger.info(f"Backup created: {self.config.backup_path}")
    
    async def validate_migration(self) -> bool:
        """Validate the migrated data."""
        logger.info("Validating migration...")
        
        try:
            # Connect to SQLite database
            conn = sqlite3.connect(self.config.sqlite_path)
            
            # Check memory count
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            logger.info(f"SQLite database contains {count} memories")
            
            # Check for required fields
            sample = conn.execute("""
                SELECT content_hash, content, tags, created_at 
                FROM memories 
                LIMIT 10
            """).fetchall()
            
            issues = []
            for row in sample:
                if not row[0]:  # content_hash
                    issues.append("Missing content_hash")
                if not row[1]:  # content
                    issues.append("Missing content")
                if row[3] is None:  # created_at
                    issues.append("Missing created_at")
            
            conn.close()
            
            if issues:
                logger.warning(f"Validation issues found: {', '.join(set(issues))}")
                return False
            
            logger.info("Validation passed!")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    async def run(self) -> bool:
        """Run the migration process."""
        try:
            # Resolve paths
            self.config.resolve_paths()
            
            # Print configuration
            print("\n" + "="*60)
            print("MCP Memory Service - Enhanced Migration Tool v2.0")
            print("="*60)
            print(f"ChromaDB source: {self.config.chroma_path}")
            print(f"SQLite-vec target: {self.config.sqlite_path}")
            if self.config.backup_path:
                print(f"Backup location: {self.config.backup_path}")
            print(f"Mode: {'DRY RUN' if self.config.dry_run else 'LIVE MIGRATION'}")
            print(f"Batch size: {self.config.batch_size}")
            print(f"Skip duplicates: {self.config.skip_duplicates}")
            print()
            
            # Check if validation only
            if self.config.validate_only:
                return await self.validate_migration()
            
            # Check if target exists
            if Path(self.config.sqlite_path).exists() and not self.config.force:
                if not self.config.dry_run:
                    response = input(f"Target database exists. Overwrite? (y/N): ")
                    if response.lower() != 'y':
                        print("Migration cancelled")
                        return False
            
            # Extract memories from ChromaDB
            memories = await self.extract_memories_from_chroma()
            self.stats['total'] = len(memories)
            
            if not memories:
                print("No memories found to migrate")
                return True
            
            # Create backup
            if self.config.backup_path and not self.config.dry_run:
                await self.create_backup(memories)
            
            # Confirm migration
            if not self.config.dry_run and not self.config.force:
                print(f"\nAbout to migrate {len(memories)} memories")
                response = input("Proceed? (y/N): ")
                if response.lower() != 'y':
                    print("Migration cancelled")
                    return False
            
            # Perform migration
            success = await self.migrate_to_sqlite(memories)
            
            # Print summary
            print("\n" + "="*60)
            print("MIGRATION SUMMARY")
            print("="*60)
            print(f"Total memories found: {self.stats['total']}")
            print(f"Successfully migrated: {self.stats['migrated']}")
            print(f"Duplicates skipped: {self.stats['skipped']}")
            print(f"Failed migrations: {self.stats['failed']}")
            
            if self.stats['errors'] and self.config.verbose:
                print("\nErrors encountered:")
                for i, error in enumerate(self.stats['errors'][:10], 1):
                    print(f"  {i}. {error}")
                if len(self.stats['errors']) > 10:
                    print(f"  ... and {len(self.stats['errors']) - 10} more")
            
            if success and not self.config.dry_run:
                # Validate migration
                if await self.validate_migration():
                    print("\n✅ Migration completed successfully!")
                    print("\nNext steps:")
                    print("1. Set environment variable:")
                    print("   export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec")
                    print(f"2. Set database path:")
                    print(f"   export MCP_MEMORY_SQLITE_PATH={self.config.sqlite_path}")
                    print("3. Restart MCP Memory Service")
                    print("4. Test that your memories are accessible")
                else:
                    print("\n⚠️ Migration completed with validation warnings")
            
            return success
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            if self.config.verbose:
                import traceback
                traceback.print_exc()
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced ChromaDB to SQLite-vec migration tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--chroma-path',
        help='Path to ChromaDB data directory (default: auto-detect)'
    )
    parser.add_argument(
        '--sqlite-path',
        help='Path for SQLite-vec database (default: same dir as ChromaDB)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of memories to migrate per batch (default: 50)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate migration without making changes'
    )
    parser.add_argument(
        '--no-skip-duplicates',
        action='store_true',
        help='Migrate all memories including duplicates'
    )
    parser.add_argument(
        '--backup',
        help='Path for JSON backup file (default: auto-generate)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing SQLite database'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompts'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = MigrationConfig.from_args(args)
    
    # Run migration
    tool = EnhancedMigrationTool(config)
    success = asyncio.run(tool.run())
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()