#!/usr/bin/env python3
"""
Migration script for moving data to Cloudflare backend.
Supports migration from SQLite-vec and ChromaDB backends.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.hashing import generate_content_hash

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataMigrator:
    """Handles migration of data to Cloudflare backend."""
    
    def __init__(self):
        self.source_storage = None
        self.cloudflare_storage = None
        
    async def export_from_sqlite_vec(self, sqlite_path: str) -> List[Dict[str, Any]]:
        """Export data from SQLite-vec backend."""
        logger.info(f"Exporting data from SQLite-vec: {sqlite_path}")
        
        try:
            from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
            
            storage = SqliteVecMemoryStorage(sqlite_path)
            await storage.initialize()
            
            # Get all memories
            memories = []
            stats = await storage.get_stats()
            total_memories = stats.get('total_memories', 0)
            
            logger.info(f"Found {total_memories} memories to export")
            
            # Get recent memories in batches
            batch_size = 100
            exported_count = 0
            
            while exported_count < total_memories:
                batch = await storage.get_recent_memories(batch_size)
                if not batch:
                    break
                
                for memory in batch:
                    memory_data = {
                        'content': memory.content,
                        'content_hash': memory.content_hash,
                        'tags': memory.tags,
                        'memory_type': memory.memory_type,
                        'metadata': memory.metadata,
                        'created_at': memory.created_at,
                        'created_at_iso': memory.created_at_iso,
                        'updated_at': memory.updated_at,
                        'updated_at_iso': memory.updated_at_iso
                    }
                    memories.append(memory_data)
                    exported_count += 1
                
                logger.info(f"Exported {exported_count}/{total_memories} memories")
                
                # Break if we got fewer memories than batch size
                if len(batch) < batch_size:
                    break
            
            logger.info(f"Successfully exported {len(memories)} memories from SQLite-vec")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to export from SQLite-vec: {e}")
            raise
    
    async def export_from_chroma(self, chroma_path: str) -> List[Dict[str, Any]]:
        """Export data from ChromaDB backend."""
        logger.info(f"Exporting data from ChromaDB: {chroma_path}")
        
        try:
            from mcp_memory_service.storage.chroma import ChromaMemoryStorage
            
            storage = ChromaMemoryStorage(chroma_path, preload_model=False)
            await storage.initialize()
            
            # Get all memories
            memories = []
            stats = await storage.get_stats()
            total_memories = stats.get('total_memories', 0)
            
            logger.info(f"Found {total_memories} memories to export")
            
            # Get recent memories
            recent_memories = await storage.get_recent_memories(total_memories)
            
            for memory in recent_memories:
                memory_data = {
                    'content': memory.content,
                    'content_hash': memory.content_hash,
                    'tags': memory.tags,
                    'memory_type': memory.memory_type,
                    'metadata': memory.metadata,
                    'created_at': memory.created_at,
                    'created_at_iso': memory.created_at_iso,
                    'updated_at': memory.updated_at,
                    'updated_at_iso': memory.updated_at_iso
                }
                memories.append(memory_data)
            
            logger.info(f"Successfully exported {len(memories)} memories from ChromaDB")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to export from ChromaDB: {e}")
            raise
    
    async def import_to_cloudflare(self, memories: List[Dict[str, Any]]) -> bool:
        """Import data to Cloudflare backend."""
        logger.info(f"Importing {len(memories)} memories to Cloudflare backend")
        
        try:
            # Initialize Cloudflare storage
            from mcp_memory_service.storage.cloudflare import CloudflareStorage
            
            # Get configuration from environment
            api_token = os.getenv('CLOUDFLARE_API_TOKEN')
            account_id = os.getenv('CLOUDFLARE_ACCOUNT_ID')
            vectorize_index = os.getenv('CLOUDFLARE_VECTORIZE_INDEX')
            d1_database_id = os.getenv('CLOUDFLARE_D1_DATABASE_ID')
            r2_bucket = os.getenv('CLOUDFLARE_R2_BUCKET')
            
            if not all([api_token, account_id, vectorize_index, d1_database_id]):
                raise ValueError("Missing required Cloudflare environment variables")
            
            storage = CloudflareStorage(
                api_token=api_token,
                account_id=account_id,
                vectorize_index=vectorize_index,
                d1_database_id=d1_database_id,
                r2_bucket=r2_bucket
            )
            
            await storage.initialize()
            
            # Import memories in batches
            batch_size = 10  # Smaller batches for Cloudflare API limits
            imported_count = 0
            failed_count = 0
            
            for i in range(0, len(memories), batch_size):
                batch = memories[i:i + batch_size]
                
                for memory_data in batch:
                    try:
                        # Convert to Memory object
                        memory = Memory(
                            content=memory_data['content'],
                            content_hash=memory_data['content_hash'],
                            tags=memory_data.get('tags', []),
                            memory_type=memory_data.get('memory_type'),
                            metadata=memory_data.get('metadata', {}),
                            created_at=memory_data.get('created_at'),
                            created_at_iso=memory_data.get('created_at_iso'),
                            updated_at=memory_data.get('updated_at'),
                            updated_at_iso=memory_data.get('updated_at_iso')
                        )
                        
                        # Store in Cloudflare
                        success, message = await storage.store(memory)
                        
                        if success:
                            imported_count += 1
                            logger.debug(f"Imported memory: {memory.content_hash[:16]}...")
                        else:
                            failed_count += 1
                            logger.warning(f"Failed to import memory {memory.content_hash[:16]}: {message}")
                        
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"Error importing memory: {e}")
                
                # Progress update
                processed = min(i + batch_size, len(memories))
                logger.info(f"Progress: {processed}/{len(memories)} processed, {imported_count} imported, {failed_count} failed")
                
                # Rate limiting - small delay between batches
                await asyncio.sleep(0.5)
            
            # Final cleanup
            await storage.close()
            
            logger.info(f"Migration completed: {imported_count} imported, {failed_count} failed")
            return failed_count == 0
            
        except Exception as e:
            logger.error(f"Failed to import to Cloudflare: {e}")
            raise
    
    async def export_to_file(self, source_backend: str, source_path: str, output_file: str) -> bool:
        """Export data from source backend to JSON file."""
        try:
            if source_backend == 'sqlite_vec':
                memories = await self.export_from_sqlite_vec(source_path)
            elif source_backend == 'chroma':
                memories = await self.export_from_chroma(source_path)
            else:
                raise ValueError(f"Unsupported source backend: {source_backend}")
            
            # Save to JSON file
            export_data = {
                'source_backend': source_backend,
                'source_path': source_path,
                'export_timestamp': time.time(),
                'total_memories': len(memories),
                'memories': memories
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(memories)} memories to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    async def import_from_file(self, input_file: str) -> bool:
        """Import data from JSON file to Cloudflare backend."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                export_data = json.load(f)
            
            memories = export_data.get('memories', [])
            logger.info(f"Loaded {len(memories)} memories from {input_file}")
            
            return await self.import_to_cloudflare(memories)
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False
    
    async def migrate_direct(self, source_backend: str, source_path: str) -> bool:
        """Direct migration from source backend to Cloudflare."""
        try:
            # Export data
            if source_backend == 'sqlite_vec':
                memories = await self.export_from_sqlite_vec(source_path)
            elif source_backend == 'chroma':
                memories = await self.export_from_chroma(source_path)
            else:
                raise ValueError(f"Unsupported source backend: {source_backend}")
            
            # Import to Cloudflare
            return await self.import_to_cloudflare(memories)
            
        except Exception as e:
            logger.error(f"Direct migration failed: {e}")
            return False


async def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description='Migrate data to Cloudflare backend')
    
    subparsers = parser.add_subparsers(dest='command', help='Migration commands')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export data to JSON file')
    export_parser.add_argument('--source', choices=['sqlite_vec', 'chroma'], required=True,
                             help='Source backend type')
    export_parser.add_argument('--source-path', required=True,
                             help='Path to source database')
    export_parser.add_argument('--output', required=True,
                             help='Output JSON file path')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import data from JSON file')
    import_parser.add_argument('--input', required=True,
                             help='Input JSON file path')
    
    # Direct migration command
    migrate_parser = subparsers.add_parser('migrate', help='Direct migration to Cloudflare')
    migrate_parser.add_argument('--source', choices=['sqlite_vec', 'chroma'], required=True,
                              help='Source backend type')
    migrate_parser.add_argument('--source-path', required=True,
                              help='Path to source database')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    migrator = DataMigrator()
    
    try:
        if args.command == 'export':
            success = await migrator.export_to_file(
                args.source, args.source_path, args.output
            )
        elif args.command == 'import':
            success = await migrator.import_from_file(args.input)
        elif args.command == 'migrate':
            success = await migrator.migrate_direct(args.source, args.source_path)
        
        if success:
            logger.info("Migration completed successfully!")
            sys.exit(0)
        else:
            logger.error("Migration failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())