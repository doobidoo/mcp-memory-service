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
Memory import functionality for database synchronization.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Set

from ..compat import _sanitize_log_value
from ..models.memory import Memory
from ..storage.base import MemoryStorage

logger = logging.getLogger(__name__)


class MemoryImporter:
    """
    Imports memories from JSON format into a storage backend.
    
    Handles deduplication based on content hash and preserves original
    timestamps while adding import metadata.
    """
    
    def __init__(self, storage: MemoryStorage):
        """
        Initialize the importer.
        
        Args:
            storage: The memory storage backend to import into
        """
        self.storage = storage
    
    async def import_from_json(
        self,
        json_files: List[Path],
        deduplicate: bool = True,
        add_source_tags: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Import memories from one or more JSON export files.
        
        Args:
            json_files: List of JSON export files to import
            deduplicate: Whether to skip memories with duplicate content hashes
            add_source_tags: Whether to add source machine tags
            dry_run: If True, analyze imports without actually storing
            
        Returns:
            Import statistics and results
        """
        logger.info("Starting import from %s JSON files", len(json_files))
        
        # Get existing content hashes for deduplication
        existing_hashes = await self._get_existing_hashes() if deduplicate else set()
        
        import_stats = {
            "files_processed": 0,
            "total_processed": 0,
            "imported": 0,
            "duplicates_skipped": 0,
            "errors": 0,
            "sources": {},
            "dry_run": dry_run,
            "start_time": datetime.now().isoformat()
        }
        
        # Process each JSON file
        for json_file in json_files:
            try:
                file_stats = await self._import_single_file(
                    json_file, existing_hashes, add_source_tags, dry_run
                )
                
                # Merge file stats into overall stats
                import_stats["files_processed"] += 1
                import_stats["total_processed"] += file_stats["processed"]
                import_stats["imported"] += file_stats["imported"]
                import_stats["duplicates_skipped"] += file_stats["duplicates"]
                import_stats["sources"].update(file_stats["sources"])
                
                logger.info(
                    "Processed %s: %s/%s imported",
                    _sanitize_log_value(json_file), file_stats['imported'], file_stats['processed'],
                )
                
            except Exception as e:
                logger.error("Error processing %s: %s", _sanitize_log_value(json_file), _sanitize_log_value(e))
                import_stats["errors"] += 1
        
        import_stats["end_time"] = datetime.now().isoformat()
        
        # Log final summary
        logger.info("Import completed:")
        logger.info("  Files processed: %s", import_stats['files_processed'])
        logger.info("  Total memories processed: %s", import_stats['total_processed'])
        logger.info("  Successfully imported: %s", import_stats['imported'])
        logger.info("  Duplicates skipped: %s", import_stats['duplicates_skipped'])
        logger.info("  Errors: %s", _sanitize_log_value(import_stats['errors']))
        
        for source_machine, stats in import_stats["sources"].items():
            logger.info("  %s: %s/%s imported", _sanitize_log_value(source_machine), stats['imported'], stats['total'])
        
        return import_stats
    
    async def _import_single_file(
        self,
        json_file: Path,
        existing_hashes: Set[str],
        add_source_tags: bool,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Import memories from a single JSON file."""
        logger.info("Processing %s", _sanitize_log_value(json_file))
        
        # Load and validate JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            export_data = json.load(f)
        
        # Validate export format
        if "export_metadata" not in export_data or "memories" not in export_data:
            raise ValueError(f"Invalid export format in {json_file}")
        
        export_metadata = export_data["export_metadata"]
        source_machine = export_metadata.get("source_machine", "unknown")
        memories_data = export_data["memories"]
        
        file_stats = {
            "processed": len(memories_data),
            "imported": 0,
            "duplicates": 0,
            "sources": {
                source_machine: {
                    "total": len(memories_data),
                    "imported": 0,
                    "duplicates": 0
                }
            }
        }
        
        # Process each memory
        for memory_data in memories_data:
            content_hash = memory_data.get("content_hash")
            
            if not content_hash:
                logger.warning("Memory missing content_hash, skipping")
                continue
            
            # Check for duplicates
            if content_hash in existing_hashes:
                file_stats["duplicates"] += 1
                file_stats["sources"][source_machine]["duplicates"] += 1
                continue
            
            # Create Memory object
            try:
                memory = await self._create_memory_from_dict(
                    memory_data, source_machine, add_source_tags, json_file
                )
                
                # Store the memory (unless dry run)
                if not dry_run:
                    await self.storage.store(memory)
                
                # Track success
                existing_hashes.add(content_hash)
                file_stats["imported"] += 1
                file_stats["sources"][source_machine]["imported"] += 1
                
            except Exception as e:
                logger.error("Error creating memory from data: %s", _sanitize_log_value(e))
                continue
        
        return file_stats
    
    async def _create_memory_from_dict(
        self,
        memory_data: Dict[str, Any],
        source_machine: str,
        add_source_tags: bool,
        source_file: Path
    ) -> Memory:
        """Create a Memory object from imported dictionary data."""
        
        # Prepare tags
        tags = memory_data.get("tags", []).copy()
        if add_source_tags and f"source:{source_machine}" not in tags:
            tags.append(f"source:{source_machine}")
        
        # Prepare metadata
        metadata = memory_data.get("metadata", {}).copy()
        metadata["import_info"] = {
            "imported_at": datetime.now().isoformat(),
            "source_machine": source_machine,
            "source_file": str(source_file),
            "importer_version": "4.5.0"
        }
        
        # Create Memory object preserving original timestamps
        memory = Memory(
            content=memory_data["content"],
            content_hash=memory_data["content_hash"],
            tags=tags,
            created_at=memory_data["created_at"],  # Preserve original
            updated_at=memory_data.get("updated_at", memory_data["created_at"]),
            memory_type=memory_data.get("memory_type", "note"),
            metadata=metadata
        )
        
        return memory
    
    async def _get_existing_hashes(self) -> Set[str]:
        """Get all existing content hashes for deduplication."""
        try:
            all_memories = await self.storage.get_all_memories()
            return {memory.content_hash for memory in all_memories}
        except Exception as e:
            logger.warning("Could not load existing memories for deduplication: %s", _sanitize_log_value(e))
            return set()
    
    async def analyze_import(self, json_files: List[Path]) -> Dict[str, Any]:
        """
        Analyze what would be imported without actually importing.
        
        Args:
            json_files: List of JSON export files to analyze
            
        Returns:
            Analysis results including potential duplicates and statistics
        """
        logger.info("Analyzing potential import from %s files", len(json_files))
        
        existing_hashes = await self._get_existing_hashes()
        
        analysis = {
            "files": [],
            "total_memories": 0,
            "unique_memories": 0,
            "potential_duplicates": 0,
            "sources": {},
            "conflicts": []
        }
        
        all_import_hashes = set()
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    export_data = json.load(f)
                
                export_metadata = export_data.get("export_metadata", {})
                memories_data = export_data.get("memories", [])
                source_machine = export_metadata.get("source_machine", "unknown")
                
                file_analysis = {
                    "file": str(json_file),
                    "source_machine": source_machine,
                    "export_date": export_metadata.get("export_timestamp"),
                    "total_memories": len(memories_data),
                    "new_memories": 0,
                    "existing_duplicates": 0,
                    "import_conflicts": 0
                }
                
                # Analyze each memory
                for memory_data in memories_data:
                    content_hash = memory_data.get("content_hash")
                    if not content_hash:
                        continue
                    
                    analysis["total_memories"] += 1
                    
                    # Check against existing database
                    if content_hash in existing_hashes:
                        file_analysis["existing_duplicates"] += 1
                        analysis["potential_duplicates"] += 1
                    # Check against other import files
                    elif content_hash in all_import_hashes:
                        file_analysis["import_conflicts"] += 1
                        analysis["conflicts"].append({
                            "content_hash": content_hash,
                            "source_machine": source_machine,
                            "conflict_type": "duplicate_in_imports"
                        })
                    else:
                        file_analysis["new_memories"] += 1
                        analysis["unique_memories"] += 1
                        all_import_hashes.add(content_hash)
                
                # Track source statistics
                if source_machine not in analysis["sources"]:
                    analysis["sources"][source_machine] = {
                        "files": 0,
                        "total_memories": 0,
                        "new_memories": 0
                    }
                
                analysis["sources"][source_machine]["files"] += 1
                analysis["sources"][source_machine]["total_memories"] += file_analysis["total_memories"]
                analysis["sources"][source_machine]["new_memories"] += file_analysis["new_memories"]
                
                analysis["files"].append(file_analysis)
                
            except Exception as e:
                logger.error("Error analyzing %s: %s", _sanitize_log_value(json_file), _sanitize_log_value(e))
                analysis["files"].append({
                    "file": str(json_file),
                    "error": str(e)
                })
        
        return analysis