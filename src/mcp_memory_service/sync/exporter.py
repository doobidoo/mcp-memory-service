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
Memory export functionality for database synchronization.
"""

import json
import os
import platform
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..models.memory import Memory
from ..storage.base import MemoryStorage

logger = logging.getLogger(__name__)


class MemoryExporter:
    """
    Exports memories from a storage backend to JSON format.
    
    Preserves all metadata, timestamps, and adds source tracking for
    multi-machine synchronization workflows.
    """
    
    def __init__(self, storage: MemoryStorage):
        """
        Initialize the exporter.
        
        Args:
            storage: The memory storage backend to export from
        """
        self.storage = storage
        self.machine_name = self._get_machine_name()
    
    def _get_machine_name(self) -> str:
        """Get a unique machine identifier."""
        # Try various methods to get a unique machine name
        machine_name = (
            os.environ.get('COMPUTERNAME') or  # Windows
            os.environ.get('HOSTNAME') or      # Linux/macOS
            platform.node() or                # Platform module
            'unknown-machine'
        )
        return machine_name.lower().replace(' ', '-')
    
    async def export_to_json(
        self, 
        output_file: Path,
        include_embeddings: bool = False,
        filter_tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Export all memories to JSON format.
        
        Args:
            output_file: Path to write the JSON export
            include_embeddings: Whether to include embedding vectors (large)
            filter_tags: Only export memories with these tags (optional)
            
        Returns:
            Export metadata dictionary with statistics
        """
        logger.info(f"Starting memory export to {output_file}")
        
        # Get all memories from storage
        all_memories = await self._get_filtered_memories(filter_tags)
        
        # Create export metadata
        export_metadata = {
            "source_machine": self.machine_name,
            "export_timestamp": datetime.now().isoformat(),
            "total_memories": len(all_memories),
            "database_path": str(self.storage.db_path) if hasattr(self.storage, 'db_path') else 'unknown',
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "include_embeddings": include_embeddings,
            "filter_tags": filter_tags,
            "exporter_version": "5.0.1"
        }
        
        # Convert memories to exportable format
        exported_memories = []
        for memory in all_memories:
            memory_dict = await self._memory_to_dict(memory, include_embeddings)
            exported_memories.append(memory_dict)
        
        # Create final export structure
        export_data = {
            "export_metadata": export_metadata,
            "memories": exported_memories
        }
        
        # Write to JSON file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        # Log success
        file_size = output_file.stat().st_size
        logger.info(f"Export completed: {len(all_memories)} memories written to {output_file}")
        logger.info(f"File size: {file_size / 1024 / 1024:.2f} MB")
        
        # Return summary statistics
        return {
            "success": True,
            "exported_count": len(all_memories),
            "output_file": str(output_file),
            "file_size_bytes": file_size,
            "source_machine": self.machine_name,
            "export_timestamp": export_metadata["export_timestamp"]
        }
    
    async def _get_filtered_memories(self, filter_tags: Optional[List[str]]) -> List[Memory]:
        """Get memories with optional tag filtering."""
        if not filter_tags:
            # Get all memories
            return await self.storage.get_all_memories()
        
        # Filter by tags if specified
        filtered_memories = []
        all_memories = await self.storage.get_all_memories()
        
        for memory in all_memories:
            if any(tag in memory.tags for tag in filter_tags):
                filtered_memories.append(memory)
        
        return filtered_memories
    
    async def _memory_to_dict(self, memory: Memory, include_embeddings: bool) -> Dict[str, Any]:
        """Convert a Memory object to a dictionary for JSON export."""
        memory_dict = {
            "content": memory.content,
            "content_hash": memory.content_hash,
            "tags": memory.tags,
            "created_at": memory.created_at,  # Preserve original timestamp
            "updated_at": memory.updated_at,
            "memory_type": memory.memory_type,
            "metadata": memory.metadata or {},
            "export_source": self.machine_name
        }
        
        # Add embedding if requested and available
        if include_embeddings and hasattr(memory, 'embedding') and memory.embedding:
            memory_dict["embedding"] = memory.embedding.tolist() if hasattr(memory.embedding, 'tolist') else memory.embedding
        
        return memory_dict
    
    async def export_summary(self) -> Dict[str, Any]:
        """
        Get a summary of what would be exported without actually exporting.
        
        Returns:
            Summary statistics about the memories in the database
        """
        all_memories = await self.storage.get_all_memories()
        
        # Analyze tags
        tag_counts = {}
        memory_types = {}
        date_range = {"earliest": None, "latest": None}
        
        for memory in all_memories:
            # Count tags
            for tag in memory.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Count memory types
            memory_types[memory.memory_type] = memory_types.get(memory.memory_type, 0) + 1
            
            # Track date range
            if date_range["earliest"] is None or memory.created_at < date_range["earliest"]:
                date_range["earliest"] = memory.created_at
            if date_range["latest"] is None or memory.created_at > date_range["latest"]:
                date_range["latest"] = memory.created_at
        
        return {
            "total_memories": len(all_memories),
            "machine_name": self.machine_name,
            "tag_counts": dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)),
            "memory_types": memory_types,
            "date_range": date_range,
            "estimated_json_size_mb": len(all_memories) * 0.001  # Rough estimate
        }