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
Document ingestion handler functions for MCP server.

PDF, DOCX, PPTX, TXT/MD document processing and batch directory ingestion.
Extracted from server_impl.py Phase 2.4 refactoring.
"""

import logging
import traceback
from typing import List

from mcp import types

logger = logging.getLogger(__name__)


async def handle_memory_ingest(server, arguments: dict) -> List[types.TextContent]:
    """Unified handler for document/directory ingestion."""
    import json

    file_path = arguments.get("file_path")
    directory_path = arguments.get("directory_path")

    if file_path and directory_path:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": "Provide either file_path OR directory_path, not both"}, indent=2)
        )]

    if file_path:
        # Route to document ingestion
        return await handle_ingest_document(server, arguments)
    elif directory_path:
        # Route to directory ingestion
        return await handle_ingest_directory(server, arguments)
    else:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": "Either file_path or directory_path required"}, indent=2)
        )]


async def handle_ingest_document(server, arguments: dict) -> List[types.TextContent]:
    """Handle document ingestion requests."""
    try:
        from pathlib import Path
        from ...ingestion import get_loader_for_file
        from ...models.memory import Memory
        from ...utils import create_memory_from_chunk
        import time

        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()

        from ...services.memory_service import normalize_tags

        file_path = Path(arguments["file_path"])
        tags = normalize_tags(arguments.get("tags", []))
        chunk_size = arguments.get("chunk_size", 1000)
        chunk_overlap = arguments.get("chunk_overlap", 200)
        memory_type = arguments.get("memory_type", "document")

        logger.info(f"Starting document ingestion: {file_path}")
        start_time = time.time()

        # Validate file exists and get appropriate document loader
        if not file_path.exists():
            return [types.TextContent(
                type="text",
                text=f"Error: File not found: {file_path.resolve()}"
            )]

        loader = get_loader_for_file(file_path)
        if loader is None:
            from ...ingestion import SUPPORTED_FORMATS
            supported_exts = ", ".join(f".{ext}" for ext in SUPPORTED_FORMATS.keys())
            return [types.TextContent(
                type="text",
                text=f"Error: Unsupported file format: {file_path.suffix}. Supported formats: {supported_exts}"
            )]

        # Configure loader
        loader.chunk_size = chunk_size
        loader.chunk_overlap = chunk_overlap

        chunks_processed = 0
        chunks_stored = 0
        errors = []

        # Extract and store chunks
        async for chunk in loader.extract_chunks(file_path):
            chunks_processed += 1

            try:
                # Combine document tags with chunk metadata tags
                all_tags = tags.copy()
                if chunk.metadata.get('tags'):
                    # Handle tags from chunk metadata (can be string or list)
                    chunk_tags = chunk.metadata['tags']
                    if isinstance(chunk_tags, str):
                        # Split comma-separated string into list
                        chunk_tags = [tag.strip() for tag in chunk_tags.split(',') if tag.strip()]
                    all_tags.extend(chunk_tags)

                # Create memory object
                from ...utils import generate_content_hash
                memory = Memory(
                    content=chunk.content,
                    content_hash=generate_content_hash(chunk.content, chunk.metadata),
                    tags=list(set(all_tags)),  # Remove duplicates
                    memory_type=memory_type,
                    metadata=chunk.metadata
                )

                # Store the memory
                success, error = await storage.store(memory)
                if success:
                    chunks_stored += 1
                else:
                    errors.append(f"Chunk {chunk.chunk_index}: {error}")

            except Exception as e:
                errors.append(f"Chunk {chunk.chunk_index}: {str(e)}")

        processing_time = time.time() - start_time
        success_rate = (chunks_stored / chunks_processed * 100) if chunks_processed > 0 else 0

        # Prepare result message
        result_lines = [
            f"✅ Document ingestion completed: {file_path.name}",
            f"📄 Chunks processed: {chunks_processed}",
            f"💾 Chunks stored: {chunks_stored}",
            f"⚡ Success rate: {success_rate:.1f}%",
            f"⏱️  Processing time: {processing_time:.2f} seconds"
        ]

        if errors:
            result_lines.append(f"⚠️  Errors encountered: {len(errors)}")
            if len(errors) <= 5:  # Show first few errors
                result_lines.extend([f"   - {error}" for error in errors[:5]])
            else:
                result_lines.extend([f"   - {error}" for error in errors[:3]])
                result_lines.append(f"   ... and {len(errors) - 3} more errors")

        logger.info(f"Document ingestion completed: {chunks_stored}/{chunks_processed} chunks stored")
        return [types.TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        logger.error(f"Error in document ingestion: {str(e)}")
        return [types.TextContent(
            type="text",
            text=f"Error ingesting document: {str(e)}"
        )]


async def handle_ingest_directory(server, arguments: dict) -> List[types.TextContent]:
    """Handle directory ingestion requests."""
    try:
        from pathlib import Path
        import time
        from ...services.memory_service import normalize_tags
        from ...utils.directory_ingestion import (
            DirectoryFileDiscovery,
            FileIngestionProcessor,
            IngestionResultFormatter
        )

        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()

        # Parse arguments
        directory_path = Path(arguments["directory_path"])
        tags = normalize_tags(arguments.get("tags", []))
        recursive = arguments.get("recursive", True)
        file_extensions = arguments.get("file_extensions", ["pdf", "txt", "md", "json"])
        chunk_size = arguments.get("chunk_size", 1000)
        max_files = arguments.get("max_files", 100)

        # Validate directory
        if not directory_path.exists() or not directory_path.is_dir():
            return [types.TextContent(
                type="text",
                text=f"Error: Directory not found: {directory_path}"
            )]

        logger.info(f"Starting directory ingestion: {directory_path}")
        start_time = time.time()

        # Discover files
        discovery = DirectoryFileDiscovery(
            directory_path=directory_path,
            file_extensions=file_extensions,
            recursive=recursive,
            max_files=max_files
        )
        files_to_process = discovery.discover_files()

        if not files_to_process:
            return [types.TextContent(
                type="text",
                text=f"No supported files found in directory: {directory_path}"
            )]

        # Process files
        processor = FileIngestionProcessor(
            storage=storage,
            chunk_size=chunk_size,
            base_tags=tags
        )

        for index, file_path in enumerate(files_to_process, start=1):
            await processor.process_file(
                file_path=file_path,
                file_index=index,
                total_files=len(files_to_process),
                directory_name=directory_path.name
            )

        # Format results
        processing_time = time.time() - start_time
        stats = processor.get_statistics()

        result_lines = IngestionResultFormatter.format_result(
            directory_name=directory_path.name,
            files_processed=stats["files_processed"],
            total_files=len(files_to_process),
            total_chunks_processed=stats["total_chunks_processed"],
            total_chunks_stored=stats["total_chunks_stored"],
            files_failed=stats["files_failed"],
            all_errors=stats["all_errors"],
            processing_time=processing_time
        )

        logger.info(f"Directory ingestion completed: {stats['total_chunks_stored']}/{stats['total_chunks_processed']} chunks from {stats['files_processed']} files")
        return [types.TextContent(type="text", text="\n".join(result_lines))]

    except Exception as e:
        logger.error(f"Error in directory ingestion: {str(e)}")
        return [types.TextContent(
            type="text",
            text=f"Error ingesting directory: {str(e)}"
        )]
