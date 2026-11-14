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
Directory ingestion processing utilities.

Extracted from server/handlers/documents.py Phase 3.3 refactoring to reduce
handle_ingest_directory complexity.
"""

import logging
from pathlib import Path
from typing import List, Set, Any

from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)


class DirectoryFileDiscovery:
    """Discover and filter files in a directory for ingestion."""

    def __init__(self, directory_path: Path, file_extensions: List[str],
                 recursive: bool = True, max_files: int = 100):
        """
        Initialize directory file discovery.

        Args:
            directory_path: Path to directory to search
            file_extensions: List of file extensions to include (e.g., ['pdf', 'txt'])
            recursive: Whether to search subdirectories
            max_files: Maximum number of files to discover
        """
        self.directory_path = directory_path
        self.file_extensions = file_extensions
        self.recursive = recursive
        self.max_files = max_files

    def discover_files(self) -> List[Path]:
        """
        Discover all supported files in directory.

        Returns:
            List of unique file paths, limited to max_files
        """
        from ..ingestion import is_supported_file

        all_files = []

        # Find files by extension
        for ext in self.file_extensions:
            ext_pattern = f"*.{ext.lstrip('.')}"
            if self.recursive:
                files = list(self.directory_path.rglob(ext_pattern))
            else:
                files = list(self.directory_path.glob(ext_pattern))
            all_files.extend(files)

        # Remove duplicates and filter supported files
        unique_files = []
        seen: Set[Path] = set()
        for file_path in all_files:
            if file_path not in seen and is_supported_file(file_path):
                unique_files.append(file_path)
                seen.add(file_path)

        # Limit number of files
        return unique_files[:self.max_files]


class FileIngestionProcessor:
    """Process individual files for directory ingestion."""

    def __init__(self, storage: Any, chunk_size: int, base_tags: List[str], store: str = "default"):
        """
        Initialize file ingestion processor.

        Args:
            storage: Storage backend instance
            chunk_size: Size of text chunks for processing
            base_tags: Base tags to apply to all memories
            store: Target store partition
        """
        self.storage = storage
        self.chunk_size = chunk_size
        self.base_tags = base_tags
        self.store = store
        self.logger = logging.getLogger(__name__)

        # Statistics
        self.total_chunks_processed = 0
        self.total_chunks_stored = 0
        self.files_processed = 0
        self.files_failed = 0
        self.all_errors: List[str] = []

    async def process_file(self, file_path: Path, file_index: int,
                          total_files: int, directory_name: str) -> None:
        """
        Process a single file and update statistics.

        Args:
            file_path: Path to file to process
            file_index: Current file index (1-based)
            total_files: Total number of files to process
            directory_name: Name of source directory (for tagging)
        """
        from ..ingestion import get_loader_for_file
        from .document_processing import _process_and_store_chunk

        try:
            self.logger.info(
                "Processing file %d/%d: %s",
                file_index,
                total_files,
                _sanitize_log_value(file_path.name),
            )

            # Get appropriate document loader
            loader = get_loader_for_file(file_path)
            if loader is None:
                self.all_errors.append(f"{file_path.name}: Unsupported format")
                self.files_failed += 1
                return

            # Configure loader
            loader.chunk_size = self.chunk_size

            file_chunks_processed = 0
            file_chunks_stored = 0

            # Extract and store chunks from this file
            async for chunk in loader.extract_chunks(file_path):
                file_chunks_processed += 1
                self.total_chunks_processed += 1

                # Process and store the chunk
                success, error = await _process_and_store_chunk(
                    chunk,
                    self.storage,
                    file_path.name,
                    base_tags=self.base_tags.copy(),
                    context_tags={
                        "source_dir": directory_name,
                        "file_type": file_path.suffix.lstrip('.')
                    },
                    store=self.store,
                )

                if success:
                    file_chunks_stored += 1
                    self.total_chunks_stored += 1
                else:
                    self.all_errors.append(error)

            if file_chunks_stored > 0:
                self.files_processed += 1
            else:
                self.files_failed += 1

        except Exception as e:
            self.files_failed += 1
            self.all_errors.append(f"{file_path.name}: {str(e)}")

    def get_statistics(self) -> dict:
        """
        Get processing statistics.

        Returns:
            Dictionary containing processing statistics
        """
        return {
            "total_chunks_processed": self.total_chunks_processed,
            "total_chunks_stored": self.total_chunks_stored,
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "all_errors": self.all_errors
        }


class IngestionResultFormatter:
    """Format directory ingestion results."""

    @staticmethod
    def format_result(directory_name: str, files_processed: int, total_files: int,
                     total_chunks_processed: int, total_chunks_stored: int,
                     files_failed: int, all_errors: List[str],
                     processing_time: float) -> List[str]:
        """
        Format ingestion results into human-readable lines.

        Args:
            directory_name: Name of ingested directory
            files_processed: Number of successfully processed files
            total_files: Total number of files attempted
            total_chunks_processed: Total chunks processed
            total_chunks_stored: Total chunks stored successfully
            files_failed: Number of failed files
            all_errors: List of error messages
            processing_time: Total processing time in seconds

        Returns:
            List of formatted result lines
        """
        success_rate = (total_chunks_stored / total_chunks_processed * 100) if total_chunks_processed > 0 else 0

        result_lines = [
            f"✅ Directory ingestion completed: {directory_name}",
            f"📁 Files processed: {files_processed}/{total_files}",
            f"📄 Total chunks processed: {total_chunks_processed}",
            f"💾 Total chunks stored: {total_chunks_stored}",
            f"⚡ Success rate: {success_rate:.1f}%",
            f"⏱️  Processing time: {processing_time:.2f} seconds"
        ]

        if files_failed > 0:
            result_lines.append(f"❌ Files failed: {files_failed}")

        if all_errors:
            result_lines.append(f"⚠️  Total errors: {len(all_errors)}")
            # Show first few errors
            error_limit = 5
            for error in all_errors[:error_limit]:
                result_lines.append(f"   - {error}")
            if len(all_errors) > error_limit:
                result_lines.append(f"   ... and {len(all_errors) - error_limit} more errors")

        return result_lines
