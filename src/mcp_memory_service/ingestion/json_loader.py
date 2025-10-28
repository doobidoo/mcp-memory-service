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
JSON document loader for structured data files.
"""

import json
import logging
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Union, List
import asyncio

from .base import DocumentLoader, DocumentChunk
from .chunker import TextChunker, ChunkingStrategy

logger = logging.getLogger(__name__)


class JSONLoader(DocumentLoader):
    """
    Document loader for JSON data files.

    Features:
    - Flattens nested JSON structures to searchable text
    - Preserves key-value context (e.g., "config.database.host: localhost")
    - Handles arrays and nested objects recursively
    - Supports configurable flattening strategies
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize JSON loader.

        Args:
            chunk_size: Target size for text chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        super().__init__(chunk_size, chunk_overlap)
        self.supported_extensions = ['json']

        self.chunker = TextChunker(ChunkingStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            respect_paragraph_boundaries=False,  # JSON doesn't have paragraphs
            respect_sentence_boundaries=False,   # JSON doesn't have sentences
            min_chunk_size=10  # Allow smaller chunks for structured data
        ))

    def can_handle(self, file_path: Path) -> bool:
        """
        Check if this loader can handle the given JSON file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this loader can process the JSON file
        """
        if not file_path.exists() or not file_path.is_file():
            return False

        extension = file_path.suffix.lower().lstrip('.')
        return extension in self.supported_extensions

    async def extract_chunks(self, file_path: Path, **kwargs) -> AsyncGenerator[DocumentChunk, None]:
        """
        Extract text chunks from a JSON file.

        Args:
            file_path: Path to the JSON file
            **kwargs: Additional options:
                - flatten_strategy: How to flatten nested structures ('dot_notation', 'bracket_notation')
                - max_depth: Maximum nesting depth to flatten (default: unlimited)
                - include_types: Whether to include value types in flattened text (default: False)
                - array_handling: How to handle arrays ('expand', 'summarize', 'flatten')

        Yields:
            DocumentChunk objects containing extracted text and metadata

        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            ValueError: If the JSON file can't be parsed or processed
        """
        await self.validate_file(file_path)

        flatten_strategy = kwargs.get('flatten_strategy', 'dot_notation')
        max_depth = kwargs.get('max_depth', None)
        include_types = kwargs.get('include_types', False)
        array_handling = kwargs.get('array_handling', 'expand')

        logger.info(f"Extracting chunks from JSON file: {file_path}")

        try:
            # Read and parse JSON
            data, encoding = await self._read_json_file(file_path)

            # Flatten the JSON structure
            flattened_text = self._flatten_json(
                data,
                flatten_strategy=flatten_strategy,
                max_depth=max_depth,
                include_types=include_types,
                array_handling=array_handling
            )

            # Create base metadata
            base_metadata = self.get_base_metadata(file_path)
            base_metadata.update({
                'encoding': encoding,
                'content_type': 'json',
                'flatten_strategy': flatten_strategy,
                'array_handling': array_handling,
                'include_types': include_types,
                'max_depth': max_depth,
                'original_keys_count': self._count_keys(data),
                'flattened_text_length': len(flattened_text)
            })

            # Chunk the flattened text
            chunks = self.chunker.chunk_text(flattened_text, base_metadata)

            for i, (chunk_text, chunk_metadata) in enumerate(chunks):
                yield DocumentChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    chunk_index=i,
                    source_file=file_path
                )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {str(e)}")
            raise ValueError(f"Invalid JSON format: {str(e)}") from e
        except Exception as e:
            logger.error(f"Error extracting from JSON file {file_path}: {type(e).__name__} - {str(e)}")
            raise ValueError(f"Failed to extract JSON content: {str(e)}") from e

    async def _read_json_file(self, file_path: Path) -> tuple:
        """
        Read and parse JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Tuple of (parsed_data, encoding_used)
        """
        def _read_sync():
            # Try UTF-8 first (most common for JSON)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                data = json.loads(content)
                return data, 'utf-8'
            except UnicodeDecodeError:
                # Fallback to other encodings
                encodings_to_try = ['utf-16', 'utf-32', 'latin-1']
                for encoding in encodings_to_try:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            content = file.read()
                        data = json.loads(content)
                        return data, encoding
                    except UnicodeDecodeError:
                        continue
                    except json.JSONDecodeError:
                        continue

                # Last resort with error replacement
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    content = file.read()
                data = json.loads(content)
                return data, 'utf-8'

        # Run file reading in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _read_sync)

    def _flatten_json(
        self,
        data: Any,
        prefix: str = "",
        flatten_strategy: str = 'dot_notation',
        max_depth: int = None,
        current_depth: int = 0,
        include_types: bool = False,
        array_handling: str = 'expand'
    ) -> str:
        """
        Flatten JSON structure to searchable text.

        Args:
            data: JSON data to flatten
            prefix: Current key prefix
            flatten_strategy: Flattening strategy
            max_depth: Maximum depth to flatten
            current_depth: Current nesting depth
            include_types: Whether to include value types
            array_handling: How to handle arrays

        Returns:
            Flattened text representation
        """
        if max_depth is not None and current_depth >= max_depth:
            return f"{prefix}: [nested structure truncated at depth {max_depth}]\n"

        lines = []

        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = self._build_prefix(prefix, key, flatten_strategy)
                flattened_value = self._flatten_json(
                    value, new_prefix, flatten_strategy, max_depth,
                    current_depth + 1, include_types, array_handling
                )
                lines.append(flattened_value)
        elif isinstance(data, list):
            if array_handling == 'summarize':
                lines.append(f"{prefix}: [array with {len(data)} items]\n")
            elif array_handling == 'flatten':
                # Flatten all array items with indexed keys
                for i, item in enumerate(data):
                    new_prefix = self._build_prefix(prefix, str(i), flatten_strategy)
                    flattened_item = self._flatten_json(
                        item, new_prefix, flatten_strategy, max_depth,
                        current_depth + 1, include_types, array_handling
                    )
                    lines.append(flattened_item)
            else:  # 'expand' - default
                # Expand arrays as separate entries
                for i, item in enumerate(data):
                    if isinstance(item, (dict, list)):
                        new_prefix = f"{prefix}[{i}]"
                        flattened_item = self._flatten_json(
                            item, new_prefix, flatten_strategy, max_depth,
                            current_depth + 1, include_types, array_handling
                        )
                        lines.append(flattened_item)
                    else:
                        # Simple values in arrays
                        type_info = f" ({type(item).__name__})" if include_types else ""
                        lines.append(f"{prefix}[{i}]: {item}{type_info}\n")
        else:
            # Primitive values
            type_info = f" ({type(data).__name__})" if include_types else ""
            lines.append(f"{prefix}: {data}{type_info}\n")

        return "".join(lines)

    def _build_prefix(self, current_prefix: str, key: str, strategy: str) -> str:
        """
        Build the prefix for nested keys.

        Args:
            current_prefix: Current prefix
            key: Key to add
            strategy: Flattening strategy

        Returns:
            New prefix string
        """
        if not current_prefix:
            return key

        if strategy == 'bracket_notation':
            return f"{current_prefix}[{key}]"
        else:  # 'dot_notation' - default
            return f"{current_prefix}.{key}"

    def _count_keys(self, data: Any) -> int:
        """
        Count total number of keys in JSON structure.

        Args:
            data: JSON data to count keys in

        Returns:
            Total number of keys
        """
        if isinstance(data, dict):
            count = len(data)
            for value in data.values():
                count += self._count_keys(value)
            return count
        elif isinstance(data, list):
            count = 0
            for item in data:
                count += self._count_keys(item)
            return count
        else:
            return 0


# Register the JSON loader
def _register_json_loader():
    """Register JSON loader with the registry."""
    try:
        from .registry import register_loader
        register_loader(JSONLoader, ['json'])
        logger.debug("JSON loader registered successfully")
    except ImportError:
        logger.debug("Registry not available during import")


# Auto-register when module is imported
_register_json_loader()
