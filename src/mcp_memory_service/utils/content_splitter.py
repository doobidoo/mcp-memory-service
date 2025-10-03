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
Content splitting utility for backend-specific length limits.

Provides intelligent content chunking that respects natural boundaries
like sentences, paragraphs, and code blocks to maintain readability.
"""

import re
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def split_content(
    content: str,
    max_length: int,
    preserve_boundaries: bool = True,
    overlap: int = 50
) -> List[str]:
    """
    Split content into chunks respecting natural boundaries.

    Args:
        content: The content to split
        max_length: Maximum length for each chunk
        preserve_boundaries: If True, respect sentence/paragraph boundaries
        overlap: Number of characters to overlap between chunks (for context)

    Returns:
        List of content chunks

    Example:
        >>> content = "First sentence. Second sentence. Third sentence."
        >>> chunks = split_content(content, max_length=30, preserve_boundaries=True)
        >>> len(chunks)
        2
    """
    if not content:
        return []

    if len(content) <= max_length:
        return [content]

    logger.info(f"Splitting content of {len(content)} chars into chunks of max {max_length} chars")

    if not preserve_boundaries:
        # Simple character-based splitting with overlap
        return _split_by_characters(content, max_length, overlap)

    # Intelligent splitting that respects boundaries
    return _split_preserving_boundaries(content, max_length, overlap)


def _split_by_characters(content: str, max_length: int, overlap: int) -> List[str]:
    """Split content by character count with overlap."""
    chunks = []
    start = 0

    while start < len(content):
        end = start + max_length
        chunk = content[start:end]
        chunks.append(chunk)

        # Move start position with overlap
        start = end - overlap if end < len(content) else end

    return chunks


def _split_preserving_boundaries(content: str, max_length: int, overlap: int) -> List[str]:
    """
    Split content while preserving natural boundaries.

    Priority order for split points:
    1. Double newlines (paragraph breaks)
    2. Single newlines
    3. Sentence endings (. ! ? followed by space)
    4. Spaces (word boundaries)
    5. Character position (last resort)
    """
    chunks = []
    remaining = content

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Find the best split point within max_length
        split_point = _find_best_split_point(remaining, max_length)

        # Extract chunk and prepare next iteration
        chunk = remaining[:split_point].rstrip()
        chunks.append(chunk)

        # Calculate overlap start (go back overlap characters but respect boundaries)
        overlap_start = max(0, split_point - overlap)
        # Find a good boundary for overlap start if possible
        if overlap > 0 and overlap_start > 0:
            # Try to start overlap at a space
            space_pos = remaining[overlap_start:split_point].find(' ')
            if space_pos != -1:
                overlap_start += space_pos + 1

        remaining = remaining[overlap_start:].lstrip()

        # Prevent infinite loop in edge cases
        if not remaining or len(chunk) == 0:
            break

    return chunks


def _find_best_split_point(text: str, max_length: int) -> int:
    """
    Find the best position to split text within max_length.

    Returns the character index where the split should occur.
    """
    # If text is within limit, return full length
    if len(text) <= max_length:
        return len(text)

    # Search window (look back up to 20% for a good boundary)
    search_start = max(int(max_length * 0.8), max_length - 100)
    search_text = text[search_start:max_length]

    # Priority 1: Double newline (paragraph break)
    para_match = search_text.rfind('\n\n')
    if para_match != -1:
        return search_start + para_match

    # Priority 2: Single newline
    newline_match = search_text.rfind('\n')
    if newline_match != -1:
        return search_start + newline_match

    # Priority 3: Sentence ending
    # Look for '. ', '! ', '? ' patterns
    sentence_pattern = r'[.!?]\s'
    matches = list(re.finditer(sentence_pattern, search_text))
    if matches:
        last_match = matches[-1]
        return search_start + last_match.end()

    # Priority 4: Word boundary (space)
    space_match = search_text.rfind(' ')
    if space_match != -1:
        return search_start + space_match + 1

    # Priority 5: Hard cutoff at max_length (last resort)
    return max_length


def estimate_chunks_needed(content_length: int, max_length: int) -> int:
    """
    Estimate the number of chunks needed for content of given length.

    Args:
        content_length: Length of content to split
        max_length: Maximum length per chunk

    Returns:
        Estimated number of chunks
    """
    import math
    return max(1, math.ceil(content_length / max_length))


def validate_chunk_lengths(chunks: List[str], max_length: int) -> bool:
    """
    Validate that all chunks are within the specified length limit.

    Args:
        chunks: List of content chunks
        max_length: Maximum allowed length

    Returns:
        True if all chunks are valid, False otherwise
    """
    for i, chunk in enumerate(chunks):
        if len(chunk) > max_length:
            logger.error(f"Chunk {i} exceeds max length: {len(chunk)} > {max_length}")
            return False
    return True
