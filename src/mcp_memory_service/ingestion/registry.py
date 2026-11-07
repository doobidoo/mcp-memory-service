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
Document loader registry for automatic format detection and loader selection.
"""

import logging
import mimetypes
from pathlib import Path
from typing import Dict, Type, List, Optional

from .base import DocumentLoader

logger = logging.getLogger(__name__)


from ..compat import _sanitize_log_value

# Registry of document loaders by file extension
_LOADER_REGISTRY: Dict[str, Type[DocumentLoader]] = {}

# Supported file formats
SUPPORTED_FORMATS = {
    'pdf': 'PDF documents',
    'docx': 'Word documents (requires semtools)',
    'doc': 'Word documents (requires semtools)',
    'pptx': 'PowerPoint presentations (requires semtools)',
    'xlsx': 'Excel spreadsheets (requires semtools)',
    'txt': 'Plain text files',
    'md': 'Markdown documents',
    'json': 'JSON data files',
    'csv': 'CSV data files',
}


def register_loader(loader_class: Type[DocumentLoader], extensions: List[str]) -> None:
    """
    Register a document loader for specific file extensions.
    
    Args:
        loader_class: The DocumentLoader subclass to register
        extensions: List of file extensions this loader handles (without dots)
    """
    for ext in extensions:
        ext = ext.lower().lstrip('.')
        _LOADER_REGISTRY[ext] = loader_class
        logger.debug("Registered %s for .%s files", _sanitize_log_value(loader_class.__name__), _sanitize_log_value(ext))


def get_loader_for_file(file_path: Path) -> Optional[DocumentLoader]:
    """
    Get appropriate document loader for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        DocumentLoader instance that can handle the file, or None if unsupported
    """
    # Resolve to an absolute, canonical path to prevent path traversal
    try:
        resolved_path = file_path.resolve()  # codeql[py/path-injection]
    except (OSError, ValueError) as exc:
        logger.warning("Could not resolve file path: %s", _sanitize_log_value(exc))
        return None

    if not resolved_path.exists():  # codeql[py/path-injection]
        logger.warning("File does not exist: %s", _sanitize_log_value(resolved_path))
        return None

    # Work with the resolved path from here on
    file_path = resolved_path

    # Try by file extension first
    extension = file_path.suffix.lower().lstrip('.')
    if extension in _LOADER_REGISTRY:
        loader_class = _LOADER_REGISTRY[extension]
        loader = loader_class()
        if loader.can_handle(file_path):
            return loader

    # Try by MIME type detection
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type:
        loader = _get_loader_by_mime_type(mime_type)
        if loader and loader.can_handle(file_path):
            return loader

    # Try all registered loaders as fallback
    for loader_class in _LOADER_REGISTRY.values():
        loader = loader_class()
        if loader.can_handle(file_path):
            logger.debug("Found fallback loader %s for %s", loader_class.__name__, _sanitize_log_value(file_path))
            return loader

    logger.warning("No suitable loader found for file: %s", _sanitize_log_value(file_path))
    return None


def _get_loader_by_mime_type(mime_type: str) -> Optional[DocumentLoader]:
    """
    Get loader based on MIME type.
    
    Args:
        mime_type: MIME type string
        
    Returns:
        DocumentLoader instance or None
    """
    mime_to_extension = {
        'application/pdf': 'pdf',
        'text/plain': 'txt',
        'text/markdown': 'md',
        'application/json': 'json',
        'text/csv': 'csv',
    }
    
    extension = mime_to_extension.get(mime_type)
    if extension and extension in _LOADER_REGISTRY:
        loader_class = _LOADER_REGISTRY[extension]
        return loader_class()
    
    return None


def get_supported_extensions() -> List[str]:
    """
    Get list of all supported file extensions.
    
    Returns:
        List of supported extensions (without dots)
    """
    return list(_LOADER_REGISTRY.keys())


def is_supported_file(file_path: Path) -> bool:
    """
    Check if a file format is supported.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file format is supported
    """
    return get_loader_for_file(file_path) is not None


def list_registered_loaders() -> Dict[str, str]:
    """
    Get mapping of extensions to loader class names.
    
    Returns:
        Dictionary mapping extensions to loader class names
    """
    return {ext: loader_class.__name__ for ext, loader_class in _LOADER_REGISTRY.items()}
