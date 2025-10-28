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
Document Ingestion Module

Provides functionality to side-load documents into the memory database,
supporting multiple formats including PDF, text, and structured data.

This module enables users to pre-populate the vector database with
documentation, knowledge bases, and other content for semantic retrieval.
"""

from .base import DocumentLoader, DocumentChunk, IngestionResult
from .chunker import TextChunker
from .registry import get_loader_for_file, register_loader, SUPPORTED_FORMATS, is_supported_file

# Import loaders to trigger registration
# Order matters! Import SemtoolsLoader first, then specialized loaders
# This allows specialized loaders to override if semtools is unavailable
from . import text_loader
from . import semtools_loader
from . import pdf_loader
from . import json_loader
from . import csv_loader

__all__ = [
    'DocumentLoader',
    'DocumentChunk', 
    'IngestionResult',
    'TextChunker',
    'get_loader_for_file',
    'register_loader',
    'SUPPORTED_FORMATS',
    'is_supported_file'
]