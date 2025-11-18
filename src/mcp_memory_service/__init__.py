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

"""MCP Memory Service initialization."""

# CRITICAL: Set offline mode BEFORE any other imports to prevent model downloads
import os
import platform

# Force offline mode for HuggingFace models - this MUST be done before any ML library imports
def setup_offline_mode():
    """Setup offline mode environment variables to prevent model downloads."""
    # Set offline environment variables
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    # Configure cache paths
    username = os.environ.get('USERNAME', os.environ.get('USER', ''))
    if platform.system() == "Windows" and username:
        default_hf_home = f"C:\\Users\\{username}\\.cache\\huggingface"
        default_transformers_cache = f"C:\\Users\\{username}\\.cache\\huggingface\\transformers"
        default_sentence_transformers_home = f"C:\\Users\\{username}\\.cache\\torch\\sentence_transformers"
    else:
        default_hf_home = os.path.expanduser("~/.cache/huggingface")
        default_transformers_cache = os.path.expanduser("~/.cache/huggingface/transformers")
        default_sentence_transformers_home = os.path.expanduser("~/.cache/torch/sentence_transformers")
    
    # Set cache paths if not already set
    if 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = default_hf_home
    if 'TRANSFORMERS_CACHE' not in os.environ:
        os.environ['TRANSFORMERS_CACHE'] = default_transformers_cache
    if 'SENTENCE_TRANSFORMERS_HOME' not in os.environ:
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = default_sentence_transformers_home

# Setup offline mode immediately when this module is imported
setup_offline_mode()

__version__ = "8.27.2"

from .models import Memory, MemoryQueryResult
from .storage import MemoryStorage
from .utils import generate_content_hash

# Conditional imports
__all__ = [
    'Memory',
    'MemoryQueryResult', 
    'MemoryStorage',
    'generate_content_hash'
]

# Import storage backends conditionally
try:
    from .storage import SqliteVecMemoryStorage
    __all__.append('SqliteVecMemoryStorage')
except ImportError:
    SqliteVecMemoryStorage = None

