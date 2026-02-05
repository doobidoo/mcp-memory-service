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

# CRITICAL: Set cache paths BEFORE any ML library imports to prevent model downloads
import os

from platformdirs import user_cache_dir


def setup_ml_cache_paths():
    """
    Setup cache paths for HuggingFace/PyTorch models using platformdirs.

    Uses platform-appropriate cache locations:
    - Linux: ~/.cache/huggingface (XDG_CACHE_HOME compliant)
    - macOS: ~/Library/Caches/huggingface
    - Windows: C:\\Users\\<user>\\AppData\\Local\\huggingface\\Cache

    Environment variables take precedence if already set.
    """
    # HuggingFace cache (unified cache for transformers/datasets/hub)
    # Note: TRANSFORMERS_CACHE is deprecated in favor of HF_HOME in transformers v5+
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = user_cache_dir("huggingface", ensure_exists=True)

    # Sentence-transformers cache (uses torch cache structure)
    if "SENTENCE_TRANSFORMERS_HOME" not in os.environ:
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(
            user_cache_dir("torch", ensure_exists=True), "sentence_transformers"
        )


# Setup cache paths immediately when this module is imported
setup_ml_cache_paths()

__version__ = "10.0.0"

from .models import Memory, MemoryQueryResult  # noqa: E402
from .storage import MemoryStorage  # noqa: E402
from .utils import generate_content_hash  # noqa: E402

# Conditional imports
__all__ = ["Memory", "MemoryQueryResult", "MemoryStorage", "generate_content_hash"]

# Import storage backends conditionally
try:
    from .storage import SqliteVecMemoryStorage

    __all__.append("SqliteVecMemoryStorage")
except ImportError:
    SqliteVecMemoryStorage = None
