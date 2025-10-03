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

from .base import MemoryStorage

# Conditional imports based on available dependencies
__all__ = ['MemoryStorage']

try:
    from .chroma import ChromaMemoryStorage
    __all__.append('ChromaMemoryStorage')
except ImportError:
    ChromaMemoryStorage = None

try:
    from .sqlite_vec import SqliteVecMemoryStorage
    __all__.append('SqliteVecMemoryStorage')
except ImportError:
    SqliteVecMemoryStorage = None

try:
    from .cloudflare import CloudflareStorage
    __all__.append('CloudflareStorage')
except ImportError:
    CloudflareStorage = None

try:
    from .hybrid import HybridMemoryStorage
    __all__.append('HybridMemoryStorage')
except ImportError:
    HybridMemoryStorage = None