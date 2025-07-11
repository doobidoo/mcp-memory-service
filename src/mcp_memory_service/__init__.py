"""MCP Memory Service initialization."""

__version__ = "0.1.0"

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
    from .storage import ChromaMemoryStorage
    __all__.append('ChromaMemoryStorage')
except ImportError:
    ChromaMemoryStorage = None

try:
    from .storage import SqliteVecMemoryStorage
    __all__.append('SqliteVecMemoryStorage')
except ImportError:
    SqliteVecMemoryStorage = None

