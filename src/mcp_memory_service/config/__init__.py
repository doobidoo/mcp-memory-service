"""Configuration package for MCP Memory Service.

Split from monolithic config.py for maintainability.
All symbols are re-exported here for backward compatibility:
    from mcp_memory_service.config import X  # still works
"""

from .base import *  # noqa: F401,F403
from .storage import *  # noqa: F401,F403
from .embedding import *  # noqa: F401,F403
from .transport import *  # noqa: F401,F403
from .oauth import *  # noqa: F401,F403
from .oauth import _load_pem_from_env  # noqa: F401 — tests import this directly
# The /metrics opt-in flag is env-driven and the tests toggle MCP_METRICS_ENABLED
# then reload *this package* (not the submodule). A bare ``from .metrics import *``
# would re-bind the stale, already-imported submodule attribute, so force the
# submodule to re-evaluate against the current environment on every (re)import.
import importlib as _importlib
from . import metrics as _metrics_mod
_importlib.reload(_metrics_mod)
from .metrics import *  # noqa: F401,F403,E402
from .documents import *  # noqa: F401,F403
from .backup import *  # noqa: F401,F403
from .consolidation import *  # noqa: F401,F403
from .quality import *  # noqa: F401,F403
from .search import *  # noqa: F401,F403
from .graph import *  # noqa: F401,F403
from .ontology import *  # noqa: F401,F403
from .validation import *  # noqa: F401,F403
