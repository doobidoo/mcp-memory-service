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
# then reload *this package* (not the submodule). ``importlib.reload(config_pkg)``
# re-runs the import below but does NOT re-execute the already-imported
# ``config.metrics`` submodule, so its module-level ``METRICS_ENABLED`` would
# stay bound to the stale (import-time) value. Force the submodule to
# re-evaluate against the current environment, then import the flag explicitly
# (no ``import *`` — the submodule's ``__all__`` is just METRICS_ENABLED anyway,
# but an explicit import keeps the config namespace clean and lint-clean).
import importlib as _importlib
from . import metrics as _metrics_mod
_importlib.reload(_metrics_mod)
from .metrics import METRICS_ENABLED  # noqa: F401,E402
from .documents import *  # noqa: F401,F403
from .backup import *  # noqa: F401,F403
from .consolidation import *  # noqa: F401,F403
from .quality import *  # noqa: F401,F403
from .search import *  # noqa: F401,F403
from .graph import *  # noqa: F401,F403
from .ontology import *  # noqa: F401,F403
from .validation import *  # noqa: F401,F403
