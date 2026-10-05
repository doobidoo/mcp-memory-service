"""Public ontology API.

The taxonomy and validators live in ``_ontology_data``. Pair rules from issue
#1458 are implemented in ``relationship_pairs`` and re-exported here so callers
keep using ``mcp_memory_service.models.ontology``.
"""

from mcp_memory_service.models import _ontology_data as _base
from mcp_memory_service.models._ontology_data import *  # noqa: F401,F403
from mcp_memory_service.models.relationship_pairs import (
    clear_pair_cache,
    is_allowed_pair,
)

def clear_ontology_caches():
    """Clear taxonomy caches and the parsed relationship-pair cache."""
    _base.clear_ontology_caches()
    clear_pair_cache()
