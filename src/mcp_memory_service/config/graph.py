"""Graph database configuration — storage mode, typed edges."""
import os
import logging

from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)

# =============================================================================
# Graph Database Configuration (v8.51.0+)
# =============================================================================

# Graph storage mode controls how memory associations are stored
# Options:
#   - 'memories_only': Store associations in memories.metadata.associations (backward compatible, v8.48.0 behavior)
#   - 'dual_write': Write to both memories.metadata.associations AND memory_graph table (migration mode, default)
#   - 'graph_only': Write to memory_graph table only (future mode, requires migration complete)
GRAPH_STORAGE_MODE = os.getenv('MCP_GRAPH_STORAGE_MODE', 'dual_write').lower()

# Validate graph storage mode
VALID_GRAPH_MODES = ['memories_only', 'dual_write', 'graph_only']
if GRAPH_STORAGE_MODE not in VALID_GRAPH_MODES:
    logger.warning(
        "Invalid graph storage mode: %s, must be one of %s. Using default 'dual_write'",
        _sanitize_log_value(GRAPH_STORAGE_MODE),
        VALID_GRAPH_MODES,
    )
    GRAPH_STORAGE_MODE = 'dual_write'

logger.info("Graph Storage Mode: %s", GRAPH_STORAGE_MODE)

# Whether consolidation should write association entries to the memories table.
# Associations are already stored in memory_graph (the structured store).
# Set to false to avoid search-result pollution and wasted embedding computation.
# Default: true for backward compatibility.
CONSOLIDATION_STORE_ASSOCIATIONS = os.getenv(
    'MCP_CONSOLIDATION_STORE_ASSOCIATIONS', 'true'
).lower() == 'true'
logger.info("Consolidation store associations in memories table: %s", CONSOLIDATION_STORE_ASSOCIATIONS)

# Whether consolidation supersedes the older memory of a pair that relationship
# inference labels "contradicts" (confidence >= 0.75), which hides it from
# default retrieval. The typed edge is written either way; set to false to keep
# the edges and leave both memories visible.
# Default: true for backward compatibility.
CONSOLIDATION_AUTO_SUPERSEDE = os.getenv(
    'MCP_CONSOLIDATION_AUTO_SUPERSEDE', 'true'
).lower() == 'true'
logger.info("Consolidation auto-supersede on contradicts edges: %s", CONSOLIDATION_AUTO_SUPERSEDE)

# Whether the RelationshipInferenceEngine assigns typed edges (fixes, causes,
# contradicts, etc.) during consolidation. Set to false to keep all inferred
# edges as "related", avoiding false-positive typed labels.
# Default: true for backward compatibility.
TYPED_EDGES_ENABLED = os.getenv(
    'MCP_TYPED_EDGES_ENABLED', 'true'
).lower() == 'true'
logger.info("Typed edge inference enabled: %s", TYPED_EDGES_ENABLED)

# =============================================================================
# End Graph Database Configuration
# =============================================================================
