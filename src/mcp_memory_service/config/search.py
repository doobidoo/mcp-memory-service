"""Search configuration — summarization policy, hybrid fusion, and weights."""
import os
import logging

from ..compat import _sanitize_log_value
from .base import safe_get_int_env, safe_get_bool_env

logger = logging.getLogger(__name__)

# Operator permission to send search results to the Harvest LLM providers.
# Independent of provider configuration and the caller's summarize flag.
MCP_SEARCH_SUMMARIZE_ENABLED = safe_get_bool_env('MCP_SEARCH_SUMMARIZE_ENABLED', False)

# =============================================================================
# Hybrid Search Configuration (v10.8.0+)
# =============================================================================

# Enable hybrid BM25 + Vector search
MCP_HYBRID_SEARCH_ENABLED = safe_get_bool_env('MCP_HYBRID_SEARCH_ENABLED', True)

# Fusion method: 'weighted_average' (default, legacy) or 'rrf' (Reciprocal Rank Fusion)
MCP_HYBRID_FUSION_METHOD = os.getenv('MCP_HYBRID_FUSION_METHOD', 'weighted_average').lower()
if MCP_HYBRID_FUSION_METHOD not in ('weighted_average', 'rrf'):
    logger.warning("Invalid fusion method: %s. Using 'weighted_average'", _sanitize_log_value(MCP_HYBRID_FUSION_METHOD))
    MCP_HYBRID_FUSION_METHOD = 'weighted_average'

# RRF parameters (only used when fusion_method='rrf')
MCP_HYBRID_RRF_K = safe_get_int_env('MCP_HYBRID_RRF_K', 60, min_value=1, max_value=1000)
MCP_HYBRID_RRF_CONSENSUS_BOOST = float(os.getenv('MCP_HYBRID_RRF_CONSENSUS_BOOST', '0.1'))

# Score fusion weights (must sum to 1.0)
MCP_HYBRID_KEYWORD_WEIGHT = float(os.getenv('MCP_HYBRID_KEYWORD_WEIGHT', '0.3'))
MCP_HYBRID_SEMANTIC_WEIGHT = float(os.getenv('MCP_HYBRID_SEMANTIC_WEIGHT', '0.7'))

# Mistake Notes configuration
MCP_MISTAKE_NOTE_DEDUP_THRESHOLD = max(0.0, min(1.0, float(os.getenv('MCP_MISTAKE_NOTE_DEDUP_THRESHOLD', '0.85'))))

# Validate weights
if not 0.0 <= MCP_HYBRID_KEYWORD_WEIGHT <= 1.0:
    logger.warning("Invalid keyword weight: %s. Using default 0.3", MCP_HYBRID_KEYWORD_WEIGHT)
    MCP_HYBRID_KEYWORD_WEIGHT = 0.3

if not 0.0 <= MCP_HYBRID_SEMANTIC_WEIGHT <= 1.0:
    logger.warning("Invalid semantic weight: %s. Using default 0.7", MCP_HYBRID_SEMANTIC_WEIGHT)
    MCP_HYBRID_SEMANTIC_WEIGHT = 0.7

# Warn if weights don't sum to 1.0 (within tolerance)
weight_sum = MCP_HYBRID_KEYWORD_WEIGHT + MCP_HYBRID_SEMANTIC_WEIGHT
if abs(weight_sum - 1.0) > 0.01:
    logger.warning("Hybrid weights sum to %s, expected 1.0. Normalizing...", weight_sum)
    total = MCP_HYBRID_KEYWORD_WEIGHT + MCP_HYBRID_SEMANTIC_WEIGHT
    MCP_HYBRID_KEYWORD_WEIGHT /= total
    MCP_HYBRID_SEMANTIC_WEIGHT /= total

if MCP_HYBRID_FUSION_METHOD == 'rrf':
    logger.info("Hybrid Search: enabled=%s, fusion=%s, keyword_weight=%.2f, semantic_weight=%.2f, rrf_k=%s, consensus_boost=%s",
                MCP_HYBRID_SEARCH_ENABLED, _sanitize_log_value(MCP_HYBRID_FUSION_METHOD),
                MCP_HYBRID_KEYWORD_WEIGHT, MCP_HYBRID_SEMANTIC_WEIGHT,
                MCP_HYBRID_RRF_K, MCP_HYBRID_RRF_CONSENSUS_BOOST)
else:
    logger.info("Hybrid Search: enabled=%s, fusion=%s, keyword_weight=%.2f, semantic_weight=%.2f",
                MCP_HYBRID_SEARCH_ENABLED, _sanitize_log_value(MCP_HYBRID_FUSION_METHOD),
                MCP_HYBRID_KEYWORD_WEIGHT, MCP_HYBRID_SEMANTIC_WEIGHT)

# =============================================================================
# End Hybrid Search Configuration
# =============================================================================
