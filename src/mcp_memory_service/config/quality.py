"""Quality system configuration — scoring, retention, maintenance."""
import os
import logging

from ..compat import _sanitize_log_value
from .base import safe_get_int_env, safe_get_bool_env

logger = logging.getLogger(__name__)

# =============================================================================
# Quality System Configuration (Memento-Inspired Quality System)
# =============================================================================

# Quality system master toggle
MCP_QUALITY_SYSTEM_ENABLED = safe_get_bool_env('MCP_QUALITY_SYSTEM_ENABLED', True)

# Quality scoring provider configuration
# Options: 'local' (ONNX ranker), 'groq', 'gemini', 'auto' (fallback chain), 'none' (disabled)
MCP_QUALITY_AI_PROVIDER = os.getenv('MCP_QUALITY_AI_PROVIDER', 'local').lower()

# Local ONNX model configuration
MCP_QUALITY_LOCAL_MODEL = os.getenv('MCP_QUALITY_LOCAL_MODEL', 'ms-marco-MiniLM-L-6-v2')
MCP_QUALITY_LOCAL_DEVICE = os.getenv('MCP_QUALITY_LOCAL_DEVICE', 'auto').lower()  # auto|cpu|cuda|mps|directml

# Quality-Boosted Search Configuration
MCP_QUALITY_BOOST_ENABLED = safe_get_bool_env('MCP_QUALITY_BOOST_ENABLED', False)  # Opt-in by default
MCP_QUALITY_BOOST_WEIGHT = float(os.getenv('MCP_QUALITY_BOOST_WEIGHT', '0.3'))  # 30% quality, 70% semantic

# Validate quality boost weight
if not 0.0 <= MCP_QUALITY_BOOST_WEIGHT <= 1.0:
    logger.warning("Invalid quality boost weight: %s, must be 0.0-1.0. Using default 0.3", MCP_QUALITY_BOOST_WEIGHT)
    MCP_QUALITY_BOOST_WEIGHT = 0.3

# Quality-Based Retention Policy (Consolidation)
MCP_QUALITY_RETENTION_HIGH = safe_get_int_env('MCP_QUALITY_RETENTION_HIGH', 365, min_value=1, max_value=3650)       # days for quality ≥0.7
MCP_QUALITY_RETENTION_MEDIUM = safe_get_int_env('MCP_QUALITY_RETENTION_MEDIUM', 180, min_value=1, max_value=3650)  # days for quality 0.5-0.7
MCP_QUALITY_RETENTION_LOW_MIN = safe_get_int_env('MCP_QUALITY_RETENTION_LOW_MIN', 30, min_value=1, max_value=365)  # minimum days for quality <0.5
MCP_QUALITY_RETENTION_LOW_MAX = safe_get_int_env('MCP_QUALITY_RETENTION_LOW_MAX', 90, min_value=1, max_value=365)  # maximum days for quality <0.5

# Log quality system configuration
logger.info("Quality System: enabled=%s, provider=%s", MCP_QUALITY_SYSTEM_ENABLED, _sanitize_log_value(MCP_QUALITY_AI_PROVIDER))
if MCP_QUALITY_SYSTEM_ENABLED:
    logger.info("Quality Boost Search: enabled=%s, weight=%s", MCP_QUALITY_BOOST_ENABLED, MCP_QUALITY_BOOST_WEIGHT)
    logger.info("Quality Retention: high=%sd, medium=%sd, low=%s-%sd",
                MCP_QUALITY_RETENTION_HIGH, MCP_QUALITY_RETENTION_MEDIUM,
                MCP_QUALITY_RETENTION_LOW_MIN, MCP_QUALITY_RETENTION_LOW_MAX)

# =============================================================================
# End Quality System Configuration
# =============================================================================

# =============================================================================
# Maintenance Configuration (memory_quality action="maintain")
# =============================================================================
MAINTAIN_STALE_DAYS = safe_get_int_env('MCP_MAINTAIN_STALE_DAYS', 30, min_value=1, max_value=3650)
# WARNING: auto_resolve=true enables automatic conflict resolution — memories above
# the similarity threshold will be silently merged. Use with caution at scale.
MAINTAIN_AUTO_RESOLVE = safe_get_bool_env('MCP_MAINTAIN_AUTO_RESOLVE', False)
try:
    MAINTAIN_AUTO_RESOLVE_THRESHOLD = float(os.getenv('MCP_MAINTAIN_AUTO_RESOLVE_THRESHOLD', '0.95'))
except (ValueError, TypeError):
    logger.error("Invalid value for MCP_MAINTAIN_AUTO_RESOLVE_THRESHOLD, using default 0.95")
    MAINTAIN_AUTO_RESOLVE_THRESHOLD = 0.95
# Two-signal guard: only auto-resolve when both memories share the same type
# AND their age difference exceeds this threshold (prevents resolving recent updates)
MAINTAIN_AUTO_RESOLVE_AGE_DAYS = safe_get_int_env('MCP_MAINTAIN_AUTO_RESOLVE_AGE_DAYS', 7, min_value=0, max_value=365)

# Insight Cards (Phase 3, #732) — opt-in, generates pattern/trend/gap insights during maintain
MCP_INSIGHT_CARDS_ENABLED = safe_get_bool_env('MCP_INSIGHT_CARDS_ENABLED', False)

# Maximum memories to scan per maintain cycle for entity extraction and insight cards.
# 0 = unlimited (scans all memories — may be slow on large corpora).
MAINTAIN_SCAN_LIMIT = safe_get_int_env('MCP_MAINTAIN_SCAN_LIMIT', 2000, min_value=0)

# Custom entity terms (comma-separated). Supplements regex extraction.
# Example: MCP_ENTITY_CUSTOM_TERMS=roma-connect,kiro,api-gateway
MCP_ENTITY_CUSTOM_TERMS = os.environ.get("MCP_ENTITY_CUSTOM_TERMS", "")

# Tag filters for entities: comma-separated fnmatch patterns, include then exclude.
# Example: MCP_ENTITY_TAG_INCLUDE=task:*
MCP_ENTITY_TAG_INCLUDE = os.environ.get("MCP_ENTITY_TAG_INCLUDE", "")
MCP_ENTITY_TAG_EXCLUDE = os.environ.get("MCP_ENTITY_TAG_EXCLUDE", "")

# =============================================================================
# Auto-capture (RFC #1008 §3) — inline extraction on memory_store / memory_observe
# =============================================================================

MCP_AUTO_EXTRACT_DEFAULT = safe_get_bool_env('MCP_AUTO_EXTRACT_DEFAULT', False)
try:
    MCP_AUTO_EXTRACT_MIN_CONFIDENCE = float(os.getenv('MCP_AUTO_EXTRACT_MIN_CONFIDENCE', '0.6'))
except (TypeError, ValueError):
    MCP_AUTO_EXTRACT_MIN_CONFIDENCE = 0.6
if not 0.0 <= MCP_AUTO_EXTRACT_MIN_CONFIDENCE <= 1.0:
    logger.warning(
        "Invalid MCP_AUTO_EXTRACT_MIN_CONFIDENCE=%s, using 0.6",
        MCP_AUTO_EXTRACT_MIN_CONFIDENCE,
    )
    MCP_AUTO_EXTRACT_MIN_CONFIDENCE = 0.6
