"""Consolidation configuration — scheduling, archive, quality boost."""
import os
import logging

from .base import BASE_DIR, safe_get_int_env, safe_get_bool_env, validate_and_create_path

logger = logging.getLogger(__name__)

# Dream-inspired consolidation configuration
CONSOLIDATION_ENABLED = os.getenv('MCP_CONSOLIDATION_ENABLED', 'false').lower() == 'true'

# Machine identification configuration
INCLUDE_HOSTNAME = os.getenv('MCP_MEMORY_INCLUDE_HOSTNAME', 'false').lower() == 'true'

# Consolidation archive location
consolidation_archive_path = None
for env_var in ['MCP_CONSOLIDATION_ARCHIVE_PATH', 'MCP_MEMORY_ARCHIVE_PATH']:
    if path := os.getenv(env_var):
        consolidation_archive_path = path
        logger.info(f"Using {env_var}={path} for consolidation archive path")
        break

if not consolidation_archive_path:
    consolidation_archive_path = os.path.join(BASE_DIR, 'consolidation_archive')
    logger.info(f"No consolidation archive path environment variable found, using default: {consolidation_archive_path}")

try:
    CONSOLIDATION_ARCHIVE_PATH = validate_and_create_path(consolidation_archive_path)
    logger.info(f"Using consolidation archive path: {CONSOLIDATION_ARCHIVE_PATH}")
except Exception as e:
    logger.error(f"Error creating consolidation archive path: {e}")
    CONSOLIDATION_ARCHIVE_PATH = None

# Consolidation settings with environment variable overrides
CONSOLIDATION_CONFIG = {
    # Decay settings
    'decay_enabled': os.getenv('MCP_DECAY_ENABLED', 'true').lower() == 'true',
    'retention_periods': {
        'critical': safe_get_int_env('MCP_RETENTION_CRITICAL', 365, min_value=1, max_value=3650),
        'reference': safe_get_int_env('MCP_RETENTION_REFERENCE', 180, min_value=1, max_value=3650),
        'standard': safe_get_int_env('MCP_RETENTION_STANDARD', 30, min_value=1, max_value=3650),
        'temporary': safe_get_int_env('MCP_RETENTION_TEMPORARY', 7, min_value=1, max_value=365)
    },
    
    # Association settings
    'associations_enabled': os.getenv('MCP_ASSOCIATIONS_ENABLED', 'true').lower() == 'true',
    'min_similarity': float(os.getenv('MCP_ASSOCIATION_MIN_SIMILARITY', '0.3')),
    'max_similarity': float(os.getenv('MCP_ASSOCIATION_MAX_SIMILARITY', '0.7')),
    'max_pairs_per_run': int(os.getenv('MCP_ASSOCIATION_MAX_PAIRS', '1000')),
    
    # Clustering settings
    'clustering_enabled': os.getenv('MCP_CLUSTERING_ENABLED', 'true').lower() == 'true',
    'min_cluster_size': int(os.getenv('MCP_CLUSTERING_MIN_SIZE', '5')),
    # 'auto' = dbscan when scikit-learn is installed, simple otherwise. Naming
    # 'dbscan'/'hierarchical' without scikit-learn is a hard error, not a downgrade.
    'clustering_algorithm': os.getenv('MCP_CLUSTERING_ALGORITHM', 'auto'),  # 'auto', 'dbscan', 'hierarchical', 'simple'
    
    # Compression settings
    'compression_enabled': os.getenv('MCP_COMPRESSION_ENABLED', 'true').lower() == 'true',
    'max_summary_length': int(os.getenv('MCP_COMPRESSION_MAX_LENGTH', '500')),
    'preserve_originals': os.getenv('MCP_COMPRESSION_PRESERVE_ORIGINALS', 'true').lower() == 'true',
    
    # Forgetting settings
    'forgetting_enabled': os.getenv('MCP_FORGETTING_ENABLED', 'true').lower() == 'true',
    'relevance_threshold': float(os.getenv('MCP_FORGETTING_RELEVANCE_THRESHOLD', '0.1')),
    'access_threshold_days': int(os.getenv('MCP_FORGETTING_ACCESS_THRESHOLD', '90')),
    'forgetting_min_age_days': int(os.getenv('MCP_FORGETTING_MIN_AGE_DAYS', '365')),
    'archive_location': CONSOLIDATION_ARCHIVE_PATH,

    # Incremental consolidation settings
    'batch_size': int(os.getenv('MCP_CONSOLIDATION_BATCH_SIZE', '500')),
    'incremental_mode': os.getenv('MCP_CONSOLIDATION_INCREMENTAL', 'true').lower() == 'true'
}

# Consolidation scheduling settings (for APScheduler integration)
# All schedules default to 'disabled' so consolidation is opt-in. Users must
# explicitly set MCP_SCHEDULE_* env vars to enable automatic runs (issue #808).
# Recommended values when enabling: daily='02:00', weekly='SUN 03:00',
# monthly='01 04:00'. See .env.example for full documentation.
CONSOLIDATION_SCHEDULE = {
    'daily': os.getenv('MCP_SCHEDULE_DAILY', 'disabled'),
    'weekly': os.getenv('MCP_SCHEDULE_WEEKLY', 'disabled'),
    'monthly': os.getenv('MCP_SCHEDULE_MONTHLY', 'disabled'),
    'quarterly': os.getenv('MCP_SCHEDULE_QUARTERLY', 'disabled'),
    'yearly': os.getenv('MCP_SCHEDULE_YEARLY', 'disabled')
}

logger.info(f"Consolidation enabled: {CONSOLIDATION_ENABLED}")
if CONSOLIDATION_ENABLED:
    logger.info(f"Consolidation configuration: {CONSOLIDATION_CONFIG}")
    logger.info(f"Consolidation schedule: {CONSOLIDATION_SCHEDULE}")

# =============================================================================
# Association-Based Quality Enhancement Configuration (v8.47.0+)
# =============================================================================

# Enable association-based quality boost during consolidation
MCP_CONSOLIDATION_QUALITY_BOOST_ENABLED = safe_get_bool_env('MCP_CONSOLIDATION_QUALITY_BOOST_ENABLED', True)

# Minimum connection count required to trigger quality boost
MCP_CONSOLIDATION_MIN_CONNECTIONS_FOR_BOOST = safe_get_int_env('MCP_CONSOLIDATION_MIN_CONNECTIONS_FOR_BOOST', 5, min_value=1, max_value=100)

# Quality boost multiplier (e.g., 1.2 = 20% boost)
MCP_CONSOLIDATION_QUALITY_BOOST_FACTOR = float(os.getenv('MCP_CONSOLIDATION_QUALITY_BOOST_FACTOR', '1.2'))

# Validate quality boost factor
if not 1.0 <= MCP_CONSOLIDATION_QUALITY_BOOST_FACTOR <= 2.0:
    logger.warning(f"Invalid consolidation quality boost factor: {MCP_CONSOLIDATION_QUALITY_BOOST_FACTOR}, must be 1.0-2.0. Using default 1.2")
    MCP_CONSOLIDATION_QUALITY_BOOST_FACTOR = 1.2

# Minimum average quality of connected memories to trigger boost
MCP_CONSOLIDATION_MIN_CONNECTED_QUALITY = float(os.getenv('MCP_CONSOLIDATION_MIN_CONNECTED_QUALITY', '0.7'))

# Validate minimum connected quality
if not 0.0 <= MCP_CONSOLIDATION_MIN_CONNECTED_QUALITY <= 1.0:
    logger.warning(f"Invalid consolidation minimum connected quality: {MCP_CONSOLIDATION_MIN_CONNECTED_QUALITY}, must be 0.0-1.0. Using default 0.7")
    MCP_CONSOLIDATION_MIN_CONNECTED_QUALITY = 0.7

# Log association-based quality boost configuration
if MCP_CONSOLIDATION_QUALITY_BOOST_ENABLED:
    logger.info(f"Association Quality Boost: enabled, min_connections={MCP_CONSOLIDATION_MIN_CONNECTIONS_FOR_BOOST}, "
               f"boost_factor={MCP_CONSOLIDATION_QUALITY_BOOST_FACTOR}, min_connected_quality={MCP_CONSOLIDATION_MIN_CONNECTED_QUALITY}")

# =============================================================================
# End Association-Based Quality Enhancement Configuration
# =============================================================================
