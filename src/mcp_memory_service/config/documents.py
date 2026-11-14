"""Document processing configuration — LlamaParse, chunking."""
import os
import logging

from .base import safe_get_int_env

logger = logging.getLogger(__name__)

# =============================================================================
# Document Processing Configuration (Semtools Integration)
# =============================================================================

# Semtools configuration for enhanced document parsing
# LlamaParse API key for advanced OCR and table extraction
LLAMAPARSE_API_KEY = os.getenv('LLAMAPARSE_API_KEY', None)

# Document chunking configuration
DOCUMENT_CHUNK_SIZE = safe_get_int_env('MCP_DOCUMENT_CHUNK_SIZE', 1000, min_value=100, max_value=10000)
DOCUMENT_CHUNK_OVERLAP = safe_get_int_env('MCP_DOCUMENT_CHUNK_OVERLAP', 200, min_value=0, max_value=1000)

# Log semtools configuration
if LLAMAPARSE_API_KEY:
    logger.info("LlamaParse API key configured - enhanced document parsing available")
else:
    logger.debug("LlamaParse API key not set - semtools will use basic parsing mode")

logger.info("Document chunking: size=%d, overlap=%d", DOCUMENT_CHUNK_SIZE, DOCUMENT_CHUNK_OVERLAP)

# =============================================================================
# End Document Processing Configuration
# =============================================================================
