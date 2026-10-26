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

"""
Logging configuration module for MCP Memory Service.

Provides client-aware logging that adjusts output behavior based on MCP client type.
Claude Desktop requires strict JSON mode (stderr only), while LM Studio supports
dual-stream output (stdout for INFO/DEBUG, stderr for WARNING+).
"""

import sys
import os
import logging
from .client_detection import MCP_CLIENT

# Custom logging handler that routes INFO/DEBUG to stdout, WARNING/ERROR to stderr
class DualStreamHandler(logging.Handler):
    """Client-aware handler that adjusts logging behavior based on MCP client."""

    def __init__(self, client_type='claude_desktop', strict_level=logging.WARNING):
        super().__init__()
        self.client_type = client_type
        # Lowest level emitted in strict mode; configure_logging() passes LOG_LEVEL.
        self.strict_level = strict_level
        self.stdout_handler = logging.StreamHandler(sys.stdout)
        self.stderr_handler = logging.StreamHandler(sys.stderr)

        # Set the same formatter for both handlers
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        self.stdout_handler.setFormatter(formatter)
        self.stderr_handler.setFormatter(formatter)

    def emit(self, record):
        """Route log records based on client type and level."""
        # For Claude Desktop: strict JSON mode - stdout carries the protocol, so
        # everything goes to stderr. Strict mode is also the fallback when no known
        # client is the parent (e.g. the HTTP server under systemd), so the level
        # follows LOG_LEVEL instead of being pinned to WARNING.
        if self.client_type == 'claude_desktop':
            if record.levelno >= self.strict_level:
                self.stderr_handler.emit(record)
            return

        # For LM Studio: enhanced mode with dual-stream
        if record.levelno >= logging.WARNING:  # WARNING, ERROR, CRITICAL
            self.stderr_handler.emit(record)
        else:  # DEBUG, INFO
            self.stdout_handler.emit(record)

def configure_logging():
    """Configure root logger with client-aware handler."""
    # Configure logging with client-aware handler BEFORE any imports that use logging
    log_level = os.getenv('LOG_LEVEL', 'WARNING').upper()  # Default to WARNING for performance
    level = getattr(logging, log_level, None)
    if not isinstance(level, int):
        level = logging.WARNING
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add our custom client-aware handler
    client_aware_handler = DualStreamHandler(client_type=MCP_CLIENT, strict_level=level)
    root_logger.addHandler(client_aware_handler)

    return logging.getLogger(__name__)

# Auto-configure on import
logger = configure_logging()
