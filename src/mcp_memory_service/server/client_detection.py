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
Client detection module for MCP Memory Service.

Detects which MCP client is running the server (Claude Desktop, LM Studio, etc.)
and provides environment-aware behavior adjustments.
"""

import os
import logging
import psutil

from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)


def detect_mcp_client():
    """Detect which MCP client is running this server."""
    try:
        # Get the parent process (the MCP client)
        current_process = psutil.Process()
        parent = current_process.parent()

        if parent:
            parent_name = parent.name().lower()
            parent_exe = parent.exe() if hasattr(parent, 'exe') else ""

            # Check for Claude Desktop
            if 'claude' in parent_name or 'claude' in parent_exe.lower():
                return 'claude_desktop'

            # Check for LM Studio
            if 'lmstudio' in parent_name or 'lm-studio' in parent_name or 'lmstudio' in parent_exe.lower():
                return 'lm_studio'

            # Check command line for additional clues
            try:
                cmdline = parent.cmdline()
                cmdline_str = ' '.join(cmdline).lower()

                if 'claude' in cmdline_str:
                    return 'claude_desktop'
                if 'lmstudio' in cmdline_str or 'lm-studio' in cmdline_str:
                    return 'lm_studio'
            except (OSError, IndexError, AttributeError) as e:
                logger.debug("Could not detect client from process: %s", _sanitize_log_value(e))

        # Fallback: check environment variables
        if os.getenv('CLAUDE_DESKTOP'):
            return 'claude_desktop'
        if os.getenv('LM_STUDIO'):
            return 'lm_studio'

        # Default to Claude Desktop for strict JSON compliance
        return 'claude_desktop'

    except Exception:
        # If detection fails, default to Claude Desktop (strict mode)
        return 'claude_desktop'


# Detect the current MCP client
MCP_CLIENT = detect_mcp_client()
