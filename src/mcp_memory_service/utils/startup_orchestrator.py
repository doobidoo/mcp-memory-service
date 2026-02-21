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
Server startup orchestration utilities.

Extracted from server_impl.py Phase 3.2 refactoring to reduce async_main complexity.
This module provides orchestrator classes for:
- Startup validation checks
- Initialization with retry logic
- Server execution mode management
"""

import os
import sys
import asyncio
import logging
import traceback
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..server_impl import MemoryServer

# Import necessary functions and constants
from ..server.client_detection import MCP_CLIENT
from ..config import SERVER_NAME, SERVER_VERSION
from ..lm_studio_compat import patch_mcp_for_lm_studio, add_windows_timeout_handling
from ..dependency_check import run_dependency_check
from ..server.environment import check_uv_environment, check_version_consistency

# MCP imports
import mcp.server.stdio
from mcp.server import InitializationOptions, NotificationOptions

logger = logging.getLogger(__name__)


class StartupCheckOrchestrator:
    """Orchestrate startup validation checks."""

    @staticmethod
    def run_all_checks() -> None:
        """Run all startup checks in sequence."""
        # Apply LM Studio compatibility patch
        patch_mcp_for_lm_studio()

        # Add Windows-specific timeout handling
        add_windows_timeout_handling()

        # Run dependency check
        run_dependency_check()

        # Check if running with UV
        check_uv_environment()

        # Check for version mismatch
        check_version_consistency()


class InitializationRetryManager:
    """Manage server initialization with timeout and retry logic."""

    def __init__(self, max_retries: int = 2, timeout: float = 30.0, retry_delay: float = 2.0):
        """
        Initialize retry manager.

        Args:
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds for each initialization attempt
            retry_delay: Delay in seconds between retry attempts
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)

    async def initialize_with_retry(self, server: 'MemoryServer') -> bool:
        """
        Initialize server with timeout and retry logic.

        Args:
            server: MemoryServer instance to initialize

        Returns:
            True if initialization succeeded, False otherwise
        """
        retry_count = 0
        init_success = False

        while retry_count <= self.max_retries and not init_success:
            if retry_count > 0:
                self.logger.warning(f"Retrying initialization (attempt {retry_count}/{self.max_retries})...")

            init_task = asyncio.create_task(server.initialize())
            try:
                # Timeout for initialization
                init_success = await asyncio.wait_for(init_task, timeout=self.timeout)
                if init_success:
                    self.logger.info("Async initialization completed successfully")
                else:
                    self.logger.warning("Initialization returned failure status")
                    retry_count += 1
            except asyncio.TimeoutError:
                self.logger.warning("Async initialization timed out. Continuing with server startup.")
                # Don't cancel the task, let it complete in the background
                break
            except Exception as init_error:
                self.logger.error(f"Initialization error: {str(init_error)}")
                self.logger.error(traceback.format_exc())
                retry_count += 1

                if retry_count <= self.max_retries:
                    self.logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                    await asyncio.sleep(self.retry_delay)

        return init_success


class ServerRunManager:
    """Manage server execution modes and lifecycle."""

    def __init__(self, server: 'MemoryServer', system_info: Any):
        """
        Initialize server run manager.

        Args:
            server: MemoryServer instance to manage
            system_info: System information object (from get_system_info)
        """
        self.server = server
        self.system_info = system_info
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def is_standalone_mode() -> bool:
        """Check if running in standalone mode."""
        standalone_mode = os.environ.get('MCP_STANDALONE_MODE', '').lower() == '1'
        return standalone_mode

    @staticmethod
    def is_docker_environment() -> bool:
        """Check if running in Docker."""
        return os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)

    async def run_standalone(self) -> None:
        """Run server in standalone mode (Docker without active client)."""
        self.logger.info("Running in standalone mode - keeping server alive without active client")
        if MCP_CLIENT == 'lm_studio':
            print("MCP Memory Service running in standalone mode", file=sys.stdout, flush=True)

        # Keep the server running indefinitely
        try:
            while True:
                await asyncio.sleep(60)  # Sleep for 60 seconds at a time
                self.logger.debug("Standalone server heartbeat")
        except asyncio.CancelledError:
            self.logger.info("Standalone server cancelled")
            raise

    async def run_stdio(self) -> None:
        """Run server with stdio communication."""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            self.logger.info("Server started and ready to handle requests")

            if self.is_docker_environment():
                self.logger.info("Detected Docker environment - ensuring proper stdio handling")
                if MCP_CLIENT == 'lm_studio':
                    print("MCP Memory Service running in Docker container", file=sys.stdout, flush=True)

            try:
                await self.server.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name=SERVER_NAME,
                        server_version=SERVER_VERSION,
                        protocol_version="2024-11-05",
                        capabilities=self.server.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={
                                "hardware_info": {
                                    "architecture": self.system_info.architecture,
                                    "accelerator": self.system_info.accelerator,
                                    "memory_gb": self.system_info.memory_gb,
                                    "cpu_count": self.system_info.cpu_count
                                }
                            },
                        ),
                    ),
                )
            except asyncio.CancelledError:
                self.logger.info("Server run cancelled")
                raise
            except Exception as e:
                self._handle_server_exception(e)
            finally:
                self.logger.info("Server run completed")

    def _handle_server_exception(self, e: Exception) -> None:
        """Handle exceptions during server run."""
        # Handle ExceptionGroup specially (Python 3.11+)
        if type(e).__name__ == 'ExceptionGroup' or 'ExceptionGroup' in str(type(e)):
            error_str = str(e)
            # Check if this contains the LM Studio cancelled notification error
            if 'notifications/cancelled' in error_str or 'ValidationError' in error_str:
                self.logger.info("LM Studio sent a cancelled notification - this is expected behavior")
                self.logger.debug(f"Full error for debugging: {error_str}")
                # Don't re-raise - just continue gracefully
            else:
                self.logger.error(f"ExceptionGroup in server.run: {str(e)}")
                self.logger.error(traceback.format_exc())
                raise
        else:
            self.logger.error(f"Error in server.run: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
