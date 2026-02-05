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
Port detection utilities for multi-client HTTP server coordination.
"""

import asyncio
import logging
import socket

import aiohttp

from ..config import HTTP_PORT

logger = logging.getLogger(__name__)


async def is_port_in_use(host: str = "localhost", port: int = HTTP_PORT) -> bool:
    """
    Check if a port is in use by attempting to create a socket connection.

    Args:
        host: Host to check (default: localhost)
        port: Port to check

    Returns:
        True if port is in use, False otherwise
    """
    try:
        # Try to create a socket and connect
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)  # 1 second timeout
            result = sock.connect_ex((host, port))
            return result == 0  # 0 means connection successful (port in use)
    except Exception as e:
        logger.debug(f"Error checking port {port}: {e}")
        return False


async def is_mcp_memory_server_running(host: str = "localhost", port: int = HTTP_PORT) -> tuple[bool, str | None]:
    """
    Check if an MCP Memory Service HTTP server is running on the specified port.

    Args:
        host: Host to check
        port: Port to check

    Returns:
        Tuple of (is_running, server_info)
        - is_running: True if MCP Memory Service is detected
        - server_info: Server identification string if detected
    """
    try:
        timeout = aiohttp.ClientTimeout(total=2.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Try to hit the health endpoint
            health_url = f"http://{host}:{port}/health"
            async with session.get(health_url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Check if this is our MCP Memory Service
                    if data.get("service") == "mcp-memory-service" or "memory" in data.get("service", "").lower():
                        server_info = f"{data.get('service', 'unknown')} v{data.get('version', 'unknown')}"
                        logger.info(f"Detected MCP Memory Service at {host}:{port} - {server_info}")
                        return True, server_info
                    else:
                        logger.debug(f"Different service running at {host}:{port}: {data.get('service')}")
                        return False, None
                else:
                    logger.debug(f"HTTP server at {host}:{port} returned status {response.status}")
                    return False, None

    except aiohttp.ClientError as e:
        logger.debug(f"HTTP client error checking {host}:{port}: {e}")
        return False, None
    except asyncio.TimeoutError:
        logger.debug(f"Timeout checking {host}:{port}")
        return False, None
    except Exception as e:
        logger.debug(f"Unexpected error checking {host}:{port}: {e}")
        return False, None


async def find_available_port(start_port: int = HTTP_PORT, max_attempts: int = 10) -> int | None:
    """
    Find an available port starting from start_port.

    Args:
        start_port: Port to start checking from
        max_attempts: Maximum number of ports to check

    Returns:
        Available port number or None if none found
    """
    for port in range(start_port, start_port + max_attempts):
        if not await is_port_in_use(port=port):
            logger.debug(f"Found available port: {port}")
            return port

    logger.warning(f"No available ports found in range {start_port}-{start_port + max_attempts}")
    return None


async def detect_server_coordination_mode(host: str = "localhost", port: int = HTTP_PORT) -> str:
    """
    Detect the best coordination mode for multi-client access.

    Returns:
        - "http_client": HTTP server is running, use client mode
        - "http_server": No server running, start HTTP server
        - "direct": Use direct SQLite access (fallback)
    """
    # Check if MCP Memory Service HTTP server is already running
    is_running, server_info = await is_mcp_memory_server_running(host, port)

    if is_running:
        logger.info(f"MCP Memory Service HTTP server detected: {server_info}")
        return "http_client"

    # Check if port is available for starting our own server
    port_available = not await is_port_in_use(host, port)

    if port_available:
        logger.info(f"Port {port} available, can start HTTP server")
        return "http_server"
    else:
        logger.info(f"Port {port} in use by different service, falling back to direct access")
        return "direct"


class ServerCoordinator:
    """Helper class for managing server coordination state."""

    def __init__(self, host: str = "localhost", port: int = HTTP_PORT):
        self.host = host
        self.port = port
        self.mode = None
        self.server_info = None

    async def detect_mode(self) -> str:
        """Detect and cache the coordination mode."""
        self.mode = await detect_server_coordination_mode(self.host, self.port)

        if self.mode == "http_client":
            _, self.server_info = await is_mcp_memory_server_running(self.host, self.port)

        return self.mode

    def get_mode(self) -> str | None:
        """Get the cached coordination mode."""
        return self.mode

    def is_http_client_mode(self) -> bool:
        """Check if we should use HTTP client mode."""
        return self.mode == "http_client"

    def is_http_server_mode(self) -> bool:
        """Check if we should start HTTP server mode."""
        return self.mode == "http_server"

    def is_direct_mode(self) -> bool:
        """Check if we should use direct access mode."""
        return self.mode == "direct"
