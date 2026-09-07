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

import socket
import asyncio
import logging
from typing import Optional
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


async def find_available_port(start_port: int = HTTP_PORT, max_attempts: int = 10) -> Optional[int]:
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
