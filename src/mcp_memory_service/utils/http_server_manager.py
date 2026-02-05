"""HTTP Server Manager for MCP Memory Service multi-client coordination."""

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


async def auto_start_http_server_if_needed() -> bool:
    """
    Auto-start HTTP server if needed for multi-client coordination.

    Returns:
        bool: True if server was started or already running, False if failed
    """
    try:
        # Check if HTTP auto-start is enabled
        if os.getenv("MCP_MEMORY_HTTP_AUTO_START", "").lower() not in ("true", "1"):
            logger.debug("HTTP auto-start not enabled")
            return False

        # Check if server is already running
        from ..utils.port_detection import is_port_in_use

        port = int(os.getenv("MCP_HTTP_PORT", "8000"))

        if await is_port_in_use("localhost", port):
            logger.info(f"HTTP server already running on port {port}")
            return True

        # Try to start the HTTP server
        logger.info(f"Starting HTTP server on port {port}")

        # Get the repository root
        repo_root = Path(__file__).parent.parent.parent.parent

        # Start the HTTP server as a background process
        cmd = [sys.executable, "-m", "src.mcp_memory_service.app", "--port", str(port), "--host", "localhost"]

        process = subprocess.Popen(
            cmd, cwd=repo_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
        )

        # Wait a moment and check if the process started successfully
        await asyncio.sleep(1)

        if process.poll() is None:  # Process is still running
            # Wait a bit more and check if port is now in use
            await asyncio.sleep(2)
            if await is_port_in_use("localhost", port):
                logger.info(f"Successfully started HTTP server on port {port}")
                return True
            else:
                logger.warning("HTTP server process started but port not in use")
                return False
        else:
            logger.warning(f"HTTP server process exited with code {process.returncode}")
            return False

    except Exception as e:
        logger.error(f"Failed to auto-start HTTP server: {e}")
        return False
