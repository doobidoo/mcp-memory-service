#!/usr/bin/env python3
"""Unified server launcher - runs HTTP, MCP, or both interfaces simultaneously.

This module provides a single entry point that can start:
1. HTTP server only (MCP_HTTP_ENABLED=true)
2. MCP server only (MCP_TRANSPORT_MODE=stdio)
3. Both interfaces simultaneously (both env vars set)

The launcher shares storage backend and embedding model across interfaces,
reducing memory usage and operational complexity compared to running separate containers.

Environment Variables:
    MCP_HTTP_ENABLED: Enable HTTP/REST API server (default: false)
    MCP_TRANSPORT_MODE: MCP transport mode (stdio|streamable-http)
    HTTP_HOST: HTTP server host (default: 0.0.0.0)
    HTTP_PORT: HTTP server port (default: 8000)

Example:
    # HTTP only
    $ MCP_HTTP_ENABLED=true python -m mcp_memory_service.unified_server

    # MCP stdio only
    $ MCP_TRANSPORT_MODE=stdio python -m mcp_memory_service.unified_server

    # Both interfaces
    $ MCP_HTTP_ENABLED=true MCP_TRANSPORT_MODE=stdio python -m mcp_memory_service.unified_server
"""

import asyncio
import logging
import signal
import sys

from .shared_storage import close_shared_storage, get_shared_storage

logger = logging.getLogger(__name__)


class UnifiedServer:
    """Manages lifecycle of HTTP and MCP server interfaces."""

    def __init__(self) -> None:
        """Initialize unified server with configuration from environment."""
        import os

        from mcp_memory_service.config import HTTP_ENABLED, HTTP_HOST, HTTP_PORT

        self.http_enabled = HTTP_ENABLED
        self.http_host = HTTP_HOST
        self.http_port = HTTP_PORT

        # MCP transport mode is read directly from environment (not in Settings)
        self.mcp_transport = os.getenv("MCP_TRANSPORT_MODE", "")
        self.mcp_enabled = self.mcp_transport in ("stdio", "http", "streamable-http")

        self.shutdown_event: asyncio.Event | None = None
        self.tasks: list[asyncio.Task] = []

    def validate_configuration(self) -> None:
        """Validate server configuration and fail fast if invalid.

        Raises:
            ValueError: If no interfaces are enabled or configuration is invalid.
        """
        if not self.http_enabled and not self.mcp_enabled:
            raise ValueError("No interface enabled. Set MCP_HTTP_ENABLED=true or MCP_TRANSPORT_MODE=stdio|http")

        logger.info("Configuration validated successfully")
        if self.http_enabled:
            logger.info(f"  HTTP interface: {self.http_host}:{self.http_port}")
        if self.mcp_enabled:
            logger.info(f"  MCP interface: {self.mcp_transport} transport")

    async def run_http_server(self) -> None:
        """Start HTTP server using existing FastAPI application.

        Uses uvicorn to serve the FastAPI app with configuration from environment.
        Runs until shutdown signal received or server fails.
        """
        try:
            import uvicorn

            from mcp_memory_service.web.app import create_app

            logger.info(f"Starting HTTP server on {self.http_host}:{self.http_port}")
            app = create_app()
            config = uvicorn.Config(
                app,
                host=self.http_host,
                port=self.http_port,
                log_config=None,  # Use existing logging config
            )
            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logger.error(f"HTTP server failed: {e}", exc_info=True)
            raise

    async def run_mcp_server(self) -> None:
        """Start MCP server using existing FastMCP implementation.

        Uses the FastMCP framework to serve MCP protocol over configured transport.
        Runs until shutdown signal received or server fails.
        """
        try:
            import os

            from mcp_memory_service.mcp_server import mcp

            # Get port and host from environment
            port = int(os.getenv("MCP_SERVER_PORT", "8001"))
            host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")

            # Map old transport name to new one
            transport = "http" if self.mcp_transport == "streamable-http" else self.mcp_transport

            logger.info(f"Starting MCP server with {transport} transport on {host}:{port}")

            # FastMCP v2.0 async API
            if transport == "http":
                await mcp.run_async(transport="http", host=host, port=port, uvicorn_config={"ws": "none"})
            else:
                await mcp.run_async(transport="stdio")
        except Exception as e:
            logger.error(f"MCP server failed: {e}", exc_info=True)
            raise

    def setup_signal_handlers(self, shutdown_event: asyncio.Event) -> None:
        """Setup graceful shutdown handlers for SIGTERM and SIGINT.

        Args:
            shutdown_event: Event to set when shutdown signal received.
        """

        def handle_shutdown(signum: int, frame) -> None:
            """Handle shutdown signal by setting shutdown event."""
            signal_name = signal.Signals(signum).name
            logger.info(f"Received {signal_name}, initiating graceful shutdown")
            shutdown_event.set()

        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)
        logger.debug("Signal handlers configured for SIGTERM and SIGINT")

    async def run(self) -> None:
        """Run the unified server with configured interfaces.

        Starts all enabled interfaces and waits for completion or shutdown signal.
        Implements graceful shutdown by cancelling running tasks.

        Raises:
            ValueError: If configuration validation fails.
        """
        # Validate before starting
        self.validate_configuration()

        # Pre-initialize shared storage before starting any server
        # This ensures both servers use the same storage instance
        logger.info("Pre-initializing shared storage for all servers...")
        try:
            storage = await get_shared_storage()
            logger.info(f"✓ Shared storage pre-initialized: {type(storage).__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize shared storage: {e}")
            raise

        # Setup graceful shutdown
        self.shutdown_event = asyncio.Event()
        self.setup_signal_handlers(self.shutdown_event)

        # Start enabled interfaces
        if self.http_enabled:
            task = asyncio.create_task(self.run_http_server())
            task.set_name("http-server")
            self.tasks.append(task)

        if self.mcp_enabled:
            task = asyncio.create_task(self.run_mcp_server())
            task.set_name("mcp-server")
            self.tasks.append(task)

        logger.info(f"Started {len(self.tasks)} interface(s)")

        # Wait for shutdown signal or task completion
        shutdown_task = asyncio.create_task(self.shutdown_event.wait())
        shutdown_task.set_name("shutdown-waiter")

        done, pending = await asyncio.wait(
            self.tasks + [shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # If a server task completed, check if it failed
        for task in done:
            if task.get_name() != "shutdown-waiter":
                try:
                    task.result()  # Raises exception if task failed
                    logger.warning(f"Task {task.get_name()} completed unexpectedly")
                except Exception as e:
                    logger.error(f"Task {task.get_name()} failed: {e}")

        # Cancel remaining tasks
        logger.info("Cancelling remaining tasks")
        for task in pending:
            task.cancel()

        # Wait for cancellation to complete
        await asyncio.gather(*pending, return_exceptions=True)

        # Close shared storage
        await close_shared_storage()

        logger.info("Shutdown complete")


def main() -> None:
    """Entry point for unified server.

    Configures logging and runs the unified server with asyncio.
    Exits with code 1 on configuration errors or fatal failures.
    """
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        server = UnifiedServer()
        asyncio.run(server.run())
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
