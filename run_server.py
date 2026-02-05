#!/usr/bin/env python3
"""Run the MCP Memory Service with HTTP/HTTPS/mDNS support via FastAPI."""

import logging
import os
import sys

import uvicorn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Log configuration
    logger.info("Starting MCP Memory Service FastAPI server with the following configuration:")
    logger.info(f"  Storage Backend: {os.environ.get('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec')}")
    logger.info(f"  HTTP Port: {os.environ.get('MCP_HTTP_PORT', '8000')}")
    logger.info(f"  HTTPS Enabled: {os.environ.get('MCP_HTTPS_ENABLED', 'false')}")
    logger.info(f"  HTTPS Port: {os.environ.get('MCP_HTTPS_PORT', '8443')}")
    logger.info(f"  mDNS Enabled: {os.environ.get('MCP_MDNS_ENABLED', 'false')}")
    logger.info(f"  API Key Set: {'Yes' if os.environ.get('MCP_API_KEY') else 'No'}")

    http_port = int(os.environ.get("MCP_HTTP_PORT", 8000))

    # Check if HTTPS is enabled
    if os.environ.get("MCP_HTTPS_ENABLED", "false").lower() == "true":
        https_port = int(os.environ.get("MCP_HTTPS_PORT", 8443))

        # Check for environment variable certificates first
        cert_file = os.environ.get("MCP_SSL_CERT_FILE")
        key_file = os.environ.get("MCP_SSL_KEY_FILE")

        if cert_file and key_file:
            # Use provided certificates
            if not os.path.exists(cert_file):
                logger.error(f"Certificate file not found: {cert_file}")
                sys.exit(1)
            if not os.path.exists(key_file):
                logger.error(f"Key file not found: {key_file}")
                sys.exit(1)
            logger.info(f"Using provided certificates: {cert_file}")
        else:
            # Generate self-signed certificate if needed
            cert_dir = os.path.expanduser("~/.mcp_memory_certs")
            os.makedirs(cert_dir, exist_ok=True)
            cert_file = os.path.join(cert_dir, "cert.pem")
            key_file = os.path.join(cert_dir, "key.pem")

            if not os.path.exists(cert_file) or not os.path.exists(key_file):
                logger.info("Generating self-signed certificate for HTTPS...")
                import subprocess

                subprocess.run(
                    [
                        "openssl",
                        "req",
                        "-x509",
                        "-newkey",
                        "rsa:4096",
                        "-keyout",
                        key_file,
                        "-out",
                        cert_file,
                        "-days",
                        "365",
                        "-nodes",
                        "-subj",
                        "/C=US/ST=State/L=City/O=MCP/CN=localhost",
                    ],
                    check=True,
                )
                logger.info(f"Certificate generated at {cert_dir}")

        # Run with HTTPS
        logger.info(f"Starting HTTPS server on port {https_port}")
        uvicorn.run(
            "mcp_memory_service.web.app:app",
            host="0.0.0.0",
            port=https_port,
            ssl_keyfile=key_file,
            ssl_certfile=cert_file,
            reload=False,
            log_level="info",
        )
    else:
        # Run HTTP only
        logger.info(f"Starting HTTP server on port {http_port}")
        uvicorn.run("mcp_memory_service.web.app:app", host="0.0.0.0", port=http_port, reload=False, log_level="info")
