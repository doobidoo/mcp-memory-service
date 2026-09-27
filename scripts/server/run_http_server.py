#!/usr/bin/env python3
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
Run the MCP Memory Service HTTP server.

This script starts the FastAPI server with uvicorn.
"""

import logging
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from mcp_memory_service.cli.lifecycle import (
    CertificateGenerationError,
    generate_self_signed_certificate,
)


def main():
    """Run the HTTP server."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set default environment variables for testing
    os.environ.setdefault('MCP_HTTP_ENABLED', 'true')
    # Don't override MCP_MEMORY_STORAGE_BACKEND - respect .env and environment settings
    # os.environ.setdefault('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec')
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    
    try:
        import uvicorn  # inline import: environment defaults must be applied first
        from mcp_memory_service.web.app import app
        from mcp_memory_service.config import (
            HTTP_HOST, HTTP_PORT, HTTPS_ENABLED, SSL_CERT_FILE, SSL_KEY_FILE,
            validate_config,
        )

        # Log any configuration issues (HTTP server handles HTTPS gracefully itself)
        config_issues = validate_config()
        if config_issues:
            for issue in config_issues:
                print(f"CONFIG WARNING: {issue}", file=sys.stderr)
        
        # SSL configuration
        ssl_keyfile = None
        ssl_certfile = None
        protocol = "http"
        
        if HTTPS_ENABLED:
            protocol = "https"
            
            if SSL_CERT_FILE and SSL_KEY_FILE:
                # Use provided certificates
                if os.path.exists(SSL_CERT_FILE) and os.path.exists(SSL_KEY_FILE):
                    ssl_certfile = SSL_CERT_FILE
                    ssl_keyfile = SSL_KEY_FILE
                    print(f"Using provided SSL certificates: {SSL_CERT_FILE}")
                else:
                    print(f"Error: Provided SSL certificates not found!")
                    print(f"Cert file: {SSL_CERT_FILE}")
                    print(f"Key file: {SSL_KEY_FILE}")
                    sys.exit(1)
            else:
                # Generate self-signed certificate
                try:
                    ssl_certfile, ssl_keyfile = generate_self_signed_certificate(
                        additional_ips=os.getenv("MCP_SSL_ADDITIONAL_IPS"),
                        additional_hostnames=os.getenv(
                            "MCP_SSL_ADDITIONAL_HOSTNAMES"
                        ),
                    )
                except CertificateGenerationError as exc:
                    print(f"Failed to generate SSL certificate: {exc}")
                    print("Refusing to fall back to unencrypted HTTP.")
                    sys.exit(1)
        
        # Display startup information
        host_display = HTTP_HOST if HTTP_HOST != '0.0.0.0' else 'localhost'
        print(f"Starting MCP Memory Service {protocol.upper()} server on {HTTP_HOST}:{HTTP_PORT}")
        print(f"Dashboard: {protocol}://{host_display}:{HTTP_PORT}")
        print(f"API Docs: {protocol}://{host_display}:{HTTP_PORT}/api/docs")
        
        if protocol == "https":
            print(f"SSL Certificate: {ssl_certfile}")
            print(f"SSL Key: {ssl_keyfile}")
            print("NOTE: Browsers may show security warnings for self-signed certificates")
        
        print("Press Ctrl+C to stop")
        
        # Start uvicorn server
        uvicorn_kwargs = {
            "app": app,
            "host": HTTP_HOST,
            "port": HTTP_PORT,
            "log_level": "info",
            "access_log": True
        }
        
        if ssl_certfile and ssl_keyfile:
            uvicorn_kwargs["ssl_certfile"] = ssl_certfile
            uvicorn_kwargs["ssl_keyfile"] = ssl_keyfile
        
        uvicorn.run(**uvicorn_kwargs)
        
    except ImportError as e:
        print(f"Error: Missing dependencies. Please run 'python install.py' first.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
