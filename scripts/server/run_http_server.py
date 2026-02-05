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
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


def generate_self_signed_cert():
    """Generate a self-signed certificate for development."""
    try:
        # Create temporary directory for certificates
        cert_dir = os.path.join(tempfile.gettempdir(), "mcp-memory-certs")
        os.makedirs(cert_dir, exist_ok=True)

        cert_file = os.path.join(cert_dir, "cert.pem")
        key_file = os.path.join(cert_dir, "key.pem")

        # Check if certificates already exist and are still valid
        if os.path.exists(cert_file) and os.path.exists(key_file):
            try:
                # Check certificate expiration
                result = subprocess.run(
                    ["openssl", "x509", "-in", cert_file, "-noout", "-enddate"], capture_output=True, text=True, check=True
                )

                # Parse expiration date
                end_date_str = result.stdout.split("=")[1].strip()
                end_date = datetime.strptime(end_date_str, "%b %d %H:%M:%S %Y %Z")

                # If certificate expires in more than 7 days, reuse it
                if end_date > datetime.now() + timedelta(days=7):
                    print(f"Using existing self-signed certificate: {cert_file}")
                    return cert_file, key_file

            except Exception:
                pass  # Fall through to generate new certificate

        print("Generating self-signed certificate for HTTPS...")

        # Generate private key
        subprocess.run(["openssl", "genrsa", "-out", key_file, "2048"], check=True, capture_output=True)

        # Generate certificate with Subject Alternative Names for better compatibility
        # Get local IP addresses dynamically
        import socket

        local_ips = []
        try:
            # Get primary local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            local_ips.append(local_ip)
        except Exception:
            pass

        # Build SAN list with common names and detected IPs
        san_entries = [
            "DNS:memory.local",
            "DNS:localhost",
            "DNS:*.local",
            "IP:127.0.0.1",
            "IP:::1",  # IPv6 localhost
        ]

        # Add detected local IPs
        for ip in local_ips:
            if ip not in ["127.0.0.1"]:
                san_entries.append(f"IP:{ip}")

        # Add additional IPs from environment variable if specified
        additional_ips = os.getenv("MCP_SSL_ADDITIONAL_IPS", "")
        if additional_ips:
            for ip in additional_ips.split(","):
                ip = ip.strip()
                if ip and ip not in [entry.split(":")[1] for entry in san_entries if entry.startswith("IP:")]:
                    san_entries.append(f"IP:{ip}")

        # Add additional hostnames from environment variable if specified
        additional_hostnames = os.getenv("MCP_SSL_ADDITIONAL_HOSTNAMES", "")
        if additional_hostnames:
            for hostname in additional_hostnames.split(","):
                hostname = hostname.strip()
                if hostname and f"DNS:{hostname}" not in san_entries:
                    san_entries.append(f"DNS:{hostname}")

        san_string = ",".join(san_entries)

        print(f"Generating certificate with SANs: {san_string}")

        subprocess.run(
            [
                "openssl",
                "req",
                "-new",
                "-x509",
                "-key",
                key_file,
                "-out",
                cert_file,
                "-days",
                "365",
                "-subj",
                "/C=US/ST=Local/L=Local/O=MCP Memory Service/CN=memory.local",
                "-addext",
                f"subjectAltName={san_string}",
            ],
            check=True,
            capture_output=True,
        )

        print(f"Generated self-signed certificate: {cert_file}")
        print("WARNING: This is a development certificate. Use proper certificates in production.")

        return cert_file, key_file

    except subprocess.CalledProcessError as e:
        print(f"Error generating certificate: {e}")
        print("Make sure OpenSSL is installed and available in PATH")
        return None, None
    except Exception as e:
        print(f"Unexpected error generating certificate: {e}")
        return None, None


def main():
    """Run the HTTP server."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Set default environment variables for testing
    os.environ.setdefault("MCP_HTTP_ENABLED", "true")
    # Don't override MCP_MEMORY_STORAGE_BACKEND - respect .env and environment settings
    # os.environ.setdefault('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec')
    os.environ.setdefault("LOG_LEVEL", "INFO")

    try:
        import uvicorn

        from mcp_memory_service.config import HTTP_HOST, HTTP_PORT, HTTPS_ENABLED, SSL_CERT_FILE, SSL_KEY_FILE
        from mcp_memory_service.web.app import app

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
                    print("Error: Provided SSL certificates not found!")
                    print(f"Cert file: {SSL_CERT_FILE}")
                    print(f"Key file: {SSL_KEY_FILE}")
                    sys.exit(1)
            else:
                # Generate self-signed certificate
                ssl_certfile, ssl_keyfile = generate_self_signed_cert()
                if not ssl_certfile or not ssl_keyfile:
                    print("Failed to generate SSL certificate. Falling back to HTTP.")
                    protocol = "http"
                    ssl_certfile = ssl_keyfile = None

        # Display startup information
        host_display = HTTP_HOST if HTTP_HOST != "0.0.0.0" else "localhost"
        print(f"Starting MCP Memory Service {protocol.upper()} server on {HTTP_HOST}:{HTTP_PORT}")
        print(f"Dashboard: {protocol}://{host_display}:{HTTP_PORT}")
        print(f"API Docs: {protocol}://{host_display}:{HTTP_PORT}/api/docs")

        if protocol == "https":
            print(f"SSL Certificate: {ssl_certfile}")
            print(f"SSL Key: {ssl_keyfile}")
            print("NOTE: Browsers may show security warnings for self-signed certificates")

        print("Press Ctrl+C to stop")

        # Start uvicorn server
        uvicorn_kwargs = {"app": app, "host": HTTP_HOST, "port": HTTP_PORT, "log_level": "info", "access_log": True}

        if ssl_certfile and ssl_keyfile:
            uvicorn_kwargs["ssl_certfile"] = ssl_certfile
            uvicorn_kwargs["ssl_keyfile"] = ssl_keyfile

        uvicorn.run(**uvicorn_kwargs)

    except ImportError as e:
        print("Error: Missing dependencies. Please run 'python install.py' first.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
