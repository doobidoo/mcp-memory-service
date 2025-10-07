#!/usr/bin/env python3
"""
Check if the MCP Memory Service HTTP server is running.

This script checks if the HTTP server is accessible and provides
helpful feedback to users about how to start it if it's not running.
"""

import sys
import os
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import json
import ssl


def check_http_server(verbose: bool = False) -> bool:
    """
    Check if the HTTP server is running.

    Args:
        verbose: If True, print detailed status messages

    Returns:
        bool: True if server is running, False otherwise
    """
    # Determine the endpoint from environment
    https_enabled = os.getenv('MCP_HTTPS_ENABLED', 'false').lower() == 'true'
    http_port = int(os.getenv('MCP_HTTP_PORT', '8000'))
    https_port = int(os.getenv('MCP_HTTPS_PORT', '8443'))

    if https_enabled:
        endpoint = f"https://localhost:{https_port}/api/health"
    else:
        endpoint = f"http://localhost:{http_port}/api/health"

    try:
        # Create SSL context that doesn't verify certificates (for self-signed certs)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = Request(endpoint)
        with urlopen(req, timeout=3, context=ctx) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                if verbose:
                    print("[OK] HTTP server is running")
                    print(f"   Version: {data.get('version', 'unknown')}")
                    print(f"   Endpoint: {endpoint}")
                    print(f"   Status: {data.get('status', 'unknown')}")
                return True
            else:
                if verbose:
                    print(f"[WARN] HTTP server responded with status {response.status}")
                return False
    except (URLError, HTTPError, json.JSONDecodeError) as e:
        if verbose:
            print("[ERROR] HTTP server is NOT running")
            print(f"\nTo start the HTTP server, run:")
            print(f"   uv run python scripts/server/run_http_server.py")
            print(f"\n   Or for HTTPS:")
            print(f"   MCP_HTTPS_ENABLED=true uv run python scripts/server/run_http_server.py")
            print(f"\nError: {str(e)}")
        return False


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check if MCP Memory Service HTTP server is running"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only return exit code (0=running, 1=not running), no output."
    )

    args = parser.parse_args()

    is_running = check_http_server(verbose=not args.quiet)
    sys.exit(0 if is_running else 1)


if __name__ == "__main__":
    main()
