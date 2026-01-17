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
Entry point for running the server package as a module.

Allows running the server with:
    python -m mcp_memory_service.server [args]

This is required for backward compatibility with CI/CD workflows
and Docker containers that use `python -m` invocation.
"""

import sys
import argparse
from . import main

try:
    from .. import __version__
except (ImportError, AttributeError):
    __version__ = "0.0.0.dev0"


def run_with_args():
    """Handle command-line arguments before starting server."""
    # Simple argument parsing for --version and --help
    parser = argparse.ArgumentParser(
        prog='python -m mcp_memory_service.server',
        description='MCP Memory Service - Model Context Protocol Server',
        add_help=True
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    # Parse known args to allow --version/--help while passing through other args
    args, unknown = parser.parse_known_args()

    # If we get here, no --version or --help was provided
    # Start the server normally
    main()


if __name__ == '__main__':
    run_with_args()
