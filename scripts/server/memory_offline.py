#!/usr/bin/env python3
"""
Memory service launcher with forced offline mode.
This script sets offline mode BEFORE importing anything else.
"""

import os
import platform
import sys


def setup_offline_mode():
    """Setup offline mode environment variables BEFORE any imports."""
    print("Setting up offline mode...", file=sys.stderr)

    # Force offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Configure cache paths for Windows
    username = os.environ.get("USERNAME", os.environ.get("USER", ""))
    if platform.system() == "Windows" and username:
        hf_home = f"C:\\Users\\{username}\\.cache\\huggingface"
        transformers_cache = f"C:\\Users\\{username}\\.cache\\huggingface\\transformers"
        sentence_transformers_home = f"C:\\Users\\{username}\\.cache\\torch\\sentence_transformers"
    else:
        hf_home = os.path.expanduser("~/.cache/huggingface")
        transformers_cache = os.path.expanduser("~/.cache/huggingface/transformers")
        sentence_transformers_home = os.path.expanduser("~/.cache/torch/sentence_transformers")

    # Set cache paths
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = sentence_transformers_home

    print(f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}", file=sys.stderr)
    print(f"HF_HOME: {os.environ.get('HF_HOME')}", file=sys.stderr)

    # Add src to Python path
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


if __name__ == "__main__":
    # Setup offline mode FIRST
    setup_offline_mode()

    # Now import and run the memory server
    print("Starting MCP Memory Service in offline mode...", file=sys.stderr)
    from mcp_memory_service.server import main

    main()
