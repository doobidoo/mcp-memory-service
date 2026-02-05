#!/usr/bin/env python3
"""
Pre-load sentence-transformers models to avoid startup delays.

This script downloads and caches the default embedding models used by
MCP Memory Service so they don't need to be downloaded during server startup,
which can cause timeout errors in Claude Desktop.
"""

import os
import sys


def preload_sentence_transformers():
    """Pre-load the default sentence-transformers model."""
    try:
        print("[INFO] Pre-loading sentence-transformers models...")
        from sentence_transformers import SentenceTransformer

        # Default model used by the memory service
        model_name = "all-MiniLM-L6-v2"
        print(f"[INFO] Downloading and caching model: {model_name}")

        model = SentenceTransformer(model_name)
        print(f"[OK] Model loaded successfully on device: {model.device}")

        # Test the model to ensure it works
        print("[INFO] Testing model functionality...")
        test_text = "This is a test sentence for embedding."
        embedding = model.encode(test_text)
        print(f"[OK] Model test successful - embedding shape: {embedding.shape}")

        return True

    except ImportError:
        print("[WARNING] sentence-transformers not available - skipping model preload")
        return False
    except Exception as e:
        print(f"[ERROR] Error preloading model: {e}")
        return False


def check_cache_status():
    """Check if models are already cached."""
    cache_locations = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/torch/sentence_transformers"),
    ]

    for cache_dir in cache_locations:
        if os.path.exists(cache_dir):
            try:
                contents = os.listdir(cache_dir)
                for item in contents:
                    if "sentence-transformers" in item.lower() or "minilm" in item.lower():
                        print(f"[OK] Found cached model: {item}")
                        return True
            except (OSError, PermissionError):
                continue

    print("[INFO] No cached models found")
    return False


def main():
    print("MCP Memory Service - Model Preloader")
    print("=" * 50)

    # Check current cache status
    print("\n[1] Checking cache status...")
    models_cached = check_cache_status()

    if models_cached:
        print("[OK] Models are already cached - no download needed")
        return True

    # Preload models
    print("\n[2] Preloading models...")
    success = preload_sentence_transformers()

    if success:
        print("\n[OK] Model preloading complete!")
        print("[INFO] MCP Memory Service should now start without downloading models")
    else:
        print("\n[WARNING] Model preloading failed - server may need to download during startup")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
