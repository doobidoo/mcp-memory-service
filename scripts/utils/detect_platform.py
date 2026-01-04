#!/usr/bin/env python3
"""
Platform and hardware detection helper for update_and_restart.sh

This script provides JSON output for bash scripts to consume, using the same
GPU detection logic as install.py for consistency.

Usage:
    python detect_platform.py

Output (JSON):
    {
        "os": "darwin|linux|windows",
        "arch": "arm64|x86_64|amd64",
        "is_arm": true|false,
        "accelerator": "mps|cuda|rocm|directml|cpu",
        "pytorch_index_url": "...",
        "needs_directml": true|false
    }
"""

import sys
import platform
import json
from pathlib import Path

# Add src to path for imports
script_dir = Path(__file__).parent
project_dir = script_dir.parent.parent
src_dir = project_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from mcp_memory_service.utils.gpu_detection import detect_gpu
except ImportError as e:
    # Log to stderr for debugging, then provide graceful fallback to stdout
    print(f"Warning: Failed to import 'mcp_memory_service.utils.gpu_detection', falling back to CPU-only. Error: {e}", file=sys.stderr)

    # Fallback: Output minimal CPU-only config to stdout (graceful degradation)
    # This allows bash scripts to continue with safe defaults
    print(json.dumps({
        "os": platform.system().lower(),
        "arch": platform.machine().lower(),
        "is_arm": platform.machine().lower() in ("arm64", "aarch64"),
        "is_x86": platform.machine().lower() in ("x86_64", "amd64", "x64"),
        "accelerator": "cpu",
        "has_cuda": False,
        "has_rocm": False,
        "has_mps": False,
        "has_directml": False,
        "cuda_version": None,
        "rocm_version": None,
        "directml_version": None,
        "pytorch_index_url": "https://download.pytorch.org/whl/cpu",
        "needs_directml": False
    }))
    sys.exit(0)  # Exit with success for graceful fallback


def main():
    """Detect platform and hardware, output JSON for bash consumption."""

    # Detect system info
    system = platform.system().lower()
    machine = platform.machine().lower()
    is_windows = system == "windows"
    is_macos = system == "darwin"
    is_linux = system == "linux"
    is_arm = machine in ("arm64", "aarch64")
    is_x86 = machine in ("x86_64", "amd64", "x64")

    system_info = {
        "is_windows": is_windows,
        "is_macos": is_macos,
        "is_linux": is_linux,
        "is_arm": is_arm,
        "is_x86": is_x86
    }

    # Detect GPU using shared module
    gpu_info = detect_gpu(system_info)
    accelerator = gpu_info.get("accelerator", "cpu")

    # Determine PyTorch index URL
    pytorch_index_url = ""
    needs_directml = False

    if accelerator == "cuda":
        # CUDA detected - determine version-specific index (default to cu118)
        pytorch_index_url = "https://download.pytorch.org/whl/cu118"  # Safe default for CUDA 11+
        cuda_version = gpu_info.get("cuda_version")
        if cuda_version:
            cuda_major = cuda_version.split('.')[0]
            if cuda_major == "12":
                pytorch_index_url = "https://download.pytorch.org/whl/cu121"
            elif cuda_major == "10":
                pytorch_index_url = "https://download.pytorch.org/whl/cu102"
            # For CUDA 11 or other versions, use default cu118

    elif accelerator == "rocm":
        # ROCm detected - use ROCm index
        pytorch_index_url = "https://download.pytorch.org/whl/rocm5.6"

    elif accelerator == "mps":
        # Apple Silicon MPS - use default PyTorch (has MPS support built-in)
        # No special index needed, standard PyPI has MPS support
        pytorch_index_url = ""

    else:
        # Handles 'directml', 'cpu', and any other fallbacks
        pytorch_index_url = "https://download.pytorch.org/whl/cpu"
        if accelerator == "directml":
            # DirectML on Windows - use CPU PyTorch + torch-directml package
            needs_directml = True

    # Build output
    output = {
        "os": system,
        "arch": machine,
        "is_arm": is_arm,
        "is_x86": is_x86,
        "accelerator": accelerator,
        "has_cuda": gpu_info.get("has_cuda", False),
        "has_rocm": gpu_info.get("has_rocm", False),
        "has_mps": gpu_info.get("has_mps", False),
        "has_directml": gpu_info.get("has_directml", False),
        "cuda_version": gpu_info.get("cuda_version"),
        "rocm_version": gpu_info.get("rocm_version"),
        "directml_version": gpu_info.get("directml_version"),
        "pytorch_index_url": pytorch_index_url,
        "needs_directml": needs_directml
    }

    # Output JSON
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
