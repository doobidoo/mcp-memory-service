# Platform Detection Helper

## Overview

`detect_platform.py` provides unified hardware and OS detection for bash scripts, using the same `gpu_detection.py` module as `install.py` for consistency.

## Usage

```bash
# Run detection
python scripts/utils/detect_platform.py

# Output (JSON):
{
  "os": "darwin",
  "arch": "arm64",
  "is_arm": true,
  "is_x86": false,
  "accelerator": "mps",
  "has_cuda": false,
  "has_rocm": false,
  "has_mps": true,
  "has_directml": false,
  "cuda_version": null,
  "rocm_version": null,
  "directml_version": null,
  "pytorch_index_url": "",
  "needs_directml": false
}
```

## Supported Platforms

| Platform | Detection | PyTorch Index |
|----------|-----------|---------------|
| **Apple Silicon (M1/M2/M3)** | MPS via system_profiler | Default PyPI (MPS built-in) |
| **NVIDIA GPU (CUDA)** | nvcc version | cu121/cu118/cu102 |
| **AMD GPU (ROCm)** | rocminfo | rocm5.6 |
| **Windows DirectML** | torch-directml import | CPU + directml package |
| **CPU-only** | Fallback | CPU index |

## Integration with update_and_restart.sh

The script automatically:
1. Detects hardware platform (MPS/CUDA/ROCm/DirectML/CPU)
2. Selects optimal PyTorch index URL
3. Installs DirectML package if needed (Windows)
4. Falls back to basic detection if Python helper unavailable

## Benefits vs. Old Logic

**Old (Bash-only):**
- ❌ Only detected macOS vs. Linux with nvidia-smi
- ❌ Treated all macOS as CPU-only (performance loss on M-series)
- ❌ No ROCm, DirectML, or MPS support

**New (Python-based):**
- ✅ Detects MPS, CUDA, ROCm, DirectML, CPU
- ✅ Consistent with install.py logic
- ✅ Optimal PyTorch selection per platform
- ✅ Graceful fallback to old logic if detection fails

## Example Output (macOS M2)

```bash
▶  Installing dependencies (editable mode)...
ℹ  Existing venv Python version: 3.13
ℹ  Installing with venv pip (this may take 1-2 minutes)...
ℹ  Apple Silicon MPS detected - using MPS-optimized PyTorch
```

## Example Output (Linux with NVIDIA)

```bash
▶  Installing dependencies (editable mode)...
ℹ  CUDA detected (12.1) - using optimized PyTorch
  Installing with: --extra-index-url https://download.pytorch.org/whl/cu121
```

## Maintenance

The detection logic is centralized in `src/mcp_memory_service/utils/gpu_detection.py`. Updates to that module automatically benefit both `install.py` and `update_and_restart.sh`.
