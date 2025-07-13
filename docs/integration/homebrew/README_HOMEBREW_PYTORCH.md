# Using MCP Memory Service with Homebrew PyTorch

This guide provides instructions for running MCP Memory Service using Homebrew PyTorch installation on macOS.

## Background

The MCP Memory Service normally installs PyTorch and other ML dependencies via pip. However, on some older Macs or specific Python versions, the pip-installed PyTorch may have compatibility issues. This guide shows how to use your existing Homebrew PyTorch installation instead.

## Prerequisites

- macOS system with Homebrew installed
- PyTorch installed via Homebrew (`brew install pytorch`)
- Python 3.10 or newer

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mcp-memory-service.git
   cd mcp-memory-service
   ```

2. Run the installer with special options:
   ```bash
   python install.py --storage-backend sqlite_vec --skip-pytorch
   ```
   
   Alternatively, you can automatically detect Homebrew PyTorch:
   ```bash
   python install.py --storage-backend sqlite_vec --use-homebrew-pytorch
   ```

3. The installer will:
   - Detect your Homebrew PyTorch installation
   - Install the SQLite-vec backend (lightweight alternative to ChromaDB)
   - Configure the system to use ONNX runtime with your Homebrew PyTorch
   - Skip attempting to install PyTorch via pip

## Running the Service

1. Set the necessary environment variables:
   ```bash
   export MCP_MEMORY_USE_ONNX=1
   export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
   ```

2. Start the memory service:
   ```bash
   memory
   ```

3. To test the installation, run:
   ```bash
   python test_memory.py
   ```

## How It Works

This configuration uses:

1. **SQLite-vec Backend**: A lightweight alternative to ChromaDB that works well on older hardware
2. **ONNX Runtime**: Provides optimized inference using your existing PyTorch installation
3. **Homebrew PyTorch**: Uses your system-installed PyTorch rather than attempting to install via pip

## Troubleshooting

If you encounter issues:

1. Ensure your Homebrew PyTorch is up to date:
   ```bash
   brew upgrade pytorch
   ```

2. Try reinstalling with explicit options:
   ```bash
   python install.py --storage-backend sqlite_vec --skip-pytorch
   ```

3. Check the compatibility of your Python version with Homebrew PyTorch

4. Ensure environment variables are set correctly:
   ```bash
   export MCP_MEMORY_USE_ONNX=1
   export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
   ```

5. Try running the test script:
   ```bash
   python test_memory.py
   ```
