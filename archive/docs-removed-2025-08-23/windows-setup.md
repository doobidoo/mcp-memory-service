# Windows Setup Guide for MCP Memory Service

This guide provides comprehensive instructions for setting up and running the MCP Memory Service on Windows systems, including handling common Windows-specific issues.

## Installation

### Prerequisites
- Python 3.10 or newer
- Git for Windows
- Visual Studio Build Tools (for PyTorch)

### Recommended Installation (Using UV)

1. Install UV:
```bash
pip install uv
```

2. Clone and setup:
```bash
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install -e .
```

### Alternative: Windows-Specific Installation

If you encounter issues with UV, use our Windows-specific installation script:

```bash
python scripts/install_windows.py
```

This script handles:
1. Detecting CUDA availability
2. Installing the correct PyTorch version
3. Setting up dependencies without conflicts
4. Verifying the installation

## Configuration

### Claude Desktop Configuration

1. Create or edit your Claude Desktop configuration file:
   - Location: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the following configuration:
```json
{
  "memory": {
    "command": "python",
    "args": [
      "C:\\path\\to\\mcp-memory-service\\memory_wrapper.py"
    ],
    "env": {
      "MCP_MEMORY_CHROMA_PATH": "C:\\Users\\YourUsername\\AppData\\Local\\mcp-memory\\chroma_db",
      "MCP_MEMORY_BACKUPS_PATH": "C:\\Users\\YourUsername\\AppData\\Local\\mcp-memory\\backups"
    }
  }
}
```

### Environment Variables

Important Windows-specific environment variables:
```
MCP_MEMORY_USE_DIRECTML=1 # Enable DirectML acceleration if CUDA is not available
PYTORCH_ENABLE_MPS_FALLBACK=0 # Disable MPS (not needed on Windows)
```

## Common Windows-Specific Issues

### PyTorch Installation Issues

If you see errors about PyTorch installation:

1. Use the Windows-specific installation script:
```bash
python scripts/install_windows.py
```

2. Or manually install PyTorch with the correct index URL:
```bash
pip install torch==2.1.0 torchvision==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### JSON Parsing Errors

If you see "Unexpected token" errors in Claude Desktop:

**Symptoms:**
```
Unexpected token 'U', "Using Chro"... is not valid JSON
Unexpected token 'I', "[INFO] Star"... is not valid JSON
```

**Solution:**
- Update to the latest version which includes Windows-specific stream handling fixes
- Use the memory wrapper script which properly handles stdout/stderr separation

### Recursion Errors

If you encounter recursion errors:

1. Run the sitecustomize fix script:
```bash
python scripts/fix_sitecustomize.py
```

2. Restart your Python environment

## Debugging Tools

Windows-specific debugging tools:

```bash
# Verify PyTorch installation
python scripts/verify_pytorch_windows.py

# Check environment compatibility
python scripts/verify_environment_enhanced.py

# Test the memory server
python scripts/run_memory_server.py
```

## Log Files

Important log locations on Windows:
- Claude Desktop logs: `%APPDATA%\Claude\logs\mcp-server-memory.log`
- Memory service logs: `%LOCALAPPDATA%\mcp-memory\logs\memory_service.log`

## Performance Optimization

### GPU Acceleration

1. CUDA (recommended if available):
- Ensure NVIDIA drivers are up to date
- CUDA toolkit is not required (bundled with PyTorch)

2. DirectML (alternative):
- Enable with `MCP_MEMORY_USE_DIRECTML=1`
- Useful for AMD GPUs or when CUDA is not available

### Memory Usage

If experiencing memory issues:
1. Reduce batch size:
```bash
set MCP_MEMORY_BATCH_SIZE=4
```

2. Use a smaller model:
```bash
set MCP_MEMORY_MODEL_NAME=paraphrase-MiniLM-L3-v2
```

## Getting Help

If you encounter Windows-specific issues:
1. Check the logs in `%APPDATA%\Claude\logs\`
2. Run verification tools mentioned above
3. Contact support via Telegram: t.me/doobeedoo