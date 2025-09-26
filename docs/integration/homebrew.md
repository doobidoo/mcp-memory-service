# Homebrew PyTorch Integration Guide

This guide covers the integration of MCP Memory Service with Homebrew-installed PyTorch, providing a lightweight solution for systems with complex Python environments or limited resources.

## Overview

The Homebrew PyTorch integration allows MCP Memory Service to use system-installed PyTorch via Homebrew, avoiding package conflicts and reducing installation complexity. This solution uses SQLite-vec as the storage backend with ONNX runtime for CPU-optimized embeddings.

### Key Components

- **SQLite-vec**: Lightweight vector database backend
- **ONNX Runtime**: CPU-optimized inference engine
- **Subprocess Isolation**: Process isolation to avoid import conflicts
- **Custom Integration Layer**: Bridge between MCP protocol and Homebrew environment

## Quick Start

### Prerequisites

- Homebrew installed
- Python 3.10+
- PyTorch via Homebrew: `brew install pytorch`

### Installation

```bash
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Install with Homebrew PyTorch integration
python install.py --use-homebrew-pytorch --storage-backend sqlite_vec
```

### Running the Service

```bash
# Using the provided script
./scripts/run_with_homebrew_pytorch.sh

# Or manually
python scripts/homebrew/homebrew_server.py
```

### Claude Desktop Configuration

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "memory": {
      "command": "/path/to/mcp-memory-service/scripts/run_with_homebrew_pytorch.sh",
      "env": {
        "MCP_MEMORY_USE_HOMEBREW_PYTORCH": "true",
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec"
      }
    }
  }
}
```

## Detailed Setup Instructions

### 1. Verify Homebrew PyTorch Installation

```bash
# Check if PyTorch is installed via Homebrew
brew list | grep pytorch

# Verify PyTorch accessibility
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### 2. Install MCP Memory Service

```bash
# Clone repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with Homebrew integration
python install.py --use-homebrew-pytorch --skip-pytorch
```

### 3. Configure Environment Variables

```bash
export MCP_MEMORY_USE_HOMEBREW_PYTORCH=true
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_MEMORY_USE_ONNX=true
export MCP_MEMORY_SQLITE_PATH="$HOME/.mcp_memory_sqlite/memory.db"
```

### 4. Test Installation

```bash
# Run diagnostic tests
python scripts/verify_environment.py

# Test Homebrew PyTorch detection
python -c "from src.mcp_memory_service.integrations.homebrew.embeddings import HomebrewPyTorchEmbeddings; print('Homebrew integration working')"

# Test server startup
python scripts/homebrew/homebrew_server.py --test
```

## Technical Implementation

### Architecture Overview

The integration uses a subprocess-based architecture to isolate the Homebrew PyTorch environment from the MCP server process:

```
MCP Server Process
    ↓ (subprocess call)
Homebrew PyTorch Process
    ↓ (file-based exchange)
Embedding Results
```

### Module Override Patterns

The integration implements several technical patterns for compatibility:

#### 1. Runtime Class Replacement

```python
# Override storage backend selection
def get_storage_backend():
    if os.getenv('MCP_MEMORY_USE_HOMEBREW_PYTORCH'):
        return SqliteVecStorage
    return ChromaStorage
```

#### 2. Subprocess Execution Pattern

```python
def generate_embeddings_subprocess(texts, model_name):
    """Generate embeddings using subprocess isolation"""
    script = f"""
import sys
sys.path.insert(0, '/opt/homebrew/lib/python3.11/site-packages')
import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('{model_name}')
embeddings = model.encode({texts!r})
print(embeddings.tolist())
"""
    
    result = subprocess.run([
        sys.executable, '-c', script
    ], capture_output=True, text=True, env=homebrew_env)
    
    return json.loads(result.stdout)
```

#### 3. MCP Protocol Compliance

```python
def wrap_mcp_handler(handler_func):
    """Wrapper to ensure MCP protocol compliance"""
    async def wrapper(*args, **kwargs):
        try:
            # Redirect stderr to prevent protocol pollution
            with redirect_stderr(StringIO()):
                result = await handler_func(*args, **kwargs)
            return result
        except Exception as e:
            # Convert to MCP-compliant error format
            return {"error": str(e)}
    return wrapper
```

### Environment Detection

The system automatically detects Homebrew PyTorch availability:

```python
def detect_homebrew_pytorch():
    """Detect if Homebrew PyTorch is available"""
    homebrew_paths = [
        '/opt/homebrew/lib/python3.11/site-packages',
        '/usr/local/lib/python3.11/site-packages'
    ]
    
    for path in homebrew_paths:
        torch_path = os.path.join(path, 'torch')
        if os.path.exists(torch_path):
            return path
    return None
```

## Troubleshooting

### Diagnostic Commands

#### Check Environment Status

```bash
# Verify Homebrew PyTorch detection
python -c "
import os
import sys
print('Homebrew paths:')
for path in ['/opt/homebrew/lib/python3.11/site-packages', '/usr/local/lib/python3.11/site-packages']:
    exists = os.path.exists(os.path.join(path, 'torch'))
    print(f'  {path}: {\"✓\" if exists else \"✗\"}')"

# Check environment variables
env | grep MCP_MEMORY
```

#### Service Health Check

```bash
# Test server startup
python scripts/homebrew/homebrew_server.py --health-check

# Check database connectivity
python -c "
from src.mcp_memory_service.storage.sqlite_vec import SqliteVecStorage
storage = SqliteVecStorage()
print('Database connection: ✓')
"
```

#### Advanced Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export MCP_MEMORY_DEBUG=true

# Run with subprocess tracing
strace -e trace=execve -f python scripts/homebrew/homebrew_server.py 2>&1 | grep execve

# Database analysis
sqlite3 ~/.mcp_memory_sqlite/memory.db ".schema"
```

### Common Issues and Solutions

#### 1. Import Conflicts

**Symptom**: `ImportError` or version conflicts
**Solution**: Ensure virtual environment isolation:

```bash
# Check current Python path
python -c "import sys; print('\\n'.join(sys.path))"

# Reset virtual environment
deactivate
rm -rf venv
python -m venv venv --clear
source venv/bin/activate
```

#### 2. Subprocess Communication Failures

**Symptom**: Embedding generation timeouts or empty results
**Solution**: Test subprocess execution manually:

```bash
# Test subprocess isolation
python -c "
import subprocess
import sys
result = subprocess.run([sys.executable, '-c', 'import torch; print(torch.__version__)'], 
                       capture_output=True, text=True)
print(f'stdout: {result.stdout}')
print(f'stderr: {result.stderr}')
print(f'returncode: {result.returncode}')
"
```

#### 3. Storage Backend Issues

**Symptom**: Database creation or access errors
**Solution**: Check SQLite-vec installation and permissions:

```bash
# Verify SQLite-vec availability
python -c "import sqlite_vec; print('SQLite-vec available')"

# Check database permissions
ls -la ~/.mcp_memory_sqlite/
chmod 755 ~/.mcp_memory_sqlite/
```

#### 4. MCP Protocol Errors

**Symptom**: Claude Desktop connection failures
**Solution**: Verify protocol compliance:

```bash
# Test MCP protocol directly
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}' | \
  python scripts/homebrew/homebrew_server.py --stdin
```

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_MEMORY_USE_HOMEBREW_PYTORCH` | `false` | Enable Homebrew PyTorch integration |
| `MCP_MEMORY_STORAGE_BACKEND` | `auto` | Force SQLite-vec backend |
| `MCP_MEMORY_USE_ONNX` | `false` | Use ONNX runtime for inference |
| `MCP_MEMORY_SQLITE_PATH` | `~/.mcp_memory_sqlite/memory.db` | SQLite-vec database file path |
| `MCP_MEMORY_HOMEBREW_PYTHON_PATH` | `auto-detect` | Override Homebrew Python path |
| `MCP_MEMORY_DEBUG` | `false` | Enable debug logging |

## Performance Considerations

### Memory Usage

The Homebrew integration is optimized for systems with limited memory:

- **Subprocess isolation**: Prevents memory leaks in the main process
- **On-demand loading**: Models loaded only when needed
- **SQLite-vec efficiency**: Minimal memory footprint for vector storage

### CPU Optimization

- **ONNX runtime**: CPU-optimized inference
- **Batch processing**: Efficient handling of multiple embeddings
- **Caching**: Avoid redundant model loading

## Advanced Configuration

### Custom Model Selection

```bash
export MCP_MEMORY_SENTENCE_TRANSFORMER_MODEL="all-MiniLM-L6-v2"
export MCP_MEMORY_ONNX_MODEL_PATH="/path/to/custom/model.onnx"
```

### Multi-Client Setup

For shared access across multiple MCP clients:

```bash
# Install with multi-client support
python install.py --use-homebrew-pytorch --multi-client

# Configure shared database location
export MCP_MEMORY_SQLITE_PATH="/shared/mcp_memory/memory.db"
```

## Development and Contributions

### Testing the Integration

```bash
# Run integration tests
pytest tests/homebrew/ -v

# Run performance benchmarks
python tests/performance/test_homebrew_performance.py
```

### Adding New Features

When extending the Homebrew integration:

1. Follow the subprocess isolation pattern
2. Maintain MCP protocol compliance
3. Add comprehensive error handling
4. Update environment variable documentation
5. Include diagnostic commands for troubleshooting

## Related Documentation

- [Installation Guide](../installation/master-guide.md) - General installation instructions
- [Storage Backends](../guides/STORAGE_BACKENDS.md) - SQLite-vec configuration
- [Troubleshooting](../troubleshooting/general.md) - General troubleshooting guide
- [macOS Intel Setup](../platforms/macos-intel.md) - Platform-specific considerations
