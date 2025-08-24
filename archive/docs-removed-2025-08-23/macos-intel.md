# macOS Intel Setup Guide

This guide addresses the specific challenges of running MCP Memory Service on Intel-based Mac systems, including both legacy (2013-2017) and modern (2018+) Intel Macs.

## Hardware Profiles

### Legacy Intel Macs (2013-2017)
**Target Hardware**: 2015 MacBook Pro, older Intel Macs without dedicated GPU  
**Optimization**: Maximum compatibility, minimal resource usage  
**Recommended Backend**: SQLite-vec with ONNX runtime

**Typical specs this applies to:**
- MacBook Pro (15-inch, Mid 2015)
- MacBook Pro (13-inch, Early 2015)
- MacBook Air (11-inch/13-inch, 2013-2017)
- iMac (21.5-inch/27-inch, 2013-2017) with integrated graphics

### Modern Intel Macs (2018+)
**Target Hardware**: 2018+ Intel Macs with better GPU support  
**Optimization**: Balanced performance and compatibility  
**Recommended Backend**: ChromaDB with CPU optimization

## Why Special Setup is Needed

Intel-based Mac systems require special consideration for several reasons:

1. **PyTorch Compatibility**: PyTorch has moved toward optimizing for Apple Silicon, with some compatibility challenges on Intel Macs
2. **NumPy Version Conflicts**: Newer NumPy 2.x can cause compatibility issues with other ML libraries
3. **Python Version Sensitivity**: Python 3.13+ has introduced breaking changes that affect ML libraries
4. **Memory Constraints**: Limited RAM on older systems requires careful resource management
5. **ChromaDB Installation Issues**: Complex dependencies often fail on older systems

## Installation

### Prerequisites

- Python 3.10 (recommended for best compatibility)
- Git to clone the repository
- Xcode Command Line Tools: `xcode-select --install`

### Automatic Installation (Recommended)

The installer automatically detects Intel Mac hardware:

```bash
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# For legacy hardware (2013-2017)
python install.py --legacy-hardware

# For modern Intel Macs (2018+)
python install.py --intel-mac
```

### Manual Installation

If you prefer manual control:

#### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Create Python 3.10 virtual environment
python3.10 -m venv venv_py310
source venv_py310/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### 2. Install Dependencies

For **Legacy Intel Macs (2013-2017)**:

```bash
# Install with SQLite-vec backend
pip install -e .
pip install sentence-transformers onnx onnxruntime

# Downgrade NumPy for compatibility
pip uninstall -y numpy
pip install numpy==1.25.2

# Configure for SQLite-vec
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_MEMORY_USE_ONNX=true
```

For **Modern Intel Macs (2018+)**:

```bash
# Install with ChromaDB support
pip install -e .
pip install chromadb sentence-transformers

# Install CPU-optimized PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Configure for ChromaDB
export MCP_MEMORY_STORAGE_BACKEND=chromadb
```

### Hardware Detection

The installer automatically detects legacy hardware by checking:

```python
# System detection criteria
is_legacy_mac = (
    platform.system() == "Darwin" and           # macOS
    platform.machine() in ("x86_64", "x64") and # Intel processor
    year_of_hardware < 2018 and                 # Pre-2018 models
    not has_dedicated_gpu                       # No discrete GPU
)
```

## Configuration

### Environment Variables

#### For Legacy Intel Macs

```bash
# Core configuration
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_MEMORY_USE_ONNX=true
export MCP_MEMORY_SQLITE_VEC_PATH="$HOME/.mcp_memory_sqlite"

# Performance optimization
export MCP_MEMORY_CPU_ONLY=true
export MCP_MEMORY_MAX_MEMORY_MB=2048
export MCP_MEMORY_SENTENCE_TRANSFORMER_MODEL="all-MiniLM-L6-v2"

# Compatibility settings
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MCP_MEMORY_USE_ONNX_RUNTIME=true
```

#### For Modern Intel Macs

```bash
# Core configuration
export MCP_MEMORY_STORAGE_BACKEND=chromadb
export MCP_MEMORY_CHROMA_PATH="$HOME/.mcp_memory_chroma"

# Performance optimization
export MCP_MEMORY_CPU_OPTIMIZATION=true
export MCP_MEMORY_SENTENCE_TRANSFORMER_MODEL="all-MiniLM-L12-v2"

# Intel-specific settings
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
```

### Claude Desktop Configuration

#### Legacy Intel Mac Configuration

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["/path/to/mcp-memory-service/scripts/legacy_intel_mac/run_mcp_memory.sh"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_MEMORY_USE_ONNX": "true",
        "MCP_MEMORY_CPU_ONLY": "true"
      }
    }
  }
}
```

#### Modern Intel Mac Configuration

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["/path/to/mcp-memory-service/src/mcp_memory_service/server.py"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "chromadb",
        "MCP_MEMORY_CPU_OPTIMIZATION": "true"
      }
    }
  }
}
```

## Provided Scripts

The repository includes several Intel Mac-specific scripts:

### Legacy Intel Mac Scripts

- `scripts/legacy_intel_mac/run_mcp_memory.sh` - Standard startup script
- `scripts/legacy_intel_mac/run_mcp_memory_foreground.sh` - Foreground mode with debugging
- `scripts/legacy_intel_mac/start_memory_for_claude.sh` - Claude-optimized startup

### Usage Examples

```bash
# For foreground mode (shows all output, can be stopped with Ctrl+C)
./scripts/legacy_intel_mac/run_mcp_memory_foreground.sh

# For background mode (runs in background, logs to file)
./scripts/legacy_intel_mac/run_mcp_memory.sh

# For Claude Desktop integration
./scripts/legacy_intel_mac/start_memory_for_claude.sh
```

## Performance Optimization

### For Legacy Intel Macs

1. **Use SQLite-vec Backend**: Lighter weight than ChromaDB
2. **ONNX Runtime**: CPU-optimized inference
3. **Memory Management**: Limited model loading and caching
4. **Smaller Models**: Use compact sentence transformer models

```bash
# Optimization settings
export MCP_MEMORY_BATCH_SIZE=16
export MCP_MEMORY_CACHE_SIZE=100
export MCP_MEMORY_MODEL_CACHE_SIZE=1
```

### For Modern Intel Macs

1. **CPU Optimization**: Multi-threaded processing
2. **Intelligent Caching**: Larger cache sizes
3. **Better Models**: Higher quality embeddings

```bash
# Performance tuning
export MCP_MEMORY_BATCH_SIZE=32
export MCP_MEMORY_CACHE_SIZE=1000
export MCP_MEMORY_MODEL_CACHE_SIZE=3
```

## Troubleshooting

### Common Issues

#### 1. NumPy Compatibility Errors

**Symptom**: 
```
AttributeError: module 'numpy' has no attribute 'float'
```

**Solution**:
```bash
pip uninstall -y numpy
pip install numpy==1.25.2
```

#### 2. PyTorch Installation Issues

**Symptom**: PyTorch fails to install or import

**Solution**:
```bash
# For legacy Macs - use CPU-only PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Set fallback environment variable
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

#### 3. ChromaDB Installation Failures

**Symptom**: ChromaDB dependency issues on legacy hardware

**Solution**: Switch to SQLite-vec backend:
```bash
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
python install.py --storage-backend sqlite_vec
```

#### 4. Memory Issues

**Symptom**: Out of memory errors during embedding generation

**Solution**: Reduce batch size and enable memory optimization:
```bash
export MCP_MEMORY_BATCH_SIZE=8
export MCP_MEMORY_MAX_MEMORY_MB=1024
export MCP_MEMORY_LOW_MEMORY_MODE=true
```

### Diagnostic Commands

#### System Information

```bash
# Check macOS version
sw_vers

# Check available memory
system_profiler SPMemoryDataType | grep Size

# Check CPU information
sysctl -n machdep.cpu.brand_string

# Check Python version and location
python --version
which python
```

#### Environment Verification

```bash
# Check virtual environment
echo $VIRTUAL_ENV

# Verify key packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sentence_transformers; print('SentenceTransformers: OK')"
python -c "import sqlite3; print('SQLite3: OK')"

# Test ONNX runtime (for legacy Macs)
python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')"
```

#### Server Testing

```bash
# Test server startup
python scripts/verify_environment.py

# Test memory operations
python -c "
from src.mcp_memory_service.storage.sqlite_vec import SqliteVecStorage
storage = SqliteVecStorage()
print('Storage backend: OK')
"

# Test embedding generation
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(['test'])
print(f'Embedding generated: {len(embedding[0])} dimensions')
"
```

## Homebrew Integration

For Intel Macs with Homebrew-installed PyTorch, see the dedicated [Homebrew Integration Guide](../integration/homebrew.md).

## Performance Benchmarks

### Typical Performance (Legacy Intel Mac)

- **Memory Storage**: ~100ms per memory
- **Search Operations**: ~200ms for 100 memories
- **Embedding Generation**: ~500ms for short text
- **Memory Usage**: ~200MB baseline

### Typical Performance (Modern Intel Mac)

- **Memory Storage**: ~50ms per memory
- **Search Operations**: ~100ms for 1000 memories
- **Embedding Generation**: ~200ms for short text
- **Memory Usage**: ~400MB baseline

## Related Documentation

- [Installation Guide](../installation/master-guide.md) - General installation instructions
- [Homebrew Integration](../integration/homebrew.md) - Homebrew PyTorch setup
- [Troubleshooting](../troubleshooting/general.md) - macOS-specific troubleshooting
- [Performance Tuning](../implementation/performance.md) - Performance optimization guide