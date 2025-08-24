# MCP Memory Service - Installation Guide

**Version**: 0.2.2+  
**Last Updated**: 2025-07-26  
**Supports**: ChromaDB + SQLite-vec backends, HTTP/SSE API

## Prerequisites

- Python 3.10 or newer
- pip (latest version recommended)
- A virtual environment (venv or conda)
- Git (to clone the repository)

## Quick Installation by Hardware Type

### 🖥️ Legacy Hardware (2013-2017 Intel Macs)
**Best for**: 2015 MacBook Pro, older Intel Macs without GPU

```bash
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service
python install.py --legacy-hardware
```

**What this does:**
- ✅ Detects older Intel Mac hardware
- ✅ Recommends SQLite-vec backend (lightweight)
- ✅ Uses Homebrew PyTorch if available (better compatibility)
- ✅ Configures ONNX runtime for CPU-only inference
- ✅ Optimizes for limited memory systems

### 🚀 Modern Hardware (2018+ Macs, GPU-enabled systems)
**Best for**: M1/M2/M3 Macs, modern Intel systems, Windows with GPU

```bash
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service
python install.py
```

**What this does:**
- ✅ Auto-detects available hardware acceleration
- ✅ Recommends ChromaDB for full features
- ✅ Configures GPU acceleration (CUDA/MPS/DirectML)
- ✅ Installs latest PyTorch and sentence-transformers

### 🌐 Server/Headless Installation
**Best for**: Linux servers, Docker deployments, CI/CD

```bash
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service
python install.py --server-mode --storage-backend sqlite_vec
```

## Standard Installation Steps

### 1. Clone and Setup Environment

```bash
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Create and activate virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 2. Run Installation

```bash
python install.py
```

🌟 **Multi-Client Setup**: The installer will automatically detect MCP applications (Claude Desktop, VS Code, Continue, etc.) and offer to configure shared memory access. Choose 'y' for universal multi-client setup.

### 3. Verify Installation

```bash
python scripts/verify_environment.py
```

## Docker Installation

For cross-platform deployment:

```bash
# Clone repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Build and run with Docker Compose
docker-compose up -d
```

## Command Line Options

### Basic Usage
```bash
python install.py [OPTIONS]
```

### Core Options

| Option | Description | Example |
|--------|-------------|---------|
| `--dev` | Install in development mode | `python install.py --dev` |
| `--chroma-path PATH` | Custom ChromaDB storage path | `python install.py --chroma-path /custom/path` |
| `--backups-path PATH` | Custom backup storage path | `python install.py --backups-path /custom/backups` |

### Storage Backend Options

| Backend | Description | Best For |
|---------|-------------|----------|
| `chromadb` | Full-featured vector database | High-memory systems, full features |
| `sqlite_vec` | Lightweight alternative | Resource-constrained systems |
| `auto_detect` | Auto-selection with fallback | Uncertain hardware capabilities |

```bash
# Force SQLite-vec backend
python install.py --storage-backend sqlite_vec

# Force ChromaDB backend  
python install.py --storage-backend chromadb

# Auto-detection with fallback
python install.py --storage-backend auto_detect
```

### Hardware-Specific Options

| Option | Description | Use Case |
|--------|-------------|----------|
| `--legacy-hardware` | Optimize for older systems | 2013-2017 Intel Macs |
| `--server-mode` | Headless server installation | Linux servers, Docker |
| `--force-cpu` | Disable GPU acceleration | Troubleshooting GPU issues |

### Multi-Client Options

| Option | Description | Example |
|--------|-------------|---------|
| `--multi-client` | Enable shared memory access | `python install.py --multi-client` |
| `--claude-only` | Configure for Claude Desktop only | `python install.py --claude-only` |

### Claude Code Integration (v2.2.0)

| Option | Description | Example |
|--------|-------------|---------|
| `--install-claude-commands` | Install conversational memory commands | `python install.py --install-claude-commands` |
| `--skip-claude-commands-prompt` | Skip interactive commands installation prompt | `python install.py --skip-claude-commands-prompt` |

## Platform-Specific Installation

- **Windows**: See [windows-setup.md](../platforms/windows.md)
- **Ubuntu/Linux**: See [ubuntu-setup.md](../platforms/ubuntu.md)
- **macOS Intel (Legacy)**: See [macos-intel.md](../platforms/macos-intel.md)

## Troubleshooting

Common installation issues and solutions can be found in [troubleshooting.md](../troubleshooting/general.md).

## Next Steps

After installation:
1. Configure your MCP client (Claude Desktop, VS Code, etc.)
2. Test the connection with `python scripts/test-connection.py`
3. Read the [User Guide](../guides/claude_integration.md) for usage instructions