# MCP Memory Service

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![smithery badge](https://smithery.ai/badge/@doobidoo/mcp-memory-service)](https://smithery.ai/server/@doobidoo/mcp-memory-service)
[![Verified on MseeP](https://mseep.ai/badge.svg)](https://mseep.ai/app/0513fb92-e941-4fe0-9948-2a1dbb870dcf)

An MCP server providing semantic memory and persistent storage capabilities for Claude Desktop using ChromaDB and sentence transformers. This service enables long-term memory storage with semantic search capabilities, making it ideal for maintaining context across conversations and instances.

<img width="240" alt="grafik" src="https://github.com/user-attachments/assets/eab1f341-ca54-445c-905e-273cd9e89555" />
<a href="https://glama.ai/mcp/servers/bzvl3lz34o"><img width="380" height="200" src="https://glama.ai/mcp/servers/bzvl3lz34o/badge" alt="Memory Service MCP server" /></a>

## Help
- Talk to the Repo with [TalkToGitHub](https://talktogithub.com/doobidoo/mcp-memory-service)!
- Use Gitprobe to digg deeper: [GitProbe](https://gitprobe.com/doobidoo/mcp-memory-service)!

## Features

- Semantic search using sentence transformers
- **Natural language time-based recall** (e.g., "last week", "yesterday morning")
- **Enhanced tag deletion system** with flexible multi-tag support
- Tag-based memory retrieval system
- **Dual storage backends**: ChromaDB (full-featured) and SQLite-vec (lightweight, fast)
- Automatic database backups
- Memory optimization tools
- Exact match retrieval
- Debug mode for similarity analysis
- Database health monitoring
- Duplicate detection and cleanup
- Customizable embedding model
- **Cross-platform compatibility** (Apple Silicon, Intel, Windows, Linux)
- **Hardware-aware optimizations** for different environments
- **Graceful fallbacks** for limited hardware resources

### Recent Enhancements

- ✅ **PyTorch Optional**: Now works without PyTorch for basic functionality when using SQLite-vec backend
- ✅ **Improved SQLite-vec**: Robust error handling and validation for the lightweight backend
- ✅ **Intelligent Health Checks**: Backend-specific health monitoring with detailed diagnostics
- ✅ **Comprehensive Testing**: Added test scripts for all critical functions
- ✅ **API Consistency**: Enhanced `delete_by_tag` to support both single and multiple tags
- ✅ **New Delete Methods**: Added `delete_by_tags` (OR logic) and `delete_by_all_tags` (AND logic)
- ✅ **Backward Compatibility**: All existing code continues to work unchanged
- ✅ **Dashboard Integration**: Enhanced UI with multiple tag selection capabilities

## Installation

### 🚀 Intelligent Installer (Recommended)

The new unified installer automatically detects your hardware and selects the optimal configuration:

```bash
# Clone the repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the intelligent installer
python install.py
```

### 🎯 Hardware-Specific Installation

**For Intel Macs:**
For detailed setup instructions specific to Intel Macs, see our [Intel Mac Setup Guide](docs/INTEL_MAC_SETUP.md). Intel Mac users should also check out our [Legacy Intel Mac Scripts](scripts/legacy_intel_mac/README.md) for specialized startup scripts.

**For Legacy Hardware (2013-2017 Intel Macs):**
```bash
python install.py --legacy-hardware
```

**For Server/Headless Deployment:**
```bash
python install.py --server-mode
```

**For HTTP/SSE API Development:**
```bash
python install.py --enable-http-api
```

**For Migration from ChromaDB:**
```bash
python install.py --migrate-from-chromadb
```

### 🧠 What the Installer Does

1. **Hardware Detection**: CPU, GPU, memory, and platform analysis
2. **Intelligent Backend Selection**: ChromaDB vs SQLite-vec based on your hardware
3. **Platform Optimization**: macOS Intel fixes, Windows CUDA setup, Linux variations
4. **Dependency Management**: Compatible PyTorch and ML library versions
5. **Auto-Configuration**: Claude Desktop config and environment variables
6. **Migration Support**: Seamless ChromaDB to SQLite-vec migration

### 📊 Storage Backend Selection

MCP Memory Service supports two optimized storage backends:

#### SQLite-vec 🪶 (Lightweight & Fast)
**Best for**: 2015 MacBook Pro, older Intel Macs, low-memory systems, Docker deployments

- ✅ **10x faster startup** (2-3 seconds vs 15-30 seconds)
- ✅ **Single file database** (easy backup/sharing)
- ✅ **Minimal memory usage** (~150MB vs ~600MB)
- ✅ **No external dependencies**
- ✅ **HTTP/SSE API support**

#### ChromaDB 📦 (Full-Featured)
**Best for**: Modern Macs (M1/M2/M3), GPU-enabled systems, production deployments

- ✅ **Advanced vector search** with multiple metrics
- ✅ **Rich metadata support** and complex queries
- ✅ **Battle-tested scalability**
- ✅ **Extensive ecosystem** integration

**The installer automatically recommends the best backend for your hardware**, but you can override with:
```bash
python install.py --storage-backend sqlite_vec    # Lightweight
python install.py --storage-backend chromadb      # Full-featured
```

### Docker Installation

#### Docker Hub (Recommended)

The easiest way to run the Memory Service is using our pre-built Docker images:

```bash
# Pull the latest image
docker pull doobidoo/mcp-memory-service:latest

# Run with default settings (for MCP clients like Claude Desktop)
docker run -d -p 8000:8000 \
  -v $(pwd)/data/chroma_db:/app/chroma_db \
  -v $(pwd)/data/backups:/app/backups \
  doobidoo/mcp-memory-service:latest

# Run in standalone mode (for testing/development)
docker run -d -p 8000:8000 \
  -e MCP_STANDALONE_MODE=1 \
  -v $(pwd)/data/chroma_db:/app/chroma_db \
  -v $(pwd)/data/backups:/app/backups \
  doobidoo/mcp-memory-service:latest
```

#### Docker Compose

We provide multiple Docker Compose configurations for different scenarios:
- `docker-compose.yml` - Standard configuration for MCP clients (Claude Desktop)
- `docker-compose.standalone.yml` - **Standalone mode** for testing/development (prevents boot loops)
- `docker-compose.uv.yml` - Alternative configuration using UV package manager
- `docker-compose.pythonpath.yml` - Configuration with explicit PYTHONPATH settings

```bash
# Using Docker Compose (recommended)
docker-compose up

# Standalone mode (prevents boot loops)
docker-compose -f docker-compose.standalone.yml up
```

#### Building from Source

If you need to build the Docker image yourself:

```bash
# Build the image
docker build -t mcp-memory-service .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/data/chroma_db:/app/chroma_db \
  -v $(pwd)/data/backups:/app/backups \
  mcp-memory-service
```

### uvx Installation

You can install and run the Memory Service using uvx for isolated execution:

```bash
# Install uv (which includes uvx) if not already installed
pip install uv
# Or use the installer script:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Install and run the memory service
uvx mcp-memory-service

# Or install from GitHub
uvx --from git+https://github.com/doobidoo/mcp-memory-service.git mcp-memory-service
```

### Windows Installation (Special Case)

Windows users may encounter PyTorch installation issues due to platform-specific wheel availability. Use our Windows-specific installation script:

```bash
# After activating your virtual environment
python scripts/install_windows.py
```

This script handles:
1. Detecting CUDA availability and version
2. Installing the appropriate PyTorch version from the correct index URL
3. Installing other dependencies without conflicting with PyTorch
4. Verifying the installation

### Installing via Smithery

To install Memory Service for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@doobidoo/mcp-memory-service):

```bash
npx -y @smithery/cli install @doobidoo/mcp-memory-service --client claude
```

### Detailed Installation Guide

For comprehensive installation instructions and troubleshooting, see the [Installation Guide](docs/guides/installation.md).

## Claude MCP Configuration

### Standard Configuration

Add the following to your `claude_desktop_config.json` file:

```json
{
  "memory": {
    "command": "uv",
    "args": [
      "--directory",
      "your_mcp_memory_service_directory",  // e.g., "C:\\REPOSITORIES\\mcp-memory-service"
      "run",
      "memory"
    ],
    "env": {
      "MCP_MEMORY_CHROMA_PATH": "your_chroma_db_path",  // e.g., "C:\\Users\\John.Doe\\AppData\\Local\\mcp-memory\\chroma_db"
      "MCP_MEMORY_BACKUPS_PATH": "your_backups_path"  // e.g., "C:\\Users\\John.Doe\\AppData\\Local\\mcp-memory\\backups"
    }
  }
}
```

### Windows-Specific Configuration (Recommended)

For Windows users, we recommend using the wrapper script to ensure PyTorch is properly installed:

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

### SQLite-vec Configuration (Lightweight)

For a lighter-weight configuration that doesn't require PyTorch:

```json
{
  "memory": {
    "command": "python",
    "args": ["-m", "mcp_memory_service.server"],
    "cwd": "/path/to/mcp-memory-service",
    "env": {
      "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
      "MCP_MEMORY_SQLITE_PATH": "/path/to/mcp-memory/sqlite_vec.db",
      "MCP_MEMORY_BACKUPS_PATH": "/path/to/mcp-memory/backups",
      "MCP_MEMORY_USE_ONNX": "1",
      "PYTHONPATH": "/path/to/mcp-memory-service"
    }
  }
}
```

The wrapper script will:
1. Check if PyTorch is installed and properly configured
2. Install PyTorch with the correct index URL if needed
3. Run the memory server with the appropriate configuration

## Usage Guide

For detailed instructions on how to interact with the memory service in Claude Desktop:

- [Invocation Guide](docs/guides/invocation_guide.md) - Learn the specific keywords and phrases that trigger memory operations in Claude
- [Installation Guide](docs/guides/installation.md) - Detailed setup instructions

The memory service is invoked through natural language commands in your conversations with Claude. For example:
- To store: "Please remember that my project deadline is May 15th."
- To retrieve: "Do you remember what I told you about my project deadline?"
- To delete: "Please forget what I told you about my address."

See the [Invocation Guide](docs/guides/invocation_guide.md) for a complete list of commands and detailed usage examples.

## Storage Backends

The MCP Memory Service supports multiple storage backends to suit different use cases:

### ChromaDB (Default)
- **Best for**: Large memory collections (>100K entries), high-performance requirements
- **Features**: Advanced vector indexing, excellent query performance, rich ecosystem
- **Memory usage**: Higher (~200MB for 1K memories)
- **Setup**: Automatically configured, no additional dependencies

### SQLite-vec (Lightweight Alternative)
- **Best for**: Smaller collections (<100K entries), resource-constrained environments
- **Features**: Single-file database, 75% lower memory usage, better portability
- **Memory usage**: Lower (~50MB for 1K memories)
- **Setup**: Requires `sqlite-vec` package

#### Quick Setup for SQLite-vec

```bash
# Install sqlite-vec (if using installation script, this is handled automatically)
pip install sqlite-vec

# Configure the backend
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_MEMORY_SQLITE_PATH=/path/to/sqlite_vec.db

# Optional: For CPU-only mode without PyTorch (much lighter resource usage)
export MCP_MEMORY_USE_ONNX=1

# Restart Claude Desktop
```

#### SQLite-vec with Optional PyTorch

The SQLite-vec backend now works with or without PyTorch installed:

- **With PyTorch**: Full functionality including embedding generation
- **Without PyTorch**: Basic functionality using pre-computed embeddings and ONNX runtime
- **With Homebrew PyTorch**: Integration with macOS Homebrew PyTorch installation
  
To install optional machine learning dependencies:

```bash
# Add ML dependencies for embedding generation
pip install 'mcp-memory-service[ml]'
```

#### Homebrew PyTorch Integration

For macOS users who prefer to use Homebrew's PyTorch installation:

```bash
# Install PyTorch via Homebrew
brew install pytorch

# Run MCP Memory Service with Homebrew PyTorch integration
./run_with_homebrew.sh
```

This integration offers several benefits:
- Uses Homebrew's isolated Python environment for PyTorch
- Avoids dependency conflicts with Claude Desktop
- Reduces memory usage in the main process
- Provides better stability in resource-constrained environments

For detailed documentation on the Homebrew PyTorch integration:
- [HOMEBREW_INTEGRATION_LESSONS.md](HOMEBREW_INTEGRATION_LESSONS.md) - Technical journey and solution architecture
- [TECHNICAL_PATTERNS.md](TECHNICAL_PATTERNS.md) - Code patterns and implementation details
- [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md) - Diagnostic commands and common solutions

#### Migration Between Backends

```bash
# Migrate from ChromaDB to SQLite-vec
python migrate_to_sqlite_vec.py

# Full migration with backup
python scripts/migrate_storage.py \
  --from chroma --to sqlite_vec \
  --backup --backup-path backup.json
```

For detailed SQLite-vec setup, migration, and troubleshooting, see the [SQLite-vec Backend Guide](docs/sqlite-vec-backend.md).

## Memory Operations

The memory service provides the following operations through the MCP server:

### Core Memory Operations

1. `store_memory` - Store new information with optional tags
2. `retrieve_memory` - Perform semantic search for relevant memories
3. `recall_memory` - Retrieve memories using natural language time expressions 
4. `search_by_tag` - Find memories using specific tags
5. `exact_match_retrieve` - Find memories with exact content match
6. `debug_retrieve` - Retrieve memories with similarity scores

### Database Management

7. `create_backup` - Create database backup
8. `get_stats` - Get memory statistics
9. `optimize_db` - Optimize database performance
10. `check_database_health` - Get database health metrics
11. `check_embedding_model` - Verify model status

### Memory Management

12. `delete_memory` - Delete specific memory by hash
13. `delete_by_tag` - **Enhanced**: Delete memories with specific tag(s) - supports both single tags and multiple tags
14. `delete_by_tags` - **New**: Explicitly delete memories containing any of the specified tags (OR logic)
15. `delete_by_all_tags` - **New**: Delete memories containing all specified tags (AND logic)
16. `cleanup_duplicates` - Remove duplicate entries

### API Consistency Improvements

**Issue 5 Resolution**: Enhanced tag deletion functionality for consistent API design.

- **Before**: `search_by_tag` accepted arrays, `delete_by_tag` only accepted single strings
- **After**: Both operations now support flexible tag handling

```javascript
// Single tag deletion (backward compatible)
delete_by_tag("temporary")

// Multiple tag deletion (new!)
delete_by_tag(["temporary", "outdated", "test"])  // OR logic

// Explicit methods for clarity
delete_by_tags(["tag1", "tag2"])                  // OR logic  
delete_by_all_tags(["urgent", "important"])       // AND logic
```

### Example Usage

```javascript
// Store memories with tags
store_memory("Project deadline is May 15th", {tags: ["work", "deadlines", "important"]})
store_memory("Grocery list: milk, eggs, bread", {tags: ["personal", "shopping"]})
store_memory("Meeting notes from sprint planning", {tags: ["work", "meetings", "important"]})

// Search by multiple tags (existing functionality)
search_by_tag(["work", "important"])  // Returns memories with either tag

// Enhanced deletion options (new!)
delete_by_tag("temporary")                    // Delete single tag (backward compatible)
delete_by_tag(["temporary", "outdated"])     // Delete memories with any of these tags
delete_by_tags(["personal", "shopping"])     // Explicit multi-tag deletion
delete_by_all_tags(["work", "important"])    // Delete only memories with BOTH tags
```

## Configuration Options

Configure through environment variables:

```
CHROMA_DB_PATH: Path to ChromaDB storage
BACKUP_PATH: Path for backups
AUTO_BACKUP_INTERVAL: Backup interval in hours (default: 24)
MAX_MEMORIES_BEFORE_OPTIMIZE: Threshold for auto-optimization (default: 10000)
SIMILARITY_THRESHOLD: Default similarity threshold (default: 0.7)
MAX_RESULTS_PER_QUERY: Maximum results per query (default: 10)
BACKUP_RETENTION_DAYS: Number of days to keep backups (default: 7)
LOG_LEVEL: Logging level (default: INFO)

# Hardware and backend configuration
MCP_MEMORY_STORAGE_BACKEND: Storage backend to use (chromadb or sqlite_vec)
MCP_MEMORY_SQLITE_PATH: Path to SQLite-vec database file
PYTORCH_ENABLE_MPS_FALLBACK: Enable MPS fallback for Apple Silicon (default: 1)
MCP_MEMORY_USE_ONNX: Use ONNX Runtime for CPU-only deployments (default: 0)
MCP_MEMORY_USE_DIRECTML: Use DirectML for Windows acceleration (default: 0)
MCP_MEMORY_MODEL_NAME: Override the default embedding model
MCP_MEMORY_BATCH_SIZE: Override the default batch size
```

## Hardware Compatibility

| Platform | Architecture | Accelerator | Status | Notes |
|----------|--------------|-------------|--------|-------|
| macOS | Apple Silicon (M1/M2/M3) | MPS | ✅ Fully supported | Best performance |
| macOS | Apple Silicon under Rosetta 2 | CPU | ✅ Supported with fallbacks | Good performance |
| macOS | Intel | CPU | ✅ Fully supported | Good with optimized settings |
| Windows | x86_64 | CUDA | ✅ Fully supported | Best performance |
| Windows | x86_64 | DirectML | ✅ Supported | Good performance |
| Windows | x86_64 | CPU | ✅ Supported with fallbacks | Slower but works |
| Linux | x86_64 | CUDA | ✅ Fully supported | Best performance |
| Linux | x86_64 | ROCm | ✅ Supported | Good performance |
| Linux | x86_64 | CPU | ✅ Supported with fallbacks | Slower but works |
| Linux | ARM64 | CPU | ✅ Supported with fallbacks | Slower but works |
| Any | Any | No PyTorch | ✅ Supported with SQLite-vec | Limited functionality, very lightweight |

## Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_memory_ops.py
pytest tests/test_semantic_search.py
pytest tests/test_database.py

# Verify environment compatibility
python scripts/verify_environment_enhanced.py

# Verify PyTorch installation on Windows
python scripts/verify_pytorch_windows.py

# Perform comprehensive installation verification
python scripts/test_installation.py
```

## FAQ

### Can I run the MCP Memory Service in both Claude Desktop and Claude Code simultaneously?

**Yes!** The MCP Memory Service is designed to support concurrent access from multiple clients. Both Claude Desktop and Claude Code can safely use the same memory service instance simultaneously, and they will share the same memory store.

**Key benefits:**
- Shared memory across both applications
- No file conflicts or locking issues
- Seamless experience when switching between clients

**Technical details:**
- ChromaDB uses SQLite which handles concurrent database access safely
- No application-level file locking that would prevent multiple instances
- Each client creates its own connection but accesses the same shared database

**Configuration tip:** Ensure both clients use the same database paths by setting identical `MCP_MEMORY_CHROMA_PATH` and `MCP_MEMORY_BACKUPS_PATH` environment variables.

## Troubleshooting

See the [Installation Guide](docs/guides/installation.md#troubleshooting-common-installation-issues) for detailed troubleshooting steps.

### Quick Troubleshooting Tips

- **Windows PyTorch errors**: Use `python scripts/install_windows.py`
- **macOS Intel dependency conflicts**: Use `python install.py --force-compatible-deps`
- **Recursion errors**: Run `python scripts/fix_sitecustomize.py` 
- **Environment verification**: Run `python scripts/verify_environment_enhanced.py`
- **Memory issues**: Set `MCP_MEMORY_BATCH_SIZE=4` and try a smaller model
- **Apple Silicon**: Ensure Python 3.10+ built for ARM64, set `PYTORCH_ENABLE_MPS_FALLBACK=1`
- **Installation testing**: Run `python scripts/test_installation.py`

## 📚 Comprehensive Documentation

### Installation & Setup
- **[Master Installation Guide](docs/guides/INSTALLATION_MASTER.md)** - Complete installation guide with hardware-specific paths
- **[Storage Backend Comparison](docs/guides/STORAGE_BACKENDS.md)** - Detailed comparison and selection guide
- **[Migration Guide](MIGRATION_GUIDE.md)** - ChromaDB to SQLite-vec migration instructions

### Platform-Specific Guides
- **[Intel Mac Setup Guide](docs/INTEL_MAC_SETUP.md)** - Comprehensive guide for Intel Mac users
- **[Legacy Mac Guide](docs/platforms/macos-intel-legacy.md)** - Optimized for 2015 MacBook Pro and older Intel Macs
- **[Windows Setup](docs/guides/windows-setup.md)** - Windows-specific installation and troubleshooting
- **[Ubuntu Setup](docs/guides/UBUNTU_SETUP.md)** - Linux server installation guide

### API & Integration
- **[HTTP/SSE API](docs/IMPLEMENTATION_PLAN_HTTP_SSE.md)** - New web interface documentation
- **[Claude Desktop Integration](docs/guides/claude_integration.md)** - Configuration examples
- **[Integrations](docs/integrations.md)** - Third-party tools and extensions

### Advanced Topics
- **[Homebrew PyTorch Integration](docs/integration/homebrew/HOMEBREW_PYTORCH_README.md)** - Using system PyTorch
- **[Docker Deployment](docs/guides/docker.md)** - Container-based deployment
- **[Performance Optimization](docs/implementation/performance.md)** - Tuning for different hardware

### Troubleshooting & Support
- **[General Troubleshooting](docs/guides/troubleshooting.md)** - Common issues and solutions
- **[Hardware Compatibility](docs/DOCUMENTATION_AUDIT.md)** - Compatibility matrix and known issues

### Quick Commands
```bash
# Get personalized setup recommendations
python install.py --help-detailed

# Generate hardware-specific setup guide
python install.py --generate-docs

# Test your installation
python scripts/test_memory_simple.py
```

## Project Structure

```
mcp-memory-service/
├── src/mcp_memory_service/      # Core package code
│   ├── __init__.py
│   ├── config.py                # Configuration utilities
│   ├── models/                  # Data models
│   ├── storage/                 # Storage implementations
│   ├── utils/                   # Utility functions
│   └── server.py                # Main MCP server
├── scripts/                     # Helper scripts
├── memory_wrapper.py            # Windows wrapper script
├── install.py                   # Enhanced installation script
└── tests/                       # Test suite
```

## Development Guidelines

- Python 3.10+ with type hints
- Use dataclasses for models
- Triple-quoted docstrings for modules and functions
- Async/await pattern for all I/O operations
- Follow PEP 8 style guidelines
- Include tests for new features

## License

MIT License - See LICENSE file for details

## Acknowledgments

- ChromaDB team for the vector database
- Sentence Transformers project for embedding models
- MCP project for the protocol specification

## 🎯 Why Sponsor MCP Memory Service?

## 🏆 In Production
- Deployed on Glama.ai
- Managing 300+ enterprise memories
- Processing queries in <1 second

### Production Impact
- **319+ memories** actively managed
- **828ms** average query response time
- **100%** cache hit ratio performance
- **20MB** efficient vector storage

### Developer Community
- Complete MCP protocol implementation
- Cross-platform compatibility
- React dashboard with real-time statistics
- Comprehensive documentation

### Enterprise Features
- Semantic search with sentence-transformers
- Tag-based categorization system
- Automatic backup and optimization
- Health monitoring dashboard

## Contact

[Telegram](t.me/doobeedoo)

## Integrations

The MCP Memory Service can be extended with various tools and utilities. See [Integrations](docs/integrations.md) for a list of available options, including:

- [MCP Memory Dashboard](https://github.com/doobidoo/mcp-memory-dashboard) - Web UI for browsing and managing memories
- [Claude Memory Context](https://github.com/doobidoo/claude-memory-context) - Inject memory context into Claude project instructions
