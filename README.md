# MCP Memory Service

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![smithery badge](https://smithery.ai/badge/@doobidoo/mcp-memory-service)](https://smithery.ai/server/@doobidoo/mcp-memory-service)
[![Verified on MseeP](https://mseep.ai/badge.svg)](https://mseep.ai/app/0513fb92-e941-4fe0-9948-2a1dbb870dcf)

An MCP server providing semantic memory and persistent storage capabilities for Claude Desktop using ChromaDB and sentence transformers. This service enables long-term memory storage with semantic search capabilities, making it ideal for maintaining context across conversations and instances.

<img width="240" alt="grafik" src="https://github.com/user-attachments/assets/eab1f341-ca54-445c-905e-273cd9e89555" />
<a href="https://glama.ai/mcp/servers/bzvl3lz34o"><img width="380" height="200" src="https://glama.ai/mcp/servers/bzvl3lz34o/badge" alt="Memory Service MCP server" /></a>

## Help
Talk to the Repo with [TalkToGitHub](https://talktogithub.com/doobidoo/mcp-memory-service)!
## Features

- Semantic search using sentence transformers
- **Natural language time-based recall** (e.g., "last week", "yesterday morning")
- **Enhanced tag deletion system** with flexible multi-tag support
- Tag-based memory retrieval system
- Persistent storage using ChromaDB
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

- ✅ **API Consistency**: Enhanced `delete_by_tag` to support both single and multiple tags
- ✅ **New Delete Methods**: Added `delete_by_tags` (OR logic) and `delete_by_all_tags` (AND logic)
- ✅ **Backward Compatibility**: All existing code continues to work unchanged
- ✅ **Dashboard Integration**: Enhanced UI with multiple tag selection capabilities

## Installation

### Quick Start (Recommended)

The enhanced installation script automatically detects your system and installs the appropriate dependencies:

```bash
# Clone the repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the installation script
python install.py
```

The `install.py` script will:
1. Detect your system architecture and available hardware accelerators
2. Install the appropriate dependencies for your platform
3. Configure the optimal settings for your environment
4. Verify the installation and provide diagnostics if needed

### Docker Installation

You can run the Memory Service using Docker:

```bash
# Using Docker Compose (recommended)
docker-compose up

# Using Docker directly
docker build -t mcp-memory-service .
docker run -p 8000:8000 -v /path/to/data:/app/chroma_db -v /path/to/backups:/app/backups mcp-memory-service
```

We provide multiple Docker Compose configurations for different scenarios:
- `docker-compose.yml` - Standard configuration using pip install
- `docker-compose.uv.yml` - Alternative configuration using UV package manager
- `docker-compose.pythonpath.yml` - Configuration with explicit PYTHONPATH settings

To use an alternative configuration:
```bash
docker-compose -f docker-compose.uv.yml up
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

# Hardware-specific environment variables
PYTORCH_ENABLE_MPS_FALLBACK: Enable MPS fallback for Apple Silicon (default: 1)
MCP_MEMORY_USE_ONNX: Use ONNX Runtime for CPU-only deployments (default: 0)
MCP_MEMORY_USE_DIRECTML: Use DirectML for Windows acceleration (default: 0)
MCP_MEMORY_MODEL_NAME: Override the default embedding model
MCP_MEMORY_BATCH_SIZE: Override the default batch size
```

## Hardware Compatibility

| Platform | Architecture | Accelerator | Status |
|----------|--------------|-------------|--------|
| macOS | Apple Silicon (M1/M2/M3) | MPS | ✅ Fully supported |
| macOS | Apple Silicon under Rosetta 2 | CPU | ✅ Supported with fallbacks |
| macOS | Intel | CPU | ✅ Fully supported |
| Windows | x86_64 | CUDA | ✅ Fully supported |
| Windows | x86_64 | DirectML | ✅ Supported |
| Windows | x86_64 | CPU | ✅ Supported with fallbacks |
| Linux | x86_64 | CUDA | ✅ Fully supported |
| Linux | x86_64 | ROCm | ✅ Supported |
| Linux | x86_64 | CPU | ✅ Supported with fallbacks |
| Linux | ARM64 | CPU | ✅ Supported with fallbacks |

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

## Contact

[Telegram](t.me/doobeedoo)

## Integrations

The MCP Memory Service can be extended with various tools and utilities. See [Integrations](docs/integrations.md) for a list of available options, including:

- [MCP Memory Dashboard](https://github.com/doobidoo/mcp-memory-dashboard) - Web UI for browsing and managing memories
- [Claude Memory Context](https://github.com/doobidoo/claude-memory-context) - Inject memory context into Claude project instructions
