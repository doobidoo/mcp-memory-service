# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MCP Memory Service is a Model Context Protocol server that provides semantic memory and persistent storage capabilities for Claude Desktop using ChromaDB and sentence transformers. The project enables long-term memory storage with semantic search across conversations.

## Key Commands

### Development
- **Install dependencies**: `python install.py` (platform-aware installation)
- **Run server**: `python scripts/run_memory_server.py` or `uv run memory`
- **Run tests**: `pytest tests/`
- **Run specific test**: `pytest tests/unit/test_memory_models.py::TestMemoryModel::test_memory_creation`
- **Check environment**: `python scripts/verify_environment.py`
- **Debug with MCP Inspector**: `npx @modelcontextprotocol/inspector uv --directory /path/to/repo run memory`
- **Check documentation links**: `python scripts/check_documentation_links.py` (validates all internal markdown links)
- **Test Docker functionality**: `python scripts/test_docker_functionality.py` (comprehensive Docker container verification)

### Build & Package
- **Build package**: `python -m build`
- **Install locally**: `pip install -e .`

### Remote Memory Connection
- **Setup remote connection**: Copy `examples/claude-desktop-memory-config.json` to Claude Desktop config directory
- **Test remote connection**: `node examples/http-mcp-bridge-robust.js` (with proper environment variables)
- **Test remote API**: `curl -k -H "Authorization: Bearer API_KEY" https://server/api/health`
- **Debug bridge connection**: Check Claude Desktop logs at `~/Library/Logs/Claude/mcp-server-memory.log`

## Architecture

### Core Components

1. **Server Layer** (`src/mcp_memory_service/server.py`)
   - Implements MCP protocol with async request handlers
   - Global model and embedding caches for performance
   - Handles all memory operations (store, retrieve, search, delete)

2. **Storage Abstraction** (`src/mcp_memory_service/storage/`)
   - `base.py`: Abstract interface for storage backends
   - `chroma.py`: ChromaDB implementation
   - `chroma_enhanced.py`: Extended features (time parsing, advanced search)

3. **Models** (`src/mcp_memory_service/models/memory.py`)
   - `Memory`: Core dataclass for memory entries
   - `MemoryMetadata`: Metadata structure
   - All models use Python dataclasses with type hints

4. **Configuration** (`src/mcp_memory_service/config.py`)
   - Environment-based configuration
   - Platform-specific optimizations
   - Hardware acceleration detection

### Key Design Patterns

- **Async/Await**: All I/O operations are async
- **Type Safety**: Comprehensive type hints (Python 3.10+)
- **Error Handling**: Specific exception types with clear messages
- **Caching**: Global caches for models and embeddings to improve performance
- **Platform Detection**: Automatic hardware optimization (CUDA, MPS, DirectML, ROCm)

### MCP Protocol Operations

Memory operations implemented:
- `store_memory`: Store new memories with tags and metadata
- `retrieve_memory`: Basic retrieval by query
- `recall_memory`: Natural language time-based retrieval
- `search_by_tag`: Tag-based search
- `delete_memory`: Delete specific memories
- `delete_by_tag/tags`: Bulk deletion by tags
- `optimize_db`: Database optimization
- `check_database_health`: Health monitoring
- `debug_retrieve`: Similarity analysis for debugging

### Testing

Tests are organized by type:
- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Integration tests for full workflows
- `tests/performance/`: Performance benchmarks

Run tests with coverage: `pytest --cov=src/mcp_memory_service tests/`

### Environment Variables

Key configuration:
- `MCP_MEMORY_CHROMA_PATH`: ChromaDB storage location (default: `~/.mcp_memory_chroma`)
- `MCP_MEMORY_BACKUPS_PATH`: Backup location (default: `~/.mcp_memory_backups`)
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- Platform-specific: `PYTORCH_ENABLE_MPS_FALLBACK`, `MCP_MEMORY_USE_ONNX`

### Platform Support

The codebase includes platform-specific optimizations:
- **macOS**: MPS acceleration for Apple Silicon, CPU fallback for Intel
- **Windows**: CUDA, DirectML, or CPU
- **Linux**: CUDA, ROCm, or CPU

Hardware detection is automatic via `utils/system_detection.py`.

### Development Tips

1. When modifying storage backends, ensure compatibility with the abstract base class
2. Memory operations should handle duplicates gracefully (content hashing)
3. Time parsing supports natural language ("yesterday", "last week")
4. Use the debug_retrieve operation to analyze similarity scores
5. The server maintains global state for models - be careful with concurrent modifications
6. All new features should include corresponding tests
7. Use semantic commit messages for version management

### Common Issues

1. **MPS Fallback**: On macOS, if MPS fails, set `PYTORCH_ENABLE_MPS_FALLBACK=1`
2. **ONNX Runtime**: For compatibility issues, use `MCP_MEMORY_USE_ONNX=true`
3. **ChromaDB Persistence**: Ensure write permissions for storage paths
4. **Memory Usage**: Model loading is deferred until first use to reduce startup time