# AI Agent Instructions

AI coding agent instructions for MCP Memory Service - a universal memory service providing semantic search and persistent storage for AI assistants.

## Project Overview

MCP Memory Service implements the Model Context Protocol (MCP) to provide semantic memory capabilities for AI assistants. It supports multiple storage backends (SQLite-vec, ChromaDB, Cloudflare) and works with 13+ AI applications including Claude Desktop, VS Code, Cursor, and Continue.

## Setup Commands

```bash
# Install dependencies (platform-aware)
python install.py

# Alternative: UV installation (faster)
uv sync

# Start development server
uv run memory server

# Run with inspector for debugging
npx @modelcontextprotocol/inspector uv run memory server

# Start HTTP API server (dashboard at https://localhost:8443)
uv run memory server --http --port 8443
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_server.py          # Server tests
pytest tests/test_storage.py         # Storage backend tests
pytest tests/test_embeddings.py      # Embedding tests

# Run with coverage
pytest --cov=mcp_memory_service tests/

# Verify environment setup
python scripts/validation/verify_environment.py

# Check database health
python scripts/database/db_health_check.py
```

## Code Style

- **Python 3.10+** with type hints everywhere
- **Async/await** for all I/O operations
- **Black** formatter with 88-char line length
- **Import order**: stdlib, third-party, local (use `isort`)
- **Docstrings**: Google style for all public functions
- **Error handling**: Always catch specific exceptions
- **Logging**: Use structured logging with appropriate levels

## Project Structure

```
src/mcp_memory_service/
├── server.py           # Main MCP server implementation
├── mcp_server.py       # MCP protocol handler
├── storage/            # Storage backend implementations
│   ├── base.py        # Abstract base class
│   ├── sqlite_vec.py  # SQLite-vec backend (default)
│   ├── chroma.py      # ChromaDB backend
│   └── cloudflare.py  # Cloudflare D1/Vectorize backend
├── embeddings/         # Embedding model implementations
├── consolidation/      # Memory consolidation algorithms
└── web/               # FastAPI dashboard and REST API
```

## Key Files to Understand

- `src/mcp_memory_service/server.py` - Entry point and server initialization
- `src/mcp_memory_service/storage/base.py` - Storage interface all backends must implement
- `src/mcp_memory_service/web/app.py` - FastAPI application for HTTP mode
- `pyproject.toml` - Project dependencies and configuration
- `install.py` - Platform-aware installer script

## Common Development Tasks

### Adding a New Storage Backend
1. Create new file in `src/mcp_memory_service/storage/`
2. Inherit from `BaseStorage` abstract class
3. Implement all required methods
4. Add backend to `STORAGE_BACKENDS` in `server.py`
5. Write tests in `tests/test_storage.py`

### Modifying MCP Tools
1. Edit tool definitions in `src/mcp_memory_service/mcp_server.py`
2. Update tool handlers in the same file
3. Test with MCP inspector: `npx @modelcontextprotocol/inspector uv run memory server`
4. Update documentation in `docs/api/tools.md`

### Adding Environment Variables
1. Define in `src/mcp_memory_service/config.py`
2. Document in README.md and CLAUDE.md
3. Add to Docker configurations in `tools/docker/`
4. Update `scripts/validation/verify_environment.py`

### Database Migrations
```bash
# Check for needed migrations
python scripts/migration/verify_mcp_timestamps.py

# Migrate from ChromaDB to SQLite-vec
python scripts/migration/migrate_chroma_to_sqlite.py

# Validate existing memories
python scripts/validation/validate_memories.py
```

## Performance Considerations

- **Embedding caching**: Models are cached globally to avoid reloading
- **Batch operations**: Use batch methods for multiple memory operations
- **Connection pooling**: Storage backends use connection pools
- **Async operations**: All I/O is async to prevent blocking
- **Hardware acceleration**: Auto-detects CUDA, MPS, DirectML, ROCm

## Security Guidelines

- **Never commit secrets**: API keys, tokens must use environment variables
- **Input validation**: Always validate and sanitize user inputs
- **SQL injection**: Use parameterized queries in SQLite backend
- **API authentication**: HTTP mode requires API key authentication
- **Path traversal**: Validate all file paths before operations
- **Memory content**: Never log full memory content (may contain sensitive data)

## Debugging Tips

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Check service health
curl https://localhost:8443/api/health

# Monitor logs
tail -f ~/.mcp-memory-service/logs/service.log

# Inspect MCP communication
npx @modelcontextprotocol/inspector uv run memory server

# Database debugging
sqlite3 ~/.mcp-memory-service/sqlite_vec.db ".tables"
```

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md with changes
3. Run full test suite: `pytest tests/`
4. Create git tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
5. Push tag: `git push origin vX.Y.Z`
6. GitHub Actions will handle PyPI release

## Common Issues and Solutions

- **SQLite extension errors on macOS**: Use Homebrew Python or pyenv with `--enable-loadable-sqlite-extensions`
- **Model download hangs**: Check network connectivity, models are ~25MB
- **Import errors**: Run `python install.py` to ensure all dependencies installed
- **MCP connection fails**: Restart Claude Desktop to refresh MCP connections
- **Memory not persisting**: Check file permissions in `~/.mcp-memory-service/`

## Contributing

- Follow existing code patterns and conventions
- Add tests for new features
- Update documentation for API changes
- Use semantic commit messages
- Run tests before submitting PRs

---

*This file follows the [agents.md](https://agents.md/) standard for AI coding agent instructions.*
