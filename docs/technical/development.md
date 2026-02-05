# MCP Memory Service - Development Guidelines

## Commands
- Run memory server: `python scripts/run_memory_server.py`
- Run tests: `pytest tests/`
- Run specific test: `pytest tests/test_memory_ops.py::test_store_memory -v`
- Check environment: `python scripts/verify_environment_enhanced.py`
- Windows installation: `python scripts/install_windows.py`
- Build package: `python -m build`

## Installation Guidelines
- Always install in a virtual environment: `python -m venv venv`
- Use `install.py` for cross-platform installation
- Windows requires special PyTorch installation with correct index URL:
  ```bash
  pip install torch==2.1.0 torchvision==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
  ```
- For recursion errors, run: `python scripts/fix_sitecustomize.py`

## Code Style
- Python 3.10+ with type hints
- Use dataclasses for models (see `models/memory.py`)
- Triple-quoted docstrings for modules and functions
- Async/await pattern for all I/O operations
- Error handling with specific exception types and informative messages
- Logging with appropriate levels for different severity
- Commit messages follow semantic release format: `type(scope): message`

## Project Structure
- `src/mcp_memory_service/` - Core package code
  - `models/` - Data models
  - `storage/` - Database abstraction
  - `utils/` - Helper functions
  - `server.py` - MCP protocol implementation
- `scripts/` - Utility scripts
- `memory_wrapper.py` - Windows wrapper script
- `install.py` - Cross-platform installation script

## Dependencies
- ChromaDB (0.5.23) for vector database
- sentence-transformers (>=2.2.2) for embeddings
- PyTorch (platform-specific installation)
- MCP protocol (>=1.0.0, <2.0.0) for client-server communication

## Troubleshooting

### Common Issues

- For Windows installation issues, use `scripts/install_windows.py`
- Apple Silicon requires Python 3.10+ built for ARM64
- CUDA issues: verify with `torch.cuda.is_available()`
- For MCP protocol issues, check `server.py` for required methods

### MCP Server Configuration Issues

If you encounter MCP server failures or "ModuleNotFoundError" issues:

#### Missing http_server_manager Module
**Symptoms:**
- Server fails with "No module named 'mcp_memory_service.utils.http_server_manager'"
- MCP server shows as "failed" in Claude Code

**Diagnosis:**
1. Test server directly: `python -m src.mcp_memory_service.server --debug`
2. Check if the error occurs during eager storage initialization
3. Look for HTTP server coordination mode detection

**Solution:**
The `http_server_manager.py` module handles multi-client coordination. If missing, create it with:
- `auto_start_http_server_if_needed()` function
- Port detection and server startup logic
- Integration with existing `port_detection.py` utilities

#### Storage Backend Issues
**Symptoms:**
- "vec0 constructor error: Unknown table option" (older sqlite-vec versions)
- Server initialization fails with storage errors

**Diagnosis:**
1. Test each backend independently:
   - ChromaDB: `python scripts/run_memory_server.py --debug`
   - SQLite-vec: `MCP_MEMORY_STORAGE_BACKEND=sqlite_vec python -m src.mcp_memory_service.server --debug`
2. Check database health: Use MCP tools to call `check_database_health`

**Solution:**
- Ensure sqlite-vec is properly installed and compatible
- Both backends should work when properly configured
- SQLite-vec uses direct storage mode when HTTP coordination fails

#### MCP Configuration Cleanup
When multiple servers conflict or fail:

1. **Backup configurations:**
   ```bash
   cp .mcp.json .mcp.json.backup
   ```

2. **Remove failing servers** from `.mcp.json` while keeping working ones

3. **Test each backend separately:**
   - ChromaDB backend: Uses `scripts/run_memory_server.py`
   - SQLite-vec backend: Uses `python -m src.mcp_memory_service.server` with `MCP_MEMORY_STORAGE_BACKEND=sqlite_vec`

4. **Verify functionality** through MCP tools before re-enabling servers

### Collaborative Debugging with Claude Code

When troubleshooting complex MCP issues, consider this collaborative approach between developer and Claude Code:

#### Systematic Problem-Solving Partnership
1. **Developer identifies the symptom** (e.g., "memory server shows as failed")
2. **Claude Code conducts comprehensive diagnosis:**
   - Tests current functionality with MCP tools
   - Examines server logs and error messages
   - Checks configuration files and git history
   - Identifies root causes through systematic analysis

3. **Collaborative investigation approach:**
   - **Developer requests:** "Check both ways - verify current service works AND fix the failing alternative"
   - **Claude Code responds:** Creates todo lists, tests each component independently, provides detailed analysis
   - **Developer provides context:** Domain knowledge, constraints, preferences

4. **Methodical fix implementation:**
   - Back up configurations before changes
   - Fix issues incrementally with testing at each step
   - Document the process for future reference
   - Commit meaningful changes with proper commit messages

#### Benefits of This Approach
- **Comprehensive coverage:** Nothing gets missed when both perspectives combine
- **Knowledge transfer:** Claude Code documents the process, developer retains understanding
- **Systematic methodology:** Todo lists and step-by-step verification prevent overlooked issues
- **Persistent knowledge:** Using the memory service itself to store troubleshooting solutions

#### Example Workflow
```
Developer: "MCP memory server is failing"
↓
Claude Code: Creates todo list, tests current state, identifies missing module
↓
Developer: "Let's fix it but keep the working parts"
↓
Claude Code: Backs up configs, fixes incrementally, tests each component
↓
Developer: "This should be documented"
↓
Claude Code: Updates documentation, memorizes the solution
↓
Both: Commit, push, and ensure knowledge is preserved
```

This collaborative model leverages Claude Code's systematic analysis capabilities with the developer's domain expertise and decision-making authority.

## Debugging with MCP-Inspector

To debug the MCP-MEMORY-SERVICE using the [MCP-Inspector](https://modelcontextprotocol.io/docs/tools/inspector) tool, you can use the following command pattern:

```bash
MCP_MEMORY_CHROMA_PATH="/path/to/your/chroma_db" MCP_MEMORY_BACKUPS_PATH="/path/to/your/backups" npx @modelcontextprotocol/inspector uv --directory /path/to/mcp-memory-service run memory
```

Replace the paths with your specific directories:
- `/path/to/your/chroma_db`: Location where Chroma database files are stored
- `/path/to/your/backups`: Location for memory backups
- `/path/to/mcp-memory-service`: Directory containing the MCP-MEMORY-SERVICE code

For example:
```bash
MCP_MEMORY_CHROMA_PATH="~/Library/Mobile Documents/com~apple~CloudDocs/AI/claude-memory/chroma_db" MCP_MEMORY_BACKUPS_PATH="~/Library/Mobile Documents/com~apple~CloudDocs/AI/claude-memory/backups" npx @modelcontextprotocol/inspector uv --directory ~/Documents/GitHub/mcp-memory-service run memory
```

This command sets the required environment variables and runs the memory service through the inspector tool for debugging purposes.
