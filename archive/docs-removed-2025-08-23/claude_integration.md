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

## Memory Service Invocation
- See the comprehensive [Invocation Guide](invocation_guide.md) for full details
- Key trigger phrases:
  - **Storage**: "remember that", "remember this", "save to memory", "store in memory"
  - **Retrieval**: "do you remember", "recall", "retrieve from memory", "search your memory for"
  - **Tag-based**: "find memories with tag", "search for tag", "retrieve memories tagged"
  - **Deletion**: "forget", "delete from memory", "remove from memory"

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
- For Windows installation issues, use `scripts/install_windows.py`
- Apple Silicon requires Python 3.10+ built for ARM64
- CUDA issues: verify with `torch.cuda.is_available()`
- For MCP protocol issues, check `server.py` for required methods