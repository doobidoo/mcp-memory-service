# MCP Memory Service with Homebrew PyTorch

This solution allows you to run the MCP Memory Service on macOS using your existing Homebrew PyTorch installation instead of installing PyTorch via pip, which can be problematic on older MacBooks.

## Quick Start

1. **Install PyTorch via Homebrew** (if not already installed):
   ```bash
   brew install pytorch
   ```

2. **Install sentence-transformers in the Homebrew Python environment**:
   ```bash
   $(brew --prefix pytorch)/libexec/bin/python3 -m pip install sentence-transformers
   ```

3. **Run the memory service**:
   ```bash
   ./run_mcp_memory.sh
   ```

## How It Works

This solution provides a seamless integration between MCP Memory Service and Homebrew PyTorch:

1. **SQLite-vec Backend**: Uses the lightweight SQLite-vec backend instead of ChromaDB
2. **ONNX Runtime**: Leverages ONNX runtime for better compatibility
3. **Homebrew PyTorch**: Uses your existing Homebrew PyTorch installation for generating embeddings
4. **Custom Integration**: Provides a bridge between the MCP Memory Service and Homebrew PyTorch

## Components

- **`homebrew_integration.py`**: A module that integrates Homebrew PyTorch with MCP Memory Service
- **`run_mcp_memory.sh`**: A script to run the memory service with the Homebrew PyTorch integration
- **`test_improved_integration.py`**: A test script to verify the integration works correctly

## Configuration

The solution uses the following environment variables:

- `MCP_MEMORY_STORAGE_BACKEND="sqlite_vec"`: Uses the lightweight SQLite-vec backend
- `MCP_MEMORY_USE_ONNX="1"`: Enables ONNX runtime for better compatibility
- `MCP_MEMORY_USE_HOMEBREW_PYTORCH="1"`: Enables Homebrew PyTorch integration

## Troubleshooting

If you encounter issues:

1. **Ensure Homebrew PyTorch is installed**:
   ```bash
   brew list pytorch
   ```

2. **Check sentence-transformers installation**:
   ```bash
   $(brew --prefix pytorch)/libexec/bin/python3 -c "import sentence_transformers; print(sentence_transformers.__version__)"
   ```

3. **Run the test script**:
   ```bash
   python test_improved_integration.py
   ```

4. **Check the database paths**:
   ```bash
   ls -la ~/Library/Application\ Support/mcp-memory/
   ```

## Integration with Claude Desktop

To use this solution with Claude Desktop:

1. Copy the settings in Claude Desktop config:
   ```json
   {
     "mcpServers": {
       "memory": {
         "command": "/path/to/run_mcp_memory.sh",
         "env": {
           "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
           "MCP_MEMORY_USE_ONNX": "1",
           "MCP_MEMORY_USE_HOMEBREW_PYTORCH": "1"
         }
       }
     }
   }
   ```

2. Set the full path to `run_mcp_memory.sh` in the `command` field
