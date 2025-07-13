# Setting Up MCP Memory Service with Homebrew PyTorch

This guide explains how to set up and run the MCP Memory Service using a Homebrew PyTorch installation on macOS, which avoids the need to install PyTorch via pip.

## Prerequisites

1. **Homebrew**: Make sure you have Homebrew installed
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Homebrew PyTorch**: Install PyTorch via Homebrew
   ```bash
   brew install pytorch
   ```

3. **Python 3.10+**: You should have Python 3.10 or newer installed

## Installation Steps

1. **Clone the repository and create a virtual environment**
   ```bash
   git clone https://github.com/yourusername/mcp-memory-service.git
   cd mcp-memory-service
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install the service with SQLite-vec backend and ONNX runtime**
   ```bash
   python install.py --storage-backend sqlite_vec --skip-pytorch
   ```

3. **Install sentence-transformers in the Homebrew Python environment**
   ```bash
   /usr/local/opt/pytorch/libexec/bin/python3 -m pip install sentence-transformers
   ```

4. **Copy the Homebrew PyTorch embeddings bridge script**
   ```bash
   # The homebrew_pytorch_embeddings.py script is already included in the repository
   chmod +x homebrew_pytorch_embeddings.py
   ```

## Running the Service

1. **Using the provided script**
   ```bash
   ./run_memory_server_with_homebrew.sh
   ```

2. **Or manually with environment variables**
   ```bash
   export MCP_MEMORY_STORAGE_BACKEND="sqlite_vec"
   export MCP_MEMORY_SQLITE_PATH="$HOME/Library/Application Support/mcp-memory/sqlite_vec.db"
   export MCP_MEMORY_BACKUPS_PATH="$HOME/Library/Application Support/mcp-memory/backups"
   export MCP_MEMORY_USE_ONNX="1"
   
   source venv/bin/activate
   memory
   ```

## Testing the Installation

1. **Run the test script**
   ```bash
   python test_with_homebrew_embeddings.py
   ```

2. **Verify that:**
   - The database is properly connected
   - Memory storage works
   - Memory retrieval by tag works
   - Semantic search works with Homebrew PyTorch embeddings

## Troubleshooting

If you encounter issues:

1. **Make sure Homebrew PyTorch is installed correctly**
   ```bash
   brew info pytorch
   ```

2. **Verify sentence-transformers is installed in the Homebrew Python environment**
   ```bash
   /usr/local/opt/pytorch/libexec/bin/python3 -c "import sentence_transformers; print(sentence_transformers.__version__)"
   ```

3. **Check the database paths**
   ```bash
   ls -la ~/Library/Application\ Support/mcp-memory/
   ```

4. **Test the embeddings directly**
   ```bash
   python homebrew_pytorch_embeddings.py
   ```

5. **Look for error messages in the logs**
   - Check terminal output when running the server
   - Try running with more verbose logging

## How It Works

This solution uses:
1. **SQLite-vec backend**: A lightweight vector database that doesn't require ChromaDB
2. **ONNX Runtime**: For CPU-based operations without PyTorch
3. **Homebrew PyTorch**: For generating embeddings through an external process
4. **Custom bridge script**: To connect your Python environment with the Homebrew PyTorch installation

The `homebrew_pytorch_embeddings.py` script creates a bridge between your virtual environment and the Homebrew PyTorch installation, allowing the memory service to generate embeddings without needing to install PyTorch via pip.
