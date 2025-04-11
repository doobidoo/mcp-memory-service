# MCP Memory Service

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![smithery badge](https://smithery.ai/badge/@doobidoo/mcp-memory-service)](https://smithery.ai/server/@doobidoo/mcp-memory-service)

An MCP server providing semantic memory and persistent storage capabilities for Claude Desktop using ChromaDB and sentence transformers. This service enables long-term memory storage with semantic search capabilities, making it ideal for maintaining context across conversations and instances.

<img width="240" alt="grafik" src="https://github.com/user-attachments/assets/eab1f341-ca54-445c-905e-273cd9e89555" />
<a href="https://glama.ai/mcp/servers/bzvl3lz34o"><img width="380" height="200" src="https://glama.ai/mcp/servers/bzvl3lz34o/badge" alt="Memory Service MCP server" /></a>

## Features

- Semantic search using sentence transformers
- **Natural language time-based recall** (e.g., "last week", "yesterday morning")
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

## Quick Start

For the fastest way to get started:

```bash
# Install UV if not already installed
pip install uv

# Clone and install
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install -e .

# Run the service
uv run memory
```

## Docker and Smithery Integration

### Docker Usage

The service can be run in a Docker container for better isolation and deployment:

```bash
# Build the Docker image
docker build -t mcp-memory-service .

# Run the container
# Note: On macOS, paths must be within Docker's allowed file sharing locations
# Default allowed locations include:
# - /Users
# - /Volumes
# - /private
# - /tmp
# - /var/folders

# Example with proper macOS paths:
docker run -it \
  -v $HOME/mcp-memory/chroma_db:/app/chroma_db \
  -v $HOME/mcp-memory/backups:/app/backups \
  mcp-memory-service

# For production use, you might want to run it in detached mode:
docker run -d \
  -v $HOME/mcp-memory/chroma_db:/app/chroma_db \
  -v $HOME/mcp-memory/backups:/app/backups \
  --name mcp-memory \
  mcp-memory-service
```

To configure Docker's file sharing on macOS:
1. Open Docker Desktop
2. Go to Settings (Preferences)
3. Navigate to Resources -> File Sharing
4. Add any additional paths you need to share
5. Click "Apply & Restart"

### Smithery Integration

The service is configured for Smithery integration through `smithery.yaml`. This configuration enables stdio-based communication with MCP clients like Claude Desktop.

To use with Smithery:

1. Ensure your `claude_desktop_config.json` points to the correct paths:
```json
{
  "memory": {
    "command": "docker",
    "args": [
      "run",
      "-i",
      "--rm",
      "-v", "$HOME/mcp-memory/chroma_db:/app/chroma_db",
      "-v", "$HOME/mcp-memory/backups:/app/backups",
      "mcp-memory-service"
    ],
    "env": {
      "MCP_MEMORY_CHROMA_PATH": "/app/chroma_db",
      "MCP_MEMORY_BACKUPS_PATH": "/app/backups"
    }
  }
}
```

2. The `smithery.yaml` configuration handles stdio communication and environment setup automatically.

### Testing with Claude Desktop

To verify your Docker-based memory service is working correctly with Claude Desktop:

1. Build the Docker image with `docker build -t mcp-memory-service .`
2. Create the necessary directories for persistent storage:
   ```bash
   mkdir -p $HOME/mcp-memory/chroma_db $HOME/mcp-memory/backups
   ```
3. Update your Claude Desktop configuration file:
   - On macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - On Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - On Linux: `~/.config/Claude/claude_desktop_config.json`
4. Restart Claude Desktop
5. When Claude starts up, you should see the memory service initialize with a message:
   ```
   MCP Memory Service initialization completed
   ```
6. Test the memory feature:
   - Ask Claude to remember something: "Please remember that my favorite color is blue"
   - Later in the conversation or in a new conversation, ask: "What is my favorite color?"
   - Claude should retrieve the information from the memory service

If you experience any issues:
- Check the Claude Desktop console for error messages
- Verify Docker has the necessary permissions to access the mounted directories
- Ensure the Docker container is running with the correct parameters
- Try running the container manually to see any error output

For detailed installation instructions, platform-specific guides, and troubleshooting, see our [documentation](docs/):

- [Installation Guide](docs/guides/installation.md) - Comprehensive installation instructions for all platforms
- [Troubleshooting Guide](docs/guides/troubleshooting.md) - Solutions for common issues
- [Technical Documentation](docs/technical/) - Detailed technical procedures and specifications
- [Scripts Documentation](docs/guides/scripts.md) - Overview of available scripts and their usage

## Configuration

### Standard Configuration (Recommended)

Add the following to your `claude_desktop_config.json` file to use UV (recommended for best performance):

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

For Windows users, we recommend using the wrapper script to ensure PyTorch is properly installed. See our [Windows Setup Guide](docs/guides/windows-setup.md) for detailed instructions.

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

## Memory Operations

The memory service provides the following operations through the MCP server:

### Core Memory Operations

1. `store_memory` - Store new information with optional tags
2. `retrieve_memory` - Perform semantic search for relevant memories
3. `recall_memory` - Retrieve memories using natural language time expressions 
4. `search_by_tag` - Find memories using specific tags
5. `exact_match_retrieve` - Find memories with exact content match
6. `debug_retrieve` - Retrieve memories with similarity scores

For detailed information about tag storage and management, see our [Tag Storage Documentation](docs/technical/tag_storage.md).

### Database Management

7. `create_backup` - Create database backup
8. `get_stats` - Get memory statistics
9. `optimize_db` - Optimize database performance
10. `check_database_health` - Get database health metrics
11. `check_embedding_model` - Verify model status

### Memory Management

12. `delete_memory` - Delete specific memory by hash
13. `delete_by_tag` - Delete all memories with specific tag
14. `cleanup_duplicates` - Remove duplicate entries

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

## Getting Help

If you encounter any issues:
1. Check our [Troubleshooting Guide](docs/guides/troubleshooting.md)
2. Review the [Installation Guide](docs/guides/installation.md)
3. For Windows-specific issues, see our [Windows Setup Guide](docs/guides/windows-setup.md)
4. Contact the developer via Telegram: t.me/doobeedoo

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
│   ├── convert_to_uv.py         # Script to migrate to UV
│   └── install_uv.py            # UV installation helper
├── .uv/                         # UV configuration
├── memory_wrapper.py            # Windows wrapper script
├── memory_wrapper_uv.py         # UV-based wrapper script
├── uv_wrapper.py                # UV wrapper script
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

[t.me/doobidoo](https://t.me/+MJtKdOWzmQdhY2Vi)

## Cloudflare Worker Implementation

A serverless implementation of the MCP Memory Service is now available using Cloudflare Workers. This implementation:

- Uses **Cloudflare D1** for storage (serverless SQLite)
- Uses **Workers AI** for embeddings generation
- Communicates via **Server-Sent Events (SSE)** for MCP protocol
- Requires no local installation or dependencies
- Scales automatically with usage

### Benefits of the Cloudflare Implementation

- **Zero local installation**: No Python, dependencies, or local storage needed
- **Cross-platform compatibility**: Works on any device that can connect to the internet
- **Automatic scaling**: Handles multiple users without configuration
- **Global distribution**: Low latency access from anywhere
- **No maintenance**: Updates and maintenance handled automatically

### Available Tools in the Cloudflare Implementation

The Cloudflare Worker implementation supports all the same tools as the Python implementation:

| Tool | Description |
|------|-------------|
| `store_memory` | Store new information with optional tags |
| `retrieve_memory` | Find relevant memories based on query |
| `recall_memory` | Retrieve memories using natural language time expressions |
| `search_by_tag` | Search memories by tags |
| `delete_memory` | Delete a specific memory by its hash |
| `delete_by_tag` | Delete all memories with a specific tag |
| `cleanup_duplicates` | Find and remove duplicate entries |
| `get_embedding` | Get raw embedding vector for content |
| `check_embedding_model` | Check if embedding model is loaded and working |
| `debug_retrieve` | Retrieve memories with debug information |
| `exact_match_retrieve` | Retrieve memories using exact content match |
| `check_database_health` | Check database health and get statistics |
| `recall_by_timeframe` | Retrieve memories within a specific timeframe |
| `delete_by_timeframe` | Delete memories within a specific timeframe |
| `delete_before_date` | Delete memories before a specific date |

### Configuring Claude to Use the Cloudflare Memory Service

Add the following to your Claude configuration to use the Cloudflare-based memory service:

```json
{
  "mcpServers": [
    {
      "name": "cloudflare-memory",
      "url": "https://your-worker-subdomain.workers.dev/mcp",
      "type": "sse"
    }
  ]
}
```

Replace `your-worker-subdomain` with your actual Cloudflare Worker subdomain.

### Deploying Your Own Cloudflare Memory Service

1. Clone the repository and navigate to the Cloudflare Worker directory:
   ```bash
   git clone https://github.com/doobidoo/mcp-memory-service.git
   cd mcp-memory-service/cloudflare_worker
   ```

2. Install Wrangler (Cloudflare's CLI tool):
   ```bash
   npm install -g wrangler
   ```

3. Login to your Cloudflare account:
   ```bash
   wrangler login
   ```

4. Create a D1 database:
   ```bash
   wrangler d1 create mcp_memory_service
   ```

5. Update the `wrangler.toml` file with your database ID from the previous step.

6. Initialize the database schema:
   ```bash
   wrangler d1 execute mcp_memory_service --local --file=./schema.sql
   ```

   Where `schema.sql` contains:
   ```sql
   CREATE TABLE IF NOT EXISTS memories (
     id TEXT PRIMARY KEY,
     content TEXT NOT NULL,
     embedding TEXT NOT NULL,
     tags TEXT,
     memory_type TEXT,
     metadata TEXT,
     created_at INTEGER
   );
   CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at);
   ```

7. Deploy the worker:
   ```bash
   wrangler deploy
   ```

8. Update your Claude configuration to use your new worker URL.

### Testing Your Cloudflare Memory Service

After deployment, you can test your memory service using curl:

1. List available tools:
   ```bash
   curl https://your-worker-subdomain.workers.dev/list_tools
   ```

2. Store a memory:
   ```bash
   curl -X POST https://your-worker-subdomain.workers.dev/mcp \
     -H "Content-Type: application/json" \
     -d '{"method":"store_memory","arguments":{"content":"This is a test memory","metadata":{"tags":["test"]}}}'
   ```

3. Retrieve memories:
   ```bash
   curl -X POST https://your-worker-subdomain.workers.dev/mcp \
     -H "Content-Type: application/json" \
     -d '{"method":"retrieve_memory","arguments":{"query":"test memory","n_results":5}}'
   ```

### Limitations

- Free tier limits on Cloudflare Workers and D1 may apply
- Workers AI embedding models may differ slightly from the local sentence-transformers models
- No direct access to the underlying database for manual operations
- Cloudflare Workers have a maximum execution time of 30 seconds on free plans
