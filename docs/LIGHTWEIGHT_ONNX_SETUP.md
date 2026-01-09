# MCP Memory Service - Portable Multi-Machine Setup

**Author:** Sundeepg98
**Purpose:** Quickly replicate optimized MCP memory setup on any machine

---

## Quick Setup (One Command)

```bash
# Install from YOUR fork with ONNX patches
pipx install "git+https://github.com/Sundeepg98/mcp-memory-service.git"
```

---

## Step-by-Step Setup

### 1. Install from Fork

```bash
# Install the patched version from your fork
pipx install "git+https://github.com/Sundeepg98/mcp-memory-service.git"

# Verify installation
pipx list | grep mcp-memory
```

### 2. Add to Claude Code Settings

Add to `~/.claude/settings.json` in the `mcpServers` section:

```json
{
  "mcpServers": {
    "memory": {
      "type": "stdio",
      "command": "~/.local/share/pipx/venvs/mcp-memory-service/bin/python",
      "args": ["-m", "mcp_memory_service.server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_QUALITY_BOOST_ENABLED": "true"
      }
    }
  }
}
```

### 3. Add Environment Variables

Add to `~/.claude/settings.json` in the `env` section:

```json
{
  "env": {
    "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
    "MCP_MEMORY_USE_ONNX": "true",
    "MCP_QUALITY_BOOST_ENABLED": "true",
    "MCP_CONSOLIDATION_ENABLED": "true",
    "MCP_DECAY_ENABLED": "true",
    "MCP_CLUSTERING_ENABLED": "true"
  }
}
```

---

## Automated Setup Script

Save this as `setup-mcp-memory.sh`:

```bash
#!/bin/bash
# MCP Memory Service - Automated Setup
# Installs from Sundeepg98's fork with ONNX patches

set -e

echo "🔧 Installing MCP Memory Service from fork..."
pipx install "git+https://github.com/Sundeepg98/mcp-memory-service.git" --force

echo "📁 Creating data directory..."
mkdir -p ~/.local/share/mcp-memory

echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Add MCP server config to ~/.claude/settings.json (see mcp-memory-portable-setup.md)"
echo "2. Restart Claude Code"
echo ""
echo "ONNX models will auto-download on first use (~255MB)"
```

---

## What Gets Created

| Location | Content | Per-Machine? |
|----------|---------|--------------|
| `~/.local/share/pipx/venvs/mcp-memory-service/` | Patched code | Same across machines |
| `~/.local/share/mcp-memory/sqlite_vec.db` | Your memories | Different per machine |
| `~/.cache/mcp_memory/onnx_models/` | AI models (255MB) | Auto-downloaded |

---

## Benefits of This Setup

- **90% smaller** than transformers+PyTorch install (805MB vs 7.7GB)
- **Quality scoring** works with ONNX Runtime only
- **Your patches** included (no waiting for upstream merge)
- **One command** to replicate on any machine

---

## Updating

```bash
# Update to latest from your fork
pipx upgrade mcp-memory-service --pip-args="--upgrade"

# Or reinstall fresh
pipx uninstall mcp-memory-service
pipx install "git+https://github.com/Sundeepg98/mcp-memory-service.git"
```

---

## Syncing Memories Between Machines

Your memories (`sqlite_vec.db`) are per-machine. To sync:

```bash
# Option 1: Copy database file
scp ~/.local/share/mcp-memory/sqlite_vec.db user@other-machine:~/.local/share/mcp-memory/

# Option 2: Use hybrid backend with Cloudflare (cloud sync)
# Set MCP_MEMORY_STORAGE_BACKEND=hybrid and configure Cloudflare
```

---

*Last updated: 2026-01-09*
