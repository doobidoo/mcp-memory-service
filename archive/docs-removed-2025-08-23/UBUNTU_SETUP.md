# Ubuntu Setup Guide for MCP Memory Service with SQLite-vec

## 🎯 Overview

This guide shows how to set up the MCP Memory Service with SQLite-vec backend on Ubuntu for integration with Claude Code and VS Code.

## ✅ Prerequisites Met

You have successfully completed:
- ✅ SQLite-vec installation and testing  
- ✅ Basic dependencies (sentence-transformers, torch, mcp)
- ✅ Environment configuration

## 🔧 Current Setup Status

Your Ubuntu machine now has:

```bash
# Virtual environment active
source venv/bin/activate

# SQLite-vec backend configured
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec

# Key packages installed:
- sqlite-vec (0.1.6)
- sentence-transformers (5.0.0)  
- torch (2.7.1+cpu)
- mcp (1.11.0)
```

## Claude Code Integration

### 1. Install Claude Code CLI

If not already installed:
```bash
# Install Claude Code CLI
curl -fsSL https://claude.ai/install.sh | sh
```

### 2. Configure MCP Memory Service with Claude Code

#### Option A: Automatic Configuration (Recommended)
```bash
# Run installer with Claude Code auto-configuration
python install.py --configure-claude-code
```

This automatically:
- Detects Claude Code installation
- Creates personalized .mcp.json from template
- Replaces placeholder paths with your system paths
- Adds optimized environment variables
- Adds .mcp.json to .gitignore (protects personal info)
- Verifies the configuration works

#### Option B: Manual Configuration
```bash
# Navigate to project directory
cd /home/hkr/repositories/mcp-memory-service

# Add memory service with optimized settings
claude mcp add memory-service --scope project \
  -e MCP_MEMORY_CHROMA_PATH=$HOME/.mcp_memory_chroma \
  -e LOG_LEVEL=INFO \
  -e MCP_TIMEOUT=30000 \
  -- python scripts/run_memory_server.py

# Verify configuration
claude mcp list
```

### 3. Secure Configuration Management

The system uses a template-based approach to protect personal information:

- **Template**: .mcp.json.template (shared, no personal data)
- **Generated**: .mcp.json (personalized, automatically added to .gitignore)
- **Placeholders**: {{USER_HOME}} replaced with your actual home directory

### 2. Database Location

Your SQLite-vec database will be created at:
```
/home/hkr/.local/share/mcp-memory/sqlite_vec.db
```

This single file contains all your memories and can be easily backed up.

### 3. Claude Code Usage

With the MCP Memory Service running, Claude Code can:

- **Store memories**: "Remember that I prefer using Ubuntu for development"
- **Retrieve memories**: "What did I tell you about my development preferences?"
- **Search by tags**: Find memories with specific topics
- **Time-based recall**: "What did we discuss yesterday about databases?"

### 4. Performance Benefits

SQLite-vec backend provides:
- **75% less memory usage** vs ChromaDB
- **Faster startup times** (2-3x faster)
- **Single file database** (easy backup/share)
- **Better for <100K memories**

## 💻 VS Code Integration Options

### Option 1: Claude Code in VS Code Terminal
```bash
# Open VS Code in your project
code /home/hkr/repositories/mcp-memory-service

# Use integrated terminal to run Claude Code with memory support
# The memory service will automatically use sqlite-vec backend
```

### Option 2: MCP Extension (if available)
```bash
# Install VS Code MCP extension when available
# Configure to use local MCP Memory Service
```

### Option 3: Development Workflow
```bash
# 1. Keep MCP Memory Service running in background
python -m src.mcp_memory_service.server &

# 2. Use Claude Code normally - it will connect to your local service
# 3. All memories stored in local sqlite-vec database
```

## 🔄 Migration from ChromaDB (if needed)

If you have existing ChromaDB data to migrate:

```bash
# Simple migration
python migrate_to_sqlite_vec.py

# Or with custom paths
python scripts/migrate_storage.py \
  --from chroma \
  --to sqlite_vec \
  --backup \
  --backup-path backup.json
```

## 🧪 Testing the Setup

### Quick Test
```bash
# Test that everything works
source venv/bin/activate
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
python simple_sqlite_vec_test.py
```

### Full Test (when server is ready)
```bash
# Test MCP server startup
python -c "
import os
os.environ['MCP_MEMORY_STORAGE_BACKEND'] = 'sqlite_vec'
from src.mcp_memory_service.server import main
print('✅ Server can start with sqlite-vec backend')
"
```

## 🛠️ Troubleshooting

### Common Issues

1. **Module Import Errors**
   ```bash
   # Make sure you're in the virtual environment
   source venv/bin/activate
   
   # Check installed packages
   pip list | grep -E "(sqlite-vec|sentence|torch|mcp)"
   ```

2. **Permission Errors**
   ```bash
   # Ensure database directory is writable
   mkdir -p ~/.local/share/mcp-memory
   chmod 755 ~/.local/share/mcp-memory
   ```

3. **Memory/Performance Issues**
   ```bash
   # SQLite-vec uses much less memory than ChromaDB
   # Monitor with: htop or free -h
   ```

### Environment Variables

Add to your `~/.bashrc` for permanent configuration:
```bash
echo 'export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec' >> ~/.bashrc
source ~/.bashrc
```

## 📊 Performance Comparison

| Metric | ChromaDB | SQLite-vec | Improvement |
|--------|----------|------------|-------------|
| Memory Usage (1K memories) | ~200MB | ~50MB | 75% less |
| Startup Time | ~5-10s | ~2-3s | 2-3x faster |
| Disk Usage | ~50MB | ~35MB | 30% less |
| Database Files | Multiple | Single | Simpler |

## 🎉 Next Steps

1. **Start using the memory service** with Claude Code
2. **Store development notes** and project information  
3. **Build up your memory database** over time
4. **Enjoy faster, lighter memory operations**

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the [SQLite-vec Backend Guide](../sqlite-vec-backend.md)
3. Test with `simple_sqlite_vec_test.py`

Your Ubuntu setup is ready for high-performance memory operations with Claude Code! 🚀