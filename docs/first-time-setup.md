# First-Time Setup Guide

This guide explains what to expect when running MCP Memory Service for the first time.

## 🎯 What to Expect on First Run

When you start the MCP Memory Service for the first time, you'll see several warnings and messages. **This is completely normal behavior** as the service initializes and downloads necessary components.

## 📋 Normal First-Time Warnings

### 1. Snapshots Directory Warning
```
WARNING:mcp_memory_service.storage.sqlite_vec:Failed to load from cache: No snapshots directory
```

**What it means:**
- The service is checking for previously downloaded embedding models
- On first run, no cache exists yet, so this warning appears
- The service will automatically download the model

**This is normal:** ✅ Expected on first run

### 2. TRANSFORMERS_CACHE Warning
```
WARNING: Using TRANSFORMERS_CACHE is deprecated
```

**What it means:**
- This is an informational warning from the Hugging Face library
- It doesn't affect the service functionality
- The service handles caching internally

**This is normal:** ✅ Can be safely ignored

### 3. Model Download Progress
```
Downloading model 'all-MiniLM-L6-v2'...
```

**What it means:**
- The service is downloading the embedding model (approximately 25MB)
- This happens only once on first setup
- Download time: 1-2 minutes on average internet connection

**This is normal:** ✅ One-time download

## 🚦 Success Indicators

After successful first-time setup, you should see:

```
INFO: SQLite-vec storage initialized successfully with embedding dimension: 384
INFO: Memory service started on port 8443
INFO: Ready to accept connections
```

## 📊 First-Time Setup Timeline

| Step | Duration | What's Happening |
|------|----------|-----------------|
| 1. Service Start | Instant | Loading configuration |
| 2. Cache Check | 1-2 seconds | Checking for existing models |
| 3. Model Download | 1-2 minutes | Downloading embedding model (~25MB) |
| 4. Model Loading | 5-10 seconds | Loading model into memory |
| 5. Database Init | 2-3 seconds | Creating database structure |
| 6. Ready | - | Service is ready to use |

**Total first-time setup:** 2-3 minutes

## 🔄 Subsequent Runs

After the first successful run:
- No download warnings will appear
- Model loads from cache (5-10 seconds)
- Service starts much faster (10-15 seconds total)

## 🐍 Python 3.13 Compatibility

### Known Issues
Python 3.13 users may encounter installation issues with **sqlite-vec** due to missing pre-built wheels. The installer now includes automatic fallback methods:

1. **Automatic Retry Logic**: Tries multiple installation strategies
2. **Source Building**: Attempts to build from source if wheels unavailable
3. **GitHub Installation**: Falls back to installing directly from repository
4. **Backend Switching**: Option to switch to ChromaDB if sqlite-vec fails

### Recommended Solutions
If you encounter sqlite-vec installation failures on Python 3.13:

**Option 1: Use Python 3.12 (Recommended)**
```bash
# macOS
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
python install.py

# Ubuntu/Linux
sudo apt install python3.12 python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
python install.py
```

**Option 2: Use ChromaDB Backend**
```bash
python install.py --storage-backend chromadb
```

**Option 3: Manual sqlite-vec Installation**
```bash
# Try building from source
pip install --no-binary :all: sqlite-vec

# Or install from GitHub
pip install git+https://github.com/asg017/sqlite-vec.git#subdirectory=python
```

## 🍎 macOS SQLite Extension Issues

### Problem: `AttributeError: 'sqlite3.Connection' object has no attribute 'enable_load_extension'`

This error occurs on **macOS with system Python** because it's not compiled with SQLite extension support.

**Why this happens:**
- macOS system Python lacks `--enable-loadable-sqlite-extensions`
- The bundled SQLite library doesn't support loadable extensions
- This is a security-focused default configuration

**Solutions:**

**Option 1: Homebrew Python (Recommended)**
```bash
# Install Python via Homebrew (includes extension support)
brew install python
hash -r  # Refresh command cache
python3 --version  # Verify you're using Homebrew Python

# Then install MCP Memory Service
python3 install.py
```

**Option 2: pyenv with Extension Support**
```bash
# Install pyenv if not already installed
brew install pyenv

# Install Python with extension support
PYTHON_CONFIGURE_OPTS="--enable-loadable-sqlite-extensions" pyenv install 3.12.0
pyenv local 3.12.0

# Verify extension support
python3 -c "import sqlite3; conn=sqlite3.connect(':memory:'); conn.enable_load_extension(True); print('Extensions supported!')"
```

**Option 3: Use ChromaDB Backend**
```bash
# ChromaDB doesn't require SQLite extensions
python3 install.py --storage-backend chromadb
```

### Verification
Check if your Python supports extensions:
```bash
python3 -c "
import sqlite3
conn = sqlite3.connect(':memory:')
print('✅ Extension support available' if hasattr(conn, 'enable_load_extension') else '❌ No extension support')
"
```

## 🐧 Ubuntu/Linux Specific Notes

For Ubuntu 24 and other Linux distributions:

### Prerequisites
```bash
# System dependencies
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev python3-pip
sudo apt install build-essential libblas3 liblapack3 liblapack-dev libblas-dev gfortran
```

### Recommended Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the service
python install.py

# Start the service
uv run memory server
```

## 🔧 Troubleshooting First-Time Issues

### Issue: Download Fails
**Solution:**
- Check internet connection
- Verify firewall/proxy settings
- Clear cache and retry: `rm -rf ~/.cache/huggingface`

### Issue: "No module named 'sentence_transformers'"
**Solution:**
```bash
pip install sentence-transformers torch
```

### Issue: Permission Denied
**Solution:**
```bash
# Fix permissions
chmod +x scripts/*.sh
sudo chown -R $USER:$USER ~/.mcp_memory_service/
```

### Issue: Service Doesn't Start After Download
**Solution:**
1. Check logs: `uv run memory server --debug`
2. Verify installation: `python scripts/verify_environment.py`
3. Restart with clean state:
   ```bash
   rm -rf ~/.mcp_memory_service
   uv run memory server
   ```

## ✅ Verification

To verify successful setup:

```bash
# Check service health
curl -k https://localhost:8443/api/health

# Or using the CLI
uv run memory health
```

Expected response:
```json
{
  "status": "healthy",
  "storage_backend": "sqlite_vec",
  "model_loaded": true
}
```

## 🎉 Setup Complete!

Once you see the success indicators and the warnings have disappeared on subsequent runs, your MCP Memory Service is fully operational and ready to use!

### Next Steps:
- [Configure Claude Desktop](../README.md#claude-desktop-integration)
- [Store your first memory](../README.md#basic-usage)
- [Explore the API](https://github.com/doobidoo/mcp-memory-service/wiki)

## 📝 Notes

- The model download is a one-time operation
- Downloaded models are cached in `~/.cache/huggingface/`
- The service creates a database in `~/.mcp_memory_service/`
- All warnings shown during first-time setup are expected behavior
- If you see different errors (not warnings), check the [Troubleshooting Guide](troubleshooting/general.md)

---

Remember: **First-time warnings are normal!** The service is working correctly and setting itself up for optimal performance.
