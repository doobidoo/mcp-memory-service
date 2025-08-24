# Claude Desktop Setup Guide - Windows

This guide helps you configure the MCP Memory Service to work with Claude Desktop on Windows without repeated PyTorch downloads.

## Problem and Solution

**Issue**: Claude Desktop was downloading PyTorch models (230MB+) on every startup due to missing offline configuration.

**Solution**: Added offline mode environment variables to your Claude Desktop config to use cached models.

## What Was Fixed

✅ **Your Claude Desktop Config Updated**:
- Added offline mode environment variables (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`)
- Added cache path configurations 
- Kept your existing SQLite-vec backend setup

✅ **Verified Components Working**:
- SQLite-vec database: 434 memories accessible ✅
- sentence-transformers: Loading models without downloads ✅
- Offline mode: No network requests when properly configured ✅

## Current Working Configuration

Your active config at `%APPDATA%\Claude\claude_desktop_config.json` now has:

- **Backend**: SQLite-vec (single database file)
- **Database**: `memory_migrated.db` with 434 memories
- **Offline Mode**: Enabled to prevent downloads
- **UV Package Manager**: For better dependency management

## For Other Users

See `examples/claude_desktop_config_windows.json` for an anonymized template with:
- SQLite-vec backend configuration (recommended)
- ChromaDB alternative configuration  
- Offline mode settings
- Performance optimizations

Replace `YOUR_USERNAME` with your actual Windows username.

## Key Changes Made

### 1. Config Template Updates
- Removed `PYTHONNOUSERSITE=1`, `PIP_NO_DEPENDENCIES=1`, `PIP_NO_INSTALL=1`
- These were blocking access to globally installed packages

### 2. Server Path Detection
Enhanced `src/mcp_memory_service/server.py`:
- Better virtual environment detection
- Windows-specific path handling
- Global site-packages access when not blocked

### 3. Dependency Checking
Improved `src/mcp_memory_service/dependency_check.py`:
- Enhanced model cache detection for Windows
- Better first-run detection logic
- Multiple cache location checking

### 4. Storage Backend Fixes
Updated both ChromaDB and SQLite-vec storage:
- Fixed hardcoded Linux paths
- Added offline mode configuration
- Better cache path detection

## Verification

After updating your Claude Desktop config:

1. **Restart Claude Desktop** completely
2. **Check the logs** - you should see:
   ```
   ✅ All dependencies are installed
   DEBUG: Found cached model in C:\Users\[username]\.cache\huggingface\hub
   ```
3. **No more downloads** - The 230MB PyTorch download should not occur

## Testing

You can test the server directly:
```bash
python scripts/run_memory_server.py --debug
```

You should see dependency checking passes and models load from cache.

## Troubleshooting

If you still see downloads:
1. Verify your username in the config paths
2. Check that models exist in `%USERPROFILE%\.cache\huggingface\hub`
3. Ensure Claude Desktop has been fully restarted

## Files Modified

- `examples/claude_desktop_config_template.json` - Removed blocking env vars
- `examples/claude_desktop_config_windows.json` - New Windows-specific config
- `src/mcp_memory_service/server.py` - Enhanced path detection
- `src/mcp_memory_service/dependency_check.py` - Better cache detection
- `src/mcp_memory_service/storage/sqlite_vec.py` - Fixed hardcoded paths
- `src/mcp_memory_service/storage/chroma.py` - Added offline mode support