# Quick Setup: Cloudflare Backend for Claude Desktop + Claude Code

This guide provides streamlined instructions to configure Cloudflare backend for both Claude Desktop and Claude Code simultaneously.

## 🎯 Overview

This setup ensures both environments use the same Cloudflare backend for consistent memory storage across Claude Desktop and Claude Code.

**Expected Result:**
- Claude Desktop: ✅ Cloudflare backend with 1000+ memories
- Claude Code: ✅ Cloudflare backend with same memories
- Health checks show: `"backend": "cloudflare"` and `"storage_type": "CloudflareStorage"`

## ⚡ Quick Setup (5 minutes)

### Step 1: Prepare Cloudflare Resources

If you don't have Cloudflare resources yet:

```bash
# Install wrangler CLI
npm install -g wrangler

# Login and create resources
wrangler login
wrangler vectorize create mcp-memory-index --dimensions=768 --metric=cosine
wrangler d1 create mcp-memory-db

# Note the database ID from output
```

### Step 2: Create Environment Configuration

Create `.env` file in the project root:

```bash
cd C:/REPOSITORIES/mcp-memory-service

# Create .env file with your Cloudflare credentials
cat > .env << 'EOF'
# MCP Memory Service Environment Configuration
MCP_MEMORY_STORAGE_BACKEND=cloudflare

# Cloudflare D1 Database Configuration
CLOUDFLARE_API_TOKEN=your-api-token-here
CLOUDFLARE_ACCOUNT_ID=your-account-id-here
CLOUDFLARE_D1_DATABASE_ID=your-d1-database-id-here
CLOUDFLARE_VECTORIZE_INDEX=mcp-memory-index

# Backup paths (for fallback)
MCP_MEMORY_BACKUPS_PATH=C:\Users\your-username\AppData\Local\mcp-memory\backups
MCP_MEMORY_SQLITE_PATH=C:\Users\your-username\AppData\Local\mcp-memory\backups\sqlite_vec.db
EOF
```

### Step 3: Configure Claude Desktop

Update `~/.claude.json` (or `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "mcp_memory_service.server"],
      "cwd": "C:/REPOSITORIES/mcp-memory-service",
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "cloudflare",
        "CLOUDFLARE_API_TOKEN": "your-api-token-here",
        "CLOUDFLARE_ACCOUNT_ID": "your-account-id-here",
        "CLOUDFLARE_D1_DATABASE_ID": "your-d1-database-id-here",
        "CLOUDFLARE_VECTORIZE_INDEX": "mcp-memory-index",
        "MCP_MEMORY_BACKUPS_PATH": "C:\\Users\\your-username\\AppData\\Local\\mcp-memory\\backups",
        "MCP_MEMORY_SQLITE_PATH": "C:\\Users\\your-username\\AppData\\Local\\mcp-memory\\backups\\sqlite_vec.db"
      }
    }
  }
}
```

### Step 4: Configure Claude Code

```bash
# Navigate to project directory
cd C:/REPOSITORIES/mcp-memory-service

# Add memory server with explicit environment variables
claude mcp add memory python \
  -e MCP_MEMORY_STORAGE_BACKEND=cloudflare \
  -e CLOUDFLARE_API_TOKEN=your-api-token-here \
  -e CLOUDFLARE_ACCOUNT_ID=your-account-id-here \
  -e CLOUDFLARE_D1_DATABASE_ID=your-d1-database-id-here \
  -e CLOUDFLARE_VECTORIZE_INDEX=mcp-memory-index \
  -e MCP_MEMORY_BACKUPS_PATH="C:\Users\your-username\AppData\Local\mcp-memory\backups" \
  -e MCP_MEMORY_SQLITE_PATH="C:\Users\your-username\AppData\Local\mcp-memory\backups\sqlite_vec.db" \
  -- -m mcp_memory_service.server
```

### Step 5: Verify Configuration

**Test Claude Desktop:**
1. Restart Claude Desktop
2. Open a new conversation
3. Ask: "Check memory health"
4. Should show: `"backend": "cloudflare"` and `"storage_type": "CloudflareStorage"`

**Test Claude Code:**
```bash
# Check MCP server status
claude mcp list

# Should show: memory: python -m mcp_memory_service.server - ✓ Connected
```

## 🔧 Configuration Templates

### Claude Desktop Template (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "mcp_memory_service.server"],
      "cwd": "C:/REPOSITORIES/mcp-memory-service",
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "cloudflare",
        "CLOUDFLARE_API_TOKEN": "YOUR_TOKEN_HERE",
        "CLOUDFLARE_ACCOUNT_ID": "YOUR_ACCOUNT_ID_HERE",
        "CLOUDFLARE_D1_DATABASE_ID": "YOUR_D1_DATABASE_ID_HERE",
        "CLOUDFLARE_VECTORIZE_INDEX": "mcp-memory-index",
        "MCP_MEMORY_BACKUPS_PATH": "C:\\Users\\USERNAME\\AppData\\Local\\mcp-memory\\backups",
        "MCP_MEMORY_SQLITE_PATH": "C:\\Users\\USERNAME\\AppData\\Local\\mcp-memory\\backups\\sqlite_vec.db"
      }
    }
  }
}
```

### Project Environment Template (`.env`)

```bash
# Storage Backend Configuration
MCP_MEMORY_STORAGE_BACKEND=cloudflare

# Required Cloudflare Settings
CLOUDFLARE_API_TOKEN=YOUR_TOKEN_HERE
CLOUDFLARE_ACCOUNT_ID=YOUR_ACCOUNT_ID_HERE
CLOUDFLARE_D1_DATABASE_ID=YOUR_D1_DATABASE_ID_HERE
CLOUDFLARE_VECTORIZE_INDEX=mcp-memory-index

# Optional Settings
CLOUDFLARE_R2_BUCKET=mcp-memory-content
CLOUDFLARE_EMBEDDING_MODEL=@cf/baai/bge-base-en-v1.5
CLOUDFLARE_LARGE_CONTENT_THRESHOLD=1048576
CLOUDFLARE_MAX_RETRIES=3
CLOUDFLARE_BASE_DELAY=1.0

# Backup Configuration
MCP_MEMORY_BACKUPS_PATH=C:\Users\USERNAME\AppData\Local\mcp-memory\backups
MCP_MEMORY_SQLITE_PATH=C:\Users\USERNAME\AppData\Local\mcp-memory\backups\sqlite_vec.db

# Logging
LOG_LEVEL=INFO
```

## ✅ Validation Commands

### Quick Health Check

```bash
# Test configuration loading
cd C:/REPOSITORIES/mcp-memory-service
python -c "
from src.mcp_memory_service.config import STORAGE_BACKEND, CLOUDFLARE_API_TOKEN
print(f'Backend: {STORAGE_BACKEND}')
print(f'Token set: {bool(CLOUDFLARE_API_TOKEN)}')
"

# Test server initialization
python scripts/validation/diagnose_backend_config.py
```

### Expected Health Check Results

**Cloudflare Backend (Correct):**
```json
{
  "validation": {
    "status": "healthy",
    "message": "Cloudflare storage validation successful"
  },
  "statistics": {
    "backend": "cloudflare",
    "storage_backend": "cloudflare",
    "total_memories": 1073,
    "vectorize_index": "mcp-memory-index",
    "d1_database_id": "f745e9b4-ba8e-4d47-b38f-12af91060d5a"
  },
  "performance": {
    "server": {
      "storage_type": "CloudflareStorage"
    }
  }
}
```

**SQLite-vec Fallback (Incorrect):**
```json
{
  "statistics": {
    "backend": "sqlite-vec",
    "storage_backend": "sqlite-vec"
  },
  "performance": {
    "server": {
      "storage_type": "SqliteVecMemoryStorage"
    }
  }
}
```

## 🚨 Troubleshooting

### Issue: Health Check Shows SQLite-vec Instead of Cloudflare

**Root Cause:** Environment variables not loading properly in execution context.

**Solutions:**

1. **Claude Desktop:**
   - Ensure `cwd` is set to project directory
   - Use explicit `env` variables in MCP configuration
   - Restart Claude Desktop after config changes

2. **Claude Code:**
   - Use explicit `-e` environment variables in `claude mcp add`
   - Ensure command runs from project directory
   - Remove and re-add memory server to pick up changes

3. **Both Environments:**
   - Verify `.env` file exists and contains correct values
   - Check API token permissions (Vectorize:Edit, D1:Edit, Workers AI:Read)
   - Test Cloudflare connectivity manually

### Issue: "Missing required environment variables"

```bash
# Check if variables are being loaded
cd C:/REPOSITORIES/mcp-memory-service
python -c "
import os
from dotenv import load_dotenv
load_dotenv('.env')
print('CLOUDFLARE_API_TOKEN:', 'SET' if os.getenv('CLOUDFLARE_API_TOKEN') else 'NOT SET')
print('CLOUDFLARE_ACCOUNT_ID:', os.getenv('CLOUDFLARE_ACCOUNT_ID', 'NOT SET'))
"
```

### Issue: Different Memory Counts Between Environments

This indicates environments are using different backends:
- **Same count (e.g., 1073):** Both using Cloudflare ✅
- **Different counts:** One using SQLite-vec fallback ❌

**Fix:** Follow troubleshooting steps above to ensure both use Cloudflare.

### Issue: Connection Failed or Authentication Errors

1. **Verify API Token:**
   ```bash
   curl -X GET "https://api.cloudflare.com/client/v4/user/tokens/verify" \
     -H "Authorization: Bearer YOUR_API_TOKEN"
   ```

2. **Check Resource IDs:**
   ```bash
   # List Vectorize indexes
   curl -X GET "https://api.cloudflare.com/client/v4/accounts/YOUR_ACCOUNT_ID/vectorize/v2/indexes" \
     -H "Authorization: Bearer YOUR_API_TOKEN"

   # List D1 databases
   curl -X GET "https://api.cloudflare.com/client/v4/accounts/YOUR_ACCOUNT_ID/d1/database" \
     -H "Authorization: Bearer YOUR_API_TOKEN"
   ```

## 🔄 Migration from SQLite-vec

If you have existing memories in SQLite-vec:

```bash
# Export existing memories
python scripts/export_sqlite_vec.py --output cloudflare_export.json

# Switch to Cloudflare (follow setup above)

# Import to Cloudflare
python scripts/import_to_cloudflare.py --input cloudflare_export.json
```

## 📝 Configuration Management

### Single Source of Truth

- **Global Config:** `~/.claude.json` (Claude Desktop) - authoritative
- **Project Config:** `.env` file (development) - for local development
- **Avoid:** Multiple conflicting configurations

### Environment Variable Precedence

1. Explicit MCP server `env` variables (highest priority)
2. System environment variables
3. `.env` file variables
4. Default values (lowest priority)

## 🎯 Success Criteria

Both Claude Desktop and Claude Code should show:

✅ **Health Check:** `"backend": "cloudflare"`
✅ **Storage Type:** `"CloudflareStorage"`
✅ **Memory Count:** Same number across environments
✅ **Database ID:** Same Cloudflare D1 database ID
✅ **Index:** Same Vectorize index name

When successful, memories will be synchronized across both environments automatically!
