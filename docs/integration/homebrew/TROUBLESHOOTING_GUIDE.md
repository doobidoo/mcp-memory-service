# MCP Memory Service Troubleshooting Guide

## Homebrew PyTorch Integration

This guide provides practical steps for diagnosing and resolving common issues with the MCP Memory Service, particularly focusing on the Homebrew PyTorch integration.

## Diagnostic Commands

### Verifying Homebrew PyTorch Installation

```bash
# Check if PyTorch is installed via Homebrew
brew list | grep pytorch

# Get the path to Homebrew PyTorch
brew --prefix pytorch

# Verify Homebrew Python location
PYTORCH_PREFIX=$(brew --prefix pytorch)
echo "$PYTORCH_PREFIX/libexec/bin/python3"

# Verify that the Python interpreter exists
ls -la "$PYTORCH_PREFIX/libexec/bin/python3"
```

### Checking Python Packages in Homebrew Environment

```bash
# Get Homebrew Python path
HOMEBREW_PYTHON="$(brew --prefix pytorch)/libexec/bin/python3"

# List installed packages
$HOMEBREW_PYTHON -m pip list

# Check if sentence-transformers is installed
$HOMEBREW_PYTHON -c "import sentence_transformers; print(sentence_transformers.__version__)"

# Check if PyTorch is working correctly
$HOMEBREW_PYTHON -c "import torch; print(torch.__version__); print(torch.cuda.is_available() if hasattr(torch, 'cuda') else 'CUDA not available')"
```

### Testing Embedding Generation

```bash
# Simple test of embedding generation
$HOMEBREW_PYTHON -c """
import torch
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
emb = model.encode(['This is a test'])
print(f'Generated embedding with shape {emb.shape}')
"""
```

### Verifying MCP Memory Service Configuration

```bash
# Check current environment variables
env | grep MCP_MEMORY

# Verify database location
ls -la "$HOME/Library/Application Support/mcp-memory/"

# Check database file
file "$HOME/Library/Application Support/mcp-memory/sqlite_vec.db"
```

### Testing MCP Memory Service Health

```bash
# Check if the server is running
ps aux | grep -i memory

# Test server health endpoint
curl -X POST http://localhost:8000/health -H "Content-Type: application/json" -d '{}'

# Test simple memory storage
curl -X POST http://localhost:8000/store_memory -H "Content-Type: application/json" -d '{"memory": {"content": "Test memory", "tags": ["test"]}}'
```

## Common Failure Modes and Solutions

### 1. Homebrew PyTorch Not Found

**Symptoms:**
- Error messages about missing PyTorch in Homebrew
- "pytorch not found in brew list"

**Solutions:**

1. Install PyTorch via Homebrew:
   ```bash
   brew install pytorch
   ```

2. If installation fails, try updating Homebrew:
   ```bash
   brew update && brew upgrade
   brew install pytorch
   ```

3. Verify the installation:
   ```bash
   brew list | grep pytorch
   ```

### 2. sentence-transformers Not Available in Homebrew Python

**Symptoms:**
- "ImportError: No module named sentence_transformers"
- Embedding generation failures

**Solutions:**

1. Install sentence-transformers in the Homebrew Python environment:
   ```bash
   HOMEBREW_PYTHON="$(brew --prefix pytorch)/libexec/bin/python3"
   $HOMEBREW_PYTHON -m pip install sentence-transformers
   ```

2. If installation fails, check for dependency issues:
   ```bash
   $HOMEBREW_PYTHON -m pip install --upgrade pip
   $HOMEBREW_PYTHON -m pip install sentence-transformers --verbose
   ```

3. Verify the installation:
   ```bash
   $HOMEBREW_PYTHON -c "import sentence_transformers; print(sentence_transformers.__version__)"
   ```

### 3. Module Override Failures

**Symptoms:**
- Log messages indicating "Failed to override SqliteVecMemoryStorage"
- Server using original implementation despite Homebrew being enabled

**Solutions:**

1. Verify the environment variable is set correctly:
   ```bash
   export MCP_MEMORY_USE_HOMEBREW_PYTORCH=1
   ```

2. Restart the server completely (not just the process):
   ```bash
   # Find and kill the server
   pkill -f memory
   # Start with correct environment
   MCP_MEMORY_USE_HOMEBREW_PYTORCH=1 memory
   ```

3. Use the dedicated Homebrew startup script:
   ```bash
   ./run_with_homebrew.sh
   ```

### 4. Subprocess Communication Errors

**Symptoms:**
- "Error generating embeddings" in logs
- Timeout errors in subprocess execution

**Solutions:**

1. Check if temporary files are being created and cleaned up:
   ```bash
   ls -la /tmp/tmp*
   ```

2. Increase subprocess timeout values in the code:
   ```python
   # Increase timeout for model loading
   result = subprocess.run([self.homebrew_python, "-c", script], capture_output=True, text=True, timeout=120)
   ```

3. Verify file permissions for temporary directories:
   ```bash
   ls -la /tmp
   ```

### 5. Database Connection Issues

**Symptoms:**
- "Database not initialized" errors
- SQLite errors in logs

**Solutions:**

1. Check if the database directory exists and is writable:
   ```bash
   mkdir -p "$HOME/Library/Application Support/mcp-memory/"
   touch "$HOME/Library/Application Support/mcp-memory/test.txt" && rm "$HOME/Library/Application Support/mcp-memory/test.txt"
   ```

2. Verify sqlite-vec extension is properly loaded:
   ```bash
   python -c "import sqlite_vec; print(sqlite_vec.__version__)"
   ```

3. Check for database corruption and create a new one if needed:
   ```bash
   mv "$HOME/Library/Application Support/mcp-memory/sqlite_vec.db" "$HOME/Library/Application Support/mcp-memory/sqlite_vec.db.bak"
   ```

### 6. MCP Protocol Compliance Issues

**Symptoms:**
- "Error parsing response" in client
- Non-JSON output in stdout

**Solutions:**

1. Verify logging is configured to use stderr:
   ```python
   logging.basicConfig(stream=sys.stderr)
   ```

2. Check for print statements in the code that might be writing to stdout

3. Redirect all diagnostic output to stderr in subprocess scripts:
   ```python
   print(f"Debug: {message}", file=sys.stderr)
   ```

## Environment Variable Verification

### Critical Environment Variables

| Variable | Purpose | Valid Values | Default |
|----------|---------|--------------|--------|
| `MCP_MEMORY_USE_HOMEBREW_PYTORCH` | Enable Homebrew PyTorch integration | `1`, `true`, `yes` | Not set (disabled) |
| `MCP_MEMORY_STORAGE_BACKEND` | Storage backend selection | `sqlite_vec`, `chroma` | `chroma` |
| `MCP_MEMORY_SQLITE_PATH` | Path to SQLite database | Valid file path | `~/Library/Application Support/mcp-memory/sqlite_vec.db` |
| `MCP_MEMORY_USE_ONNX` | Enable ONNX runtime (fallback) | `1`, `true`, `yes` | Not set (disabled) |
| `LOG_LEVEL` | Logging verbosity | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |

### Verifying Environment Configuration

```bash
# Create a script to verify all environment variables
cat > verify_env.sh << 'EOF'
#!/bin/bash
echo "===== MCP Memory Service Environment Check ====="
echo "Homebrew PyTorch enabled: ${MCP_MEMORY_USE_HOMEBREW_PYTORCH:-Not set (disabled)}"
echo "Storage backend: ${MCP_MEMORY_STORAGE_BACKEND:-chroma (default)}"
echo "SQLite database path: ${MCP_MEMORY_SQLITE_PATH:-Not set}"
echo "ONNX runtime enabled: ${MCP_MEMORY_USE_ONNX:-Not set (disabled)}"
echo "Log level: ${LOG_LEVEL:-INFO (default)}"
echo "================="
echo "Checking Homebrew PyTorch installation:"
if brew list | grep -q pytorch; then
    echo "✅ PyTorch is installed via Homebrew"
    PYTORCH_PREFIX=$(brew --prefix pytorch)
    HOMEBREW_PYTHON="$PYTORCH_PREFIX/libexec/bin/python3"
    if [ -f "$HOMEBREW_PYTHON" ]; then
        echo "✅ Homebrew Python found at: $HOMEBREW_PYTHON"
        if $HOMEBREW_PYTHON -c "import sentence_transformers" 2>/dev/null; then
            echo "✅ sentence-transformers is installed"
        else
            echo "❌ sentence-transformers is NOT installed in Homebrew Python"
        fi
    else
        echo "❌ Homebrew Python NOT found at: $HOMEBREW_PYTHON"
    fi
else
    echo "❌ PyTorch is NOT installed via Homebrew"
fi
echo "================="
echo "Checking database:"
DB_PATH="${MCP_MEMORY_SQLITE_PATH:-$HOME/Library/Application Support/mcp-memory/sqlite_vec.db}"
DB_DIR=$(dirname "$DB_PATH")
if [ -d "$DB_DIR" ]; then
    echo "✅ Database directory exists: $DB_DIR"
    if [ -w "$DB_DIR" ]; then
        echo "✅ Database directory is writable"
    else
        echo "❌ Database directory is NOT writable"
    fi
else
    echo "❌ Database directory does NOT exist: $DB_DIR"
fi
if [ -f "$DB_PATH" ]; then
    echo "✅ Database file exists: $DB_PATH"
    echo "   File size: $(du -h "$DB_PATH" | cut -f1)"
else
    echo "ℹ️  Database file does not exist yet (will be created on first run)"
fi
echo "===== Environment Check Complete ====="
EOF

chmod +x verify_env.sh
./verify_env.sh
```

## Service Startup and Health Check

### Standard Startup Procedure

```bash
# Set environment variables
export MCP_MEMORY_STORAGE_BACKEND="sqlite_vec"
export MCP_MEMORY_SQLITE_PATH="$HOME/Library/Application Support/mcp-memory/sqlite_vec.db"
export MCP_MEMORY_USE_HOMEBREW_PYTORCH="1"

# Create necessary directories
mkdir -p "$HOME/Library/Application Support/mcp-memory"

# Start the server
memory
```

### Homebrew-Enabled Startup

```bash
# Use the provided script
./run_with_homebrew.sh

# Or manually with full environment setup
MCP_MEMORY_STORAGE_BACKEND="sqlite_vec" \
MCP_MEMORY_SQLITE_PATH="$HOME/Library/Application Support/mcp-memory/sqlite_vec.db" \
MCP_MEMORY_USE_HOMEBREW_PYTORCH="1" \
LOG_LEVEL="DEBUG" \
python -m mcp_memory_service.homebrew_server
```

### Health Check Procedure

```bash
# Create a health check script
cat > check_health.sh << 'EOF'
#!/bin/bash
echo "Testing MCP Memory Service health..."
RESPONSE=$(curl -s -X POST http://localhost:8000/health -H "Content-Type: application/json" -d '{}')
if [[ $RESPONSE == *"healthy"* ]]; then
    echo "✅ Server is healthy: $RESPONSE"
else
    echo "❌ Server health check failed: $RESPONSE"
fi

# Test basic memory storage
echo "Testing memory storage..."
CONTENT="Test memory created at $(date)"
STORE_RESPONSE=$(curl -s -X POST http://localhost:8000/store_memory -H "Content-Type: application/json" -d "{\"memory\": {\"content\": \"$CONTENT\", \"tags\": [\"test\"]}}")
if [[ $STORE_RESPONSE == *"success"* ]]; then
    echo "✅ Successfully stored test memory"
else
    echo "❌ Failed to store test memory: $STORE_RESPONSE"
fi

# Test memory retrieval
echo "Testing memory retrieval..."
RETRIEVE_RESPONSE=$(curl -s -X POST http://localhost:8000/retrieve_memory -H "Content-Type: application/json" -d '{"query": "test", "n_results": 1}')
if [[ $RETRIEVE_RESPONSE == *"success"* ]] && [[ $RETRIEVE_RESPONSE != *"results\":[]"* ]]; then
    echo "✅ Successfully retrieved memories"
else
    echo "❌ Failed to retrieve memories: $RETRIEVE_RESPONSE"
fi
EOF

chmod +x check_health.sh
./check_health.sh
```

## Advanced Debugging Techniques

### Tracing Subprocess Execution

```bash
# Add detailed tracing to subprocess execution
cat > trace_subprocess.py << 'EOF'
import os
import sys
import subprocess
import tempfile
import time

def trace_subprocess(cmd, input_data=None):
    print(f"[TRACE] Executing subprocess: {cmd}", file=sys.stderr)
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        duration = time.time() - start_time
        print(f"[TRACE] Subprocess completed in {duration:.2f}s with return code {result.returncode}", file=sys.stderr)
        
        if result.stdout:
            print(f"[TRACE] stdout ({len(result.stdout)} chars): {result.stdout[:200]}{'...' if len(result.stdout) > 200 else ''}", file=sys.stderr)
        else:
            print(f"[TRACE] stdout: <empty>", file=sys.stderr)
            
        if result.stderr:
            print(f"[TRACE] stderr ({len(result.stderr)} chars): {result.stderr[:200]}{'...' if len(result.stderr) > 200 else ''}", file=sys.stderr)
        else:
            print(f"[TRACE] stderr: <empty>", file=sys.stderr)
            
        return result
    except subprocess.TimeoutExpired as e:
        print(f"[TRACE] Subprocess TIMEOUT after {e.timeout}s", file=sys.stderr)
        raise
    except Exception as e:
        print(f"[TRACE] Subprocess ERROR: {str(e)}", file=sys.stderr)
        raise

# Example usage
if __name__ == "__main__":
    homebrew_python = subprocess.check_output(["brew", "--prefix", "pytorch"]).decode().strip() + "/libexec/bin/python3"
    print(f"Using Homebrew Python: {homebrew_python}", file=sys.stderr)
    
    # Test basic Python execution
    result = trace_subprocess([homebrew_python, "-c", "print('Hello from Homebrew Python');"])
    
    # Test importing sentence_transformers
    result = trace_subprocess([homebrew_python, "-c", "import sentence_transformers; print(sentence_transformers.__version__);"])
    
    # Test embedding generation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as f:
        f.write('''
import time
import sys
print("Starting embedding test...", file=sys.stderr)
start_time = time.time()
try:
    from sentence_transformers import SentenceTransformer
    print(f"Loaded sentence_transformers in {time.time() - start_time:.2f}s", file=sys.stderr)
    
    model_start = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Loaded model in {time.time() - model_start:.2f}s", file=sys.stderr)
    
    embed_start = time.time()
    embedding = model.encode(["This is a test sentence"])
    print(f"Generated embedding in {time.time() - embed_start:.2f}s", file=sys.stderr)
    
    print(embedding.shape)
except Exception as e:
    print(f"Error: {str(e)}", file=sys.stderr)
    sys.exit(1)
''')
        f.flush()
        result = trace_subprocess([homebrew_python, f.name])
EOF

python trace_subprocess.py
```

### Analyzing SQLite Database

```bash
# Create a script to analyze the SQLite database
cat > analyze_db.py << 'EOF'
import os
import sys
import sqlite3
import json

# Get database path from environment or use default
db_path = os.environ.get(
    'MCP_MEMORY_SQLITE_PATH',
    os.path.expanduser('~/Library/Application Support/mcp-memory/sqlite_vec.db')
)

print(f"Analyzing database at: {db_path}")

if not os.path.exists(db_path):
    print(f"Database file not found: {db_path}")
    sys.exit(1)

# Connect to database
conn = sqlite3.connect(db_path)

# Get list of tables
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"Tables in database: {tables}")

# Analyze each table
for table in tables:
    print(f"\nTable: {table}")
    # Get column info
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [(row[1], row[2]) for row in cursor.fetchall()]
    print(f"Columns: {', '.join([f'{name} ({type_})' for name, type_ in columns])}")
    
    # Count rows
    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
    row_count = cursor.fetchone()[0]
    print(f"Row count: {row_count}")
    
    # Show sample data if not empty
    if row_count > 0:
        cursor = conn.execute(f"SELECT * FROM {table} LIMIT 1")
        sample_row = cursor.fetchone()
        print("Sample row:")
        for i, col in enumerate(columns):
            if col[0] in ['metadata', 'tags'] and sample_row[i]:
                try:
                    # Try to pretty-print JSON
                    formatted = json.dumps(json.loads(sample_row[i]), indent=2) if sample_row[i] else 'NULL'
                except:
                    formatted = sample_row[i]
                print(f"  {col[0]}: {formatted}")
            else:
                value = sample_row[i]
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + '...'
                print(f"  {col[0]}: {value}")

# Check for virtual tables (for sqlite-vec)
print("\nChecking for virtual tables:")
cursor = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql LIKE '%USING vec%'")
virtual_tables = cursor.fetchall()
if virtual_tables:
    for name, sql in virtual_tables:
        print(f"Virtual table: {name}")
        print(f"Creation SQL: {sql}")
else:
    print("No vector virtual tables found")

# Close connection
conn.close()
print("\nDatabase analysis complete")
EOF

python analyze_db.py
```

## Conclusion

This troubleshooting guide provides practical steps for diagnosing and resolving common issues with the MCP Memory Service, particularly focusing on the Homebrew PyTorch integration. By following these systematic debugging procedures, you can identify and fix most integration issues that may arise.

Remember that the Homebrew PyTorch integration involves multiple components working together:

1. A Homebrew installation of PyTorch with its own Python environment
2. The MCP Memory Service using SQLite-vec as a storage backend
3. A subprocess-based communication mechanism between these environments
4. Proper module overrides to enable the integration

Most issues stem from problems in one of these components or in their interactions. The diagnostic tools and procedures in this guide will help you identify which component is causing the issue and how to resolve it.