# Jules Bug Fix Plan - Post WSL2 Testing

## Confirmed Working
- ✅ All dependencies install correctly in Linux (WSL2)
- ✅ ONNX Runtime works without crashes
- ✅ PyTorch loads properly
- ✅ ChromaDB initializes
- ✅ MCP server starts and runs
- ✅ Service credentials are valid (tested with real values)

## Infrastructure Issues (Not Code)

### 1. Neon Database Schema
- **Problem**: Missing pgvector extension functions
- **Error**: `function array_length(vector, integer) does not exist`
- **Fix**: Run migrations to install pgvector extension
- **Command**: `CREATE EXTENSION IF NOT EXISTS vector;`

### 2. R2 Bucket
- **Problem**: 400 Bad Request on bucket operations
- **Fix**: Create bucket or fix permissions in Cloudflare dashboard

### 3. Qdrant
- **Status**: Not tested (USE_QDRANT=false by default)
- **Fix**: Set USE_QDRANT=true in .env to enable

## Required Code Fixes

### 1. Import Path Issues
**Files Affected**: All Python files in src/
**Problem**: Need PYTHONPATH configuration
**Fix**:
```python
# Add to top of each entry point file:
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### 2. Missing --use-echovault Flag
**File**: `src/mcp_memory_service/server.py`
**Problem**: Flag not implemented
**Fix**:
```python
parser.add_argument("--use-echovault", action="store_true", 
                    help="Enable EchoVault features")
# Then check: args.use_echovault or os.getenv('USE_ECHOVAULT', '').lower() == 'true'
```

### 3. Async Test Fixtures
**Files**: `tests/test_database.py` and others
**Problem**: `'async_generator' object has no attribute 'store_memory'`
**Fix**: Use `@pytest_asyncio.fixture` decorator properly

### 4. Windows ARM64 Compatibility
**Create**: `src/mcp_memory_service/utils/platform_compat.py`
**Purpose**: Detect platform and provide fallbacks
```python
import platform
import sys

def is_windows_arm64():
    return sys.platform == 'win32' and platform.machine() == 'ARM64'

def get_onnx_providers():
    if is_windows_arm64():
        return ['CPUExecutionProvider']  # Fallback
    return None  # Use defaults
```

### 5. Memory Wrapper Updates
**File**: `memory_wrapper.py`
**Issues**:
- PyTorch version detection
- UV environment conflicts
**Fix**: Check for existing PyTorch before installing

## Testing Strategy

### Phase 1: Fix Import Paths
1. Update all entry points with proper sys.path
2. Test each module can be imported
3. Verify no circular imports

### Phase 2: Implement EchoVault Flag
1. Add argument to server.py
2. Create factory pattern for storage selection
3. Test both modes (ChromaDB and EchoVault)

### Phase 3: Fix Async Tests
1. Update test fixtures to use proper decorators
2. Ensure event loops are managed correctly
3. Run full test suite

### Phase 4: Windows Compatibility
1. Add platform detection
2. Create graceful fallbacks
3. Document Windows limitations

## Branch Strategy
```bash
# From feature/echovault-complete
git checkout -b jules-fix-imports
# Fix import paths

git checkout -b jules-fix-echovault-flag  
# Implement --use-echovault

git checkout -b jules-fix-async-tests
# Fix test fixtures

git checkout -b jules-fix-windows-compat
# Add platform detection
```

## Priority Order
1. **Import paths** - Blocking everything
2. **EchoVault flag** - Core functionality
3. **Async tests** - Quality assurance
4. **Windows compat** - Nice to have

## Success Criteria
- [ ] Can run `python -m mcp_memory_service.server` without PYTHONPATH
- [ ] Can run `python -m mcp_memory_service.server --use-echovault`
- [ ] All tests pass with `pytest tests/`
- [ ] Graceful fallback on Windows ARM64

## Notes for Jules
- WSL2 is the recommended development environment
- Real credentials have been tested and work
- Database schema needs setup (not a code issue)
- Focus on code fixes, not infrastructure 