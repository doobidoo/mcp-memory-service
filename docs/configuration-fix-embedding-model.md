# Configuration Fix: Embedding Model Environment Variables

## Problem Summary

**Issue**: Environment variable `MCP_MEMORY_EMBEDDING_MODEL` was being ignored in Docker deployments, causing the application to use the hardcoded default `all-MiniLM-L6-v2` instead of the configured `sentence-transformers/all-mpnet-base-v2`.

**Root Cause**: Pydantic-settings `alias` parameter conflicted with `env_prefix`, preventing proper environment variable resolution.

## Technical Details

### Original Code (Broken)

```python
class StorageSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='MCP_MEMORY_',
        # ...
    )

    embedding_model: str = Field(
        default='all-MiniLM-L6-v2',
        alias='EMBEDDING_MODEL',  # ❌ CONFLICTS WITH env_prefix
        description="Embedding model name"
    )
```

**Why this failed:**
- `env_prefix='MCP_MEMORY_'` tells pydantic to look for `MCP_MEMORY_*` env vars
- `alias='EMBEDDING_MODEL'` tells pydantic to look for EXACT name `EMBEDDING_MODEL`
- These two directives conflict - pydantic chooses alias, ignoring the prefix
- Docker sets `MCP_MEMORY_EMBEDDING_MODEL` (with prefix), but pydantic only checks `EMBEDDING_MODEL` (without prefix)
- Result: Environment variable never read, default value used

### Fixed Code

```python
class StorageSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='MCP_MEMORY_',
        # ...
    )

    embedding_model: str = Field(
        default='all-MiniLM-L6-v2',
        description="Embedding model name (env: MCP_MEMORY_EMBEDDING_MODEL)"  # ✅ WORKS CORRECTLY
    )
```

**Why this works:**
- Removed conflicting `alias` parameter
- Pydantic now correctly applies `env_prefix` to field name
- Looks for: `MCP_MEMORY_` + `EMBEDDING_MODEL` = `MCP_MEMORY_EMBEDDING_MODEL`
- Docker env var matches expected name exactly
- Result: Environment variable correctly read and applied

## Changes Made

### 1. `src/mcp_memory_service/config.py`

**Line 187-195**: Removed `alias` parameters from `StorageSettings`:

```diff
- embedding_model: str = Field(
-     default='all-MiniLM-L6-v2',
-     alias='EMBEDDING_MODEL',
-     description="Embedding model name"
- )
+ embedding_model: str = Field(
+     default='all-MiniLM-L6-v2',
+     description="Embedding model name (env: MCP_MEMORY_EMBEDDING_MODEL)"
+ )

- use_onnx: bool = Field(
-     default=False,
-     alias='USE_ONNX',
-     description="Use ONNX for embeddings (PyTorch-free)"
- )
+ use_onnx: bool = Field(
+     default=False,
+     description="Use ONNX for embeddings (PyTorch-free) (env: MCP_MEMORY_USE_ONNX)"
+ )
```

### 2. Lazy Configuration Loading (Bonus Fix)

**Lines 788-971**: Implemented lazy configuration loading via module-level `__getattr__`:

- **Why**: Prevents import-time freezing of config values
- **Benefit**: Environment variables are read when values are accessed, not when module is imported
- **Impact**: Better compatibility with Docker's environment variable propagation timing

**Key improvements**:
- All config constants now lazy-loaded via `__getattr__()`
- Settings instantiation deferred until first config access
- OAuth validation functions updated to use lazy settings
- ONNX cache directory creation moved to lazy helper function

## Verification

### Test Case 1: Docker-style Environment Variable

```bash
MCP_MEMORY_EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2" python3 -c "
from src.mcp_memory_service import config
print(config.EMBEDDING_MODEL_NAME)
"
```

**Expected Output**: `sentence-transformers/all-mpnet-base-v2`

### Test Case 2: Default Value When Env Var Not Set

```bash
python3 -c "
from src.mcp_memory_service import config
print(config.EMBEDDING_MODEL_NAME)
"
```

**Expected Output**: `all-MiniLM-L6-v2`

### Test Case 3: Storage Factory Integration

```bash
MCP_MEMORY_EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2" \
MCP_MEMORY_STORAGE_BACKEND="sqlite_vec" \
python3 -c "
import asyncio
from src.mcp_memory_service.storage.factory import create_storage_instance
asyncio.run(create_storage_instance('/tmp/test.db'))
"
```

**Expected Behavior**: Attempts to load `sentence-transformers/all-mpnet-base-v2` (not default `all-MiniLM-L6-v2`)

## Impact

### Before Fix
- ❌ Docker env vars ignored
- ❌ Always used hardcoded default model
- ❌ Semantic search quality degraded
- ❌ Model download failures (wrong model not cached)

### After Fix
- ✅ Docker env vars properly read
- ✅ Configured model correctly used
- ✅ Semantic search quality as expected
- ✅ Model downloads/caching work correctly

## Breaking Changes

**None**. This fix maintains full backward compatibility:

- Default value unchanged (`all-MiniLM-L6-v2`)
- Config API surface unchanged (all constants still accessible)
- Existing code continues to work
- Only affects users who were trying (and failing) to override the embedding model via env vars

## Related Files

- `/Users/68824/code/27B/mcp/mcp-memory-service/src/mcp_memory_service/config.py` (main fix)
- `/Users/68824/code/27B/mcp/mcp-memory-service/src/mcp_memory_service/storage/factory.py` (uses fixed config)
- `/Users/68824/code/27B/mcp/mcp-memory-service/src/mcp_memory_service/storage/sqlite_vec.py` (embedding model initialization)

## Deployment Notes

For users experiencing this issue in Docker deployments:

1. **Update code** to version with this fix
2. **Rebuild Docker image**
3. **Verify configuration** using health check:
   ```bash
   docker exec mcp-memory env | grep EMBEDDING
   docker logs mcp-memory | grep "Loading embedding model"
   ```
4. **Expected log output**:
   ```
   Loading embedding model: sentence-transformers/all-mpnet-base-v2
   ```

## Pydantic-Settings Best Practices

**Lesson learned**: When using `env_prefix`, do NOT use `alias` on field definitions.

✅ **Correct**:
```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='APP_')

    database_url: str = Field(default="...")  # Reads APP_DATABASE_URL
```

❌ **Wrong**:
```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='APP_')

    database_url: str = Field(
        default="...",
        alias='DATABASE_URL'  # Conflicts with env_prefix!
    )
```

## References

- Pydantic Settings Documentation: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- Issue Report: Docker env vars not being read for embedding model configuration
- Fix Commit: [Add commit hash after committing]
