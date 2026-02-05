# SQLite-vec Embedding Fixes

This document summarizes the fixes applied to resolve issue #64 where semantic search returns 0 results in the SQLite-vec backend.

## Root Causes Identified

1. **Missing Core Dependencies**: `sentence-transformers` and `torch` were in optional dependencies, causing silent failures
2. **Dimension Mismatch**: Vector table was created with hardcoded dimensions before model initialization
3. **Silent Failures**: Missing dependencies returned zero vectors without raising exceptions
4. **Database Integrity Issues**: Potential rowid misalignment between memories and embeddings tables

## Changes Made

### 1. Fixed Dependencies (pyproject.toml)

- Moved `sentence-transformers>=2.2.2` from optional to core dependencies
- Added `torch>=1.6.0` to core dependencies
- This ensures embedding functionality is always available

### 2. Fixed Initialization Order (sqlite_vec.py)

- Moved embedding model initialization BEFORE vector table creation
- This ensures the correct embedding dimension is used for the table schema
- Added explicit check for sentence-transformers availability

### 3. Improved Error Handling

- Replaced silent failures with explicit exceptions
- Added proper error messages for missing dependencies
- Added embedding validation after generation (dimension check, finite values check)

### 4. Fixed Database Operations

#### Store Operation:
- Added try-catch for embedding generation with proper error propagation
- Added fallback for rowid insertion if direct rowid insert fails
- Added validation before storing embeddings

#### Retrieve Operation:
- Added check for empty embeddings table
- Added debug logging for troubleshooting
- Improved error handling for query embedding generation

### 5. Created Diagnostic Script

- `scripts/test_sqlite_vec_embeddings.py` - comprehensive test suite
- Tests dependencies, initialization, embedding generation, storage, and search
- Provides clear error messages and troubleshooting guidance

## Key Code Changes

### sqlite_vec.py:

1. **Initialize method**:
   - Added sentence-transformers check
   - Moved model initialization before table creation

2. **_generate_embedding method**:
   - Raises exception instead of returning zero vector
   - Added comprehensive validation

3. **store method**:
   - Better error handling for embedding generation
   - Fallback for rowid insertion

4. **retrieve method**:
   - Check for empty embeddings table
   - Better debug logging

## Testing

Run the diagnostic script to verify the fixes:

```bash
python3 scripts/test_sqlite_vec_embeddings.py
```

This will check:
- Dependency installation
- Storage initialization
- Embedding generation
- Memory storage with embeddings
- Semantic search functionality
- Database integrity

## Migration Notes

For existing installations:

1. Update dependencies: `uv pip install -e .`
2. Use the provided migration tools to save existing memories:

### Option 1: Quick Repair (Try First)
For databases with missing embeddings but correct schema:

```bash
python3 scripts/repair_sqlite_vec_embeddings.py /path/to/your/sqlite_vec.db
```

This will:
- Analyze your database
- Generate missing embeddings
- Verify search functionality

### Option 2: Full Migration (If Repair Fails)
For databases with dimension mismatches or schema issues:

```bash
python3 scripts/migrate_sqlite_vec_embeddings.py /path/to/your/sqlite_vec.db
```

This will:
- Create a backup of your database
- Extract all memories
- Create a new database with correct schema
- Regenerate all embeddings
- Restore all memories

**Important**: The migration creates a timestamped backup before making any changes.

## Future Improvements

1. ~~Add migration script for existing databases~~ ✓ Done
2. Add batch embedding generation for better performance
3. ~~Add embedding regeneration capability for existing memories~~ ✓ Done
4. Implement better rowid synchronization between tables
5. Add automatic detection and repair on startup
6. Add embedding model versioning to handle model changes
