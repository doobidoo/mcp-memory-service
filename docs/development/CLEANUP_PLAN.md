# Cleanup Plan for MCP-MEMORY-SERVICE

## 1. Test Files Organization

### Current Test Files
- `test_chromadb.py` - Tests ChromaDB initialization with new API pattern
- `test_health_check_fixes.py` - Tests health check fixes and validation
- `test_issue_5_fix.py` - Tests tag deletion functionality
- `test_performance_optimizations.py` - Tests performance improvements

### Recommended Organization
1. **Create a structured tests directory:**
   ```
   tests/
   ├── integration/         # Integration tests between components
   │   ├── test_server.py   # Server integration tests
   │   └── test_storage.py  # Storage integration tests
   ├── unit/                # Unit tests for individual components
   │   ├── test_chroma.py   # ChromaDB-specific tests
   │   ├── test_config.py   # Configuration tests
   │   └── test_utils.py    # Utility function tests
   └── performance/         # Performance benchmarks
       ├── test_caching.py  # Cache performance tests
       └── test_queries.py  # Query performance tests
   ```

2. **Move existing test files to appropriate directories:**
   - `test_chromadb.py` → `tests/unit/test_chroma.py`
   - `test_health_check_fixes.py` → `tests/integration/test_storage.py`
   - `test_issue_5_fix.py` → `tests/unit/test_tags.py`
   - `test_performance_optimizations.py` → `tests/performance/test_caching.py`

3. **Create a proper test runner:**
   - Add `pytest.ini` configuration
   - Add `conftest.py` with common fixtures
   - Create a `.coveragerc` file for coverage reporting

## 2. Documentation Organization

### Current Documentation
- `CHANGELOG.md` - Release history and changes
- `CLAUDE.md` - Claude-specific documentation
- `CLEANUP_SUMMARY.md` - Cleanup summary
- `HEALTH_CHECK_FIXES_SUMMARY.md` - Health check fixes documentation
- `PERFORMANCE_OPTIMIZATION_SUMMARY.md` - Performance optimization documentation
- `README.md` - Main project documentation

### Recommended Organization
1. **Consolidate implementation documentation:**
   ```
   docs/
   ├── guides/                # User guides
   │   ├── getting_started.md # Quick start guide
   │   └── configuration.md   # Configuration options
   ├── implementation/        # Implementation details
   │   ├── health_checks.md   # Health check documentation
   │   ├── performance.md     # Performance optimization details
   │   └── tags.md           # Tag functionality documentation
   ├── api/                   # API documentation
   │   ├── server.md          # Server API documentation
   │   └── storage.md         # Storage API documentation
   └── examples/              # Example code
       ├── basic_usage.md     # Basic usage examples
       └── advanced.md        # Advanced usage examples
   ```

2. **Move existing documentation:**
   - `HEALTH_CHECK_FIXES_SUMMARY.md` → `docs/implementation/health_checks.md`
   - `PERFORMANCE_OPTIMIZATION_SUMMARY.md` → `docs/implementation/performance.md`
   - Keep `CHANGELOG.md` in the root directory
   - Move `CLAUDE.md` to `docs/guides/claude_integration.md`

## 3. Backup and Archive Files

### Files to Archive
- `backup_performance_optimization/` - Archive this directory
- Any development artifacts that are no longer needed

### Recommended Action
1. **Create an archive directory:**
   ```
   archive/
   ├── 2025-06-24/            # Archive by date
   │   ├── tests/             # Old test files
   │   └── docs/              # Old documentation
   ```

2. **Move backup files to the archive:**
   - Move `backup_performance_optimization/` to `archive/2025-06-24/`
   - Create a README in the archive directory explaining what's stored there

## 4. Git Cleanup Actions

### Recommended Git Actions
1. **Create a new branch for changes:**
   ```bash
   git checkout -b feature/cleanup-and-organization
   ```

2. **Add and organize files:**
   ```bash
   # Create new directories
   mkdir -p tests/integration tests/unit tests/performance
   mkdir -p docs/guides docs/implementation docs/api docs/examples
   mkdir -p archive/2025-06-24
   
   # Move test files
   git mv test_chromadb.py tests/unit/test_chroma.py
   git mv test_health_check_fixes.py tests/integration/test_storage.py
   git mv test_issue_5_fix.py tests/unit/test_tags.py
   git mv test_performance_optimizations.py tests/performance/test_caching.py
   
   # Move documentation
   git mv HEALTH_CHECK_FIXES_SUMMARY.md docs/implementation/health_checks.md
   git mv PERFORMANCE_OPTIMIZATION_SUMMARY.md docs/implementation/performance.md
   git mv CLAUDE.md docs/guides/claude_integration.md
   
   # Archive backup files
   git mv backup_performance_optimization archive/2025-06-24/
   ```

3. **Update CHANGELOG.md:**
   ```bash
   git mv CHANGELOG.md.new CHANGELOG.md
   ```

4. **Commit changes:**
   ```bash
   git add .
   git commit -m "Organize tests, documentation, and archive old files"
   ```

5. **Create new branch for hardware testing:**
   ```bash
   git checkout -b test/hardware-validation
   ```

## 5. Final Verification Steps

1. **Run tests to ensure everything still works:**
   ```bash
   cd tests
   pytest
   ```

2. **Verify documentation links are updated:**
   - Check README.md for any links to moved files
   - Update any cross-references in documentation

3. **Ensure CHANGELOG is complete:**
   - Verify all changes are documented
   - Check version numbers and dates

4. **Track changes in memory:**
   ```bash
   # Store the changes in memory
   memory store_memory --content "Reorganized MCP-MEMORY-SERVICE project structure on June 24, 2025. Created proper test directory structure, consolidated documentation in docs/ directory, and archived old backup files. Changes are in the feature/cleanup-and-organization branch, with hardware testing in test/hardware-validation branch." --tags "mcp-memory-service,cleanup,reorganization,memory-driven"
   ```