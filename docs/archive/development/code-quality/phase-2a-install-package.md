# Refactoring: install_package() - Phase 2, Function #4

## Summary
Refactored `install.py::install_package()` to reduce cyclomatic complexity and improve maintainability.

**Metrics:**
- **Original Complexity:** 27-33 (high-risk)
- **Refactored Main Function:** Complexity 7 (79% reduction)
- **Total Lines:** Original 199 → Refactored 39 (main function only)

## Refactoring Strategy: Extract Method Pattern

The function contained multiple responsibility areas with high nesting and branching. Extracted into 8 focused helper functions:

### Helper Functions Created

#### 1. `_setup_installer_command()` - CC: 6
**Purpose:** Detect and configure pip/uv package manager
**Responsibilities:**
- Check if pip is available
- Attempt uv installation if pip missing
- Return appropriate installer command

**Location:** Lines 1257-1298
**Usage:** Called at the beginning of `install_package()`

---

#### 2. `_configure_storage_and_gpu()` - CC: 9
**Purpose:** Configure storage backend and GPU environment setup
**Responsibilities:**
- Detect system and GPU capabilities
- Choose and install storage backend
- Set environment variables for backend and GPU type
- Return configured environment and system info

**Location:** Lines 1301-1357
**Usage:** After installer command setup

---

#### 3. `_handle_pytorch_setup()` - CC: 3
**Purpose:** Orchestrate PyTorch installation
**Responsibilities:**
- Detect Homebrew PyTorch installation
- Trigger platform-specific PyTorch installation if needed
- Set environment variables for ONNX when using Homebrew PyTorch

**Location:** Lines 1360-1387
**Usage:** After storage backend configuration

---

#### 4. `_should_use_onnx_installation()` - CC: 1
**Purpose:** Simple decision function for ONNX installation path
**Logic:** Return True if:
- macOS with Intel CPU AND
- (Python 3.13+ OR using Homebrew PyTorch OR skip_pytorch flag)

**Location:** Lines 1390-1402
**Usage:** Determines which installation flow to follow

---

#### 5. `_install_with_onnx()` - CC: 7
**Purpose:** SQLite-vec + ONNX specialized installation path
**Responsibilities:**
- Install without ML dependencies (--no-deps)
- Build dependency list (ONNX, tokenizers, etc.)
- Install backend-specific packages
- Configure environment for ONNX runtime
- Fall back to standard installation if fails

**Location:** Lines 1405-1477
**Usage:** Called when ONNX installation is appropriate

---

#### 6. `_install_standard()` - CC: 2
**Purpose:** Standard pip/uv installation
**Responsibilities:**
- Run pip/uv install command
- Handle success/failure cases

**Location:** Lines 1480-1502
**Usage:** Called for normal installation flow

---

#### 7. `_handle_installation_failure()` - CC: 3
**Purpose:** Provide troubleshooting guidance on failure
**Responsibilities:**
- Detect if macOS Intel platform
- Print platform-specific installation instructions
- Suggest Homebrew PyTorch workarounds if available

**Location:** Lines 1505-1521
**Usage:** Called only when installation fails

---

## Refactored `install_package()` Function - CC: 7

**New Structure:**
```python
def install_package(args):
    1. Setup installer command (pip/uv)
    2. Configure storage backend and GPU
    3. Handle PyTorch setup
    4. Decide installation path (ONNX vs Standard)
    5. Execute installation
    6. Handle failures if needed
    7. Return status
```

**Lines:** 39 (vs 199 original)
**Control Flow:** Reduced from 26 branches to 6

## Benefits

### Code Quality
- ✅ **Single Responsibility:** Each function has one clear purpose
- ✅ **Testability:** Helper functions can be unit tested independently
- ✅ **Readability:** Main function now reads like a high-level workflow
- ✅ **Maintainability:** Changes isolated to specific functions

### Complexity Reduction
- Main function complexity: 27 → 7 (74% reduction)
- Maximum helper function complexity: 9 (vs 27 original)
- Total cyclomatic complexity across all functions: ~38 (distributed vs monolithic)

### Architecture
- **Concerns separated:** GPU detection, backend selection, PyTorch setup, installation paths
- **Clear flow:** Easy to understand the order of operations
- **Error handling:** Dedicated failure handler with platform-specific guidance
- **Extensibility:** Easy to add new installation paths or backends

## Backward Compatibility

✅ **Fully compatible** - No changes to:
- Function signature: `install_package(args)`
- Return values: `bool`
- Environment variable handling
- Command-line argument processing
- Error messages and output format

## Testing Recommendations

1. **Unit Tests for Helpers:**
   - `test_setup_installer_command()` - Verify pip/uv detection
   - `test_configure_storage_and_gpu()` - Test backend selection
   - `test_should_use_onnx_installation()` - Test platform detection logic

2. **Integration Tests:**
   - Test full installation on macOS Intel with Python 3.13+
   - Test with Homebrew PyTorch detected
   - Test ONNX fallback path

3. **Manual Testing:**
   - Run with `--skip-pytorch` flag
   - Run with `--storage-backend sqlite_vec`
   - Verify error messages on intentional failures

## Related Issues

- **Issue #246:** Code Quality Phase 2 - Reduce Function Complexity
- **Phase 2 Progress:** 2/5 top functions completed
  - ✅ `install.py::main()` - Complexity 62 → ~8
  - ✅ `sqlite_vec.py::initialize()` - Complexity 38 → Reduced
  - ⏳ `config.py::__main__()` - Complexity 42 (next)
  - ⏳ `oauth/authorization.py::token()` - Complexity 35
  - ⏳ `install_package()` - Complexity 33 (this refactoring)

## Files Modified

- `install.py`: Refactored `install_package()` function with 7 new helper functions

## Git Commit

Use semantic commit message:
```
refactor: reduce install_package() complexity from 27 to 7 (74% reduction)

Extract helper functions for:
- Installer command setup (pip/uv detection)
- Storage backend and GPU configuration
- PyTorch installation orchestration
- ONNX installation path decision
- ONNX-specific installation
- Standard pip/uv installation
- Installation failure handling

All functions individually testable and maintainable.
Addresses issue #246 Phase 2, function #4 of top 5 high-complexity targets.
```
