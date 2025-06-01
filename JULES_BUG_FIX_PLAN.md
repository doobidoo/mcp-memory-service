# Jules Bug Fix Plan for EchoVault

**Created**: 2025-05-31  
**Priority**: Fix blocking issues first, then enhance

## 🚨 Priority 1: Environment Setup (Blocking All Tests)

### Issue 1.1: Missing .env Configuration
**Problem**: No .env file exists with required credentials  
**Fix**:
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add actual credentials for:
# - NEON_DSN
# - QDRANT_URL and QDRANT_API_KEY  
# - R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
```
**Verification**: Check file exists and has valid credentials

### Issue 1.2: Python 3.12 Incompatibility
**Problem**: PyTorch 2.1.0 doesn't support Python 3.12  
**Fix Option A - Recommended**: Update memory_wrapper.py to use newer PyTorch
```python
# In memory_wrapper.py, update PyTorch version:
# Line ~374: Change from torch==2.1.0 to torch>=2.4.0
subprocess.check_call([
    sys.executable, '-m', 'uv', 'pip', 'install',
    'torch>=2.4.0', 'torchvision>=0.19.0', 'torchaudio>=2.4.0',
    '--extra-index-url', 'https://download.pytorch.org/whl/cpu'
])
```

**Fix Option B**: Create Python 3.11 environment
```bash
conda create -n echovault python=3.11
conda activate echovault
pip install -r requirements.txt
pip install -r requirements-echovault.txt
```

### Issue 1.3: ONNX Runtime DLL Crash
**Problem**: ONNX Runtime crashes with access violation on Windows  
**Fix**:
```bash
# 1. Install Visual C++ Redistributables
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

# 2. Reinstall ONNX Runtime
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime==1.17.1

# 3. If still failing, try CPU-only version
pip install onnxruntime==1.17.1 --force-reinstall
```

## 🔧 Priority 2: Code Fixes (Enable Basic Functionality)

### Issue 2.1: Import Path Conflicts
**Problem**: Conflicting imports between local and installed mcp_memory_service  
**Fix**: Update imports in test files
```python
# In tests/test_echovault_integration.py, line 16
# Change from:
from src.mcp_memory_service.storage.echovault import EchoVaultStorage
# To:
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mcp_memory_service.storage.echovault import EchoVaultStorage
```

### Issue 2.2: ChromaDB Dependency on PyTorch
**Problem**: EchoVault mode still loads ChromaDB which requires PyTorch  
**Fix**: Lazy load ChromaDB in factory.py
```python
# In src/mcp_memory_service/storage/factory.py
def create_storage(path: Optional[str] = None) -> MemoryStorage:
    use_echovault = os.environ.get("USE_ECHOVAULT", "").lower() in ("true", "1", "yes")
    
    if use_echovault:
        try:
            from .echovault import EchoVaultStorage
            logger.info("Using EchoVault storage implementation")
            return EchoVaultStorage(path)
        except ImportError as e:
            logger.warning(f"Failed to import EchoVaultStorage: {e}")
    
    # Only import ChromaDB if not using EchoVault
    from .chroma import ChromaMemoryStorage
    logger.info("Using ChromaDB storage implementation")
    return ChromaMemoryStorage(path)
```

### Issue 2.3: Test Connectivity Import Error
**Problem**: test_connectivity.py fails to import modules  
**Fix**: Add proper error handling and module path
```python
# At the top of src/mcp_memory_service/test_connectivity.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
```

## ⚡ Priority 3: Dependency Conflicts (Non-blocking)

### Issue 3.1: OpenTelemetry Version Mismatch
**Problem**: Semantic conventions version conflict  
**Fix**: Update requirements-echovault.txt
```
opentelemetry-api==1.33.0
opentelemetry-sdk==1.33.0
opentelemetry-exporter-otlp==1.33.0
```

### Issue 3.2: Protobuf Version Conflicts
**Problem**: Different packages need different protobuf versions  
**Fix**: Pin to compatible version
```bash
pip install protobuf==4.25.3
```

## 📋 Testing Sequence After Fixes

1. **Environment Check**:
   ```bash
   python -c "import sys; print(f'Python: {sys.version}')"
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import onnxruntime; print(f'ONNX: {onnxruntime.__version__}')"
   ```

2. **EchoVault Connectivity**:
   ```bash
   python src/mcp_memory_service/test_connectivity.py
   ```

3. **Database Migrations**:
   ```bash
   cd migrations
   alembic upgrade head
   ```

4. **EchoVault Mode Test**:
   ```bash
   python memory_wrapper.py --use-echovault --force-cpu
   ```

5. **Integration Tests**:
   ```bash
   pytest tests/test_echovault_integration.py -v -s
   ```

## 🎯 Quick Start Commands

```bash
# 1. Fix Python environment
conda create -n echovault python=3.11 -y
conda activate echovault

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-echovault.txt
pip install onnxruntime==1.17.1

# 3. Configure environment
cp .env.example .env
# Edit .env with your credentials

# 4. Test EchoVault
python src/mcp_memory_service/test_connectivity.py
```

## Expected Outcome After Fixes

- ✅ EchoVault mode works without PyTorch
- ✅ Database connectivity test passes
- ✅ Integration tests run without crashes
- ✅ Can switch between ChromaDB and EchoVault modes
- ✅ All cloud services connect properly 