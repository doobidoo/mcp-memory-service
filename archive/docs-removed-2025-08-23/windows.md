# Windows Setup Guide

This guide provides comprehensive instructions for setting up and running the MCP Memory Service on Windows systems, including handling common Windows-specific issues.

## Prerequisites

- **Python 3.10 or newer** (Python 3.11 recommended)
- **Git for Windows** ([download here](https://git-scm.com/download/win))
- **Visual Studio Build Tools** (for PyTorch compilation)
- **PowerShell 5.1+** or **Windows Terminal** (recommended)

## Quick Installation

### Automatic Installation (Recommended)

```powershell
# Clone repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Run Windows-specific installer
python install.py --windows
```

The installer automatically:
- Detects CUDA availability
- Installs the correct PyTorch version
- Configures Windows-specific settings
- Sets up optimal storage backend

## Manual Installation

### 1. Environment Setup

```powershell
# Clone repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install Dependencies

#### For CUDA-enabled Systems

```powershell
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -e .
pip install chromadb sentence-transformers
```

#### For CPU-only Systems

```powershell
# Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install with SQLite-vec backend (recommended for CPU)
pip install -e .
pip install sentence-transformers sqlite-vec
```

### 3. Windows-Specific Installation Script

If you encounter issues, use the Windows-specific installation script:

```powershell
python scripts/install_windows.py
```

This script handles:
1. CUDA detection and appropriate PyTorch installation
2. Resolving common Windows dependency conflicts
3. Setting up Windows-specific environment variables
4. Configuring optimal storage backend based on hardware

## Configuration

### Environment Variables

#### For CUDA Systems

```powershell
# Set environment variables (PowerShell)
$env:MCP_MEMORY_STORAGE_BACKEND = "chromadb"
$env:MCP_MEMORY_USE_CUDA = "true"
$env:MCP_MEMORY_CHROMA_PATH = "$env:USERPROFILE\.mcp_memory_chroma"

# Or set permanently
[Environment]::SetEnvironmentVariable("MCP_MEMORY_STORAGE_BACKEND", "chromadb", "User")
[Environment]::SetEnvironmentVariable("MCP_MEMORY_USE_CUDA", "true", "User")
```

#### For CPU-only Systems

```powershell
# Set environment variables (PowerShell)
$env:MCP_MEMORY_STORAGE_BACKEND = "sqlite_vec"
$env:MCP_MEMORY_SQLITE_VEC_PATH = "$env:USERPROFILE\.mcp_memory_sqlite"
$env:MCP_MEMORY_CPU_ONLY = "true"

# Or set permanently
[Environment]::SetEnvironmentVariable("MCP_MEMORY_STORAGE_BACKEND", "sqlite_vec", "User")
[Environment]::SetEnvironmentVariable("MCP_MEMORY_CPU_ONLY", "true", "User")
```

### Windows Batch Scripts

The repository includes Windows batch scripts for easy startup:

#### `scripts/run/run-with-uv.bat`

```batch
@echo off
cd /d "%~dp0..\.."
call venv\Scripts\activate.bat
python src\mcp_memory_service\server.py
```

#### Usage

```powershell
# Run the server
.\scripts\run\run-with-uv.bat

# Or run directly
python src\mcp_memory_service\server.py
```

## Claude Desktop Configuration

### Windows Configuration File Location

Claude Desktop configuration is typically located at:
```
%APPDATA%\Claude\claude_desktop_config.json
```

### Configuration Examples

#### For CUDA Systems

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["C:\\path\\to\\mcp-memory-service\\src\\mcp_memory_service\\server.py"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "chromadb",
        "MCP_MEMORY_USE_CUDA": "true",
        "PATH": "C:\\path\\to\\mcp-memory-service\\venv\\Scripts;%PATH%"
      }
    }
  }
}
```

#### For CPU-only Systems

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["C:\\path\\to\\mcp-memory-service\\src\\mcp_memory_service\\server.py"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_MEMORY_CPU_ONLY": "true",
        "PATH": "C:\\path\\to\\mcp-memory-service\\venv\\Scripts;%PATH%"
      }
    }
  }
}
```

#### Using Batch Script

```json
{
  "mcpServers": {
    "memory": {
      "command": "C:\\path\\to\\mcp-memory-service\\scripts\\run\\run-with-uv.bat"
    }
  }
}
```

## Hardware Detection and Optimization

### CUDA Detection

The installer automatically detects CUDA availability:

```python
def detect_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
```

### DirectML Support

For Windows systems without CUDA but with DirectX 12 compatible GPUs:

```powershell
# Install DirectML-enabled PyTorch
pip install torch-directml
```

Configure for DirectML:
```powershell
$env:MCP_MEMORY_USE_DIRECTML = "true"
$env:MCP_MEMORY_DEVICE = "dml"
```

## Windows-Specific Features

### Windows Service Installation

To run MCP Memory Service as a Windows service:

```powershell
# Install as Windows service (requires admin privileges)
python scripts/install_windows_service.py install

# Start service
net start MCPMemoryService

# Stop service
net stop MCPMemoryService

# Remove service
python scripts/install_windows_service.py remove
```

### Task Scheduler Integration

Create a scheduled task to start MCP Memory Service on boot:

```powershell
# Create scheduled task
schtasks /create /tn "MCP Memory Service" /tr "C:\path\to\mcp-memory-service\scripts\run\run-with-uv.bat" /sc onlogon /ru "$env:USERNAME"

# Delete scheduled task
schtasks /delete /tn "MCP Memory Service" /f
```

## Troubleshooting

### Common Windows Issues

#### 1. Path Length Limitations

**Symptom**: Installation fails with "path too long" errors

**Solution**: Enable long path support:
```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

#### 2. Visual Studio Build Tools Missing

**Symptom**: 
```
Microsoft Visual C++ 14.0 is required
```

**Solution**: Install Visual Studio Build Tools:
```powershell
# Download and install from:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or install via winget
winget install Microsoft.VisualStudio.2022.BuildTools
```

#### 3. CUDA Version Mismatch

**Symptom**: PyTorch CUDA installation issues

**Solution**: Match PyTorch CUDA version to your installed CUDA:
```powershell
# Check CUDA version
nvcc --version

# Install matching PyTorch version
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Permission Issues

**Symptom**: Access denied errors when installing or running

**Solution**: Run PowerShell as Administrator and check folder permissions:
```powershell
# Check current user permissions
whoami /groups

# Run installation as Administrator if needed
# Or adjust folder permissions
icacls "C:\path\to\mcp-memory-service" /grant "$env:USERNAME:(F)" /t
```

#### 5. Windows Defender Issues

**Symptom**: Installation files deleted or blocked

**Solution**: Add exclusions to Windows Defender:
```powershell
# Add folder exclusion (run as Administrator)
Add-MpPreference -ExclusionPath "C:\path\to\mcp-memory-service"

# Add process exclusion
Add-MpPreference -ExclusionProcess "python.exe"
```

### Diagnostic Commands

#### System Information

```powershell
# Check Python version and location
python --version
Get-Command python

# Check pip version
pip --version

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check DirectML (if installed)
python -c "import torch_directml; print('DirectML available')"

# Check Windows version
Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion
```

#### Environment Verification

```powershell
# Check environment variables
Get-ChildItem Env: | Where-Object {$_.Name -like "MCP_MEMORY_*"}

# Check virtual environment
echo $env:VIRTUAL_ENV

# Verify key packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sentence_transformers; print('SentenceTransformers: OK')"
python -c "import chromadb; print('ChromaDB: OK')" # or sqlite_vec
```

#### Network and Firewall

```powershell
# Check if Windows Firewall is blocking
Get-NetFirewallRule -DisplayName "*Python*" | Format-Table

# Test network connectivity (if using HTTP mode)
Test-NetConnection -ComputerName localhost -Port 8000
```

### Performance Optimization

#### Windows-Specific Settings

```powershell
# Optimize for machine learning workloads
$env:OMP_NUM_THREADS = [Environment]::ProcessorCount
$env:MKL_NUM_THREADS = [Environment]::ProcessorCount

# Set Windows-specific memory settings
$env:MCP_MEMORY_WINDOWS_OPTIMIZATION = "true"
$env:MCP_MEMORY_BATCH_SIZE = "32"
```

#### Resource Monitoring

```powershell
# Monitor memory usage
Get-Process python | Select-Object ProcessName, WorkingSet, CPU

# Monitor GPU usage (if CUDA)
nvidia-smi

# Monitor disk I/O
Get-Counter "\PhysicalDisk(_Total)\Disk Reads/sec"
```

## Development on Windows

### Setting up Development Environment

```powershell
# Clone for development
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Create development environment
python -m venv venv-dev
venv-dev\Scripts\activate

# Install in development mode
pip install -e .
pip install pytest black isort mypy

# Run tests
pytest tests/
```

### Windows-Specific Testing

```powershell
# Run Windows-specific tests
pytest tests/platform/test_windows.py -v

# Test CUDA functionality (if available)
pytest tests/cuda/ -v

# Test DirectML functionality (if available)
pytest tests/directml/ -v
```

## Alternative Installation Methods

### Using Chocolatey

```powershell
# Install Python via Chocolatey
choco install python

# Install Git
choco install git

# Then follow standard installation
```

### Using Conda

```powershell
# Create conda environment
conda create -n mcp-memory python=3.11
conda activate mcp-memory

# Install PyTorch via conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -e .
```

### Using Docker on Windows

```powershell
# Using Docker Desktop
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Build Windows container
docker build -f Dockerfile.windows -t mcp-memory-service-windows .

# Run container
docker run -p 8000:8000 mcp-memory-service-windows
```

## Related Documentation

- [Installation Guide](../installation/master-guide.md) - General installation instructions
- [Multi-Client Setup](../integration/multi-client.md) - Multi-client configuration
- [Troubleshooting](../troubleshooting/general.md) - Windows-specific troubleshooting
- [Docker Deployment](../deployment/docker.md) - Docker setup on Windows