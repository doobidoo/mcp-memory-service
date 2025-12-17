# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
System detection utilities for hardware compatibility.
Provides functions to detect hardware architecture, available accelerators,
and determine optimal configurations for different environments.
"""
import os
import sys
import platform
import logging
import subprocess
from typing import Dict, Any, Tuple, Optional, List
import json

logger = logging.getLogger(__name__)

# Hardware acceleration types
class AcceleratorType:
    NONE = "none"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders
    CPU = "cpu"
    DIRECTML = "directml"  # DirectML for Windows
    ROCm = "rocm"  # AMD ROCm

class Architecture:
    X86_64 = "x86_64"
    ARM64 = "arm64"
    UNKNOWN = "unknown"

class SystemInfo:
    """Class to store and provide system information."""
    
    def __init__(self):
        self.os_name = platform.system().lower()
        self.os_version = platform.version()
        self.architecture = self._detect_architecture()
        self.python_version = platform.python_version()
        self.cpu_count = os.cpu_count() or 1
        self.memory_gb = self._get_system_memory()
        self.accelerator = self._detect_accelerator()
        self.is_rosetta = self._detect_rosetta()
        self.is_virtual_env = sys.prefix != sys.base_prefix
        
    def _detect_architecture(self) -> str:
        """Detect the system architecture."""
        arch = platform.machine().lower()
        
        if arch in ("x86_64", "amd64", "x64"):
            return Architecture.X86_64
        elif arch in ("arm64", "aarch64"):
            return Architecture.ARM64
        else:
            return Architecture.UNKNOWN
            
    def _get_system_memory(self) -> float:
        """Get the total system memory in GB."""
        try:
            if self.os_name == "linux":
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            # Extract the memory value (in kB)
                            memory_kb = int(line.split()[1])
                            return round(memory_kb / (1024 * 1024), 2)  # Convert to GB
                            
            elif self.os_name == "darwin":  # macOS
                output = subprocess.check_output(['sysctl', '-n', 'hw.memsize']).decode('utf-8').strip()
                memory_bytes = int(output)
                return round(memory_bytes / (1024**3), 2)  # Convert to GB
                
            elif self.os_name == "windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulonglong = ctypes.c_ulonglong
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', c_ulonglong),
                        ('ullAvailPhys', c_ulonglong),
                        ('ullTotalPageFile', c_ulonglong),
                        ('ullAvailPageFile', c_ulonglong),
                        ('ullTotalVirtual', c_ulonglong),
                        ('ullAvailVirtual', c_ulonglong),
                        ('ullAvailExtendedVirtual', c_ulonglong),
                    ]
                    
                memoryStatus = MEMORYSTATUSEX()
                memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus))
                return round(memoryStatus.ullTotalPhys / (1024**3), 2)  # Convert to GB
                
        except Exception as e:
            logger.warning(f"Failed to get system memory: {e}")
            
        # Default fallback
        return 4.0  # Assume 4GB as a conservative default
        
    def _detect_accelerator(self) -> str:
        """Detect available hardware acceleration."""
        # Try to detect CUDA
        if self._check_cuda_available():
            return AcceleratorType.CUDA
            
        # Check for Apple MPS (Metal Performance Shaders)
        if self.os_name == "darwin" and self.architecture == Architecture.ARM64:
            if self._check_mps_available():
                return AcceleratorType.MPS
                
        # Check for ROCm on Linux
        if self.os_name == "linux" and self._check_rocm_available():
            return AcceleratorType.ROCm
            
        # Check for DirectML on Windows
        if self.os_name == "windows" and self._check_directml_available():
            return AcceleratorType.DIRECTML
            
        # Default to CPU
        return AcceleratorType.CPU
        
    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            # Try to import torch and check for CUDA
            import torch
            # Check if torch is properly installed with CUDA support
            if hasattr(torch, 'cuda'):
                return torch.cuda.is_available()
            else:
                logger.warning("PyTorch installed but appears broken (no cuda attribute)")
                return False
        except (ImportError, AttributeError, OSError) as e:
            logger.debug(f"CUDA check failed: {e}")
            # If torch is not installed or broken, try to check for CUDA using environment
            return 'CUDA_HOME' in os.environ or 'CUDA_PATH' in os.environ
            
    def _check_mps_available(self) -> bool:
        """Check if Apple MPS is available."""
        try:
            import torch
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
                return torch.backends.mps.is_available()
            else:
                logger.warning("PyTorch installed but appears broken (no backends attribute)")
                return False
        except (ImportError, AttributeError, OSError) as e:
            logger.debug(f"MPS check failed: {e}")
            # Check for Metal support using system profiler
            try:
                output = subprocess.check_output(
                    ['system_profiler', 'SPDisplaysDataType'], 
                    stderr=subprocess.DEVNULL
                ).decode('utf-8')
                return 'Metal' in output
            except (subprocess.SubprocessError, FileNotFoundError):
                return False
                
    def _check_rocm_available(self) -> bool:
        """Check if AMD ROCm is available."""
        try:
            # Check for ROCm environment
            if 'ROCM_HOME' in os.environ or 'ROCM_PATH' in os.environ:
                return True
                
            # Check if ROCm libraries are installed
            try:
                output = subprocess.check_output(
                    ['rocminfo'], 
                    stderr=subprocess.DEVNULL
                ).decode('utf-8')
                return 'GPU Agent' in output
            except (subprocess.SubprocessError, FileNotFoundError):
                return False
                
        except Exception:
            return False
            
    def _check_directml_available(self) -> bool:
        """Check if DirectML is available on Windows."""
        try:
            # Check if DirectML package is installed
            import pkg_resources
            pkg_resources.get_distribution('torch-directml')
            return True
        except ImportError:
            # pkg_resources not available
            return False
        except Exception:
            # Any other error (including DistributionNotFound)
            return False
            
    def _detect_rosetta(self) -> bool:
        """Detect if running under Rosetta 2 on Apple Silicon."""
        if self.os_name != "darwin" or self.architecture != Architecture.ARM64:
            return False
            
        try:
            # Check for Rosetta by examining the process
            output = subprocess.check_output(
                ['sysctl', '-n', 'sysctl.proc_translated'], 
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            return output == '1'
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
            
    def get_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on hardware."""
        # Start with a base batch size
        if self.accelerator == AcceleratorType.CUDA:
            # Scale based on available GPU memory (rough estimate)
            try:
                import torch
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                if gpu_memory > 10:
                    return 32
                elif gpu_memory > 6:
                    return 16
                else:
                    return 8
            except:
                return 8  # Default for CUDA
        elif self.accelerator == AcceleratorType.MPS:
            return 8  # Conservative for Apple Silicon
        elif self.memory_gb > 16:
            return 8  # Larger batch for systems with more RAM
        elif self.memory_gb > 8:
            return 4
        else:
            return 2  # Conservative for low-memory systems
            
    def get_optimal_model(self) -> str:
        """Determine the optimal embedding model based on hardware capabilities."""
        # Default model
        default_model = 'all-MiniLM-L6-v2'
        
        # For very constrained environments, use an even smaller model
        if self.memory_gb < 4:
            return 'paraphrase-MiniLM-L3-v2'
            
        # For high-performance environments, consider a larger model
        if (self.accelerator in [AcceleratorType.CUDA, AcceleratorType.MPS] and 
                self.memory_gb > 8):
            return 'all-mpnet-base-v2'  # Better quality but more resource intensive
            
        return default_model
        
    def get_optimal_thread_count(self) -> int:
        """Determine optimal thread count for parallel operations."""
        # Use 75% of available cores, but at least 1
        return max(1, int(self.cpu_count * 0.75))
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert system info to dictionary."""
        return {
            "os": self.os_name,
            "os_version": self.os_version,
            "architecture": self.architecture,
            "python_version": self.python_version,
            "cpu_count": self.cpu_count,
            "memory_gb": self.memory_gb,
            "accelerator": self.accelerator,
            "is_rosetta": self.is_rosetta,
            "is_virtual_env": self.is_virtual_env,
            "optimal_batch_size": self.get_optimal_batch_size(),
            "optimal_model": self.get_optimal_model(),
            "optimal_thread_count": self.get_optimal_thread_count()
        }
        
    def __str__(self) -> str:
        """String representation of system info."""
        return json.dumps(self.to_dict(), indent=2)


def get_system_info() -> SystemInfo:
    """Get system information singleton."""
    if not hasattr(get_system_info, 'instance'):
        get_system_info.instance = SystemInfo()
    return get_system_info.instance


def get_torch_device() -> str:
    """Get the optimal PyTorch device based on system capabilities."""
    system_info = get_system_info()
    
    try:
        import torch
        
        if system_info.accelerator == AcceleratorType.CUDA and torch.cuda.is_available():
            return "cuda"
        elif (system_info.accelerator == AcceleratorType.MPS and 
              hasattr(torch.backends, 'mps') and 
              torch.backends.mps.is_available()):
            return "mps"
        else:
            return "cpu"
    except (ImportError, OSError):
        return "cpu"


def get_optimal_embedding_settings() -> Dict[str, Any]:
    """Get optimal settings for embedding operations."""
    system_info = get_system_info()
    
    return {
        "model_name": system_info.get_optimal_model(),
        "batch_size": system_info.get_optimal_batch_size(),
        "device": get_torch_device(),
        "threads": system_info.get_optimal_thread_count()
    }


def print_system_diagnostics(client_type: str = 'lm_studio'):
    """Print detailed system diagnostics for troubleshooting, conditionally based on client."""
    # Only print for LM Studio to avoid JSON parsing errors in Claude Desktop
    if client_type != 'lm_studio':
        return
        
    system_info = get_system_info()
    
    print("\n=== System Diagnostics ===")
    print(f"OS: {system_info.os_name} {system_info.os_version}")
    print(f"Architecture: {system_info.architecture}")
    print(f"Python: {system_info.python_version}")
    print(f"CPU Cores: {system_info.cpu_count}")
    print(f"Memory: {system_info.memory_gb:.2f} GB")
    print(f"Accelerator: {system_info.accelerator}")
    
    if system_info.is_rosetta:
        print("⚠️ Running under Rosetta 2 translation")
        
    print("\n=== Optimal Settings ===")
    print(f"Embedding Model: {system_info.get_optimal_model()}")
    print(f"Batch Size: {system_info.get_optimal_batch_size()}")
    print(f"Thread Count: {system_info.get_optimal_thread_count()}")
    print(f"PyTorch Device: {get_torch_device()}")
    
    # Additional PyTorch diagnostics if available
    try:
        import torch
        print("\n=== PyTorch Diagnostics ===")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            
        if hasattr(torch.backends, 'mps'):
            print(f"MPS Available: {torch.backends.mps.is_available()}")
            
    except (ImportError, OSError):
        print("\nPyTorch not installed, skipping PyTorch diagnostics")
        
    print("\n=== Environment Variables ===")
    for var in ['CUDA_HOME', 'CUDA_PATH', 'ROCM_HOME', 'PYTORCH_ENABLE_MPS_FALLBACK']:
        if var in os.environ:
            print(f"{var}: {os.environ[var]}")