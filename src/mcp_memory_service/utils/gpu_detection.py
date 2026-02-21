#!/usr/bin/env python3
"""
Shared GPU detection utilities for MCP Memory Service.

This module provides unified GPU platform detection logic used across
installation and verification scripts. Supports CUDA, ROCm, MPS, and DirectML.
"""

import os
import subprocess
from typing import Dict, Any, Tuple, Optional, Callable, List, Union


# Single source of truth for GPU platform detection configuration
GPU_PLATFORM_CHECKS = {
    'cuda': {
        'windows': {
            'env_var': 'CUDA_PATH',
            'version_cmd': lambda path: [os.path.join(path, 'bin', 'nvcc'), '--version'],
            'version_pattern': 'release'
        },
        'linux': {
            'paths': ['/usr/local/cuda', lambda: os.environ.get('CUDA_HOME')],
            'version_cmd': lambda path: [os.path.join(path, 'bin', 'nvcc'), '--version'],
            'version_pattern': 'release'
        }
    },
    'rocm': {
        'linux': {
            'paths': ['/opt/rocm', lambda: os.environ.get('ROCM_HOME')],
            'version_file': lambda path: os.path.join(path, 'bin', '.rocmversion'),
            'version_cmd': ['rocminfo'],
            'version_pattern': 'Version'
        }
    },
    'mps': {
        'macos': {
            'check_cmd': ['system_profiler', 'SPDisplaysDataType'],
            'check_pattern': 'Metal',
            'requires_arm': True
        }
    },
    'directml': {
        'windows': {
            'import_name': 'torch-directml',
            'dll_name': 'DirectML.dll'
        }
    }
}


def parse_version(output: str, pattern: str = 'release') -> Optional[str]:
    """
    Parse version string from command output.

    Args:
        output: Command output to parse
        pattern: Pattern to search for ('release' or 'Version')

    Returns:
        Parsed version string or None if not found
    """
    for line in output.split('\n'):
        if pattern in line:
            if pattern == 'release':
                return line.split('release')[-1].strip().split(',')[0].strip()
            elif pattern == 'Version':
                return line.split(':')[-1].strip()
    return None


def test_gpu_platform(platform: str, system_info: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Test for a specific GPU platform and return detection status.

    Args:
        platform: Platform name ('cuda', 'rocm', 'mps', 'directml')
        system_info: System information dictionary with keys:
            - is_windows: bool
            - is_linux: bool
            - is_macos: bool
            - is_arm: bool (for ARM/Apple Silicon)

    Returns:
        Tuple of (detected: bool, version: Optional[str])
    """
    if platform not in GPU_PLATFORM_CHECKS:
        return False, None

    platform_config = GPU_PLATFORM_CHECKS[platform]

    # Determine OS-specific configuration
    if system_info.get('is_windows') and 'windows' in platform_config:
        os_config = platform_config['windows']
    elif system_info.get('is_linux') and 'linux' in platform_config:
        os_config = platform_config['linux']
    elif system_info.get('is_macos') and 'macos' in platform_config:
        os_config = platform_config['macos']
    else:
        return False, None

    # Platform-specific detection logic
    if platform == 'cuda':
        return _detect_cuda(os_config, system_info)
    elif platform == 'rocm':
        return _detect_rocm(os_config)
    elif platform == 'mps':
        return _detect_mps(os_config, system_info)
    elif platform == 'directml':
        return _detect_directml(os_config)

    return False, None


def _detect_cuda(config: Dict[str, Any], system_info: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Detect CUDA installation."""
    # Check environment variable (Windows) or paths (Linux)
    if 'env_var' in config:
        cuda_path = os.environ.get(config['env_var'])
        if not cuda_path or not os.path.exists(cuda_path):
            return False, None
        paths_to_check = [cuda_path]
    elif 'paths' in config:
        paths_to_check = []
        for path in config['paths']:
            if callable(path):
                path = path()
            if path and os.path.exists(path):
                paths_to_check.append(path)
        if not paths_to_check:
            return False, None
    else:
        return False, None

    # Try to get version
    for path in paths_to_check:
        try:
            version_cmd = config['version_cmd'](path)
            output = subprocess.check_output(
                version_cmd,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            version = parse_version(output, config.get('version_pattern', 'release'))
            return True, version
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            continue

    # Found path but couldn't get version
    return True, None


def _detect_rocm(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Detect ROCm installation."""
    paths_to_check = []
    for path in config.get('paths', []):
        if callable(path):
            path = path()
        if path and os.path.exists(path):
            paths_to_check.append(path)

    if not paths_to_check:
        return False, None

    # Try version file first
    for path in paths_to_check:
        if 'version_file' in config:
            version_file = config['version_file'](path)
            try:
                with open(version_file, 'r') as f:
                    version = f.read().strip()
                    return True, version
            except (FileNotFoundError, IOError):
                pass

    # Try version command
    if 'version_cmd' in config:
        try:
            output = subprocess.check_output(
                config['version_cmd'],
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            version = parse_version(output, config.get('version_pattern', 'Version'))
            return True, version
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass

    # Found path but couldn't get version
    return True, None


def _detect_mps(config: Dict[str, Any], system_info: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Detect Apple Metal Performance Shaders (MPS)."""
    # MPS requires ARM architecture
    if config.get('requires_arm') and not system_info.get('is_arm'):
        return False, None

    try:
        result = subprocess.run(
            config['check_cmd'],
            capture_output=True,
            text=True
        )
        if config['check_pattern'] in result.stdout:
            return True, None  # MPS doesn't have a version string
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass  # MPS check command not available; assume MPS not present

    return False, None


def _detect_directml(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Detect DirectML installation."""
    # Try importing the package
    try:
        import pkg_resources
        version = pkg_resources.get_distribution(config['import_name']).version
        return True, version
    except (ImportError, Exception):
        pass

    # Try loading the DLL
    try:
        import ctypes
        ctypes.WinDLL(config['dll_name'])
        return True, None  # Found DLL but no version
    except (ImportError, OSError, Exception):
        pass

    return False, None


def detect_gpu(system_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect all available GPU platforms and return comprehensive GPU info.

    Args:
        system_info: System information dictionary with keys:
            - is_windows: bool
            - is_linux: bool
            - is_macos: bool
            - is_arm: bool (for ARM/Apple Silicon)

    Returns:
        Dictionary containing:
            - has_cuda: bool
            - cuda_version: Optional[str]
            - has_rocm: bool
            - rocm_version: Optional[str]
            - has_mps: bool
            - has_directml: bool
            - directml_version: Optional[str]
            - accelerator: str ('cuda', 'rocm', 'mps', 'directml', or 'cpu')
    """
    gpu_info = {
        "has_cuda": False,
        "cuda_version": None,
        "has_rocm": False,
        "rocm_version": None,
        "has_mps": False,
        "has_directml": False,
        "directml_version": None,
        "accelerator": "cpu"
    }

    # Test each platform
    gpu_info["has_cuda"], gpu_info["cuda_version"] = test_gpu_platform('cuda', system_info)
    gpu_info["has_rocm"], gpu_info["rocm_version"] = test_gpu_platform('rocm', system_info)
    gpu_info["has_mps"], _ = test_gpu_platform('mps', system_info)
    gpu_info["has_directml"], gpu_info["directml_version"] = test_gpu_platform('directml', system_info)

    # Determine primary accelerator (priority order: CUDA > ROCm > MPS > DirectML > CPU)
    if gpu_info["has_cuda"]:
        gpu_info["accelerator"] = "cuda"
    elif gpu_info["has_rocm"]:
        gpu_info["accelerator"] = "rocm"
    elif gpu_info["has_mps"]:
        gpu_info["accelerator"] = "mps"
    elif gpu_info["has_directml"]:
        gpu_info["accelerator"] = "directml"

    return gpu_info
