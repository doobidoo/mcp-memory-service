#!/usr/bin/env python3
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
Enhanced environment verification script for MCP Memory Service.
This script checks the system environment, hardware capabilities,
and installed dependencies to ensure compatibility.
"""

import ctypes
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pkg_resources


class EnvironmentVerifier:
    def __init__(self):
        self.verification_results = []
        self.critical_failures = []
        self.warnings = []
        self.system_info = self.detect_system()
        self.gpu_info = self.detect_gpu()
        self.claude_config = self.load_claude_config()

    def detect_system(self):
        """Detect system architecture and platform."""
        system_info = {
            "os_name": platform.system().lower(),
            "os_version": platform.version(),
            "architecture": platform.machine().lower(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count() or 1,
            "memory_gb": self.get_system_memory(),
            "in_virtual_env": sys.prefix != sys.base_prefix,
        }

        self.verification_results.append(f"[OK] System: {platform.system()} {platform.version()}")
        self.verification_results.append(f"[OK] Architecture: {system_info['architecture']}")
        self.verification_results.append(f"[OK] Python: {system_info['python_version']}")

        if system_info["in_virtual_env"]:
            self.verification_results.append(f"[OK] Virtual environment: {sys.prefix}")
        else:
            self.warnings.append("Not running in a virtual environment")

        return system_info

    def get_system_memory(self):
        """Get the total system memory in GB."""
        try:
            if self.system_info["os_name"] == "linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            memory_kb = int(line.split()[1])
                            return round(memory_kb / (1024 * 1024), 2)

            elif self.system_info["os_name"] == "darwin":
                output = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode("utf-8").strip()
                memory_bytes = int(output)
                return round(memory_bytes / (1024**3), 2)

            elif self.system_info["os_name"] == "windows":

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                memoryStatus = MEMORYSTATUSEX()
                memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus))
                return round(memoryStatus.ullTotalPhys / (1024**3), 2)

        except Exception as e:
            self.warnings.append(f"Failed to get system memory: {e}")

        return 4.0  # Conservative default

    def detect_gpu(self):
        """Detect GPU and acceleration capabilities."""
        gpu_info = {
            "has_cuda": False,
            "cuda_version": None,
            "has_rocm": False,
            "rocm_version": None,
            "has_mps": False,
            "has_directml": False,
            "accelerator": "cpu",
        }

        # Check for CUDA
        if self.system_info["os_name"] == "windows":
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path and os.path.exists(cuda_path):
                gpu_info["has_cuda"] = True
                try:
                    nvcc_output = subprocess.check_output(
                        [os.path.join(cuda_path, "bin", "nvcc"), "--version"], stderr=subprocess.STDOUT, universal_newlines=True
                    )
                    for line in nvcc_output.split("\n"):
                        if "release" in line:
                            gpu_info["cuda_version"] = line.split("release")[-1].strip().split(",")[0].strip()
                            break
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass
        elif self.system_info["os_name"] == "linux":
            cuda_paths = ["/usr/local/cuda", os.environ.get("CUDA_HOME")]
            for path in cuda_paths:
                if path and os.path.exists(path):
                    gpu_info["has_cuda"] = True
                    try:
                        nvcc_output = subprocess.check_output(
                            [os.path.join(path, "bin", "nvcc"), "--version"], stderr=subprocess.STDOUT, universal_newlines=True
                        )
                        for line in nvcc_output.split("\n"):
                            if "release" in line:
                                gpu_info["cuda_version"] = line.split("release")[-1].strip().split(",")[0].strip()
                                break
                    except (subprocess.SubprocessError, FileNotFoundError):
                        pass
                    break

        # Check for ROCm
        if self.system_info["os_name"] == "linux":
            rocm_paths = ["/opt/rocm", os.environ.get("ROCM_HOME")]
            for path in rocm_paths:
                if path and os.path.exists(path):
                    gpu_info["has_rocm"] = True
                    try:
                        with open(os.path.join(path, "bin", ".rocmversion")) as f:
                            gpu_info["rocm_version"] = f.read().strip()
                    except (OSError, FileNotFoundError):
                        try:
                            rocm_output = subprocess.check_output(
                                ["rocminfo"], stderr=subprocess.STDOUT, universal_newlines=True
                            )
                            for line in rocm_output.split("\n"):
                                if "Version" in line:
                                    gpu_info["rocm_version"] = line.split(":")[-1].strip()
                                    break
                        except (subprocess.SubprocessError, FileNotFoundError):
                            pass
                    break

        # Check for MPS
        if self.system_info["os_name"] == "darwin" and self.system_info["architecture"] in ("arm64", "aarch64"):
            try:
                result = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True)
                gpu_info["has_mps"] = "Metal" in result.stdout
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

        # Check for DirectML
        if self.system_info["os_name"] == "windows":
            try:
                pkg_resources.get_distribution("torch-directml")
                gpu_info["has_directml"] = True
            except (ImportError, pkg_resources.DistributionNotFound):
                try:
                    ctypes.WinDLL("DirectML.dll")
                    gpu_info["has_directml"] = True
                except (ImportError, OSError):
                    pass

        # Set accelerator type
        if gpu_info["has_cuda"]:
            gpu_info["accelerator"] = "cuda"
            self.verification_results.append(f"[OK] CUDA detected: {gpu_info['cuda_version'] or 'Unknown version'}")
        elif gpu_info["has_rocm"]:
            gpu_info["accelerator"] = "rocm"
            self.verification_results.append(f"[OK] ROCm detected: {gpu_info['rocm_version'] or 'Unknown version'}")
        elif gpu_info["has_mps"]:
            gpu_info["accelerator"] = "mps"
            self.verification_results.append("[OK] Apple Metal Performance Shaders (MPS) detected")
        elif gpu_info["has_directml"]:
            gpu_info["accelerator"] = "directml"
            self.verification_results.append("[OK] DirectML detected")
        else:
            self.verification_results.append("[OK] Using CPU-only mode (no GPU acceleration detected)")

        return gpu_info

    def load_claude_config(self):
        """Load configuration from Claude Desktop config."""
        try:
            home_dir = Path.home()
            possible_paths = [
                home_dir / "Library/Application Support/Claude/claude_desktop_config.json",
                home_dir / ".config/Claude/claude_desktop_config.json",
                Path(__file__).parent.parent / "claude_config/claude_desktop_config.json",
            ]

            for config_path in possible_paths:
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                        self.verification_results.append(f"[OK] Found Claude Desktop config at {config_path}")
                        return config

            self.warnings.append("Could not find Claude Desktop config file in any standard location")
            return None

        except Exception as e:
            self.critical_failures.append(f"Error loading Claude Desktop config: {str(e)}")
            return None

    def verify_python_version(self):
        """Verify Python interpreter version matches production requirements."""
        try:
            python_version = sys.version.split()[0]
            required_version = "3.10"  # Updated to match current requirements

            if not python_version.startswith(required_version):
                self.critical_failures.append(f"Python version mismatch: Found {python_version}, required {required_version}")
            else:
                self.verification_results.append(f"[OK] Python version verified: {python_version}")
        except Exception as e:
            self.critical_failures.append(f"Failed to verify Python version: {str(e)}")

    def verify_virtual_environment(self):
        """Verify we're running in a virtual environment."""
        try:
            if sys.prefix == sys.base_prefix:
                self.critical_failures.append("Not running in a virtual environment!")
            else:
                self.verification_results.append(f"[OK] Virtual environment verified: {sys.prefix}")
        except Exception as e:
            self.critical_failures.append(f"Failed to verify virtual environment: {str(e)}")

    def verify_critical_packages(self):
        """Verify critical packages are installed with correct versions."""
        required_packages = {
            "chromadb": "0.5.23",
            "sentence-transformers": "2.2.2",
            "urllib3": "1.26.6",
            "python-dotenv": "1.0.0",
        }

        for package, required_version in required_packages.items():
            try:
                installed_version = pkg_resources.get_distribution(package).version
                if required_version and installed_version != required_version:
                    self.critical_failures.append(
                        f"Package version mismatch: {package} (found {installed_version}, required {required_version})"
                    )
                else:
                    self.verification_results.append(f"[OK] Package verified: {package} {installed_version}")
            except pkg_resources.DistributionNotFound:
                self.critical_failures.append(f"Required package not found: {package}")
            except Exception as e:
                self.critical_failures.append(f"Failed to verify package {package}: {str(e)}")

    def verify_claude_paths(self):
        """Verify paths from Claude Desktop config."""
        if not self.claude_config:
            return

        try:
            chroma_path = self.claude_config.get("mcp-memory", {}).get("chroma_db")
            backup_path = self.claude_config.get("mcp-memory", {}).get("backup_path")

            if chroma_path:
                os.environ["CHROMA_DB_PATH"] = str(chroma_path)
                self.verification_results.append(f"[OK] Set CHROMA_DB_PATH from config: {chroma_path}")
            else:
                self.critical_failures.append("CHROMA_DB_PATH not found in Claude config")

            if backup_path:
                os.environ["MCP_MEMORY_BACKUP_PATH"] = str(backup_path)
                self.verification_results.append(f"[OK] Set MCP_MEMORY_BACKUP_PATH from config: {backup_path}")
            else:
                self.critical_failures.append("MCP_MEMORY_BACKUP_PATH not found in Claude config")

        except Exception as e:
            self.critical_failures.append(f"Failed to verify Claude paths: {str(e)}")

    def verify_import_functionality(self):
        """Verify critical imports work correctly."""
        critical_imports = [
            "chromadb",
            "sentence_transformers",
        ]

        for module_name in critical_imports:
            try:
                importlib.import_module(module_name)
                self.verification_results.append(f"[OK] Successfully imported {module_name}")
            except ImportError as e:
                self.critical_failures.append(f"Failed to import {module_name}: {str(e)}")

    def verify_paths(self):
        """Verify critical paths exist and are accessible."""
        critical_paths = [os.environ.get("CHROMA_DB_PATH", ""), os.environ.get("MCP_MEMORY_BACKUP_PATH", "")]

        for path in critical_paths:
            if not path:
                continue
            try:
                path_obj = Path(path)
                if not path_obj.exists():
                    self.critical_failures.append(f"Critical path does not exist: {path}")
                elif not os.access(path, os.R_OK | os.W_OK):
                    self.critical_failures.append(f"Insufficient permissions for path: {path}")
                else:
                    self.verification_results.append(f"[OK] Path verified: {path}")
            except Exception as e:
                self.critical_failures.append(f"Failed to verify path {path}: {str(e)}")

    def run_verifications(self):
        """Run all verifications."""
        self.verify_python_version()
        self.verify_virtual_environment()
        self.verify_critical_packages()
        self.verify_claude_paths()
        self.verify_import_functionality()
        self.verify_paths()

    def print_results(self):
        """Print verification results."""
        print("\n=== Environment Verification Results ===\n")

        if self.verification_results:
            print("Successful Verifications:")
            for result in self.verification_results:
                print(f"  {result}")

        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  [!] {warning}")

        if self.critical_failures:
            print("\nCritical Failures:")
            for failure in self.critical_failures:
                print(f"  [X] {failure}")

        print("\nSummary:")
        print(f"  Passed: {len(self.verification_results)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Failed: {len(self.critical_failures)}")

        if self.critical_failures:
            print("\nTo fix these issues:")
            print("1. Create a new virtual environment:")
            print("   conda create -n mcp-env python=3.10")
            print("   conda activate mcp-env")
            print("\n2. Install requirements:")
            print("   pip install -r requirements.txt")
            print("\n3. Ensure Claude Desktop config is properly set up with required paths")

        return len(self.critical_failures) == 0


def main():
    verifier = EnvironmentVerifier()
    verifier.run_verifications()
    environment_ok = verifier.print_results()

    if not environment_ok:
        print("\n[WARNING]  Environment verification failed! Please fix the issues above.")
        sys.exit(1)
    else:
        print("\n[OK] Environment verification passed! Safe to proceed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
