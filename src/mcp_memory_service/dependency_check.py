"""
Dependency pre-check to ensure all required packages are installed.
This prevents runtime downloads during server initialization that cause timeouts.
"""

import sys
import subprocess
import platform
import logging
import os
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def detect_mcp_client_simple():
    """Simple MCP client detection for dependency checking."""
    try:
        # Check environment variables first
        if os.getenv('LM_STUDIO'):
            return 'lm_studio'
        if os.getenv('CLAUDE_DESKTOP'):
            return 'claude_desktop'
            
        import psutil
        current_process = psutil.Process()
        parent = current_process.parent()
        
        if parent:
            parent_name = parent.name().lower()
            if 'claude' in parent_name:
                return 'claude_desktop'
            if 'lmstudio' in parent_name or 'lm-studio' in parent_name:
                return 'lm_studio'
        
        # Default to Claude Desktop for strict mode
        return 'claude_desktop'
    except:
        return 'claude_desktop'

def check_torch_installed() -> Tuple[bool, Optional[str]]:
    """
    Check if PyTorch is properly installed.
    Returns (is_installed, version_string)
    """
    try:
        import torch
        # Check if torch has __version__ attribute (it should)
        version = getattr(torch, '__version__', 'unknown')
        # Also verify torch is functional
        try:
            _ = torch.tensor([1.0])
            return True, version
        except Exception:
            return False, None
    except ImportError:
        return False, None

def check_sentence_transformers_installed() -> Tuple[bool, Optional[str]]:
    """
    Check if sentence-transformers is properly installed.
    Returns (is_installed, version_string)
    """
    try:
        import sentence_transformers
        return True, sentence_transformers.__version__
    except ImportError:
        return False, None

def check_critical_dependencies() -> Tuple[bool, list]:
    """
    Check if all critical dependencies are installed.
    Returns (all_installed, missing_packages)
    """
    missing = []
    
    # Check PyTorch
    torch_installed, torch_version = check_torch_installed()
    if not torch_installed:
        missing.append("torch")
    else:
        logger.debug(f"PyTorch {torch_version} is installed")
    
    # Check sentence-transformers
    st_installed, st_version = check_sentence_transformers_installed()
    if not st_installed:
        missing.append("sentence-transformers")
    else:
        logger.debug(f"sentence-transformers {st_version} is installed")
    
    # Check other critical packages
    critical_packages = [
        "sqlite-vec",
        "mcp",
        "aiohttp",
        "fastapi",
        "uvicorn"
    ]
    
    for package in critical_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.debug(f"{package} is installed")
        except ImportError:
            missing.append(package)
    
    return len(missing) == 0, missing

def suggest_installation_command(missing_packages: list) -> str:
    """
    Generate the appropriate installation command for missing packages.
    """
    if not missing_packages:
        return ""
    
    # For Windows, suggest running install.py
    if platform.system() == "Windows":
        return "python install.py"
    else:
        return "python install.py"

def run_dependency_check() -> bool:
    """
    Run the dependency check and provide user feedback.
    Returns True if all dependencies are satisfied, False otherwise.
    """
    client_type = detect_mcp_client_simple()
    all_installed, missing = check_critical_dependencies()
    
    # Only show output for LM Studio to avoid JSON parsing errors in Claude Desktop
    if client_type == 'lm_studio':
        print("\n=== MCP Memory Service Dependency Check ===", file=sys.stdout, flush=True)
        
        if all_installed:
            print("[OK] All dependencies are installed", file=sys.stdout, flush=True)
        else:
            print(f"[MISSING] Missing dependencies detected: {', '.join(missing)}", file=sys.stdout, flush=True)
            print("\n[WARNING] IMPORTANT: Missing dependencies will cause timeouts!", file=sys.stdout, flush=True)
            print("[INSTALL] To install missing dependencies, run:", file=sys.stdout, flush=True)
            print(f"   {suggest_installation_command(missing)}", file=sys.stdout, flush=True)
            print("\nThe server will attempt to continue, but may timeout during initialization.", file=sys.stdout, flush=True)
            print("============================================\n", file=sys.stdout, flush=True)
    
    return all_installed

def is_first_run() -> bool:
    """
    Check if this appears to be the first run of the server.
    Enhanced for Windows and Claude Desktop environments.
    """
    # Enhanced cache detection for Windows and different environments
    cache_indicators = []
    
    # Standard HuggingFace cache locations
    cache_indicators.extend([
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/torch/sentence_transformers"),
    ])
    
    # Windows-specific locations
    if platform.system() == "Windows":
        username = os.environ.get('USERNAME', os.environ.get('USER', ''))
        cache_indicators.extend([
            f"C:\\Users\\{username}\\.cache\\huggingface\\hub",
            f"C:\\Users\\{username}\\.cache\\torch\\sentence_transformers",
            f"C:\\Users\\{username}\\AppData\\Local\\huggingface\\hub",
            f"C:\\Users\\{username}\\AppData\\Local\\torch\\sentence_transformers",
            os.path.expanduser("~/AppData/Local/sentence-transformers"),
        ])
    
    # Check environment variables for custom cache locations
    hf_home = os.environ.get('HF_HOME')
    if hf_home:
        cache_indicators.append(os.path.join(hf_home, 'hub'))
    
    transformers_cache = os.environ.get('TRANSFORMERS_CACHE')
    if transformers_cache:
        cache_indicators.append(transformers_cache)
    
    sentence_transformers_home = os.environ.get('SENTENCE_TRANSFORMERS_HOME')
    if sentence_transformers_home:
        cache_indicators.append(sentence_transformers_home)
    
    # Check each cache location
    for path in cache_indicators:
        if os.path.exists(path):
            try:
                contents = os.listdir(path)
                # Look for sentence-transformers models specifically
                for item in contents:
                    item_lower = item.lower()
                    # Check for common sentence-transformers model indicators
                    if any(indicator in item_lower for indicator in [
                        'sentence-transformers', 'miniml', 'all-miniml', 
                        'paraphrase', 'distilbert', 'mpnet', 'roberta'
                    ]):
                        logger.debug(f"Found cached model in {path}: {item}")
                        return False
                        
                # Also check for any model directories
                for item in contents:
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        try:
                            sub_contents = os.listdir(item_path)
                            # Look for model files
                            if any(f.endswith(('.bin', '.safetensors', '.json')) for f in sub_contents):
                                logger.debug(f"Found model files in {item_path}")
                                return False
                        except (OSError, PermissionError):
                            continue
                            
            except (OSError, PermissionError):
                logger.debug(f"Could not access cache directory: {path}")
                continue
    
    logger.debug("No cached sentence-transformers models found - this appears to be first run")
    return True

def get_recommended_timeout() -> float:
    """
    Get the recommended timeout based on system and dependencies.
    """
    # Check if dependencies are missing
    all_installed, missing = check_critical_dependencies()
    
    # Check if it's first run (models need downloading)
    first_run = is_first_run()
    
    # Base timeout
    timeout = 30.0 if platform.system() == "Windows" else 15.0
    
    # Extend timeout if dependencies are missing
    if not all_installed:
        timeout *= 2  # Double the timeout
        logger.warning(f"Dependencies missing, extending timeout to {timeout}s")
    
    # Extend timeout if it's first run
    if first_run:
        timeout *= 2  # Double the timeout
        logger.warning(f"First run detected, extending timeout to {timeout}s")
    
    return timeout