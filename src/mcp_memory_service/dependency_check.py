"""
Dependency pre-check to ensure all required packages are installed.
This prevents runtime downloads during server initialization that cause timeouts.
"""

import sys
import platform
import logging
import os
from typing import Tuple, Optional

from .compat import _sanitize_log_value

logger = logging.getLogger(__name__)

def detect_mcp_client_simple():
    """Simple MCP client detection for dependency checking."""
    try:
        # Check environment variables first
        if os.getenv('LM_STUDIO'):
            return 'lm_studio'
        if os.getenv('CLAUDE_DESKTOP'):
            return 'claude_desktop'
            
        import psutil  # inline import: optional dependency, the caller degrades when it is absent
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
    except Exception:
        return 'claude_desktop'

def check_torch_installed() -> Tuple[bool, Optional[str]]:
    """
    Check if PyTorch is properly installed.
    Returns (is_installed, version_string)
    """
    try:
        import torch  # inline import: presence of this package is exactly what is being probed
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
        import sentence_transformers  # inline import: presence of this package is exactly what is being probed
        return True, sentence_transformers.__version__
    except ImportError:
        return False, None

def check_onnx_installed() -> Tuple[bool, Optional[str]]:
    """
    Check if the ONNX embedding backend (onnxruntime + tokenizers) is installed.
    Returns (is_installed, onnxruntime_version)
    """
    try:
        import onnxruntime  # inline import: presence of this package is exactly what is being probed
        import tokenizers  # noqa: F401  # inline import: the ONNX backend cannot tokenize without it
        return True, getattr(onnxruntime, '__version__', 'unknown')
    except ImportError:
        return False, None

def check_critical_dependencies() -> Tuple[bool, list]:
    """
    Check if all critical dependencies are installed.
    Returns (all_installed, missing_packages)
    """
    missing = []

    # Embeddings need one backend: ONNX (the [sqlite] extra) or PyTorch.
    # Only with neither does the service fall back to hash embeddings.
    onnx_installed, onnx_version = check_onnx_installed()
    if onnx_installed:
        logger.debug("onnxruntime %s and tokenizers are installed", _sanitize_log_value(onnx_version))
    else:
        torch_installed, torch_version = check_torch_installed()
        st_installed, st_version = check_sentence_transformers_installed()
        if torch_installed and st_installed:
            logger.debug(
                "PyTorch %s and sentence-transformers %s are installed",
                _sanitize_log_value(torch_version),
                _sanitize_log_value(st_version),
            )
        else:
            missing.append("onnxruntime + tokenizers (or torch + sentence-transformers)")

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
            logger.debug(f"{_sanitize_log_value(package)} is installed")
        except ImportError:
            missing.append(package)
    
    return len(missing) == 0, missing

def suggest_installation_command(missing_packages: list) -> str:
    """
    Generate the appropriate installation command for missing packages.
    """
    if not missing_packages:
        return ""

    return 'pip install "mcp-memory-service[sqlite]"'

def run_dependency_check() -> bool:
    """
    Run the dependency check and provide user feedback.
    Returns True if all dependencies are satisfied, False otherwise.
    """
    all_installed, missing = check_critical_dependencies()

    # stderr, not stdout. On a stdio MCP server stdout IS the JSON-RPC channel,
    # so anything human-readable written there corrupts the protocol stream.
    #
    # This used to be `if client_type == 'lm_studio'`, with a comment about
    # avoiding JSON parsing errors in Claude Desktop. The diagnosis was right
    # and the remedy was not: LM Studio speaks stdio too, so restricting the
    # output to that client moved the corruption rather than removing it.
    # Driving the server over stdio with a single initialize request put seven
    # non-JSON lines ahead of the response (issue #275). On stderr the output
    # is safe for every client, which is why there is no longer a guard --
    # every client now gets the warning instead of only the broken one.
    print("\n=== MCP Memory Service Dependency Check ===", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel

    if all_installed:
        print("[OK] All dependencies are installed", file=sys.stderr, flush=True)  # debug: startup diagnostic on stderr
    else:
        print(f"[MISSING] Missing dependencies detected: {', '.join(missing)}", file=sys.stderr, flush=True)  # debug: startup diagnostic on stderr
        print("\n[WARNING] IMPORTANT: Missing dependencies will cause timeouts!", file=sys.stderr, flush=True)  # debug: startup diagnostic on stderr
        print("[INSTALL] To install missing dependencies, run:", file=sys.stderr, flush=True)  # debug: startup diagnostic on stderr
        print(f"   {suggest_installation_command(missing)}", file=sys.stderr, flush=True)  # debug: startup diagnostic on stderr
        print("\n[WARNING] The server will attempt to continue, but may timeout during initialization.", file=sys.stderr, flush=True)  # debug: startup diagnostic on stderr
        print("============================================\n", file=sys.stderr, flush=True)  # debug: startup diagnostic on stderr

    return all_installed

def _onnx_model_cached() -> bool:
    """
    Check the ONNX model cache for a model with both its weights and tokenizer.
    ONNXEmbeddingModel creates the model directory before downloading, so an
    interrupted download can leave it empty.
    """
    onnx_root = os.path.expanduser("~/.cache/mcp_memory/onnx_models")
    try:
        models = os.listdir(onnx_root)
    except OSError:
        return False

    for model in models:
        model_dir = os.path.join(onnx_root, model)
        # The bundled default model unpacks into <model>/onnx/, Hub exports keep
        # model.onnx in <model>/ or <model>/onnx/ and tokenizer.json in <model>/
        for weights_dir in (model_dir, os.path.join(model_dir, "onnx")):
            if os.path.isfile(os.path.join(weights_dir, "model.onnx")) and any(
                os.path.isfile(os.path.join(d, "tokenizer.json"))
                for d in (weights_dir, model_dir)
            ):
                logger.debug("Found cached ONNX model in %s", _sanitize_log_value(weights_dir))
                return True
    return False

def is_first_run() -> bool:
    """
    Check if this appears to be the first run of the server.
    Enhanced for Windows and Claude Desktop environments.
    """
    if _onnx_model_cached():
        return False

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
                        logger.debug(f"Found cached model in {_sanitize_log_value(path)}: {_sanitize_log_value(item)}")
                        return False
                        
                # Also check for any model directories
                for item in contents:
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        try:
                            sub_contents = os.listdir(item_path)
                            # Look for model files
                            if any(f.endswith(('.bin', '.safetensors', '.json')) for f in sub_contents):
                                logger.debug(f"Found model files in {_sanitize_log_value(item_path)}")
                                return False
                        except (OSError, PermissionError):
                            continue
                            
            except (OSError, PermissionError):
                logger.debug(f"Could not access cache directory: {_sanitize_log_value(path)}")
                continue
    
    logger.debug("No cached sentence-transformers models found - this appears to be first run")
    return True

def get_recommended_timeout() -> float:
    """
    Get the recommended timeout based on system and dependencies.

    Can be overridden via MCP_INIT_TIMEOUT environment variable.
    Useful for Windows systems where ONNX model initialization takes longer.

    For strict stdio clients (default: Claude Desktop/Codex-like), keep eager
    initialization timeout short so MCP handshake is not blocked by heavy startup.
    """
    # Allow explicit override via environment variable
    env_timeout = os.getenv('MCP_INIT_TIMEOUT')
    if env_timeout:
        try:
            override = float(env_timeout)
            if override > 0:
                logger.info(f"Using MCP_INIT_TIMEOUT override: {_sanitize_log_value(override)}s")
                return override
            logger.warning(f"MCP_INIT_TIMEOUT must be a positive number, got '{_sanitize_log_value(env_timeout)}'. Using automatic detection.")
        except ValueError:
            logger.warning(f"Invalid MCP_INIT_TIMEOUT value '{_sanitize_log_value(env_timeout)}', using automatic detection")

    # Check if dependencies are missing
    all_installed, missing = check_critical_dependencies()

    # Check if it's first run (models need downloading)
    first_run = is_first_run()

    # Base timeout
    timeout = 30.0 if platform.system() == "Windows" else 15.0

    # Extend timeout if dependencies are missing
    if not all_installed:
        timeout *= 2  # Double the timeout
        logger.warning(f"Dependencies missing, extending base timeout to {_sanitize_log_value(timeout)}s (before client cap)")

    # Extend timeout if it's first run
    if first_run:
        timeout *= 2  # Double the timeout
        logger.warning(f"First run detected, extending base timeout to {_sanitize_log_value(timeout)}s (before client cap)")

    # Strict stdio clients often have small handshake budgets.
    # Keep eager init conservative and rely on lazy-load fallback.
    ADAPTIVE_TIMEOUT_CLIENT = 'lm_studio'
    STRICT_CLIENT_TIMEOUT_CAP_S = 5.0

    client = detect_mcp_client_simple()
    if client != ADAPTIVE_TIMEOUT_CLIENT:
        strict_timeout = min(timeout, STRICT_CLIENT_TIMEOUT_CAP_S)
        if strict_timeout < timeout:
            logger.info(
                "Strict MCP client detected (%s); capping eager init timeout from %.1fs to %.1fs",
                client,
                timeout,
                strict_timeout,
            )
        timeout = strict_timeout

    return timeout
