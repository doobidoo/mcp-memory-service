"""Tests for dependency_check.py - especially get_recommended_timeout."""
import os
import platform
import sys
import types

import pytest


class TestGetRecommendedTimeout:
    """Tests for the get_recommended_timeout function."""

    def test_env_override_takes_precedence(self, monkeypatch):
        """MCP_INIT_TIMEOUT env var should override automatic detection."""
        monkeypatch.setenv('MCP_INIT_TIMEOUT', '120')
        from mcp_memory_service.dependency_check import get_recommended_timeout
        result = get_recommended_timeout()
        assert result == 120.0

    def test_env_override_float_value(self, monkeypatch):
        """MCP_INIT_TIMEOUT should accept float values."""
        monkeypatch.setenv('MCP_INIT_TIMEOUT', '45.5')
        from mcp_memory_service.dependency_check import get_recommended_timeout
        result = get_recommended_timeout()
        assert result == 45.5

    def test_env_override_invalid_value_falls_back(self, monkeypatch):
        """Invalid MCP_INIT_TIMEOUT should fall back to automatic detection."""
        monkeypatch.setenv('MCP_INIT_TIMEOUT', 'not-a-number')
        from mcp_memory_service.dependency_check import get_recommended_timeout
        # Should not raise, should return a positive value from automatic detection
        result = get_recommended_timeout()
        assert isinstance(result, float)
        assert result > 0

    def test_env_override_zero_falls_back(self, monkeypatch):
        """Zero MCP_INIT_TIMEOUT should fall back to automatic detection."""
        monkeypatch.setenv('MCP_INIT_TIMEOUT', '0')
        from mcp_memory_service.dependency_check import get_recommended_timeout
        result = get_recommended_timeout()
        assert result > 0

    def test_env_override_negative_falls_back(self, monkeypatch):
        """Negative MCP_INIT_TIMEOUT should fall back to automatic detection."""
        monkeypatch.setenv('MCP_INIT_TIMEOUT', '-10')
        from mcp_memory_service.dependency_check import get_recommended_timeout
        result = get_recommended_timeout()
        assert result > 0

    def test_no_env_override_returns_positive(self, monkeypatch):
        """Without env override, should return a positive timeout."""
        monkeypatch.delenv('MCP_INIT_TIMEOUT', raising=False)
        from mcp_memory_service.dependency_check import get_recommended_timeout
        result = get_recommended_timeout()
        assert isinstance(result, float)
        assert result > 0

    def test_env_override_empty_string_falls_back(self, monkeypatch):
        """Empty MCP_INIT_TIMEOUT should fall back to automatic detection."""
        monkeypatch.setenv('MCP_INIT_TIMEOUT', '')
        from mcp_memory_service.dependency_check import get_recommended_timeout
        result = get_recommended_timeout()
        assert result > 0

    def test_strict_clients_are_capped_to_5s(self, monkeypatch):
        """Strict stdio clients should not block handshake with long eager init."""
        monkeypatch.delenv('MCP_INIT_TIMEOUT', raising=False)
        from mcp_memory_service import dependency_check
        monkeypatch.setattr(dependency_check, 'detect_mcp_client_simple', lambda: 'claude_desktop')
        monkeypatch.setattr(dependency_check, 'check_critical_dependencies', lambda: (False, ['torch']))
        monkeypatch.setattr(dependency_check, 'is_first_run', lambda: True)
        result = dependency_check.get_recommended_timeout()
        assert result == 5.0

    def test_lm_studio_keeps_adaptive_timeout(self, monkeypatch):
        """LM Studio keeps longer adaptive timeout for heavyweight local startup."""
        import platform
        monkeypatch.delenv('MCP_INIT_TIMEOUT', raising=False)
        from mcp_memory_service import dependency_check
        monkeypatch.setattr(dependency_check, 'detect_mcp_client_simple', lambda: 'lm_studio')
        monkeypatch.setattr(dependency_check, 'check_critical_dependencies', lambda: (False, ['torch']))
        monkeypatch.setattr(dependency_check, 'is_first_run', lambda: True)

        base_timeout = 30.0 if platform.system() == "Windows" else 15.0
        expected_timeout = base_timeout * 2 * 2  # For missing deps and first run
        result = dependency_check.get_recommended_timeout()
        assert result == expected_timeout


def _fake_module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _install_backends(monkeypatch, onnx, torch):
    """Make the embedding backends importable or not, whatever the venv holds.

    A None entry in sys.modules makes the import raise ImportError.
    """
    if onnx:
        monkeypatch.setitem(sys.modules, 'onnxruntime', _fake_module('onnxruntime', __version__='1.20.0'))
        monkeypatch.setitem(sys.modules, 'tokenizers', _fake_module('tokenizers', __version__='0.22.2'))
    else:
        monkeypatch.setitem(sys.modules, 'onnxruntime', None)
        monkeypatch.setitem(sys.modules, 'tokenizers', None)
    if torch:
        monkeypatch.setitem(sys.modules, 'torch', _fake_module('torch', __version__='2.5.0', tensor=lambda data: data))
        monkeypatch.setitem(sys.modules, 'sentence_transformers', _fake_module('sentence_transformers', __version__='3.3.0'))
    else:
        monkeypatch.setitem(sys.modules, 'torch', None)
        monkeypatch.setitem(sys.modules, 'sentence_transformers', None)


class TestEmbeddingBackendCheck:
    """Either embedding backend satisfies the check (issue #1930)."""

    def test_onnx_only_install_passes(self, monkeypatch):
        """The [sqlite] extra (onnxruntime + tokenizers, no torch) is complete."""
        _install_backends(monkeypatch, onnx=True, torch=False)
        from mcp_memory_service.dependency_check import check_critical_dependencies
        assert check_critical_dependencies() == (True, [])

    def test_torch_only_install_passes(self, monkeypatch):
        """torch + sentence-transformers without ONNX is complete too."""
        _install_backends(monkeypatch, onnx=False, torch=True)
        from mcp_memory_service.dependency_check import check_critical_dependencies
        assert check_critical_dependencies() == (True, [])

    def test_neither_backend_is_reported(self, monkeypatch):
        """With no embedding backend at all the check fails and names both options."""
        _install_backends(monkeypatch, onnx=False, torch=False)
        from mcp_memory_service.dependency_check import check_critical_dependencies
        all_installed, missing = check_critical_dependencies()
        assert all_installed is False
        assert missing == ["onnxruntime + tokenizers (or torch + sentence-transformers)"]

    def test_torch_without_sentence_transformers_is_reported(self, monkeypatch):
        """torch alone is not a usable backend."""
        _install_backends(monkeypatch, onnx=False, torch=True)
        monkeypatch.setitem(sys.modules, 'sentence_transformers', None)
        from mcp_memory_service.dependency_check import check_critical_dependencies
        all_installed, _ = check_critical_dependencies()
        assert all_installed is False

    def test_onnx_only_startup_check_prints_no_warning(self, monkeypatch, capsys):
        """run_dependency_check() reports OK on the [sqlite] install."""
        _install_backends(monkeypatch, onnx=True, torch=False)
        from mcp_memory_service.dependency_check import run_dependency_check
        assert run_dependency_check() is True
        err = capsys.readouterr().err
        assert "[OK] All dependencies are installed" in err
        assert "[MISSING]" not in err

    def test_install_hint_names_the_sqlite_extra(self):
        """install.py is not shipped to pip, uv tool or uvx installs."""
        from mcp_memory_service.dependency_check import suggest_installation_command
        assert suggest_installation_command(["onnxruntime"]) == 'pip install "mcp-memory-service[sqlite]"'


class TestTimeoutWithEmbeddingBackends:
    """get_recommended_timeout() only doubles when no backend is usable."""

    def _timeout(self, monkeypatch, onnx, torch):
        monkeypatch.delenv('MCP_INIT_TIMEOUT', raising=False)
        _install_backends(monkeypatch, onnx=onnx, torch=torch)
        from mcp_memory_service import dependency_check
        monkeypatch.setattr(dependency_check, 'detect_mcp_client_simple', lambda: 'lm_studio')
        monkeypatch.setattr(dependency_check, 'is_first_run', lambda: False)
        return dependency_check.get_recommended_timeout()

    def test_onnx_only_timeout_is_not_doubled(self, monkeypatch):
        base_timeout = 30.0 if platform.system() == "Windows" else 15.0
        assert self._timeout(monkeypatch, onnx=True, torch=False) == base_timeout

    def test_torch_only_timeout_is_not_doubled(self, monkeypatch):
        base_timeout = 30.0 if platform.system() == "Windows" else 15.0
        assert self._timeout(monkeypatch, onnx=False, torch=True) == base_timeout

    def test_no_backend_timeout_is_doubled(self, monkeypatch):
        base_timeout = 30.0 if platform.system() == "Windows" else 15.0
        assert self._timeout(monkeypatch, onnx=False, torch=False) == base_timeout * 2


class TestIsFirstRunOnnxCache:
    """is_first_run() sees the ONNX model cache, not only sentence-transformers."""

    @pytest.fixture
    def fake_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv('HOME', str(tmp_path))
        monkeypatch.setenv('USERPROFILE', str(tmp_path))
        monkeypatch.setenv('USERNAME', 'mcp-test-no-such-user')
        for var in ('HF_HOME', 'TRANSFORMERS_CACHE', 'SENTENCE_TRANSFORMERS_HOME'):
            monkeypatch.delenv(var, raising=False)
        return tmp_path

    def test_cached_default_onnx_model_is_not_first_run(self, fake_home):
        """The bundled default model, unpacked by ONNXEmbeddingModel."""
        model_dir = fake_home / ".cache" / "mcp_memory" / "onnx_models" / "all-MiniLM-L6-v2" / "onnx"
        model_dir.mkdir(parents=True)
        (model_dir / "model.onnx").write_bytes(b"")
        (model_dir / "tokenizer.json").write_text("{}")
        from mcp_memory_service.dependency_check import is_first_run
        assert is_first_run() is False

    def test_cached_hub_onnx_model_is_not_first_run(self, fake_home):
        """A non-default model fetched from the Hugging Face Hub."""
        model_dir = fake_home / ".cache" / "mcp_memory" / "onnx_models" / "multilingual-e5-small"
        (model_dir / "onnx").mkdir(parents=True)
        (model_dir / "onnx" / "model.onnx").write_bytes(b"")
        (model_dir / "tokenizer.json").write_text("{}")
        from mcp_memory_service.dependency_check import is_first_run
        assert is_first_run() is False

    def test_model_dir_without_files_is_first_run(self, fake_home):
        """The model directory is created before the download, so it alone proves nothing."""
        (fake_home / ".cache" / "mcp_memory" / "onnx_models" / "all-MiniLM-L6-v2").mkdir(parents=True)
        from mcp_memory_service.dependency_check import is_first_run
        assert is_first_run() is True
