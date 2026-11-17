"""Tests for server-side learning — post-commit trigger + bootstrap consuming distill (TDD)."""

import pytest
import subprocess
from pathlib import Path


class TestBootstrapConsumesDistill:
    """Fase C: bootstrap inclui insights do memory-distill."""

    def test_bootstrap_searches_memory_distill_tag(self):
        """handle_get_bootstrap_profile deve buscar tag 'memory-distill' além de 'session-harvest'."""
        result = subprocess.run(
            ["grep", "-c", "memory-distill",
             "src/mcp_memory_service/server_impl.py"],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1])
        )
        # Deve aparecer no bootstrap handler (não só no distill handler)
        count = int(result.stdout.strip())
        # Precisa aparecer pelo menos 2x: 1 no distill handler (tag ao salvar) + 1 no bootstrap (busca)
        assert count >= 2, f"'memory-distill' aparece {count}x, precisa >=2 (distill + bootstrap)"


class TestPostCommitTrigger:
    """Fase A: memory_commit_session dispara learning em background."""

    def test_post_commit_learning_method_exists(self):
        """server_impl deve ter _post_commit_learning."""
        result = subprocess.run(
            ["grep", "-c", "_post_commit_learning",
             "src/mcp_memory_service/server_impl.py"],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1])
        )
        assert int(result.stdout.strip()) >= 1

    def test_commit_creates_background_task(self):
        """handle_commit_session_legacy deve chamar asyncio.create_task para learning."""
        result = subprocess.run(
            ["grep", "-c", "_post_commit_learning",
             "src/mcp_memory_service/server_impl.py"],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1])
        )
        assert int(result.stdout.strip()) >= 2, "commit handler deve ter _post_commit_learning (def + call)"

    def test_post_commit_checks_distill_threshold(self):
        """_post_commit_learning deve verificar threshold antes de rodar distill."""
        result = subprocess.run(
            ["grep", "-c", "undistilled\|threshold",
             "src/mcp_memory_service/server_impl.py"],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1])
        )
        assert int(result.stdout.strip()) >= 1

    def test_post_commit_handles_failure_gracefully(self):
        """_post_commit_learning deve ter try/except (não crashar o server)."""
        # Buscar try/except dentro do método
        result = subprocess.run(
            ["bash", "-c",
             "sed -n '/_post_commit_learning/,/^    async def /p' src/mcp_memory_service/server_impl.py | grep -c 'except'"],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1])
        )
        assert int(result.stdout.strip()) >= 1, "_post_commit_learning deve ter except"


class TestScheduledDistill:
    """Fase B+D: scheduler roda distill periodicamente com threshold."""

    def test_scheduled_distill_method_exists(self):
        """server_impl deve ter _scheduled_distill_check."""
        import subprocess  # inline import: used by this test only
        result = subprocess.run(
            ["grep", "-c", "_scheduled_distill_check",
             "src/mcp_memory_service/server_impl.py"],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1])
        )
        assert int(result.stdout.strip()) >= 1

    def test_distill_job_registered_in_scheduler(self):
        """Scheduler deve ter job 'distill_check' registrado."""
        import subprocess  # inline import: used by this test only
        result = subprocess.run(
            ["grep", "-c", "distill_check",
             "src/mcp_memory_service/server_impl.py"],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1])
        )
        assert int(result.stdout.strip()) >= 1

    def test_scheduled_distill_has_try_except(self):
        """_scheduled_distill_check deve ter try/except (não crashar scheduler)."""
        import subprocess  # inline import: used by this test only
        result = subprocess.run(
            ["bash", "-c",
             "sed -n '/_scheduled_distill_check/,/^    async def \\|^    def /p' src/mcp_memory_service/server_impl.py | grep -c 'except'"],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1])
        )
        assert int(result.stdout.strip()) >= 1
