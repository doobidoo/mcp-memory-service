"""
Shared test fixtures and helpers for unit tests.
"""

import tempfile
from pathlib import Path
from typing import List, Any, Optional

import pytest


def _refuse_real_kill(pid):
    raise AssertionError(
        f"mcp_memory_service.cli.lifecycle._kill_process was called for "
        f"real with pid={pid} -- this test needs to mock it explicitly "
        "(this default exists so a missing mock fails loudly here instead "
        "of sending a real signal to whatever process happens to own a "
        "port; two different tests once sent a real SIGTERM/SIGKILL to "
        "the memory service's own default port before this guard existed)."
    )


@pytest.fixture(autouse=True)
def _lifecycle_process_safety_net(monkeypatch, tmp_path):
    """Structural guard, not per-test discipline: any test anywhere under
    tests/unit/ that reaches lifecycle._kill_process without mocking it
    gets a loud AssertionError instead of a real signal -- this is the
    actual protection. To test _kill_process itself, re-patch it in that
    test; the raise-by-default here is just the fallback for everyone
    else. monkeypatch auto-restores it at teardown, so nothing leaks into
    later tests in the session. A no-op for tests that never touch the
    CLI lifecycle module.

    MCP_HTTP_PORT/HOST are also pinned away from this service's real
    default port (8000), covering the case where a test relies on
    launch's *default* port rather than passing --port explicitly (a
    test that passes --port 8000 directly bypasses this env entirely --
    _kill_process raising is what protects that case). This pin works
    despite mcp_memory_service.config.transport binding HTTP_PORT from
    MCP_HTTP_PORT at import time (before this fixture ever runs, and
    monkeypatch.setenv can't reach an already-bound module attribute) --
    lifecycle.py's own commands never read that cached value, they call
    os.environ.get("MCP_HTTP_PORT", "8000") directly at call time, which
    is exactly when this fixture's monkeypatch.setenv is in effect."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    from mcp_memory_service.cli import lifecycle
    monkeypatch.setattr(lifecycle, "_kill_process", _refuse_real_kill)
    monkeypatch.setenv("MCP_HTTP_HOST", "127.0.0.1")
    monkeypatch.setenv("MCP_HTTP_PORT", "59999")


async def extract_chunks_from_temp_file(
    loader: Any,
    filename: str,
    content: str,
    encoding: str = 'utf-8',
    **extract_kwargs
) -> List[Any]:
    """
    Helper to extract chunks from a temporary file.

    Args:
        loader: Loader instance (CSVLoader, JSONLoader, etc.)
        filename: Name of the temporary file to create
        content: Content to write to the file
        encoding: File encoding (default: utf-8)
        **extract_kwargs: Additional keyword arguments to pass to extract_chunks()

    Returns:
        List of extracted chunks

    Example:
        >>> loader = CSVLoader(chunk_size=1000, chunk_overlap=200)
        >>> chunks = await extract_chunks_from_temp_file(
        ...     loader,
        ...     "test.csv",
        ...     "name,age\\nJohn,25",
        ...     delimiter=','
        ... )
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / filename
        file_path.write_text(content, encoding=encoding)

        chunks = []
        async for chunk in loader.extract_chunks(file_path, **extract_kwargs):
            chunks.append(chunk)

        return chunks
