#!/usr/bin/env python3
"""
Unit tests for ingestion loader registry.

Covers dispatch behavior (which loader wins for a given file), the
INFO-level usage log emitted on every dispatch, and the priority
mechanism that lets markitdown override pdf_loader for .pdf when both
are registered.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcp_memory_service.ingestion import registry
from mcp_memory_service.ingestion.base import DocumentLoader


def _stub_loader_class(extensions=None, available: bool = True, name: str = "StubLoader"):
    """Build a minimal DocumentLoader subclass so the registry can
    instantiate it and call can_handle().

    `extensions` is a set of suffixes (without dots) the stub will accept;
    `available` simulates an optional dep being importable (False mimics
    e.g. markitdown not installed)."""
    accepted = {e.lstrip('.').lower() for e in (extensions or [])}

    class _StubLoader(DocumentLoader):
        def __init__(self, *_args, **_kwargs):
            pass

        def can_handle(self, file_path: Path) -> bool:  # type: ignore[override]
            if not available:
                return False
            if not accepted:
                return True
            return file_path.suffix.lower().lstrip('.') in accepted

        async def extract_chunks(self, file_path: Path, **_):  # pragma: no cover
            if False:
                yield None

    _StubLoader.__name__ = name
    return _StubLoader


@pytest.fixture
def isolated_registry():
    """Snapshot + clear the global registry around a test, then restore."""
    original = dict(registry._LOADER_REGISTRY)
    registry._LOADER_REGISTRY.clear()
    try:
        yield registry
    finally:
        registry._LOADER_REGISTRY.clear()
        registry._LOADER_REGISTRY.update(original)


class TestRegisterLoader:
    """register_loader populates the extension → loader mapping."""

    def test_registers_single_extension(self, isolated_registry):
        cls = _stub_loader_class(extensions=['pdf'], name="PdfStub")
        registry.register_loader(cls, ['pdf'])
        assert isolated_registry._LOADER_REGISTRY['pdf'] is cls

    def test_registers_multiple_extensions(self, isolated_registry):
        cls = _stub_loader_class(extensions=['docx', 'doc', 'pptx'], name="OfficeStub")
        registry.register_loader(cls, ['docx', 'doc', 'pptx'])
        for ext in ('docx', 'doc', 'pptx'):
            assert isolated_registry._LOADER_REGISTRY[ext] is cls

    def test_normalizes_extensions(self, isolated_registry):
        """Extensions are stored lowercase without leading dots."""
        cls = _stub_loader_class(extensions=['pdf', 'docx'], name="MixedStub")
        registry.register_loader(cls, ['.PDF', 'DocX'])
        assert 'pdf' in isolated_registry._LOADER_REGISTRY
        assert 'docx' in isolated_registry._LOADER_REGISTRY
        assert '.PDF' not in isolated_registry._LOADER_REGISTRY

    def test_later_registration_overrides_earlier(self, isolated_registry):
        """When two loaders register for the same extension, the one
        registered last wins. This is the mechanism markitdown relies on
        to override PDFLoader for .pdf when available."""
        first = _stub_loader_class(extensions=['pdf'], name="FirstLoader")
        second = _stub_loader_class(extensions=['pdf'], name="SecondLoader")

        registry.register_loader(first, ['pdf'])
        registry.register_loader(second, ['pdf'])

        assert isolated_registry._LOADER_REGISTRY['pdf'] is second


class TestGetLoaderForFile:
    """get_loader_for_file picks the right loader and logs the dispatch."""

    def test_returns_none_when_file_missing(self, isolated_registry, tmp_path):
        cls = _stub_loader_class(extensions=['pdf'])
        registry.register_loader(cls, ['pdf'])

        ghost = tmp_path / "nope.pdf"
        assert registry.get_loader_for_file(ghost) is None

    def test_returns_loader_when_extension_matches(self, isolated_registry, tmp_path):
        cls = _stub_loader_class(extensions=['pdf'], name="PdfStub")
        registry.register_loader(cls, ['pdf'])

        f = tmp_path / "report.pdf"
        f.touch()
        loader = registry.get_loader_for_file(f)
        assert loader is not None
        assert isinstance(loader, cls)

    def test_returns_none_when_extension_unregistered(self, isolated_registry, tmp_path):
        cls = _stub_loader_class(extensions=['pdf'])
        registry.register_loader(cls, ['pdf'])

        f = tmp_path / "report.xyz"
        f.touch()
        assert registry.get_loader_for_file(f) is None

    def test_falls_through_when_can_handle_returns_false(self, isolated_registry, tmp_path):
        """If the extension-mapped loader rejects the file (e.g. markitdown
        unavailable), the registry should keep looking. With only that
        single loader registered (and unavailable), the result is None."""
        cls = _stub_loader_class(extensions=['pdf'], available=False, name="UnavailableStub")
        registry.register_loader(cls, ['pdf'])

        f = tmp_path / "report.pdf"
        f.touch()
        assert registry.get_loader_for_file(f) is None

    def test_logs_dispatch_at_info_level(self, isolated_registry, tmp_path, caplog):
        """Each successful dispatch emits an [ingestion] INFO line so
        deployments can audit which loader handled which file."""
        cls = _stub_loader_class(extensions=['pdf'], name="LoggingStub")
        registry.register_loader(cls, ['pdf'])

        f = tmp_path / "report.pdf"
        f.touch()

        with caplog.at_level(logging.INFO, logger="mcp_memory_service.ingestion.registry"):
            registry.get_loader_for_file(f)

        dispatch_lines = [r.getMessage() for r in caplog.records
                          if "[ingestion] dispatch" in r.getMessage()]
        assert len(dispatch_lines) == 1
        line = dispatch_lines[0]
        assert "file=report.pdf" in line
        assert "ext=pdf" in line
        assert "loader=LoggingStub" in line
        assert "match=extension" in line


class TestMarkitdownOverridesPdfLoader:
    """End-to-end check of the registration order: when both PDFLoader
    and (an importable) markitdown register, markitdown wins for .pdf
    while PDFLoader stays the handler when markitdown is gated off."""

    def _reregister_pdf_then_markitdown(self, markitdown_available: bool):
        from mcp_memory_service.ingestion import pdf_loader
        from mcp_memory_service.ingestion import markitdown_loader as md_module

        # Mirror the production import order: pdf_loader first.
        registry.register_loader(pdf_loader.PDFLoader, ['pdf'])

        saved_flag = md_module._MARKITDOWN_AVAILABLE
        md_module._MARKITDOWN_AVAILABLE = markitdown_available
        try:
            md_module._register_markitdown_loader()
        finally:
            md_module._MARKITDOWN_AVAILABLE = saved_flag

    def test_markitdown_wins_pdf_when_available(self, isolated_registry):
        from mcp_memory_service.ingestion.markitdown_loader import MarkItDownLoader

        self._reregister_pdf_then_markitdown(markitdown_available=True)
        assert isolated_registry._LOADER_REGISTRY['pdf'] is MarkItDownLoader

    def test_pdf_loader_stays_when_markitdown_unavailable(self, isolated_registry):
        from mcp_memory_service.ingestion.pdf_loader import PDFLoader

        self._reregister_pdf_then_markitdown(markitdown_available=False)
        assert isolated_registry._LOADER_REGISTRY['pdf'] is PDFLoader


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
