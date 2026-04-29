#!/usr/bin/env python3
"""
Unit tests for the sys:-namespaced upload tags built by the documents API.

The team taxonomy reserves the `sys:` namespace for service-managed
metadata. Tags use '=' as the inner separator (not ':') because
parse_tag() splits on the first ':' to extract the namespace, so any
value containing ':' would corrupt the parse.
"""

from pathlib import Path

import pytest

from mcp_memory_service.web.api.documents import (
    SYS_UPLOAD_ID_TAG_PREFIX,
    _build_system_tags,
)


class TestBuildSystemTags:
    """Tests for _build_system_tags()."""

    def test_returns_three_tags_in_sys_namespace(self):
        tags = _build_system_tags("notes.pdf", Path("/tmp/notes.pdf"), "u-1")
        assert len(tags) == 3
        for tag in tags:
            assert tag.startswith("sys:"), f"non-sys tag: {tag}"

    def test_uses_equals_separator_not_colon(self):
        """parse_tag() splits on the first ':' for the namespace; the
        value side must use '='. Guard against accidental reverts."""
        tags = _build_system_tags("notes.pdf", Path("/tmp/notes.pdf"), "abc-123")
        for tag in tags:
            namespace, _, value = tag.partition(":")
            assert namespace == "sys"
            assert ":" not in value, f"value side contains ':': {tag}"
            assert "=" in value, f"value side missing '=': {tag}"

    def test_includes_source_file_tag_with_filename(self):
        tags = _build_system_tags("Report 2026.docx", Path("/tmp/x.docx"), "u-1")
        assert "sys:source_file=Report 2026.docx" in tags

    def test_includes_file_type_tag_from_suffix(self):
        tags = _build_system_tags("notes.PDF", Path("/tmp/notes.PDF"), "u-1")
        # Suffix preserves case in the value but always includes a leading dot
        # which the helper strips.
        assert any(t.startswith("sys:file_type=") for t in tags)
        file_type_tag = next(t for t in tags if t.startswith("sys:file_type="))
        assert "." not in file_type_tag.split("=", 1)[1]

    def test_includes_upload_id_tag_with_shared_prefix(self):
        upload_id = "ab425714-6f21-4ea5-bcae-d525d8857c4e"
        tags = _build_system_tags("notes.pdf", Path("/tmp/notes.pdf"), upload_id)
        expected = f"{SYS_UPLOAD_ID_TAG_PREFIX}{upload_id}"
        assert expected in tags

    def test_upload_id_prefix_constant_matches_tag_format(self):
        """The constant used by remove/search must reproduce the exact
        prefix the upload path emits — protects against drift between
        the writer and the lookup."""
        tags = _build_system_tags("notes.pdf", Path("/tmp/notes.pdf"), "u-42")
        upload_tag = next(t for t in tags if t.startswith("sys:upload_id="))
        assert upload_tag.startswith(SYS_UPLOAD_ID_TAG_PREFIX)

    def test_handles_extension_less_filename(self):
        """Files without an extension get an empty file_type value rather
        than crashing — uploads of plain text without a suffix still work."""
        tags = _build_system_tags("README", Path("/tmp/README"), "u-1")
        assert "sys:file_type=" in tags  # empty value, but tag exists


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
