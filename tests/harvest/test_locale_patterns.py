"""Tests for locale-based harvest pattern loading."""

import pytest
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mcp_memory_service.harvest.patterns import load_patterns


class TestLocalePatternLoader:
    """Test YAML pattern loading and merging."""

    def test_load_english_default(self):
        """English patterns load by default."""
        patterns = load_patterns(locales=["en"])
        assert "decision" in patterns
        assert "bug" in patterns
        assert len(patterns["decision"]) > 0

    def test_load_pt_br(self):
        """Portuguese patterns load and contain expected keywords."""
        patterns = load_patterns(locales=["pt_BR"])
        assert "decision" in patterns
        # Should have Portuguese patterns
        all_patterns = " ".join(str(p) for p in patterns["decision"])
        assert "decid" in all_patterns.lower() or "decisão" in all_patterns.lower()

    def test_load_de(self):
        """German patterns load."""
        patterns = load_patterns(locales=["de"])
        assert "decision" in patterns
        all_patterns = " ".join(str(p) for p in patterns["decision"])
        assert "entschied" in all_patterns.lower() or "beschloss" in all_patterns.lower()

    def test_merge_multiple_locales(self):
        """Multiple locales merge additively."""
        en_only = load_patterns(locales=["en"])
        en_pt = load_patterns(locales=["en", "pt_BR"])
        # Merged should have more patterns
        assert len(en_pt["decision"]) >= len(en_only["decision"])

    def test_unknown_locale_ignored(self):
        """Unknown locale doesn't crash, just logs warning."""
        patterns = load_patterns(locales=["en", "xx_FAKE"])
        # Should still have English patterns
        assert len(patterns["decision"]) > 0

    def test_empty_locales_returns_english(self):
        """Empty locale list falls back to English."""
        patterns = load_patterns(locales=[])
        assert len(patterns["decision"]) > 0
