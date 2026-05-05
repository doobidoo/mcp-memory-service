# Copyright 2026 Claudio Ferreira Filho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for mistake notes lifecycle integration in consolidation."""

import pytest
from unittest.mock import MagicMock

from mcp_memory_service.consolidation.decay import ExponentialDecayCalculator
from mcp_memory_service.consolidation.base import ConsolidationConfig
from mcp_memory_service.models.memory import Memory


@pytest.fixture
def consolidation_base():
    """Create a concrete ConsolidationBase subclass for testing."""
    config = ConsolidationConfig()
    return ExponentialDecayCalculator(config)


def _make_memory(memory_type="note", tags=None, metadata=None):
    """Helper to create a Memory object for testing."""
    mem = MagicMock(spec=Memory)
    mem.memory_type = memory_type
    mem.tags = tags or []
    mem.metadata = metadata
    return mem


class TestMistakeNoteProtection:
    """Test that high-value mistake notes are protected from decay/forgetting."""

    def test_regular_memory_not_protected(self, consolidation_base):
        mem = _make_memory(memory_type="note", tags=["general"])
        assert consolidation_base._is_protected_memory(mem) is False

    def test_critical_tag_protected(self, consolidation_base):
        mem = _make_memory(tags=["critical"])
        assert consolidation_base._is_protected_memory(mem) is True

    def test_mistake_note_low_count_not_protected(self, consolidation_base):
        """Mistake notes with failure_count < 3 can still be decayed."""
        mem = _make_memory(memory_type="mistake", metadata={"failure_count": 1})
        assert consolidation_base._is_protected_memory(mem) is False

    def test_mistake_note_high_count_protected(self, consolidation_base):
        """Mistake notes with failure_count >= 3 are protected."""
        mem = _make_memory(memory_type="mistake", metadata={"failure_count": 3})
        assert consolidation_base._is_protected_memory(mem) is True

    def test_mistake_note_very_high_count_protected(self, consolidation_base):
        mem = _make_memory(memory_type="mistake", metadata={"failure_count": 10})
        assert consolidation_base._is_protected_memory(mem) is True

    def test_mistake_note_no_metadata_not_protected(self, consolidation_base):
        """Mistake notes without metadata are not protected."""
        mem = _make_memory(memory_type="mistake", metadata=None)
        assert consolidation_base._is_protected_memory(mem) is False

    def test_mistake_note_json_string_metadata(self, consolidation_base):
        """Metadata stored as JSON string should be parsed."""
        mem = _make_memory(memory_type="mistake", metadata='{"failure_count": 5}')
        assert consolidation_base._is_protected_memory(mem) is True

    def test_mistake_note_json_string_low_count(self, consolidation_base):
        mem = _make_memory(memory_type="mistake", metadata='{"failure_count": 2}')
        assert consolidation_base._is_protected_memory(mem) is False

    def test_mistake_note_invalid_json_metadata(self, consolidation_base):
        """Invalid JSON metadata should not crash — just not protect."""
        mem = _make_memory(memory_type="mistake", metadata="{invalid json")
        assert consolidation_base._is_protected_memory(mem) is False

    def test_mistake_note_list_metadata(self, consolidation_base):
        """Non-dict JSON (e.g. list) should not crash."""
        mem = _make_memory(memory_type="mistake", metadata='[1, 2, 3]')
        assert consolidation_base._is_protected_memory(mem) is False
