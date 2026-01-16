"""
Integration tests for Memory dataclass with ontology validation

Tests the complete ontology validation workflow in Memory creation.
"""

import pytest
import hashlib
from pathlib import Path
import importlib.util
import sys

# Load ontology module first
ontology_path = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "models" / "ontology.py"
spec_ont = importlib.util.spec_from_file_location("ontology_test", ontology_path)
ontology_module = importlib.util.module_from_spec(spec_ont)
spec_ont.loader.exec_module(ontology_module)

# Load tag_taxonomy module
tag_tax_path = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "models" / "tag_taxonomy.py"
spec_tag = importlib.util.spec_from_file_location("tag_taxonomy_test", tag_tax_path)
tag_taxonomy_module = importlib.util.module_from_spec(spec_tag)
spec_tag.loader.exec_module(tag_taxonomy_module)

# Create a fake package for the absolute imports
fake_package = type(sys)('mcp_memory_service')
fake_models = type(sys)('models')
fake_package.models = fake_models
fake_models.ontology = ontology_module
fake_models.tag_taxonomy = tag_taxonomy_module
sys.modules['mcp_memory_service'] = fake_package
sys.modules['mcp_memory_service.models'] = fake_models
sys.modules['mcp_memory_service.models.ontology'] = ontology_module
sys.modules['mcp_memory_service.models.tag_taxonomy'] = tag_taxonomy_module

# Now load memory module
memory_path = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "models" / "memory.py"
spec = importlib.util.spec_from_file_location("memory_test", memory_path)
memory_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_module)

Memory = memory_module.Memory


class TestBurstI1MemoryOntologyValidation:
    """Tests for Burst I.1: Memory Dataclass Ontology Validation"""

    def test_valid_base_type_passes(self):
        """Valid base types should pass validation"""
        content = "Test observation"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        memory = Memory(
            content=content,
            content_hash=content_hash,
            memory_type="observation"
        )

        assert memory.memory_type == "observation"

    def test_valid_subtype_passes(self):
        """Valid subtypes should pass validation"""
        content = "Edited config.py"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        memory = Memory(
            content=content,
            content_hash=content_hash,
            memory_type="code_edit"
        )

        assert memory.memory_type == "code_edit"

    def test_invalid_type_defaults_to_observation(self, caplog):
        """Invalid types should default to 'observation' with warning"""
        content = "Test content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        memory = Memory(
            content=content,
            content_hash=content_hash,
            memory_type="invalid_type"
        )

        # Should default to observation
        assert memory.memory_type == "observation"

        # Should log warning
        assert "Invalid memory_type" in caplog.text
        assert "invalid_type" in caplog.text

    def test_none_memory_type_allowed(self):
        """None memory_type should be preserved (backward compatibility)"""
        content = "Test content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        memory = Memory(
            content=content,
            content_hash=content_hash,
            memory_type=None
        )

        assert memory.memory_type is None


class TestBurstI2TagValidation:
    """Tests for Burst I.2: Tag Validation in Memory"""

    def test_valid_namespaced_tags_pass(self):
        """Valid namespaced tags should pass without warnings"""
        content = "Test content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        memory = Memory(
            content=content,
            content_hash=content_hash,
            tags=["sys:core", "q:high", "proj:auth", "topic:security"]
        )

        assert memory.tags == ["sys:core", "q:high", "proj:auth", "topic:security"]

    def test_legacy_tags_pass(self, caplog):
        """Legacy tags (no namespace) should pass without warnings"""
        content = "Test content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        memory = Memory(
            content=content,
            content_hash=content_hash,
            tags=["python", "refactoring", "bug-fix"]
        )

        assert memory.tags == ["python", "refactoring", "bug-fix"]
        # Should not log anything for legacy tags
        assert "invalid namespaces" not in caplog.text.lower()

    def test_invalid_namespaced_tags_logged(self, caplog):
        """Invalid namespaced tags should be logged but preserved"""
        import logging
        caplog.set_level(logging.INFO)

        content = "Test content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        memory = Memory(
            content=content,
            content_hash=content_hash,
            tags=["invalid:tag", "bad:namespace"]
        )

        # Tags should still be preserved
        assert memory.tags == ["invalid:tag", "bad:namespace"]

        # Should log info message
        assert "invalid namespaces" in caplog.text.lower()
        assert "invalid:tag" in caplog.text
        assert "bad:namespace" in caplog.text

    def test_mixed_tags_only_invalid_logged(self, caplog):
        """Mix of valid, invalid, and legacy tags"""
        import logging
        caplog.set_level(logging.INFO)

        content = "Test content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        memory = Memory(
            content=content,
            content_hash=content_hash,
            tags=["sys:core", "python", "bad:namespace", "q:high", "legacy-tag"]
        )

        # All tags preserved
        assert len(memory.tags) == 5

        # Only invalid namespaced tag logged
        assert "bad:namespace" in caplog.text
        assert "sys:core" not in caplog.text  # Valid, not logged
        assert "python" not in caplog.text    # Legacy, not logged

    def test_empty_tags_no_validation(self):
        """Empty tags list should not trigger validation"""
        content = "Test content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        memory = Memory(
            content=content,
            content_hash=content_hash,
            tags=[]
        )

        assert memory.tags == []
