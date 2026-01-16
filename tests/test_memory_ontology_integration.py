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

# Create a fake package for the relative import
fake_package = type(sys)('mcp_memory_service')
fake_models = type(sys)('models')
fake_package.models = fake_models
fake_models.ontology = ontology_module
sys.modules['mcp_memory_service'] = fake_package
sys.modules['mcp_memory_service.models'] = fake_models
sys.modules['mcp_memory_service.models.ontology'] = ontology_module

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
