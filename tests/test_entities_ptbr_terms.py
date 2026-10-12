"""Tests for Portuguese term extraction in entities."""

import pytest
from mcp_memory_service.reasoning.entities import EntityExtractor


class TestEntityExtractorPtBrTerms:
    def test_extract_frequent_terms_preserves_ptbr(self):
        """Test that Portuguese accented terms are extracted intact, not fragmented."""
        # Create memories with repeated Portuguese terms that should be preserved
        memories = [
            {"content": "A configuração do sistema precisa ser alterada."},
            {"content": "Esta configuração está causando problemas no código."},
            {"content": "Vamos revisar a configuração antes da decisão final."},
        ]
        
        # Extract frequent terms with min_count=2 (appears in 2+ memories)
        frequent_terms = EntityExtractor.extract_frequent_terms(memories, min_count=2)
        
        # Assert target behavior: the complete Portuguese term should be preserved
        # This will be RED today because the ASCII regex fragments 'configuração' → 'configura'
        assert 'configuração' in frequent_terms
    
    def test_extract_frequent_terms_multiple_ptbr_words(self):
        """Test extraction of multiple Portuguese accented terms."""
        memories = [
            {"content": "A aplicação tem código de qualidade."},
            {"content": "O código da aplicação está bem estruturado."},
        ]
        
        frequent_terms = EntityExtractor.extract_frequent_terms(memories, min_count=2)
        
        # Both complete terms should be preserved
        assert 'aplicação' in frequent_terms
        assert 'código' in frequent_terms

    def test_entities_non_latin(self):
        """
        RED: extract_frequent_terms should capture non-Latin script terms but currently discards them.
        
        Cyrillic term 'конфигурация' repeated ≥min_count should be captured.
        Today [a-zA-ZÀ-ÿ]{3,} regex discards non-Latin → empty result → RED.
        Target: \\w{3,} with re.UNICODE captures ANY script.
        """
        memories = [
            {"content": "Система использует конфигурация для запуска."},
            {"content": "Проблема с конфигурация базы данных."},
            {"content": "Нужно обновить конфигурация сервера."},
        ]
        
        frequent_terms = EntityExtractor.extract_frequent_terms(memories, min_count=2)
        
        # Should capture the repeated Cyrillic term
        assert 'конфигурация' in frequent_terms, f"Expected 'конфигурация' in frequent terms, got: {frequent_terms}"