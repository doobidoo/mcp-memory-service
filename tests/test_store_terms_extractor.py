"""Tests for StoreTermsExtractor - RED phase (class doesn't exist yet)."""

import json

import pytest

from mcp_memory_service.reasoning.entities import EntityExtractor

# These imports will fail initially - expected for RED phase
from mcp_memory_service.reasoning.store_terms import StoreTermsExtractor


@pytest.fixture
def store_terms_json(tmp_path):
    """Create a fixture JSON file with store terms."""
    json_data = {
        "mir": {
            "locale": "pt_BR",
            "terms": ["F008", "SICAR", "CAR", "floresta tipo B"]
        },
        "rer": {
            "locale": "en", 
            "terms": ["CAR", "geoserver"]
        },
        "default": {
            "locale": "pt_BR,en",
            "terms": []
        }
    }
    
    json_file = tmp_path / "store_terms.json"
    json_file.write_text(json.dumps(json_data, indent=2))
    return str(json_file)


class TestStoreTermsExtractor:
    """Test suite for StoreTermsExtractor."""
    
    def test_match_por_store_extrai_termos_corretos(self, store_terms_json):
        """Test that terms are extracted correctly for specific store."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        content = "F008 SICAR floresta tipo B"
        metadata = {"store": "mir"}
        
        entities = extractor.extract(content, metadata)
        
        assert len(entities) == 3
        entity_names = [e.name for e in entities]
        assert "F008" in entity_names
        assert "SICAR" in entity_names
        assert "floresta tipo B" in entity_names
        
        for entity in entities:
            assert entity.entity_type == "domain_term"
            assert entity.source == "domain"
    
    def test_isolamento_entre_stores_nao_extrai_termos_outros_stores(self, store_terms_json):
        """Test that terms from other stores are not extracted."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        content = "F008 SICAR floresta tipo B"  # mir terms
        metadata = {"store": "rer"}
        
        entities = extractor.extract(content, metadata)
        
        # rer store doesn't have F008, SICAR, or "floresta tipo B"
        assert len(entities) == 0
    
    def test_isolamento_store_rer_extrai_seus_proprios_termos(self, store_terms_json):
        """Test that rer store extracts its own terms but not mir terms."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        content = "CAR geoserver F008"
        metadata = {"store": "rer"}
        
        entities = extractor.extract(content, metadata)
        
        assert len(entities) == 2
        entity_names = [e.name for e in entities]
        assert "CAR" in entity_names
        assert "geoserver" in entity_names
        assert "F008" not in entity_names  # F008 is mir-specific
    
    def test_sem_store_usa_default_retorna_vazio(self, store_terms_json):
        """Test that missing store metadata extracts from all stores (union behavior)."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        content = "F008 SICAR CAR"
        metadata = {}
        
        entities = extractor.extract(content, metadata)
        
        # Should extract from union of all stores
        assert len(entities) == 3
        entity_names = [e.name for e in entities]
        assert "F008" in entity_names  # from mir
        assert "SICAR" in entity_names  # from mir (and rer, but deduplicated)  
        assert "CAR" in entity_names  # from both mir and rer, deduplicated
    
    def test_sem_metadata_usa_default_retorna_vazio(self, store_terms_json):
        """Test that None metadata extracts from all stores (union behavior)."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        content = "F008 SICAR CAR"
        
        entities = extractor.extract(content, None)
        
        # Should extract from union of all stores  
        assert len(entities) == 3
        entity_names = [e.name for e in entities]
        assert "F008" in entity_names  # from mir
        assert "SICAR" in entity_names  # from mir (and rer, but deduplicated)
        assert "CAR" in entity_names  # from both mir and rer, deduplicated
    
    def test_store_desconhecido_fallback_default_retorna_vazio(self, store_terms_json):
        """Test that unknown store falls back to default."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        content = "F008 SICAR CAR"
        metadata = {"store": "inexistente"}
        
        entities = extractor.extract(content, metadata)
        
        # fallback to default store (empty terms)
        assert len(entities) == 0
    
    def test_multi_word_uma_entidade_nao_tres(self, store_terms_json):
        """Test that multi-word terms are extracted as single entities."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        content = "consultar floresta tipo B no mapa"
        metadata = {"store": "mir"}
        
        entities = extractor.extract(content, metadata)
        
        assert len(entities) == 1
        assert entities[0].name == "floresta tipo B"
        assert entities[0].entity_type == "domain_term"
        assert entities[0].source == "domain"
    
    def test_word_boundary_anti_falso_positivo_car_em_scared(self, store_terms_json):
        """Test that word boundaries prevent false positives (CAR inside 'scared')."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        content = "the child was scared"
        metadata = {"store": "rer"}
        
        entities = extractor.extract(content, metadata)
        
        # "CAR" should NOT match inside "scared"
        assert len(entities) == 0
    
    def test_word_boundary_car_isolado_casa_corretamente(self, store_terms_json):
        """Test that isolated CAR word matches correctly."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        content = "register CAR document"
        metadata = {"store": "rer"}
        
        entities = extractor.extract(content, metadata)
        
        assert len(entities) == 1
        assert entities[0].name == "CAR"
    
    def test_case_insensitive_sicar_minusculo_casa_maiusculo(self, store_terms_json):
        """Test that case-insensitive matching works."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        content = "sicar minusculo"
        metadata = {"store": "mir"}
        
        entities = extractor.extract(content, metadata)
        
        assert len(entities) == 1
        assert entities[0].name == "SICAR"  # original term is uppercase
    
    def test_case_insensitive_termo_misto_floresta_tipo_b(self, store_terms_json):
        """Test case-insensitive matching for multi-word terms."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        content = "FLORESTA TIPO b analysis"
        metadata = {"store": "mir"}
        
        entities = extractor.extract(content, metadata)
        
        assert len(entities) == 1
        assert entities[0].name == "floresta tipo B"  # original case preserved
    
    def test_arquivo_ausente_no_op_sem_excecao(self, tmp_path):
        """Test that missing file returns empty list without exception."""
        nonexistent_path = str(tmp_path / "nonexistent.json")
        extractor = StoreTermsExtractor(path=nonexistent_path)
        content = "F008 SICAR CAR"
        metadata = {"store": "mir"}
        
        # Should not raise exception
        entities = extractor.extract(content, metadata)
        
        assert entities == []
    
    def test_arquivo_corrompido_no_op_sem_excecao(self, tmp_path):
        """Test that corrupted JSON file returns empty list without exception."""
        corrupted_file = tmp_path / "corrupted.json"
        corrupted_file.write_text("{ invalid json content")
        
        extractor = StoreTermsExtractor(path=str(corrupted_file))
        content = "F008 SICAR CAR"
        metadata = {"store": "mir"}
        
        # Should not raise exception
        entities = extractor.extract(content, metadata)
        
        assert entities == []
    
    def test_integracao_via_entity_extractor_source_domain(self, store_terms_json):
        """Test integration with EntityExtractor showing source='domain'."""
        store_extractor = StoreTermsExtractor(path=store_terms_json)
        entity_extractor = EntityExtractor(domain_extractors=[store_extractor])
        
        content = "F008 SICAR analysis"
        metadata = {"store": "mir"}
        
        all_entities = entity_extractor.extract_entities(content, metadata)
        
        # Filter for domain entities
        domain_entities = [e for e in all_entities if e.source == "domain"]
        
        assert len(domain_entities) >= 2  # At least F008 and SICAR
        domain_names = [e.name for e in domain_entities]
        assert "F008" in domain_names
        assert "SICAR" in domain_names
        
        for entity in domain_entities:
            assert entity.entity_type == "domain_term"
            assert entity.source == "domain"
    
    def test_construtor_sem_path_usa_default_path(self):
        """Test constructor without path uses default path."""
        # This will likely fail to load the default file, but should not crash
        extractor = StoreTermsExtractor()
        content = "test content"
        metadata = {"store": "mir"}
        
        # Should handle missing default file gracefully
        entities = extractor.extract(content, metadata)
        
        assert isinstance(entities, list)
    
    def test_word_boundary_special_chars_f008_isolado(self, store_terms_json):
        """Word boundary treats '-' and '_' as part of the word (matches
        entities.py:88). F008 only matches when isolated by whitespace/plain
        punctuation, NOT when glued to hyphens/underscores."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        metadata = {"store": "mir"}

        # Isolated by spaces -> matches
        assert "F008" in [e.name for e in extractor.extract("veja o F008 hoje", metadata)]
        # Isolated by plain punctuation (period) -> matches
        assert "F008" in [e.name for e in extractor.extract("query F008.", metadata)]
        # Glued to hyphen/underscore/alnum -> does NOT match (consistent with upstream)
        assert "F008" not in [e.name for e in extractor.extract("document-F008-version", metadata)]
        assert "F008" not in [e.name for e in extractor.extract("F008_update", metadata)]
        assert "F008" not in [e.name for e in extractor.extract("xF008x", metadata)]
    
    def test_sem_store_extrai_uniao_todos_stores(self, store_terms_json):
        """Test that missing store extracts terms from ALL stores (union)."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        # Content with terms from different stores: F008 (mir), geoserver (rer)
        content = "F008 SICAR geoserver CAR analysis"
        metadata = {}
        
        entities = extractor.extract(content, metadata)
        
        # Should extract terms from any store
        assert len(entities) >= 4
        entity_names = [e.name for e in entities]
        assert "F008" in entity_names  # from mir
        assert "SICAR" in entity_names  # from both mir and rer, should appear once
        assert "geoserver" in entity_names  # from rer
        assert "CAR" in entity_names  # from both mir and rer, should appear once
        
        for entity in entities:
            assert entity.entity_type == "domain_term"
            assert entity.source == "domain"
    
    def test_sem_store_dedup_termo_multiplos_stores(self, store_terms_json):
        """Test that terms appearing in multiple stores are deduplicated."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        # CAR appears in both mir and rer, SICAR appears in both mir and rer
        content = "CAR SICAR document"
        metadata = {}
        
        entities = extractor.extract(content, metadata)
        
        # Should have exactly 2 entities (no duplicates)
        assert len(entities) == 2
        entity_names = [e.name for e in entities]
        assert "CAR" in entity_names
        assert "SICAR" in entity_names
        
        # Verify no duplicates by checking unique names
        assert len(entity_names) == len(set(entity_names))
    
    def test_sem_store_none_metadata_extrai_uniao(self, store_terms_json):
        """Test that None metadata extracts terms from ALL stores."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        content = "F008 geoserver analysis"
        
        entities = extractor.extract(content, None)
        
        # Should extract from union of all stores
        assert len(entities) == 2
        entity_names = [e.name for e in entities]
        assert "F008" in entity_names  # from mir
        assert "geoserver" in entity_names  # from rer
    
    def test_content_sem_nenhum_termo_vazio(self, store_terms_json):
        """Test that content with no terms returns empty list."""
        extractor = StoreTermsExtractor(path=store_terms_json)
        content = "some random text without domain terms"
        metadata = {}
        
        entities = extractor.extract(content, metadata)
        
        assert len(entities) == 0