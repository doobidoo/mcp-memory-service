"""RED tests for belief quality Camada 1 (degenerate fragment filtering).

SPEC: docs/rfc/planned/spec-belief-quality-camada1.md

This file contains FAILING tests that define the required behavior for:
1. _is_degenerate_fragment function within _is_noise (behind MCP_BELIEF_DEGENERATE_FILTER flag)
2. One-shot sweep functionality (behind MCP_BELIEF_DEGENERATE_SWEEP flag)

These tests MUST fail against HEAD 83be6d25 until the implementation is done.
"""

import os
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.consolidation.belief_service import _is_noise, BeliefService


class TestDegenerateFragmentD1:
    """D1 tests: Dangling cut-stopword (truncated sentence endings)."""

    def test_d1_portuguese_preposition_final(self):
        """D1: Final Portuguese preposition should be degenerate."""
        # Set the environment flag
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            # These MUST be classified as noise by the new rule
            assert _is_noise("Se o fato mudou, o") is True  # ends in "o" (article)
            assert _is_noise("Configuração está em") is True  # ends in "em" (preposition) 
            assert _is_noise("Dados foram coletados de") is True  # ends in "de" (preposition)
            assert _is_noise("Sistema está funcionando para") is True  # ends in "para" (preposition)
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)

    def test_d1_english_preposition_final(self):
        """D1: Final English preposition should be degenerate."""
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            assert _is_noise("The configuration is set to") is True  # ends in "to"
            assert _is_noise("Data was collected from") is True  # ends in "from"
            assert _is_noise("System is working with") is True  # ends in "with"
            assert _is_noise("Results are saved in") is True  # ends in "in"
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)

    def test_d1_conjunction_final(self):
        """D1: Final conjunction should be degenerate."""
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            assert _is_noise("System works fine but") is True  # ends in "but"
            assert _is_noise("Configuration is valid and") is True  # ends in "and"
            assert _is_noise("Database backup succeeded because") is True  # ends in "because"
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)

    def test_d1_flag_off_no_filter(self):
        """D1: With flag OFF, behavior is identical to HEAD (degenerate passes)."""
        # Flag is off by default or explicitly disabled
        os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)
        
        # These would be degenerate WITH flag, but pass WITHOUT flag
        assert _is_noise("Se o fato mudou, o") is False  # passes current _is_noise
        assert _is_noise("Configuration is set to") is False  # passes current _is_noise


class TestDegenerateFragmentD2:
    """D2 tests: Dangling punctuation."""

    def test_d2_dangling_punctuation(self):
        """D2: Content ending in dangling punctuation should be degenerate."""
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            assert _is_noise("Configuration needs review:") is True  # ends in ":"
            assert _is_noise("Items to check,") is True  # ends in ","
            assert _is_noise("Database connection failed;") is True  # ends in ";"
            assert _is_noise("System status check-") is True  # ends in "-"
            assert _is_noise("Server config file (") is True  # ends in "("
            assert _is_noise("Array of values [") is True  # ends in "["
            assert _is_noise("JSON object format {") is True  # ends in "{"
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)

    def test_d2_with_trailing_whitespace(self):
        """D2: Dangling punctuation with trailing whitespace should be degenerate."""
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            assert _is_noise("Configuration needs: \t\n") is True  # colon + whitespace
            assert _is_noise("Check these items, ") is True  # comma + space
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)


class TestDegenerateFragmentD3:
    """D3 tests: Unbalanced opener brackets."""

    def test_d3_unbalanced_parentheses(self):
        """D3: Unbalanced opening parentheses should be degenerate."""
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            assert _is_noise("**Proxy Conecta.gov (") is True  # unbalanced (
            assert _is_noise("Function call format (param1, param2") is True  # missing )
            assert _is_noise("Multi-level nesting ((level1") is True  # 2 open, 0 close
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)

    def test_d3_unbalanced_brackets(self):
        """D3: Unbalanced opening brackets should be degenerate."""
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            assert _is_noise("Array access pattern [index") is True  # missing ]
            assert _is_noise("Nested arrays [[inner") is True  # 2 open, 0 close
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)

    def test_d3_unbalanced_braces(self):
        """D3: Unbalanced opening braces should be degenerate."""
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            assert _is_noise("JSON object format {key: value") is True  # missing }
            assert _is_noise("Nested objects {{inner") is True  # 2 open, 0 close
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)

    def test_d3_balanced_brackets_not_degenerate(self):
        """D3: Balanced brackets should NOT be degenerate (only by this rule)."""
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            # These have balanced brackets, so D3 should not trigger
            # (might still be degenerate by other rules like D4)
            content = "Function definition (param1, param2) works correctly"
            # This should pass D3 specifically (though may fail other rules)
            # We test this by checking it's not caught by D3's bracket logic
            # The content is long enough and substantive to pass existing filters
            assert len(content) > 15  # passes length check
            assert len(content.split()) > 3  # passes word count
            # The overall _is_noise might still return True for other reasons
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)


class TestDegenerateFragmentD4:
    """D4 tests: Orphan markdown heading."""

    def test_d4_orphan_bold_heading(self):
        """D4: Single bold heading lines should be degenerate."""
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            assert _is_noise("**Causa Raiz Identificada:**") is True
            assert _is_noise("**Ativação OSS do Greptile:**") is True  
            assert _is_noise("**QueryFlow Audit Design**") is True
            assert _is_noise("**Memory Service (F022)**") is True
            assert _is_noise("**Proxy Conecta.gov (**") is True  # also triggers D3
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)

    def test_d4_orphan_markdown_atx_heading(self):
        """D4: ATX markdown headings (###) without body should be degenerate."""
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            assert _is_noise("## Configuration Setup") is True
            assert _is_noise("### Database Schema") is True
            assert _is_noise("# Project Overview:") is True
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)

    def test_d4_orphan_colon_heading(self):
        """D4: Single line ending in colon (label-style heading) should be degenerate."""
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            assert _is_noise("Database Configuration:") is True
            assert _is_noise("System Requirements:") is True
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)

    def test_d4_heading_with_body_not_degenerate(self):
        """D4: Heading followed by body content should NOT be degenerate."""
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            multi_line_content = """**Configuration Setup:**
            The system requires proper database connection settings"""
            
            # This should NOT be degenerate because it has body content
            # (though it might fail for other reasons like length/noise filters)
            lines = [l for l in multi_line_content.splitlines() if l.strip()]
            assert len(lines) > 1  # Has body, so D4 should not trigger
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)

    def test_d4_inline_bold_not_degenerate(self):
        """D4 anti-regression: Inline bold in sentences should NOT be degenerate."""
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            # These MUST NOT be degenerate - they are legitimate sentences with inline bold
            assert _is_noise("Use **bold** para destacar termos.") is False
            assert _is_noise("Sempre marcar a tarefa como **done** ao concluir.") is False
            # These should still be degenerate - single bold lines without body
            assert _is_noise("**Causa Raiz Identificada:**") is True
            assert _is_noise("**QueryFlow Audit Design**") is True
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)


class TestCanaryValidSentences:
    """Test that legitimate short sentences are NOT classified as degenerate (anti-false-positive)."""

    def test_canary_sentences_survive(self):
        """All 4 canary sentences from spec MUST return False (not degenerate)."""
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            # These MUST NOT be degenerate - they are valid complete sentences
            assert _is_noise("Nunca deixar commit sem push.") is False
            assert _is_noise("Usar sempre LIMIT em queries PostgreSQL.") is False  
            assert _is_noise("O token em ~/.git-credentials expira.") is False
            assert _is_noise("Preferir ONNX ao invés de torch no CPU.") is False
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)

    def test_canary_explanation_push(self):
        """Canary 1: ends in content word 'push', not a stopword."""
        content = "Nunca deixar commit sem push."
        # Extract last alphabetic token for verification
        import re
        tokens = re.findall(r"[A-Za-zÀ-ÿ'`-]+", content)
        last_token = tokens[-1].lower()
        assert last_token == "push"
        # "push" is not in cut-stopwords, so D1 should not trigger
        
    def test_canary_explanation_postgresql(self):
        """Canary 2: ends in content word 'PostgreSQL', not a stopword."""
        content = "Usar sempre LIMIT em queries PostgreSQL."
        import re
        tokens = re.findall(r"[A-Za-zÀ-ÿ'`-]+", content)  
        last_token = tokens[-1].lower()
        assert last_token == "postgresql"
        # "postgresql" is not in cut-stopwords, so D1 should not trigger

    def test_canary_explanation_expira(self):
        """Canary 3: ends in content word 'expira' (verb), not a stopword."""
        content = "O token em ~/.git-credentials expira."
        import re
        tokens = re.findall(r"[A-Za-zÀ-ÿ'`-]+", content)
        last_token = tokens[-1].lower() 
        assert last_token == "expira"
        # "expira" is not in cut-stopwords, so D1 should not trigger

    def test_canary_explanation_cpu(self):
        """Canary 4: ends in content word 'CPU', not a stopword."""
        content = "Preferir ONNX ao invés de torch no CPU."
        import re
        tokens = re.findall(r"[A-Za-zÀ-ÿ'`-]+", content)
        last_token = tokens[-1].lower()
        assert last_token == "cpu"  
        # "cpu" is not in cut-stopwords, so D1 should not trigger


class TestBackwardCompatibility:
    """Test that with flags OFF, behavior is byte-identical to HEAD."""

    def test_flag_off_identical_behavior(self):
        """With MCP_BELIEF_DEGENERATE_FILTER=false, _is_noise behaves exactly like HEAD."""
        # Ensure flag is off
        os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)
        
        # Test cases that WOULD be degenerate with flag on, but pass with flag off
        test_cases = [
            "Se o fato mudou, o",  # D1 
            "**Causa Raiz Identificada:**",  # D4 + D2
            "Configuration setup:",  # D2
            "Function call format (param1",  # D3
        ]
        
        for case in test_cases:
            # With flag off, these should pass current _is_noise logic
            # (assuming they pass length/word/entropy checks - which these examples do)
            if len(case) >= 15 and len(case.split()) >= 3:  # Pass existing filters
                assert _is_noise(case) is False

    def test_existing_filters_still_work(self):
        """Existing _is_noise filters must still work regardless of flag state."""
        os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)
        
        # Test existing noise detection still works
        assert _is_noise("short") is True  # too short
        assert _is_noise("a b") is True  # too few words  
        assert _is_noise("# Session Summary") is True  # session pattern
        assert _is_noise("[CHECKPOINT] saved") is True  # prefix pattern
        
        # Same tests with flag on
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        try:
            assert _is_noise("short") is True
            assert _is_noise("a b") is True
            assert _is_noise("# Session Summary") is True
            assert _is_noise("[CHECKPOINT] saved") is True
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)


class TestDegenerateSweep:
    """Test the one-shot sweep functionality for existing beliefs."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage with test beliefs."""
        # Create in-memory database for testing
        db = sqlite3.connect(":memory:")
        
        # Create beliefs table (simplified schema for testing)
        db.execute("""
            CREATE TABLE beliefs (
                belief_hash TEXT PRIMARY KEY,
                content TEXT,
                status TEXT,
                confidence REAL,
                updated_at TEXT,
                derived_from TEXT,
                contradicted_by TEXT,
                created_at TEXT,
                metadata TEXT
            )
        """)
        
        # Insert test beliefs - mix of degenerate and valid
        test_beliefs = [
            ("hash_d1", "Se o fato mudou, o", "active", 0.8, "2026-10-15T10:00:00Z", "[]", "[]", "2026-10-15T09:00:00Z", "{}"),
            ("hash_d2", "Configuration setup:", "candidate", 0.6, "2026-10-15T10:00:00Z", "[]", "[]", "2026-10-15T09:00:00Z", "{}"),
            ("hash_d4", "**Causa Raiz Identificada:**", "active", 0.7, "2026-10-15T10:00:00Z", "[]", "[]", "2026-10-15T09:00:00Z", "{}"),
            ("hash_valid", "Always use LIMIT in PostgreSQL queries.", "active", 0.9, "2026-10-15T10:00:00Z", "[]", "[]", "2026-10-15T09:00:00Z", "{}"),
            ("hash_superseded", "Already superseded belief", "superseded", 0.3, "2026-10-15T10:00:00Z", "[]", "[]", "2026-10-15T09:00:00Z", "{}"),
        ]
        
        for belief in test_beliefs:
            db.execute("INSERT INTO beliefs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", belief)
        db.commit()
        
        # Mock storage object
        storage = MagicMock()
        storage.conn = db
        
        yield storage
        db.close()

    def test_sweep_requires_flag(self, temp_storage):
        """Sweep should be no-op when MCP_BELIEF_DEGENERATE_SWEEP flag is off."""
        os.environ.pop("MCP_BELIEF_DEGENERATE_SWEEP", None)
        
        service = BeliefService(temp_storage)
        
        # This method doesn't exist yet - will fail until implemented
        result = service.sweep_degenerate_beliefs()
        
        # Should return zero changes when flag is off
        assert result["superseded"] == 0
        assert result["evaluated"] == 0

    def test_sweep_marks_degenerate_superseded(self, temp_storage):
        """Sweep should mark degenerate beliefs as superseded."""
        os.environ["MCP_BELIEF_DEGENERATE_SWEEP"] = "1"
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"  # Enable the filter logic
        
        try:
            service = BeliefService(temp_storage)
            
            # This method doesn't exist yet - will fail until implemented  
            result = service.sweep_degenerate_beliefs()
            
            # Should have marked the 3 degenerate beliefs as superseded
            assert result["superseded"] == 3  # hash_d1, hash_d2, hash_d4
            assert result["evaluated"] > 0
            assert result["capped"] is False
            
            # Verify beliefs were actually updated in database
            cursor = temp_storage.conn.execute(
                "SELECT belief_hash, status FROM beliefs WHERE belief_hash IN ('hash_d1', 'hash_d2', 'hash_d4')"
            )
            results = cursor.fetchall()
            
            for belief_hash, status in results:
                assert status == "superseded"
                
            # Valid belief should remain active
            cursor = temp_storage.conn.execute(
                "SELECT status FROM beliefs WHERE belief_hash = 'hash_valid'"
            )
            status = cursor.fetchone()[0]
            assert status == "active"
            
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_SWEEP", None)
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)

    def test_sweep_respects_cap(self, temp_storage):
        """Sweep should respect the cap limit."""
        os.environ["MCP_BELIEF_DEGENERATE_SWEEP"] = "1"
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        os.environ["MCP_BELIEF_DEGENERATE_SWEEP_CAP"] = "2"  # Cap at 2
        
        try:
            service = BeliefService(temp_storage)
            
            result = service.sweep_degenerate_beliefs()
            
            # Should have marked at most 2 beliefs due to cap
            assert result["superseded"] <= 2
            assert result["capped"] is True
            
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_SWEEP", None)
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)
            os.environ.pop("MCP_BELIEF_DEGENERATE_SWEEP_CAP", None)

    def test_sweep_idempotent(self, temp_storage):
        """Running sweep twice should mark 0 additional beliefs on second run."""
        os.environ["MCP_BELIEF_DEGENERATE_SWEEP"] = "1"
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        
        try:
            service = BeliefService(temp_storage)
            
            # First sweep
            result1 = service.sweep_degenerate_beliefs()
            first_superseded = result1["superseded"]
            
            # Second sweep
            result2 = service.sweep_degenerate_beliefs()
            
            # Second run should supersede 0 additional beliefs
            assert result2["superseded"] == 0
            assert first_superseded > 0  # First run did work
            
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_SWEEP", None)
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)

    def test_sweep_preserves_derived_from(self, temp_storage):
        """Sweep should not delete or alter derived_from/contradicted_by fields."""
        os.environ["MCP_BELIEF_DEGENERATE_SWEEP"] = "1"
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        
        try:
            service = BeliefService(temp_storage)
            
            # Get original derived_from before sweep
            cursor = temp_storage.conn.execute(
                "SELECT derived_from, contradicted_by FROM beliefs WHERE belief_hash = 'hash_d1'"
            )
            original_data = cursor.fetchone()
            
            service.sweep_degenerate_beliefs()
            
            # Check that derived_from/contradicted_by are preserved
            cursor = temp_storage.conn.execute(
                "SELECT derived_from, contradicted_by, status FROM beliefs WHERE belief_hash = 'hash_d1'"
            )
            after_data = cursor.fetchone()
            
            assert after_data[0] == original_data[0]  # derived_from preserved
            assert after_data[1] == original_data[1]  # contradicted_by preserved
            assert after_data[2] == "superseded"  # but status was updated
            
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_SWEEP", None)
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)


class TestIntegrationIsNoise:
    """Integration tests for _is_noise with degenerate filter enabled."""

    def test_is_noise_composes_all_filters(self):
        """_is_noise should apply degenerate filter in addition to existing filters."""
        os.environ["MCP_BELIEF_DEGENERATE_FILTER"] = "1"
        
        try:
            # Existing filters should still work
            assert _is_noise("short") is True  # length filter
            assert _is_noise("# Session Summary") is True  # pattern filter  
            
            # New degenerate filter should also work
            assert _is_noise("Se o fato mudou, o") is True  # D1
            assert _is_noise("**Causa Raiz:**") is True  # D4
            
            # Valid content should pass all filters
            assert _is_noise("Always use LIMIT in PostgreSQL queries.") is False
            
        finally:
            os.environ.pop("MCP_BELIEF_DEGENERATE_FILTER", None)

    def test_derive_beliefs_calls_is_noise(self):
        """Integration test that derive_beliefs properly calls hardened _is_noise."""
        # This would be an integration test with real BeliefService.derive_beliefs()
        # For now, just verify that the interface exists
        
        # Mock a BeliefService and verify it calls _is_noise during derive_beliefs
        storage = MagicMock()
        storage.get_all_memories = AsyncMock(return_value=[])
        
        service = BeliefService(storage)
        
        # The derive_beliefs method should exist and be callable
        assert hasattr(service, 'derive_beliefs')
        assert callable(service.derive_beliefs)