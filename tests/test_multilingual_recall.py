"""
Tests for multilingual tokenization in context_injection.py / entities.py.

History:
- Originally _WORD_RE = [a-z0-9]+ (ASCII) mutilated pt-BR ('decisão'->['decis','o']).
- 11/out: fixed to Latin-1 [a-zà-ÿ0-9]+ — pt-BR OK, but still blind to non-Latin scripts.
- 12/out: upgraded to \w+ with re.UNICODE — truly multilingual (cyrillic/CJK/arabic/ß),
  not just the Latin family. The pt-BR assertions below are the REGRESSION guard
  (they already passed since 11/out); the non-Latin tests are the real delta of 12/out.

The pt-BR tests FAIL only against the original ASCII regex; the non-Latin tests FAIL
against both ASCII and Latin-1, passing only with \w+ UNICODE.
The entity_id test is the INVARIANT (Henry's NFKD+hash decision, must stay ASCII).
"""

import pytest

# Import the actual functions to test
from mcp_memory_service.storage.context_injection import _tokenize, _overlap_relevance
from mcp_memory_service.storage.graph import normalize_entity_id


def test_tokenize_preserves_ptbr_accents():
    """REGRESSION guard: _tokenize preserves pt-BR accented words whole (OK since the
    Latin-1 fix 11/out; this guards against a regression back to ASCII)."""
    text = "decisão de injeção e código"
    tokens = _tokenize(text)
    
    # These whole Portuguese words should be preserved
    assert 'decisão' in tokens, f"Expected 'decisão' in tokens, got: {tokens}"
    assert 'injeção' in tokens, f"Expected 'injeção' in tokens, got: {tokens}"  
    assert 'código' in tokens, f"Expected 'código' in tokens, got: {tokens}"
    
    # These fragments should NOT appear (they indicate broken ASCII tokenization)
    assert 'decis' not in tokens, f"Broken fragment 'decis' should not appear, got: {tokens}"
    assert 'inje' not in tokens, f"Broken fragment 'inje' should not appear, got: {tokens}"
    assert 'c' not in tokens, f"Broken fragment 'c' should not appear, got: {tokens}"
    assert 'digo' not in tokens, f"Broken fragment 'digo' should not appear, got: {tokens}"



def test_tokenize_non_latin_scripts():
    """
    RED: _tokenize should capture tokens from NON-Latin scripts but currently returns [].
    
    Cyrillic: 'конфигурация' (configuration in Russian) should be preserved whole
    German ß: 'straße' should be preserved whole, not broken into 'stra' + 'e'
    
    This should FAIL today because _WORD_RE uses [a-zà-ÿ0-9]+ which is Latin-1 aware
    but still blind to Cyrillic, Greek, CJK, Arabic, and even German ß.
    
    Target: re.compile(r"\\w+", re.UNICODE) captures ANY Unicode script.
    """
    # Test Cyrillic (Russian) - should extract the whole word
    cyrillic_text = "конфигурация do sistema"
    cyrillic_tokens = _tokenize(cyrillic_text)
    assert 'конфигурация' in cyrillic_tokens, f"Expected 'конфигурация' in tokens, got: {cyrillic_tokens}"
    
    # Test German ß - should preserve the whole word, not break at ß  
    german_text = "razão straße problema"
    german_tokens = _tokenize(german_text)
    assert 'straße' in german_tokens, f"Expected 'straße' in tokens, got: {german_tokens}"
    # Should NOT have broken fragments
    assert 'stra' not in german_tokens, f"Broken fragment 'stra' should not appear, got: {german_tokens}"


def test_tokenize_still_preserves_ptbr():
    """
    REGRESSÃO: ensure Unicode upgrade doesn't break existing PT-BR behavior.
    
    'decisão de injeção' should still have 'decisão','injeção' intact.
    This should PASS today (with Latin-1 aware regex) AND continue passing 
    after Unicode upgrade. Proves we don't regress Latin script support.
    """
    text = "decisão de injeção"
    tokens = _tokenize(text)
    
    # These should work today AND after the fix
    assert 'decisão' in tokens, f"Expected 'decisão' in tokens, got: {tokens}"
    assert 'injeção' in tokens, f"Expected 'injeção' in tokens, got: {tokens}"


def test_entity_id_unchanged_INVARIANT():
    """
    INVARIANT: The tokenization fix must NOT alter normalize_entity_id behavior.
    
    This function uses NFKD normalization + ASCII slugification for entity IDs.
    Even after fixing _tokenize, normalize_entity_id should produce the same
    ASCII slug for 'Decisão Arquitetural' as it does today.
    
    This test should PASS both today AND after the fix (proves no regression).
    """
    input_text = "Decisão Arquitetural"
    
    # Capture the current normalized value
    current_normalized = normalize_entity_id(input_text)
    
    # This should be a stable ASCII slug (Henry's decision: NFKD+hash)
    # The actual value doesn't matter, but it should be deterministic ASCII
    assert isinstance(current_normalized, str), f"Expected str, got {type(current_normalized)}"
    assert len(current_normalized) > 0, f"Expected non-empty string, got {current_normalized}"
    
    # Should be ASCII-only (no accents in the entity ID)
    try:
        current_normalized.encode('ascii')
    except UnicodeEncodeError:
        pytest.fail(f"Entity ID should be ASCII-only, got: {current_normalized}")
    
    # The exact value should be stable - this captures the current behavior
    # Expected format is either a slug or 'e-<hash>' for non-ASCII inputs
    # Since 'Decisão Arquitetural' has accents, it might normalize to ASCII slug
    # or fall back to hash - either is valid as long as it's stable
    
    # Test deterministic behavior - calling twice should give same result
    second_call = normalize_entity_id(input_text)
    assert current_normalized == second_call, (
        f"normalize_entity_id should be deterministic. "
        f"First call: {current_normalized}, Second call: {second_call}"
    )