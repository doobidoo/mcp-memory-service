"""RED tests for the ``TYPE:`` prefix leak in HarvestRewriter batch parsing.

Observed in a real deepseek run over the curated gold corpus (02/out/2026):
the model echoes the literal placeholder word ``TYPE:`` from the prompt instead
of substituting the real type, producing candidates like:

    "TYPE: convention — Arquivos de curadoria devem seguir ..."   (prefix leak)
    "TYPE: bug"                                                   (degenerate)
    "TYPE"                                                        (degenerate)

The parser must:
  1. Strip a leaked ``TYPE: <realtype> — content`` wrapper, keeping only content
     and the real type.
  2. Drop degenerate candidates whose content is empty / just a label.
"""
from mcp_memory_service.harvest.rewriter import HarvestRewriter


def _mk():
    return HarvestRewriter.__new__(HarvestRewriter)


def test_batch_strips_leaked_type_label_with_real_type():
    """``N. TYPE: convention — text`` → content='text', type='convention'."""
    r = _mk()
    items = [{"content": "x", "memory_type": "learning"}]
    resp = "1. TYPE: convention — Arquivos de curadoria seguem o padrão curadoria-WXX.md."
    out = r._parse_batch_response(resp, items)
    assert out[0] is not None
    assert out[0].memory_type == "convention"
    assert not out[0].content.upper().startswith("TYPE")
    assert "Arquivos de curadoria" in out[0].content


def test_batch_strips_leaked_type_label_em_dash_variants():
    """Handles ' - ', ' — ', ': ' separators after the leaked label+type."""
    r = _mk()
    items = [{"content": "x", "memory_type": "context"}] * 3
    resp = (
        "1. TYPE: bug - OpenConnect 9.12 nao chama gnutls_pkcs11_init(); fix: patch 7 linhas.\n"
        "2. TYPE: learning: SPEC em JSON e mais resiliente que Markdown.\n"
        "3. TYPE: decision — Usar RRF em vez de weighted average."
    )
    out = r._parse_batch_response(resp, items)
    assert out[0].memory_type == "bug" and out[0].content.startswith("OpenConnect")
    assert out[1].memory_type == "learning" and out[1].content.startswith("SPEC em JSON")
    assert out[2].memory_type == "decision" and out[2].content.startswith("Usar RRF")
    for o in out:
        assert not o.content.upper().startswith("TYPE")


def test_batch_drops_degenerate_label_only_candidates():
    """``TYPE: bug`` / ``TYPE`` with no real content must be dropped (None)."""
    r = _mk()
    items = [{"content": "x", "memory_type": "bug"}] * 3
    resp = "1. TYPE: bug\n2. TYPE: context\n3. TYPE"
    out = r._parse_batch_response(resp, items)
    assert out == [None, None, None]


def test_single_parse_strips_leaked_type_label():
    """Single-item parser also strips a leaked ``TYPE: <realtype>`` wrapper."""
    r = _mk()
    res = r._parse_response("TYPE: convention — Para diffs, forcar side-by-side.", "learning")
    assert res is not None
    assert res.memory_type == "convention"
    assert not res.content.upper().startswith("TYPE")
    assert res.content.startswith("Para diffs")


def test_single_parse_drops_degenerate_label_only():
    """Single parser drops a label-only degenerate response."""
    r = _mk()
    assert r._parse_response("TYPE: bug", "bug") is None
    assert r._parse_response("TYPE", "bug") is None


def test_legit_content_starting_with_type_word_is_untouched():
    """Content legitimately starting with 'Type' (no colon) must NOT be mangled."""
    r = _mk()
    res = r._parse_response("Type hints em Python melhoram legibilidade e catch de bugs.", "learning")
    assert res is not None
    assert res.content == "Type hints em Python melhoram legibilidade e catch de bugs."
    assert res.memory_type == "learning"


# === P1 Bug Tests (from Greptile findings) ===

def test_p1a_prose_after_type_should_keep_whole_text():
    """P1-a: TYPE: Nunca usar force push → keep WHOLE text 'Nunca usar force push', not 'usar force push'."""
    r = _mk()
    # When content after TYPE: is prose (not a valid type), keep everything after TYPE:
    res = r._parse_response("TYPE: Nunca usar force push em branches protegidos", "convention")
    assert res is not None
    assert res.content == "Nunca usar force push em branches protegidos"
    assert not res.content.startswith("usar force push")  # Should NOT strip first word


def test_p1b_short_valid_insight_should_not_be_dropped():
    """P1-b: 'TYPE: convention — Nunca force push.' (17 chars) should be KEPT, not dropped."""
    r = _mk()
    # Short but complete insight with valid type should not be dropped
    res = r._parse_response("TYPE: convention — Nunca force push.", "learning")
    assert res is not None
    assert res.memory_type == "convention"
    assert res.content == "Nunca force push."
    assert len(res.content) < 25  # Confirm it's short but still kept


def test_p1c_angle_bracket_type_should_be_unwrapped():
    """P1-c: '<type>: convention — texto' should unwrap <type>: prefix same as TYPE:."""
    r = _mk()
    # Both single and batch parsers should handle <type>: prefix
    res = r._parse_response("<type>: convention — Nunca force push", "learning")
    assert res is not None
    assert res.memory_type == "convention"
    assert res.content == "Nunca force push"
    assert not res.content.startswith("<type>:")


def test_p1c_batch_angle_bracket_type_unwrapping():
    """P1-c batch: '1. <type>: texto' should unwrap <type>: prefix."""
    r = _mk()
    items = [{"content": "x", "memory_type": "learning"}]
    resp = "1. <type>: convention — Usar subagent dedicado"
    out = r._parse_batch_response(resp, items)
    assert out[0] is not None
    assert out[0].memory_type == "convention"
    assert out[0].content == "Usar subagent dedicado"
    assert not out[0].content.startswith("<type>:")


def test_still_drop_true_degenerates():
    """Should still drop true degenerates: bare labels without meaningful content."""
    r = _mk()
    # These should still be dropped as they have no meaningful content
    assert r._parse_response("TYPE: bug", "learning") is None  # Valid type but no content
    assert r._parse_response("TYPE", "learning") is None       # Bare TYPE
    assert r._parse_response("<type>", "learning") is None     # Bare <type>
    assert r._parse_response("<type>: bug", "learning") is None  # Valid type but no content


def test_prose_starting_with_typescript_untouched():
    """Legitimate prose starting with 'typescript' (no colon) should be untouched."""
    r = _mk()
    res = r._parse_response("typescript interfaces provide better type safety than any", "learning")
    assert res is not None
    assert res.content == "typescript interfaces provide better type safety than any"
    assert res.memory_type == "learning"
