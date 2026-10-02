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
