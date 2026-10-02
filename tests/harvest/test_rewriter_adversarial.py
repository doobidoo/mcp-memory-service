"""Adversarial (Gate 5 / JiTTesting) tests for the TYPE-leak fix in rewriter.py.

Goal: try to BREAK ``_unleak_type`` and the two parsers (``_parse_response``,
``_parse_batch_response``). These probe the boundary conditions the happy-path
``test_rewriter_type_leak.py`` does not: legitimate content that merely starts
with the word "type", case folding, unicode separators, DOTALL multi-line,
and — critically — whether the min-payload guard silently eats legitimate short
insights.

Written by the independent reviewer. The v2 fix narrowed the length guard so it
fires ONLY on a *confirmed* leaked wrapper whose unwrapped payload is a
truncated fragment (< ``_MIN_LEAKED_PAYLOAD_CHARS``). Clean content (no
``TYPE:`` wrapper) is now RETAINED verbatim regardless of size. The former
xfail that documented the DROP of short clean content is therefore converted
into a POSITIVE retention assertion.
"""
import time

import pytest

from mcp_memory_service.harvest.rewriter import (
    _unleak_type,
    VALID_TYPES,
    HarvestRewriter,
)


def _mk():
    return HarvestRewriter.__new__(HarvestRewriter)


# ---------------------------------------------------------------------------
# (a) Legitimate content starting with 'type'/'Type'/'typescript' WITHOUT a
#     colon must NEVER be mutilated.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "typescript strict mode evita muitos bugs em runtime no projeto.",
    "Type hints em Python melhoram legibilidade e catch de bugs cedo.",
    "typed arrays sao mais rapidos que listas comuns para numeros grandes.",
    "TypeScript: use strict e noImplicitAny para pegar erros em build time.",
])
def test_legit_type_prefixed_content_untouched(text):
    """No literal 'TYPE:' wrapper → content returned verbatim, type preserved."""
    content, mem_type = _unleak_type(text, "learning")
    assert content == text, "legitimate 'type...'-content was mangled"
    assert mem_type == "learning"


def test_legit_type_prefixed_via_single_parser():
    r = _mk()
    text = "TypeScript: strict mode evita bugs em runtime com facilidade total."
    res = r._parse_response(text, "learning")
    assert res is not None
    # 'TypeScript' is not a valid type and is NOT the leaked 'TYPE' token,
    # so the full response must survive intact.
    assert res.content == text
    assert res.memory_type == "learning"


# ---------------------------------------------------------------------------
# (b) 'TYPE:' with a VALID real type vs an INVALID one.
# ---------------------------------------------------------------------------

def test_leaked_type_with_valid_realtype_recovers_type():
    content, mem_type = _unleak_type(
        "TYPE: convention — Nunca usar strReplace em arquivos append-only.",
        "learning",
    )
    assert content == "Nunca usar strReplace em arquivos append-only."
    assert mem_type == "convention"


def test_leaked_type_with_invalid_realtype_falls_back():
    content, mem_type = _unleak_type(
        "TYPE: frobnicate — Esse texto tem substancia suficiente para passar.",
        "learning",
    )
    assert content == "frobnicate — Esse texto tem substancia suficiente para passar."
    # Unknown real type → must fall back, never store 'frobnicate' as type.
    assert mem_type == "learning"
    assert mem_type in VALID_TYPES


def test_leaked_type_no_realtype_token_falls_back():
    # "TYPE: <text>" with no recognizable type word after the colon.
    content, mem_type = _unleak_type(
        "TYPE: Esse texto comeca direto sem um tipo valido explicito nenhum.",
        "context",
    )
    assert content and not content.upper().startswith("TYPE")
    assert mem_type == "context"


# ---------------------------------------------------------------------------
# (c) Multi-line / DOTALL content.
# ---------------------------------------------------------------------------

def test_dotall_multiline_content_preserved():
    content, mem_type = _unleak_type(
        "TYPE: decision — Usar RRF em vez de weighted average\n"
        "porque melhora recall em hybrid search na pratica.",
        "learning",
    )
    assert mem_type == "decision"
    assert "RRF" in content and "recall" in content
    assert "\n" in content, "second line of a multi-line insight was lost"


def test_batch_dotall_does_not_swallow_next_numbered_line():
    """A leaked line must not greedily eat the following numbered memory."""
    r = _mk()
    items = [{"content": "x", "memory_type": "context"}] * 2
    resp = (
        "1. TYPE: convention — Nunca usar strReplace em append-only files aqui.\n"
        "2. decision: Usar RRF sempre em hybrid search para melhor recall real."
    )
    out = r._parse_batch_response(resp, items)
    assert out[0] is not None and out[1] is not None
    assert out[0].memory_type == "convention"
    assert out[1].memory_type == "decision"
    assert "RRF" in out[1].content
    # The convention line must not contain the decision line's text.
    assert "RRF" not in out[0].content


# ---------------------------------------------------------------------------
# (d) Unicode separators (em dash) and extra whitespace.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("resp,expected_type,needle", [
    ("TYPE: bug — OpenConnect 9.12 nao chama gnutls_pkcs11_init() nunca.", "bug", "OpenConnect"),
    ("TYPE: learning: SPEC em JSON e mais resiliente que Markdown puro.", "learning", "SPEC"),
    ("TYPE: decision - Usar RRF em vez de weighted average no ranking.", "decision", "RRF"),
    ("TYPE:   convention   —    Espacos extras ao redor do separador aqui.", "convention", "Espacos"),
])
def test_separators_and_whitespace(resp, expected_type, needle):
    content, mem_type = _unleak_type(resp, "context")
    assert mem_type == expected_type
    assert needle in content
    assert not content.upper().startswith("TYPE")


# ---------------------------------------------------------------------------
# (e) Min-payload guard: v2 fix — the guard fires ONLY on a confirmed leaked
#     wrapper. Clean short content (no wrapper) MUST be RETAINED.
#     This replaces the former xfail that documented the (now-fixed) DROP.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("short_insight", [
    "Nunca force push.",            # 17 chars — the original MEDIUM-finding case
    "Prefira uv a pip install.",
    "Nunca usar force push.",
    "Use git rebase sempre -i.",
    "Usar RRF nao weighted avg.",
])
def test_short_clean_insight_is_retained(short_insight):
    """A legitimate short convention with NO leak wrapper must be kept verbatim.

    v2 regression guard for the MEDIUM finding: the length gate must NOT run on
    clean content. Several of these are < _MIN_LEAKED_PAYLOAD_CHARS and must
    still survive untouched.
    """
    content, mem_type = _unleak_type(short_insight, "convention")
    assert content == short_insight, (
        "clean short insight was dropped/mutilated — the length gate leaked "
        "back onto the clean path (MEDIUM finding regressed)"
    )
    assert mem_type == "convention"


def test_medium_finding_force_push_retained_end_to_end():
    """Explicit pin of the exact MEDIUM-finding case through the single parser.

    'Nunca force push.' (17 chars, clean, no TYPE: wrapper) must round-trip
    intact — not be dropped by the min-payload guard.
    """
    r = _mk()
    res = r._parse_response("Nunca force push.", "convention")
    assert res is not None, "clean 17-char convention was dropped (MEDIUM regressed)"
    assert res.content == "Nunca force push."
    assert res.memory_type == "convention"


def test_leaked_wrapper_with_short_valid_insight_kept():
    """P1-b fix: Short but complete insight with valid type should be KEPT, not dropped.
    
    The old behavior dropped 'Nunca force push.' (17 chars) due to length guard,
    but this is a complete, meaningful convention that should be preserved.
    """
    content, mem_type = _unleak_type("TYPE: convention — Nunca force push.", "learning")
    assert content == "Nunca force push."  # P1-b: short but complete insight kept
    assert mem_type == "convention"  # Valid type extracted


def test_leaked_wrapper_meaningful_content_kept():
    """Content that looks truncated but is meaningful should be kept.
    
    'Usar subagent dedic' while short, conveys a meaningful convention.
    """
    content, mem_type = _unleak_type("TYPE: convention — Usar subagent dedic", "learning")
    assert content == "Usar subagent dedic"  # Meaningful content kept
    assert mem_type == "convention"


def test_leaked_bare_type_label_dropped():
    """'TYPE: bug' (label echo, no substance) must drop."""
    content, _ = _unleak_type("TYPE: bug", "learning")
    assert content is None


# ---------------------------------------------------------------------------
# (f) Case folding of the leaked label (TYPE / type / Type / mixed).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", ["TYPE", "type", "Type", "tYpE"])
def test_bare_label_any_case_dropped(label):
    content, _ = _unleak_type(label, "bug")
    assert content is None


@pytest.mark.parametrize("prefix", ["TYPE", "type", "Type", "tYpE"])
def test_leaked_wrapper_any_case_recovers(prefix):
    resp = f"{prefix}: convention — Nunca usar strReplace em append-only files."
    content, mem_type = _unleak_type(resp, "learning")
    assert mem_type == "convention"
    assert content == "Nunca usar strReplace em append-only files."


# ---------------------------------------------------------------------------
# Robustness: no catastrophic backtracking on pathological input.
# ---------------------------------------------------------------------------

def test_no_catastrophic_backtracking():
    big = "TYPE: " + ("a " * 100000)
    start = time.time()
    _unleak_type(big, "bug")
    assert time.time() - start < 1.0, "regex took too long — possible backtracking"


# ---------------------------------------------------------------------------
# Parser integration: degenerate drop propagates to None in both parsers.
# ---------------------------------------------------------------------------

def test_single_parser_keeps_short_meaningful_content():
    """P1-b fix: Short but meaningful content with valid type should be kept.""" 
    r = _mk()
    result = r._parse_response("TYPE: bug — curto.", "bug")
    assert result is not None
    assert result.content == "curto."
    assert result.memory_type == "bug"


def test_batch_parser_mixes_valid_and_degenerate():
    r = _mk()
    items = [{"content": "x", "memory_type": "context"}] * 3
    resp = (
        "1. TYPE: convention — Nunca usar strReplace em append-only files.\n"
        "2. TYPE: bug\n"
        "3. decision: Usar RRF sempre para melhor recall em hybrid search."
    )
    out = r._parse_batch_response(resp, items)
    assert out[0] is not None and out[0].memory_type == "convention"
    assert out[1] is None  # degenerate
    assert out[2] is not None and out[2].memory_type == "decision"
