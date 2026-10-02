"""Adversarial tests for the Greptile P1 fixes in ``_unleak_type``.

Gate-5 independent review (Tuvok). These attack the NEW dividing line
introduced when the dev removed the char-count truncation guard: the
candidate is now kept/dropped based on "is there prose after the type?"
rather than "how many chars?". We probe the prose-vs-label boundary and
the ``<type>:`` placeholder handling in both parsers.

Accepted tradeoff (manager decision): a real truncated fragment like
``TYPE: convention — Usar subagent dedic`` is now KEPT. These tests pin
that intentional behavior AND prove the degenerate-without-prose cases
still drop to None.
"""
from mcp_memory_service.harvest.rewriter import _unleak_type, VALID_TYPES


def _mk():
    from mcp_memory_service.harvest.rewriter import HarvestRewriter
    return HarvestRewriter.__new__(HarvestRewriter)


# ---------------------------------------------------------------------------
# Dividing line: label-only (no prose) must still drop to None.
# ---------------------------------------------------------------------------

def test_type_colon_only_spaces_drops():
    """``TYPE:`` followed by only whitespace is a bare label → None."""
    content, mem_type = _unleak_type("TYPE:     ", "learning")
    assert content is None
    assert mem_type == "learning"


def test_valid_type_with_separator_but_empty_text_drops():
    """``TYPE: bug — `` (valid type, separator, no payload) → None."""
    content, _ = _unleak_type("TYPE: bug — ", "learning")
    assert content is None


def test_bare_valid_type_drops():
    """``TYPE: bug`` (valid type, no prose) → None."""
    content, _ = _unleak_type("TYPE: bug", "learning")
    assert content is None


def test_bare_labels_drop():
    for label in ("TYPE", "<type>", "<TYPE>", "<Type>", "TYPE:", "<type>:"):
        content, _ = _unleak_type(label, "learning")
        assert content is None, f"{label!r} should drop to None"


# ---------------------------------------------------------------------------
# Dividing line: prose present → keep intact (P1-a / P1-b semantics).
# ---------------------------------------------------------------------------

def test_prose_after_type_keeps_first_word():
    """``TYPE: Nunca usar force push`` must NOT lose 'Nunca' (P1-a)."""
    content, mem_type = _unleak_type("TYPE: Nunca usar force push", "convention")
    assert content == "Nunca usar force push"
    assert mem_type == "convention"


def test_short_valid_insight_kept():
    """``TYPE: convention — Nunca force push.`` (17 chars) kept (P1-b)."""
    content, mem_type = _unleak_type("TYPE: convention — Nunca force push.", "learning")
    assert content == "Nunca force push."
    assert mem_type == "convention"


def test_accepted_truncated_fragment_now_kept():
    """Accepted tradeoff: a real truncated fragment is KEPT (char guard removed)."""
    content, mem_type = _unleak_type("TYPE: convention — Usar subagent dedic", "learning")
    assert content == "Usar subagent dedic"
    assert mem_type == "convention"


# ---------------------------------------------------------------------------
# Case / angle-bracket variants of the placeholder.
# ---------------------------------------------------------------------------

def test_uppercase_angle_type_unwrapped():
    """``<TYPE>:`` uppercase angle placeholder unwraps like ``<type>:``."""
    content, mem_type = _unleak_type("<TYPE>: convention — texto aqui com substancia", "learning")
    assert content == "texto aqui com substancia"
    assert mem_type == "convention"


def test_lowercase_type_colon_treated_as_leak():
    """``type:`` lowercase+colon is treated as a leak (regex is IGNORECASE).

    Documents current behavior: a colon after the bare word 'type' triggers
    unwrapping. This is the intended guard ('no colon' = prose, 'colon' = leak).
    """
    content, mem_type = _unleak_type("type: convention — texto aqui com substancia", "learning")
    assert content == "texto aqui com substancia"
    assert mem_type == "convention"


# ---------------------------------------------------------------------------
# Prose that merely starts with a type-word (no colon) must be untouched.
# ---------------------------------------------------------------------------

def test_prose_starting_with_valid_type_word_no_colon_untouched():
    """``convention is a good idea here`` (valid-type word, no colon) → untouched."""
    content, mem_type = _unleak_type("convention is a good idea here", "learning")
    assert content == "convention is a good idea here"
    assert mem_type == "learning"


def test_typescript_colon_not_a_leak():
    """``typescript: ...`` must NOT be unwrapped — token is not the word 'type'."""
    text = "typescript: a superset of javascript with static types"
    content, mem_type = _unleak_type(text, "learning")
    assert content == text
    assert mem_type == "learning"


def test_type_word_sentence_with_colon_is_stripped_known_limitation():
    """KNOWN LIMITATION: legit prose beginning 'Type: ...' is stripped.

    The IGNORECASE regex cannot distinguish a leaked placeholder from a
    sentence that genuinely opens with 'Type:'. Pinned to document the
    (pre-existing) false-positive surface, not to bless it.
    """
    content, _ = _unleak_type("Type: the return value is cached for performance", "learning")
    # Current behavior strips the 'Type:' prefix.
    assert content == "the return value is cached for performance"


# ---------------------------------------------------------------------------
# Multi-line payloads.
# ---------------------------------------------------------------------------

def test_multiline_payload_kept():
    """Leaked wrapper over a multi-line body keeps the whole body (DOTALL)."""
    text = "TYPE: decision\nmultiline body with real substance across lines"
    content, mem_type = _unleak_type(text, "learning")
    assert "multiline body with real substance across lines" in content
    assert mem_type == "decision"


# ---------------------------------------------------------------------------
# Parser integration for <type>: in BOTH parsers (P1-c).
# ---------------------------------------------------------------------------

def test_single_parser_unwraps_angle_type():
    r = _mk()
    res = r._parse_response("<type>: convention — Nunca force push em protegidos", "learning")
    assert res is not None
    assert res.memory_type == "convention"
    assert not res.content.startswith("<type>")


def test_batch_parser_unwraps_angle_type():
    r = _mk()
    items = [{"content": "x", "memory_type": "learning"}]
    out = r._parse_batch_response("1. <type>: convention — Usar subagent dedicado", items)
    assert out[0] is not None
    assert out[0].memory_type == "convention"
    assert not out[0].content.startswith("<type>")


def test_batch_parser_drops_angle_degenerate():
    r = _mk()
    items = [{"content": "x", "memory_type": "learning"}]
    out = r._parse_batch_response("1. <type>: bug", items)
    assert out[0] is None


# ---------------------------------------------------------------------------
# P1-d: Single unknown word after TYPE:/`<type>:` should be dropped (labels)
# Multi-word should be kept (prose). This is the dividing line for invalid types.
# ---------------------------------------------------------------------------

def test_single_unknown_word_type_dropped():
    """P1-d: ``TYPE: frobnicate`` (single unknown word) → None (not a valid type, not prose)."""
    content, mem_type = _unleak_type("TYPE: frobnicate", "learning")
    assert content is None
    assert mem_type == "learning"


def test_single_unknown_word_angle_type_dropped():
    """P1-d: ``<type>: frobnicate`` (single unknown word) → None."""
    content, mem_type = _unleak_type("<type>: frobnicate", "learning")
    assert content is None
    assert mem_type == "learning"


def test_single_word_with_punctuation_dropped():
    """P1-d: ``TYPE: whatever.`` (single word + punctuation) → None (still a label)."""
    content, mem_type = _unleak_type("TYPE: whatever.", "learning")
    assert content is None
    assert mem_type == "learning"


def test_p1a_regression_protection_multiword_prose_kept():
    """P1-d: Ensure P1-a doesn't regress - ``TYPE: Nunca usar force push`` → kept (multi-word prose)."""
    content, mem_type = _unleak_type("TYPE: Nunca usar force push", "learning")
    assert content == "Nunca usar force push"
    assert mem_type == "learning"


def test_multiword_invalid_type_kept_as_prose():
    """P1-d: ``TYPE: frobnicate the whole thing`` (multi-word with invalid type) → kept as prose."""
    content, mem_type = _unleak_type("TYPE: frobnicate the whole thing", "learning")
    assert content == "frobnicate the whole thing"
    assert mem_type == "learning"
