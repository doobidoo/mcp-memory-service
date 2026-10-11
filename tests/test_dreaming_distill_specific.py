"""
RED tests for DREAMING-LOOP Inc 1 — distill the recent SPECIFIC
(rfc-dreaming-loop-inc1-distill-specific.md).

The live pain (CLIENT, 10/out): a specific fix stored today never becomes an
injectable belief because of 4 cascading barriers in belief_service.py:
  1. derive_beliefs() reads only memory_type="observation" (:131)
  2. _NOISE_PREFIXES drops [CHECKPOINT]/[SESSÃO]
  3. _is_noise drops >500 chars / bullet-ratio>0.6
  4. PROVENANCE_FLOOR=2 — a one-time fix never promotes to 'active'

These tests are RED until Inc 1 lands:
  - R1: configurable intake types (observation+decision+learning+error)
  - R3: a single actionable observation becomes a "singular" active belief
  - R4: pure checkpoints are still dropped (regression guard)

Gated by MCP_BELIEF_INTAKE_SINGULAR (opt-in). Implementation of the "singular"
belief mechanism needs Claudio sign-off (touches derive_confidence core / live 235
beliefs) — these tests pin the contract first.
"""

import pytest

from mcp_memory_service.consolidation.belief_service import _is_noise


# --- R2/R4: noise filter contract (pure function, no storage needed) ----------

def test_pure_checkpoint_still_noise():
    """R4 regression: a pure checkpoint must STAY noise even after Inc 1."""
    assert _is_noise("[CHECKPOINT] Sessão sirdata 10/out\n\nFEITO:\n- x\n- y") is True


def test_short_fragment_still_noise():
    """Unchanged: too-short fragments remain noise."""
    assert _is_noise("TYPE:") is True


def test_long_specific_fix_is_dropped_today_RED():
    """RED: a detailed specific fix (>500 chars, actionable) is dropped by the
    length rule today. Inc 1 (R2) must let an actionable memory through.

    This asserts the DESIRED post-Inc-1 behavior via a helper that does not exist
    yet, so it is RED until implemented."""
    from mcp_memory_service.consolidation.belief_service import is_actionable_noise  # noqa: F401

    long_fix = (
        "Fix delta-sync deadlock: quando o content do evento vem vazio e o payload "
        "diverge, o pull travava em fail-stop. A correção torna o evento skippable: "
        "o resolver detecta content-vazio, registra tombstone e segue. " * 3
    )
    # Actionable (type=learning) must NOT be classified as noise by length alone (R2).
    assert is_actionable_noise(long_fix, memory_type="learning") is False


def test_intake_types_includes_actionable_RED():
    """RED: the configurable intake-type set (R1) must include decision/learning/error
    by default. The accessor does not exist yet."""
    from mcp_memory_service.consolidation.belief_service import get_belief_intake_types  # noqa: F401

    types = set(get_belief_intake_types())
    assert "observation" in types
    assert {"decision", "learning", "error"} & types, "actionable types must be intake-eligible"
