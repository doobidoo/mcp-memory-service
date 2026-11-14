"""Pure decision and config parsing for semantic-dedup supersede (#1938)."""

from types import SimpleNamespace

from mcp_memory_service.storage.semantic_dedup import (
    NEAR_IDENTICAL_DEFAULT,
    apply_semantic_dedup_config,
    normalize_dedup_text,
    semantic_dedup_action,
    verdict_for_match,
)
from tests.unit.dedup_fixtures import XERJ_UPDATE_PAIRS


def test_xerj_updates_are_inside_the_default_band():
    """The reproduced updates sit between 0.85 and 0.97, and are not the same text."""
    assert len(XERJ_UPDATE_PAIRS) == 4
    for old, new, cosine in XERJ_UPDATE_PAIRS:
        assert 0.85 <= cosine < NEAR_IDENTICAL_DEFAULT
        assert normalize_dedup_text(old) != normalize_dedup_text(new)
        assert semantic_dedup_action(
            "supersede", cosine, NEAR_IDENTICAL_DEFAULT, same_normalized_text=False
        ) == "supersede"
        assert semantic_dedup_action(
            "reject", cosine, NEAR_IDENTICAL_DEFAULT, same_normalized_text=False
        ) == "reject"


def test_near_identical_and_normalised_text_reject_in_both_modes():
    for mode in ("reject", "supersede"):
        assert semantic_dedup_action(
            mode, 0.99, 0.97, same_normalized_text=False
        ) == "reject"
        assert semantic_dedup_action(
            mode, 0.90, 0.97, same_normalized_text=True
        ) == "reject"
        assert semantic_dedup_action(
            mode, None, 0.97, same_normalized_text=False
        ) == "reject"


def test_whitespace_and_case_count_as_the_same_text():
    assert normalize_dedup_text("  Hello   World ") == normalize_dedup_text("hello world")


def test_verdict_supersedes_only_inside_the_band():
    verdict = verdict_for_match(
        mode="supersede",
        near_identical_threshold=0.97,
        is_duplicate=True,
        existing_hash="abc",
        similarity=0.90,
        matched_content="old fact",
        new_content="new fact",
    )
    assert verdict.supersede_hash == "abc"
    assert verdict.reject_message is None

    rejected = verdict_for_match(
        mode="supersede",
        near_identical_threshold=0.97,
        is_duplicate=True,
        existing_hash="abc",
        similarity=0.99,
        matched_content="old fact",
        new_content="new fact",
    )
    assert rejected.supersede_hash is None
    assert rejected.reject_message is not None
    assert rejected.reject_message.outcome == "deduplicated"
    assert rejected.reject_message.matched_hash == "abc"
    assert "semantically similar to abc" in rejected.reject_message


def test_no_hit_stores_normally():
    verdict = verdict_for_match(
        mode="supersede",
        near_identical_threshold=0.97,
        is_duplicate=False,
        existing_hash=None,
        similarity=None,
        matched_content=None,
        new_content="anything",
    )
    assert verdict.reject_message is None
    assert verdict.supersede_hash is None


def test_config_defaults_and_invalid_values(monkeypatch, caplog):
    monkeypatch.delenv("MCP_SEMANTIC_DEDUP_MODE", raising=False)
    monkeypatch.delenv("MCP_SEMANTIC_DEDUP_NEAR_IDENTICAL_THRESHOLD", raising=False)
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "true")
    target = SimpleNamespace()
    apply_semantic_dedup_config(target)
    assert target.semantic_dedup_enabled is True
    assert target.semantic_dedup_mode == "reject"
    assert target.semantic_dedup_near_identical_threshold == NEAR_IDENTICAL_DEFAULT
    assert target.semantic_dedup_threshold == 0.85

    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_MODE", "Supersede")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_NEAR_IDENTICAL_THRESHOLD", "0.96")
    apply_semantic_dedup_config(target)
    assert target.semantic_dedup_mode == "supersede"
    assert target.semantic_dedup_near_identical_threshold == 0.96

    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_MODE", "merge")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_NEAR_IDENTICAL_THRESHOLD", "nope")
    apply_semantic_dedup_config(target)
    assert target.semantic_dedup_mode == "reject"
    assert target.semantic_dedup_near_identical_threshold == NEAR_IDENTICAL_DEFAULT
    assert "Invalid MCP_SEMANTIC_DEDUP_MODE" in caplog.text
    assert "NEAR_IDENTICAL_THRESHOLD" in caplog.text


def test_cutoff_below_threshold_warns_but_is_kept(monkeypatch, caplog):
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_THRESHOLD", "0.99")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_NEAR_IDENTICAL_THRESHOLD", "0.90")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_MODE", "supersede")
    target = SimpleNamespace()
    apply_semantic_dedup_config(target)
    assert target.semantic_dedup_near_identical_threshold == 0.90
    assert "supersede band is empty" in caplog.text
    # Every match is at or above 0.99, which is past 0.90, so it still rejects.
    assert semantic_dedup_action("supersede", 0.99, 0.90, same_normalized_text=False) == "reject"
