"""Compression must not detach a rule from its qualifications."""

import pytest
from datetime import datetime

from mcp_memory_service.consolidation.base import MemoryCluster
from mcp_memory_service.consolidation.compression import SemanticCompressionEngine
from mcp_memory_service.models.memory import Memory


@pytest.mark.unit
@pytest.mark.parametrize("content", [
    "Do not start work on an issue assigned to someone else. "
    "This rule does not apply to umbrella issues when the chosen file has not been claimed.",
    "Do not start work on an issue assigned to someone else. Not always.",
    "Do not start work on an issue assigned to someone else.\n\n"
    "Exception: umbrella issues allow separate unclaimed files.",
    "Keine Arbeit an bereits zugewiesenen Issues beginnen. "
    "Ausgenommen sind freie Dateien in Sammelaufgaben.",
])
async def test_summary_keeps_selected_memory_intact(consolidation_config, content):
    engine = SemanticCompressionEngine(consolidation_config)
    memory = Memory(content=content, content_hash="qualified-rule")

    summary = await engine._generate_thematic_summary([memory], ["issue", "work"])

    assert content in summary
    assert len(summary) <= engine.max_summary_length


@pytest.mark.unit
async def test_summary_omits_whole_rule_when_qualification_exceeds_budget(consolidation_config):
    engine = SemanticCompressionEngine(consolidation_config)
    rule = "Do not start work on an issue assigned to someone else."
    content = rule + " Exception: " + "unclaimed files in umbrella issues remain available; " * 12
    memory = Memory(content=content, content_hash="long-qualified-rule")

    summary = await engine._generate_thematic_summary([memory], ["issue", "work"])

    assert rule.rstrip(".") not in summary
    assert len(summary) <= engine.max_summary_length


@pytest.mark.unit
@pytest.mark.parametrize("too_long", [False, True], ids=["intact", "over-budget"])
async def test_process_preserves_scope_or_skips_cluster(consolidation_config, too_long):
    engine = SemanticCompressionEngine(consolidation_config)
    content = "Do not start work on an issue assigned to someone else. Not always."
    if too_long:
        content += " Unclaimed files in umbrella issues remain available." * 12
    memories = [Memory(content=content, content_hash=f"rule-{i}") for i in range(2)]
    cluster = MemoryCluster(
        cluster_id="qualified-rules",
        memory_hashes=[memory.content_hash for memory in memories],
        centroid_embedding=[0.1] * 3,
        coherence_score=0.8,
        created_at=datetime.now(),
        theme_keywords=["issue", "work"],
    )

    results = await engine.process([cluster], memories)

    if too_long:
        assert results == []
    else:
        assert len(results) == 1
        assert content in results[0].compressed_memory.content
        assert len(results[0].compressed_memory.content) <= engine.max_summary_length
