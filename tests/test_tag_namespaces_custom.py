"""Tag namespaces the server itself writes, plus operator-registered ones.

The taxonomy used to list only sys/q/proj/topic/t/user/agent, while the server's own
code writes harvest:, source:, conflict: and confidence: tags, so every load of such a
memory reported its own tags as invalid. MCP_CUSTOM_TAG_NAMESPACES lets a deployment
register more, the way MCP_CUSTOM_MEMORY_TYPES does for memory types.
"""

import logging

import pytest

from mcp_memory_service.models import tag_taxonomy
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.models.tag_taxonomy import TagTaxonomy, parse_custom_namespaces


@pytest.mark.parametrize("ns", ["harvest:", "source:", "conflict:", "confidence:"])
def test_namespaces_the_server_writes_are_valid(ns):
    assert ns in TagTaxonomy.VALID_NAMESPACES
    assert tag_taxonomy.validate_tag(f"{ns}example")


@pytest.mark.parametrize("raw, expected", [
    (None, set()),
    ("", set()),
    ("project", {"project:"}),
    ("project:, Cleanup:", {"project:", "cleanup:"}),
    (" project ,, cleanup ", {"project:", "cleanup:"}),
])
def test_parse_custom_namespaces(raw, expected):
    assert parse_custom_namespaces(raw) == expected


@pytest.mark.parametrize("raw", ["bad name", "a:b", "x!"])
def test_parse_custom_namespaces_skips_malformed_entries(raw, caplog):
    with caplog.at_level(logging.WARNING):
        assert parse_custom_namespaces(f"project,{raw}") == {"project:"}
    assert "MCP_CUSTOM_TAG_NAMESPACES" in caplog.text


def test_register_custom_namespaces_extends_validation(monkeypatch):
    monkeypatch.setattr(TagTaxonomy, "VALID_NAMESPACES", set(TagTaxonomy.VALID_NAMESPACES))
    monkeypatch.setattr(tag_taxonomy, "VALID_NAMESPACES", list(tag_taxonomy.VALID_NAMESPACES))
    assert not tag_taxonomy.validate_tag("project:x")
    tag_taxonomy.register_custom_namespaces("project")
    assert tag_taxonomy.validate_tag("project:x")
    assert "project:" in TagTaxonomy.VALID_NAMESPACES


def test_invalid_tag_message_lists_every_valid_namespace(caplog):
    with caplog.at_level(logging.DEBUG, logger="mcp_memory_service.models.memory"):
        Memory(content="x", content_hash="h" * 64, tags=["nosuchns:value"])
    message = next(r.getMessage() for r in caplog.records if "invalid namespaces" in r.getMessage())
    for ns in TagTaxonomy.VALID_NAMESPACES:
        assert ns in message
