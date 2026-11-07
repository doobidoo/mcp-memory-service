"""Tag namespaces the server itself writes, plus operator-registered ones.

The taxonomy used to list only sys/q/proj/topic/t/user/agent, while the server's own
code writes harvest:, source:, conflict: and confidence: tags, so every load of such a
memory reported its own tags as invalid. MCP_CUSTOM_TAG_NAMESPACES lets a deployment
register more, the way MCP_CUSTOM_MEMORY_TYPES does for memory types.
"""

import logging
import os

import pytest
from dotenv import load_dotenv

from mcp_memory_service.models import tag_taxonomy
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.models.tag_taxonomy import TagTaxonomy, parse_custom_namespaces


@pytest.fixture
def clean_registry(monkeypatch):
    """Built-in namespaces only, nothing from the host environment, env not yet read."""
    builtin = [ns for ns in tag_taxonomy.VALID_NAMESPACES if ns not in ("project:", "cleanup:")]
    monkeypatch.setattr(tag_taxonomy, "VALID_NAMESPACES", list(builtin))
    monkeypatch.setattr(TagTaxonomy, "VALID_NAMESPACES", set(builtin))
    monkeypatch.setattr(tag_taxonomy, "_env_namespaces_registered", False)
    monkeypatch.delenv(tag_taxonomy.CUSTOM_NAMESPACES_ENV, raising=False)


@pytest.mark.parametrize("ns", ["harvest:", "source:", "conflict:", "confidence:"])
def test_namespaces_the_server_writes_are_valid(ns, clean_registry):
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


def test_register_custom_namespaces_extends_validation(clean_registry):
    assert not tag_taxonomy.validate_tag("project:x")
    tag_taxonomy.register_custom_namespaces("project")
    assert tag_taxonomy.validate_tag("project:x")
    assert "project:" in TagTaxonomy.VALID_NAMESPACES


def test_env_value_loaded_after_import_is_honoured(clean_registry, monkeypatch, tmp_path, caplog):
    """The .env startup path: config.base calls load_dotenv after this module is imported."""
    env_file = tmp_path / ".env"
    env_file.write_text("MCP_CUSTOM_TAG_NAMESPACES=project,cleanup\n")
    try:
        load_dotenv(env_file, override=False)
        with caplog.at_level(logging.DEBUG, logger="mcp_memory_service.models.memory"):
            Memory(content="x", content_hash="h" * 64, tags=["project:alpha", "cleanup:kept"])
    finally:
        os.environ.pop(tag_taxonomy.CUSTOM_NAMESPACES_ENV, None)  # load_dotenv wrote it
    assert "invalid namespaces" not in caplog.text
    assert TagTaxonomy.is_valid_namespace("project:")


def test_invalid_tag_message_lists_every_valid_namespace(clean_registry, caplog):
    with caplog.at_level(logging.DEBUG, logger="mcp_memory_service.models.memory"):
        Memory(content="x", content_hash="h" * 64, tags=["nosuchns:value"])
    message = next(r.getMessage() for r in caplog.records if "invalid namespaces" in r.getMessage())
    for ns in TagTaxonomy.VALID_NAMESPACES:
        assert ns in message
