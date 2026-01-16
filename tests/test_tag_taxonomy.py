"""
Unit tests for Tag Taxonomy with Namespaces

Tests the namespace-based tag organization system.
"""

import importlib.util
from pathlib import Path

# Load tag_taxonomy module directly without importing the package
tag_taxonomy_path = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "models" / "tag_taxonomy.py"
spec = importlib.util.spec_from_file_location("tag_taxonomy", tag_taxonomy_path)
tag_taxonomy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tag_taxonomy)

NAMESPACE_SYSTEM = tag_taxonomy.NAMESPACE_SYSTEM
NAMESPACE_QUALITY = tag_taxonomy.NAMESPACE_QUALITY
NAMESPACE_PROJECT = tag_taxonomy.NAMESPACE_PROJECT
NAMESPACE_TOPIC = tag_taxonomy.NAMESPACE_TOPIC
NAMESPACE_TEMPORAL = tag_taxonomy.NAMESPACE_TEMPORAL
NAMESPACE_USER = tag_taxonomy.NAMESPACE_USER
parse_tag = tag_taxonomy.parse_tag


class TestBurst21NamespaceConstants:
    """Tests for Burst 2.1: Namespace Constants"""

    def test_six_namespaces_defined(self):
        """Should have exactly 6 namespace constants"""
        namespaces = [
            NAMESPACE_SYSTEM,
            NAMESPACE_QUALITY,
            NAMESPACE_PROJECT,
            NAMESPACE_TOPIC,
            NAMESPACE_TEMPORAL,
            NAMESPACE_USER
        ]
        assert len(namespaces) == 6

    def test_each_namespace_ends_with_colon(self):
        """All namespace constants should end with ':' for easy concatenation"""
        namespaces = [
            NAMESPACE_SYSTEM,
            NAMESPACE_QUALITY,
            NAMESPACE_PROJECT,
            NAMESPACE_TOPIC,
            NAMESPACE_TEMPORAL,
            NAMESPACE_USER
        ]
        for namespace in namespaces:
            assert namespace.endswith(":"), f"Namespace '{namespace}' should end with ':'"

    def test_namespaces_have_correct_values(self):
        """Namespace constants should have expected values"""
        assert NAMESPACE_SYSTEM == "sys:"
        assert NAMESPACE_QUALITY == "q:"
        assert NAMESPACE_PROJECT == "proj:"
        assert NAMESPACE_TOPIC == "topic:"
        assert NAMESPACE_TEMPORAL == "t:"
        assert NAMESPACE_USER == "user:"

    def test_namespaces_are_unique(self):
        """No two namespaces should have the same value"""
        namespaces = [
            NAMESPACE_SYSTEM,
            NAMESPACE_QUALITY,
            NAMESPACE_PROJECT,
            NAMESPACE_TOPIC,
            NAMESPACE_TEMPORAL,
            NAMESPACE_USER
        ]
        assert len(namespaces) == len(set(namespaces))


class TestBurst22ParseTag:
    """Tests for Burst 2.2: Parse Tag Function"""

    def test_parse_namespaced_tag(self):
        """Should parse namespaced tags into (namespace, value)"""
        namespace, value = parse_tag("q:high")
        assert namespace == "q:"
        assert value == "high"

    def test_parse_legacy_tag_without_namespace(self):
        """Legacy tags without namespace should return (None, tag)"""
        namespace, value = parse_tag("legacy-tag")
        assert namespace is None
        assert value == "legacy-tag"

    def test_parse_handles_multiple_colons(self):
        """Should split on first colon only"""
        namespace, value = parse_tag("topic:auth:oauth2")
        assert namespace == "topic:"
        assert value == "auth:oauth2"

    def test_parse_various_namespaces(self):
        """Should correctly parse different namespace formats"""
        assert parse_tag("proj:mcp-memory") == ("proj:", "mcp-memory")
        assert parse_tag("t:2024-01") == ("t:", "2024-01")
        assert parse_tag("sys:auto-generated") == ("sys:", "auto-generated")
