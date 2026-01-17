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
validate_tag = tag_taxonomy.validate_tag
add_namespace = tag_taxonomy.add_namespace
filter_by_namespace = tag_taxonomy.filter_by_namespace
TagTaxonomy = tag_taxonomy.TagTaxonomy


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


class TestBurst23ValidateTag:
    """Tests for Burst 2.3: Validate Tag Function"""

    def test_valid_namespaced_tags(self):
        """Tags with valid namespaces should return True"""
        assert validate_tag("q:high") is True
        assert validate_tag("proj:mcp-memory") is True
        assert validate_tag("topic:authentication") is True
        assert validate_tag("sys:auto") is True

    def test_legacy_tags_validate_correctly(self):
        """Legacy tags without namespace should return True (backward compat)"""
        assert validate_tag("legacy-tag") is True
        assert validate_tag("important") is True
        assert validate_tag("mcp-memory-service") is True

    def test_invalid_namespace_returns_false(self):
        """Tags with invalid namespaces should return False"""
        assert validate_tag("invalid:tag") is False
        assert validate_tag("bad:value") is False
        assert validate_tag("unknown:namespace") is False


class TestBurst24AddNamespace:
    """Tests for Burst 2.4: Add Namespace Function"""

    def test_add_namespace_to_value(self):
        """Should add namespace prefix to value"""
        assert add_namespace("high", "q:") == "q:high"
        assert add_namespace("mcp-memory", "proj:") == "proj:mcp-memory"

    def test_strips_existing_namespace(self):
        """Should strip existing namespace before adding new one"""
        assert add_namespace("q:high", "proj:") == "proj:high"
        assert add_namespace("topic:auth", "sys:") == "sys:auth"


class TestBurst25FilterTagsByNamespace:
    """Tests for Burst 2.5: Filter Tags by Namespace"""

    def test_filters_quality_tags(self):
        """Should filter only quality namespace tags"""
        tags = ["q:high", "proj:auth", "q:medium", "legacy"]
        result = filter_by_namespace(tags, "q:")
        assert result == ["q:high", "q:medium"]

    def test_empty_list_for_non_matching_namespace(self):
        """Should return empty list when no tags match"""
        tags = ["proj:auth", "legacy"]
        result = filter_by_namespace(tags, "q:")
        assert result == []

    def test_filters_correctly_with_legacy_tags(self):
        """Should exclude legacy tags from namespace filter"""
        tags = ["legacy", "q:high"]
        result = filter_by_namespace(tags, "q:")
        assert result == ["q:high"]


class TestBurst26TagTaxonomyClass:
    """Tests for Burst 2.6: Tag Taxonomy Class"""

    def test_class_methods_accessible(self):
        """All methods should be accessible via class"""
        assert hasattr(TagTaxonomy, 'parse_tag')
        assert hasattr(TagTaxonomy, 'validate_tag')
        assert hasattr(TagTaxonomy, 'add_namespace')
        assert hasattr(TagTaxonomy, 'filter_by_namespace')

    def test_valid_namespaces_attribute_exists(self):
        """VALID_NAMESPACES class attribute should be exposed for efficient access"""
        assert hasattr(TagTaxonomy, 'VALID_NAMESPACES')
        assert isinstance(TagTaxonomy.VALID_NAMESPACES, set)
        assert len(TagTaxonomy.VALID_NAMESPACES) == 6

    def test_valid_namespaces_contains_all_namespaces(self):
        """VALID_NAMESPACES should contain all 6 namespace constants"""
        expected_namespaces = {
            NAMESPACE_SYSTEM,
            NAMESPACE_QUALITY,
            NAMESPACE_PROJECT,
            NAMESPACE_TOPIC,
            NAMESPACE_TEMPORAL,
            NAMESPACE_USER
        }
        assert TagTaxonomy.VALID_NAMESPACES == expected_namespaces

    def test_parse_tag_via_class(self):
        """parse_tag should work via class method"""
        namespace, value = TagTaxonomy.parse_tag("q:high")
        assert namespace == "q:"
        assert value == "high"

    def test_validate_tag_via_class(self):
        """validate_tag should work via class method"""
        assert TagTaxonomy.validate_tag("q:high") is True
        assert TagTaxonomy.validate_tag("invalid:tag") is False

    def test_add_namespace_via_class(self):
        """add_namespace should work via class method"""
        assert TagTaxonomy.add_namespace("high", "q:") == "q:high"

    def test_filter_by_namespace_via_class(self):
        """filter_by_namespace should work via class method"""
        tags = ["q:high", "proj:auth", "q:medium"]
        result = TagTaxonomy.filter_by_namespace(tags, "q:")
        assert result == ["q:high", "q:medium"]
