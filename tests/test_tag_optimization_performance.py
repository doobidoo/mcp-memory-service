"""
Performance test to demonstrate tag validation optimization.

Shows the efficiency gain from parsing tags only once instead of twice.
"""

import importlib.util
from pathlib import Path
import time

# Load tag_taxonomy module directly (standalone module, no dependencies)
tag_taxonomy_path = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "models" / "tag_taxonomy.py"
spec = importlib.util.spec_from_file_location("tag_taxonomy", tag_taxonomy_path)
tag_taxonomy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tag_taxonomy)

TagTaxonomy = tag_taxonomy.TagTaxonomy


class TestTagValidationOptimization:
    """Test that demonstrates the tag validation optimization"""

    def test_valid_namespaces_is_set_for_fast_lookup(self):
        """VALID_NAMESPACES should be a set for O(1) lookup performance"""
        # Sets provide O(1) membership testing vs O(n) for lists
        assert isinstance(TagTaxonomy.VALID_NAMESPACES, set)

        # Verify all namespaces are present
        assert len(TagTaxonomy.VALID_NAMESPACES) == 6

        # Verify fast membership testing works
        assert "q:" in TagTaxonomy.VALID_NAMESPACES
        assert "invalid:" not in TagTaxonomy.VALID_NAMESPACES

    def test_parse_then_validate_vs_direct_check(self):
        """Compare parse+validate (old) vs parse+set-check (new)"""
        # Test that both approaches give same results
        test_tags = [
            ("q:high", True),           # Valid namespaced tag
            ("proj:mcp-memory", True),  # Valid namespaced tag
            ("legacy-tag", True),       # Valid legacy tag (no namespace)
            ("invalid:tag", False)      # Invalid namespace
        ]

        for tag, expected_valid in test_tags:
            # Method 1: Parse then validate (old approach - parses twice)
            namespace, value = TagTaxonomy.parse_tag(tag)
            if namespace is None:
                result_old = True  # Legacy tags are valid
            else:
                result_old = TagTaxonomy.validate_tag(tag)  # Parses again!

            # Method 2: Parse then direct set check (new approach - parses once)
            namespace, value = TagTaxonomy.parse_tag(tag)
            if namespace is None:
                result_new = True  # Legacy tags are valid
            else:
                result_new = namespace in TagTaxonomy.VALID_NAMESPACES

            # Both methods should give same result
            assert result_old == result_new == expected_valid, \
                f"Tag '{tag}': expected {expected_valid}, got old={result_old}, new={result_new}"

    def test_optimization_logic_correctness(self):
        """Verify that optimized validation produces same results as original"""
        # Test case 1: Valid namespaced tags
        namespace, value = TagTaxonomy.parse_tag("q:high")
        assert namespace == "q:"

        # Original approach: validate_tag() which internally calls parse_tag() again
        # Optimized approach: direct membership check
        assert (namespace in TagTaxonomy.VALID_NAMESPACES) == True
        assert TagTaxonomy.validate_tag("q:high") == True

        # Test case 2: Invalid namespaced tag
        namespace, value = TagTaxonomy.parse_tag("invalid:tag")
        assert namespace == "invalid:"
        assert (namespace in TagTaxonomy.VALID_NAMESPACES) == False
        assert TagTaxonomy.validate_tag("invalid:tag") == False

        # Test case 3: Legacy tag (no namespace)
        namespace, value = TagTaxonomy.parse_tag("legacy-tag")
        assert namespace is None
        # Legacy tags should pass validation (backward compatibility)
        assert TagTaxonomy.validate_tag("legacy-tag") == True

    def test_performance_benchmark(self):
        """Benchmark to show optimization effect (informational, not a strict test)"""
        # Create a large set of tags to validate
        test_tags = [
            "q:high", "q:medium", "q:low",
            "proj:mcp-memory", "proj:auth", "proj:api",
            "topic:database", "topic:performance",
            "sys:auto", "t:2024-01",
            "user:important", "user:review",
            "legacy-tag-1", "legacy-tag-2",
            "invalid:tag1", "invalid:tag2"
        ] * 100  # Repeat 100 times for measurable time

        # Method 1: Original approach (parse + validate which parses again)
        start = time.perf_counter()
        invalid_count_original = 0
        for tag in test_tags:
            namespace, value = TagTaxonomy.parse_tag(tag)
            if namespace is not None:
                if not TagTaxonomy.validate_tag(tag):  # Parses again internally!
                    invalid_count_original += 1
        time_original = time.perf_counter() - start

        # Method 2: Optimized approach (parse once, direct membership check)
        start = time.perf_counter()
        invalid_count_optimized = 0
        for tag in test_tags:
            namespace, value = TagTaxonomy.parse_tag(tag)
            if namespace is not None:
                if namespace not in TagTaxonomy.VALID_NAMESPACES:  # No re-parsing!
                    invalid_count_optimized += 1
        time_optimized = time.perf_counter() - start

        # Both methods should find same number of invalid tags
        assert invalid_count_original == invalid_count_optimized

        # Optimized version should be faster (though timing can vary)
        # Just log the results for information
        print(f"\nPerformance comparison for {len(test_tags)} tag validations:")
        print(f"  Original (parse + validate):  {time_original*1000:.3f} ms")
        print(f"  Optimized (parse + set check): {time_optimized*1000:.3f} ms")
        if time_original > 0:
            speedup = (time_original - time_optimized) / time_original * 100
            print(f"  Improvement: {speedup:.1f}% faster")
        print(f"  Invalid tags found: {invalid_count_optimized}")
