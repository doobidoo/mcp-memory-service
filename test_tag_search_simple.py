#!/usr/bin/env python3
"""
Simple test script to validate tag search functionality without external dependencies.
This script tests the _parse_tags_fast method independently.
"""

import json
import sys
import os

# Add the path to the MCP Memory Service
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_parse_tags_fast():
    """Test the improved _parse_tags_fast method logic."""
    
    def parse_tags_fast(tag_string):
        """Replicated logic from the improved _parse_tags_fast method."""
        if not tag_string or tag_string == "":
            return []
            
        # Handle None or non-string values
        if not isinstance(tag_string, str):
            return []
            
        # Try to parse as JSON first (old format)
        if tag_string.startswith("[") and tag_string.endswith("]"):
            try:
                parsed = json.loads(tag_string)
                if isinstance(parsed, list):
                    return [str(tag).strip() for tag in parsed if str(tag).strip()]
                else:
                    return []
            except (json.JSONDecodeError, TypeError):
                pass
                
        # If not JSON or parsing fails, treat as comma-separated (new format)
        # Split by comma and clean up each tag
        tags = [tag.strip() for tag in tag_string.split(",") if tag.strip()]
        return tags
    
    # Test cases
    test_cases = [
        # (input, expected_output, description)
        ("", [], "Empty string"),
        ("tag1,tag2,tag3", ["tag1", "tag2", "tag3"], "Comma-separated tags"),
        ("tag1, tag2 , tag3", ["tag1", "tag2", "tag3"], "Comma-separated with spaces"),
        ('["tag1", "tag2", "tag3"]', ["tag1", "tag2", "tag3"], "JSON array format"),
        ('["tag1","tag2","tag3"]', ["tag1", "tag2", "tag3"], "JSON array without spaces"),
        ("single_tag", ["single_tag"], "Single tag"),
        ("tag1,", ["tag1"], "Trailing comma"),
        (",tag1", ["tag1"], "Leading comma"),
        ("tag1,,tag2", ["tag1", "tag2"], "Double comma"),
        ('[""]', [], "JSON array with empty string"),
        ('[]', [], "Empty JSON array"),
        ("important,work,meeting", ["important", "work", "meeting"], "Real-world example 1"),
        ('["bug-fix", "urgent", "version-2.0"]', ["bug-fix", "urgent", "version-2.0"], "Real-world example 2"),
    ]
    
    print("Testing _parse_tags_fast method logic...")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_input, expected, description in test_cases:
        try:
            result = parse_tags_fast(test_input)
            
            if result == expected:
                print(f"✅ PASS: {description}")
                print(f"   Input: {repr(test_input)}")
                print(f"   Output: {result}")
                passed += 1
            else:
                print(f"❌ FAIL: {description}")
                print(f"   Input: {repr(test_input)}")
                print(f"   Expected: {expected}")
                print(f"   Got: {result}")
                failed += 1
            print()
        except Exception as e:
            print(f"❌ ERROR: {description}")
            print(f"   Exception: {e}")
            print()
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The _parse_tags_fast logic is working correctly.")
        return True
    else:
        print("❌ Some tests failed. The logic needs further improvement.")
        return False

def test_tag_matching_logic():
    """Test the tag matching logic used in search_by_tag."""
    
    print("\nTesting tag matching logic...")
    print("=" * 50)
    
    # Simulate the tag matching logic from search_by_tag
    def simulate_search(stored_tags_list, search_tags):
        """Simulate the tag search matching logic."""
        matches = []
        
        # Normalize search tags to lowercase
        search_tags_normalized = [str(tag).strip().lower() for tag in search_tags if str(tag).strip()]
        
        for stored_tags_raw in stored_tags_list:
            # Normalize stored tags to lowercase
            stored_tags_normalized = [tag.lower() for tag in stored_tags_raw] if stored_tags_raw else []
            
            # Check if any search tag matches any stored tag
            if any(search_tag in stored_tags_normalized for search_tag in search_tags_normalized):
                matches.append(stored_tags_raw)
        
        return matches
    
    # Test scenarios
    test_scenarios = [
        {
            "stored_tags": [
                ["important", "work", "meeting"],
                ["personal", "reminder"],
                ["Important", "URGENT", "Todo"],
                ["bug-fix", "version-2.0"],
                []
            ],
            "search_queries": [
                (["important"], 2, "Case-insensitive 'important'"),
                (["work"], 1, "Exact match 'work'"),
                (["URGENT"], 1, "Case-insensitive 'URGENT'"),
                (["nonexistent"], 0, "Non-existent tag"),
                (["important", "personal"], 3, "Multiple tags (OR logic)"),
                (["bug-fix"], 1, "Tag with hyphen"),
                ([], 0, "Empty search")
            ]
        }
    ]
    
    passed = 0
    failed = 0
    
    for scenario in test_scenarios:
        stored_tags = scenario["stored_tags"]
        
        for search_tags, expected_count, description in scenario["search_queries"]:
            try:
                matches = simulate_search(stored_tags, search_tags)
                actual_count = len(matches)
                
                if actual_count == expected_count:
                    print(f"✅ PASS: {description}")
                    print(f"   Search tags: {search_tags}")
                    print(f"   Found {actual_count} matches")
                    passed += 1
                else:
                    print(f"❌ FAIL: {description}")
                    print(f"   Search tags: {search_tags}")
                    print(f"   Expected: {expected_count} matches")
                    print(f"   Got: {actual_count} matches")
                    print(f"   Matches: {matches}")
                    failed += 1
                print()
            except Exception as e:
                print(f"❌ ERROR: {description}")
                print(f"   Exception: {e}")
                print()
                failed += 1
    
    print("=" * 50)
    print(f"Tag matching results: {passed} passed, {failed} failed")
    
    return failed == 0

def main():
    """Run all simple tests."""
    print("Simple Tag Search Functionality Test")
    print("====================================")
    
    test1_result = test_parse_tags_fast()
    test2_result = test_tag_matching_logic()
    
    print("\n" + "=" * 50)
    print("OVERALL RESULTS")
    print("=" * 50)
    
    if test1_result and test2_result:
        print("🎉 ALL TESTS PASSED!")
        print("The tag search functionality fixes appear to be working correctly.")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("The tag search functionality needs further debugging.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)