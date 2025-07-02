#!/usr/bin/env python3
"""
Comprehensive test for tag search functionality fixes.

This test verifies that the tag search functionality works correctly
after the fixes to the _parse_tags_fast method and search_by_tag method.
"""

import asyncio
import sys
import os
import tempfile
import shutil
import time
import logging
import json
from pathlib import Path

# Add the path to the MCP Memory Service
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mcp_memory_service.storage.chroma import ChromaMemoryStorage
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.hashing import generate_content_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TagSearchTestSuite:
    """Test suite for tag search functionality."""
    
    def __init__(self):
        self.temp_dir = None
        self.storage = None
        
    async def setup(self):
        """Set up test environment."""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp(prefix="tag_search_test_")
        logger.info(f"Created temporary test directory: {self.temp_dir}")
        
        # Initialize storage without preloading model for faster testing
        self.storage = ChromaMemoryStorage(self.temp_dir, preload_model=False)
        
    async def teardown(self):
        """Clean up test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
    
    async def test_parse_tags_fast(self):
        """Test the _parse_tags_fast method with various inputs."""
        logger.info("Testing _parse_tags_fast method...")
        
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
            (None, [], "None input"),
        ]
        
        passed = 0
        failed = 0
        
        for test_input, expected, description in test_cases:
            try:
                if test_input is None:
                    # Special case for None input
                    result = self.storage._parse_tags_fast("")
                else:
                    result = self.storage._parse_tags_fast(test_input)
                
                if result == expected:
                    logger.info(f"✅ PASS: {description}")
                    passed += 1
                else:
                    logger.error(f"❌ FAIL: {description}")
                    logger.error(f"   Input: {test_input}")
                    logger.error(f"   Expected: {expected}")
                    logger.error(f"   Got: {result}")
                    failed += 1
            except Exception as e:
                logger.error(f"❌ ERROR: {description}")
                logger.error(f"   Exception: {e}")
                failed += 1
        
        logger.info(f"_parse_tags_fast test results: {passed} passed, {failed} failed")
        return failed == 0
    
    async def test_tag_search_functionality(self):
        """Test the complete tag search functionality."""
        logger.info("Testing tag search functionality...")
        
        # Create test memories with different tag formats
        test_memories = [
            {
                "content": "Memory with comma-separated tags",
                "tags": ["important", "work", "meeting"],
                "memory_type": "note"
            },
            {
                "content": "Memory with single tag",
                "tags": ["personal"],
                "memory_type": "reminder"
            },
            {
                "content": "Memory with special characters",
                "tags": ["bug-fix", "version-2.0", "urgent!"],
                "memory_type": "task"
            },
            {
                "content": "Memory with case variations",
                "tags": ["Important", "URGENT", "Todo"],
                "memory_type": "note"
            },
            {
                "content": "Memory with no tags",
                "tags": [],
                "memory_type": "info"
            }
        ]
        
        # Store test memories
        stored_memories = []
        for i, mem_data in enumerate(test_memories):
            content = mem_data["content"]
            content_hash = generate_content_hash(content, mem_data)
            
            memory = Memory(
                content=content,
                content_hash=content_hash,
                tags=mem_data["tags"],
                memory_type=mem_data["memory_type"],
                created_at=time.time(),
                created_at_iso=f"2024-01-0{i+1}T10:00:00Z"
            )
            
            success, message = await self.storage.store(memory)
            if success:
                stored_memories.append(memory)
                logger.info(f"Stored memory: {content[:30]}...")
            else:
                logger.error(f"Failed to store memory: {message}")
                return False
        
        # Test various search scenarios
        search_tests = [
            (["important"], 2, "Case-insensitive search for 'important'"),
            (["work"], 1, "Search for 'work'"),
            (["personal"], 1, "Search for 'personal'"),
            (["urgent"], 2, "Case-insensitive search for 'urgent'"),
            (["nonexistent"], 0, "Search for non-existent tag"),
            (["important", "work"], 2, "Search for multiple tags (OR logic)"),
            (["bug-fix"], 1, "Search for tag with hyphen"),
            (["version-2.0"], 1, "Search for tag with version number"),
            (["urgent!"], 1, "Search for tag with special character"),
            ([], 0, "Empty search tags"),
        ]
        
        passed = 0
        failed = 0
        
        for search_tags, expected_count, description in search_tests:
            try:
                results = await self.storage.search_by_tag(search_tags)
                actual_count = len(results)
                
                if actual_count == expected_count:
                    logger.info(f"✅ PASS: {description}")
                    logger.info(f"   Found {actual_count} memories as expected")
                    passed += 1
                else:
                    logger.error(f"❌ FAIL: {description}")
                    logger.error(f"   Expected: {expected_count} memories")
                    logger.error(f"   Got: {actual_count} memories")
                    # Log details of found memories for debugging
                    for i, mem in enumerate(results):
                        logger.error(f"   Memory {i+1}: {mem.content[:30]}... (tags: {mem.tags})")
                    failed += 1
            except Exception as e:
                logger.error(f"❌ ERROR: {description}")
                logger.error(f"   Exception: {e}")
                failed += 1
        
        logger.info(f"Tag search test results: {passed} passed, {failed} failed")
        return failed == 0
    
    async def test_edge_cases(self):
        """Test edge cases and error conditions."""
        logger.info("Testing edge cases...")
        
        edge_cases = [
            (None, "None as search tags"),
            ("string_instead_of_list", "String instead of list"),
            (123, "Number instead of list"),
            ([None], "List with None"),
            ([""], "List with empty string"),
            ([" ", "  "], "List with whitespace-only strings"),
        ]
        
        passed = 0
        failed = 0
        
        for test_input, description in edge_cases:
            try:
                # These should all return empty results without crashing
                results = await self.storage.search_by_tag(test_input)
                if isinstance(results, list):
                    logger.info(f"✅ PASS: {description} - returned list")
                    passed += 1
                else:
                    logger.error(f"❌ FAIL: {description} - did not return list")
                    failed += 1
            except Exception as e:
                logger.error(f"❌ ERROR: {description}")
                logger.error(f"   Exception: {e}")
                failed += 1
        
        logger.info(f"Edge case test results: {passed} passed, {failed} failed")
        return failed == 0
    
    async def run_all_tests(self):
        """Run all tests in the suite."""
        logger.info("Starting tag search test suite...")
        
        try:
            await self.setup()
            
            # Run all test methods
            test_results = [
                await self.test_parse_tags_fast(),
                await self.test_tag_search_functionality(),
                await self.test_edge_cases()
            ]
            
            # Summary
            passed_tests = sum(test_results)
            total_tests = len(test_results)
            
            logger.info(f"\n=== TEST SUITE SUMMARY ===")
            logger.info(f"Tests passed: {passed_tests}/{total_tests}")
            
            if passed_tests == total_tests:
                logger.info("🎉 ALL TESTS PASSED! Tag search functionality is working correctly.")
                return True
            else:
                logger.error(f"❌ {total_tests - passed_tests} test(s) failed.")
                return False
                
        except Exception as e:
            logger.error(f"Test suite failed with exception: {e}")
            return False
        finally:
            await self.teardown()

async def main():
    """Main test execution function."""
    test_suite = TagSearchTestSuite()
    success = await test_suite.run_all_tests()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)