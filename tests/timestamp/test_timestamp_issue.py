#!/usr/bin/env python3
"""Test script to debug timestamp issues in recall functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
import time
from datetime import datetime, timedelta
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.utils.time_parser import parse_time_expression, extract_time_expression

async def test_timestamp_issue():
    """Test timestamp storage and retrieval issues."""
    
    print("=== Testing Timestamp Issue ===")
    
    # Test 1: Precision loss when converting float to int
    print("\n1. Testing precision loss:")
    current_time = time.time()
    print(f"Current time (float): {current_time}")
    print(f"Current time (int): {int(current_time)}")
    print(f"Difference: {current_time - int(current_time)} seconds")
    
    # Test 2: Time expression parsing
    print("\n2. Testing time expression parsing:")
    test_queries = [
        "yesterday",
        "last week",
        "2 days ago",
        "last month",
        "this morning",
        "yesterday afternoon"
    ]
    
    for query in test_queries:
        cleaned_query, (start_ts, end_ts) = extract_time_expression(query)
        if start_ts and end_ts:
            start_dt = datetime.fromtimestamp(start_ts)
            end_dt = datetime.fromtimestamp(end_ts)
            print(f"\nQuery: '{query}'")
            print(f"  Cleaned: '{cleaned_query}'")
            print(f"  Start: {start_dt} (timestamp: {start_ts})")
            print(f"  End: {end_dt} (timestamp: {end_ts})")
            print(f"  Start (int): {int(start_ts)}")
            print(f"  End (int): {int(end_ts)}")
    
    # Test 3: Memory timestamp creation
    print("\n3. Testing Memory timestamp creation:")
    memory = Memory(
        content="Test memory",
        content_hash=generate_content_hash("Test memory"),
        tags=["test"]
    )
    
    print(f"Memory created_at (float): {memory.created_at}")
    print(f"Memory created_at (int): {int(memory.created_at)}")
    print(f"Memory created_at_iso: {memory.created_at_iso}")
    
    # Test 4: Timestamp comparison issue
    print("\n4. Testing timestamp comparison issue:")
    # Create a timestamp from "yesterday"
    yesterday_query = "yesterday"
    _, (yesterday_start, yesterday_end) = extract_time_expression(yesterday_query)
    
    # Create a memory with timestamp in the middle of yesterday
    yesterday_middle = (yesterday_start + yesterday_end) / 2
    test_memory_timestamp = yesterday_middle
    
    print(f"\nYesterday range:")
    print(f"  Start: {yesterday_start} ({datetime.fromtimestamp(yesterday_start)})")
    print(f"  End: {yesterday_end} ({datetime.fromtimestamp(yesterday_end)})")
    print(f"  Test memory timestamp: {test_memory_timestamp} ({datetime.fromtimestamp(test_memory_timestamp)})")
    
    # Check if memory would be included with float comparison
    print(f"\nFloat comparison:")
    print(f"  {test_memory_timestamp} >= {yesterday_start}: {test_memory_timestamp >= yesterday_start}")
    print(f"  {test_memory_timestamp} <= {yesterday_end}: {test_memory_timestamp <= yesterday_end}")
    print(f"  Would be included: {test_memory_timestamp >= yesterday_start and test_memory_timestamp <= yesterday_end}")
    
    # Check if memory would be included with int comparison (current implementation)
    print(f"\nInt comparison (current implementation):")
    print(f"  {int(test_memory_timestamp)} >= {int(yesterday_start)}: {int(test_memory_timestamp) >= int(yesterday_start)}")
    print(f"  {int(test_memory_timestamp)} <= {int(yesterday_end)}: {int(test_memory_timestamp) <= int(yesterday_end)}")
    print(f"  Would be included: {int(test_memory_timestamp) >= int(yesterday_start) and int(test_memory_timestamp) <= int(yesterday_end)}")
    
    # Test edge case: memory created at the very beginning or end of a day
    print(f"\n5. Testing edge cases:")
    # Memory at 00:00:00.5 (half second past midnight)
    edge_timestamp = yesterday_start + 0.5
    print(f"  Edge case timestamp: {edge_timestamp} ({datetime.fromtimestamp(edge_timestamp)})")
    print(f"  Float: {edge_timestamp} >= {yesterday_start}: {edge_timestamp >= yesterday_start}")
    print(f"  Int: {int(edge_timestamp)} >= {int(yesterday_start)}: {int(edge_timestamp) >= int(yesterday_start)}")
    
    # If the int values are the same but float values are different, we might miss memories
    if int(edge_timestamp) == int(yesterday_start) and edge_timestamp > yesterday_start:
        print(f"  WARNING: This memory would be missed with int comparison!")

if __name__ == "__main__":
    asyncio.run(test_timestamp_issue())
