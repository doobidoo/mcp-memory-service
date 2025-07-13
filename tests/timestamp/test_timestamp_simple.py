#!/usr/bin/env python3
"""Test script to debug timestamp issues in recall functionality."""

import time
from datetime import datetime, timedelta

def test_timestamp_precision():
    """Test timestamp storage and retrieval issues."""
    
    print("=== Testing Timestamp Precision Issue ===")
    
    # Test 1: Precision loss when converting float to int
    print("\n1. Testing precision loss:")
    current_time = time.time()
    print(f"Current time (float): {current_time}")
    print(f"Current time (int): {int(current_time)}")
    print(f"Difference: {current_time - int(current_time)} seconds")
    
    # Test 2: Edge case demonstration
    print("\n2. Testing edge case with timestamps:")
    
    # Create timestamps for yesterday at midnight
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = (today - timedelta(days=1)).timestamp()
    yesterday_end = (today - timedelta(microseconds=1)).timestamp()
    
    print(f"\nYesterday range:")
    print(f"  Start (float): {yesterday_start} ({datetime.fromtimestamp(yesterday_start)})")
    print(f"  End (float): {yesterday_end} ({datetime.fromtimestamp(yesterday_end)})")
    print(f"  Start (int): {int(yesterday_start)}")
    print(f"  End (int): {int(yesterday_end)}")
    
    # Test a memory created at various times yesterday
    test_times = [
        ("00:00:00.5", yesterday_start + 0.5),
        ("00:00:30", yesterday_start + 30),
        ("12:00:00", yesterday_start + 12*3600),
        ("23:59:59.5", yesterday_end - 0.5)
    ]
    
    print("\n3. Testing memory inclusion with float vs int comparison:")
    for time_desc, timestamp in test_times:
        print(f"\n  Memory at {time_desc} (timestamp: {timestamp}):")
        
        # Float comparison
        float_included = (timestamp >= yesterday_start and timestamp <= yesterday_end)
        print(f"    Float comparison: {float_included}")
        
        # Int comparison (current implementation)
        int_included = (int(timestamp) >= int(yesterday_start) and int(timestamp) <= int(yesterday_end))
        print(f"    Int comparison: {int_included}")
        
        if float_included != int_included:
            print(f"    ⚠️  MISMATCH! Memory would be {'excluded' if float_included else 'included'} incorrectly!")
    
    # Test 4: Demonstrate the issue with ChromaDB filtering
    print("\n4. ChromaDB filter comparison issue:")
    print("  ChromaDB uses integer comparisons for numeric fields.")
    print("  When we store timestamp as int(created_at), we lose sub-second precision.")
    print("  This can cause memories to be excluded from time-based queries.")
    
    # Example of the fix
    print("\n5. Proposed fix:")
    print("  Option 1: Store timestamp as float in metadata (if ChromaDB supports it)")
    print("  Option 2: Store timestamp with higher precision (e.g., milliseconds as int)")
    print("  Option 3: Use ISO string timestamps for filtering")
    
    # Test millisecond precision
    print("\n6. Testing millisecond precision:")
    current_ms = int(current_time * 1000)
    print(f"  Current time in ms: {current_ms}")
    print(f"  Reconstructed time: {current_ms / 1000}")
    print(f"  Precision preserved: {abs((current_ms / 1000) - current_time) < 0.001}")

if __name__ == "__main__":
    test_timestamp_precision()
