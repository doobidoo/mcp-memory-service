#!/usr/bin/env python3
"""
Verification script to check timestamp consistency in MCP Memory ChromaDB database.
Run this before and after migration to see the state of timestamps.
"""

import sqlite3
from datetime import datetime
import os

DB_PATH = "/Users/hkr/Library/Application Support/mcp-memory/chroma_db/chroma.sqlite3"

def check_timestamps():
    """Check current timestamp situation in the database."""
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at: {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("=" * 70)
    print("MCP Memory Timestamp Verification Report")
    print("=" * 70)
    print(f"Database: {DB_PATH}")
    print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Count total memories
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    total_memories = cursor.fetchone()[0]
    print(f"Total memories in database: {total_memories}")
    
    # 2. Check timestamp field distribution
    print("\n📊 Timestamp Field Analysis:")
    print("-" * 70)
    
    cursor.execute("""
        SELECT 
            key,
            COUNT(DISTINCT id) as memories,
            COUNT(CASE WHEN string_value IS NOT NULL THEN 1 END) as str_vals,
            COUNT(CASE WHEN int_value IS NOT NULL THEN 1 END) as int_vals,
            COUNT(CASE WHEN float_value IS NOT NULL THEN 1 END) as float_vals
        FROM embedding_metadata
        WHERE key IN ('timestamp', 'created_at', 'created_at_iso', 'timestamp_float', 
                      'timestamp_str', 'updated_at', 'updated_at_iso', 'date')
        GROUP BY key
        ORDER BY memories DESC
    """)
    
    results = cursor.fetchall()
    
    print(f"{'Field':<20} {'Memories':<12} {'String':<10} {'Int':<10} {'Float':<10}")
    print("-" * 70)
    
    for row in results:
        print(f"{row[0]:<20} {row[1]:<12} {row[2]:<10} {row[3]:<10} {row[4]:<10}")
    
    # 3. Check for memories without timestamps
    print("\n📍 Missing Timestamp Analysis:")
    
    cursor.execute("""
        SELECT COUNT(DISTINCT e.id)
        FROM embeddings e
        WHERE e.id NOT IN (
            SELECT id FROM embedding_metadata 
            WHERE key = 'timestamp' AND int_value IS NOT NULL
        )
    """)
    
    missing_timestamps = cursor.fetchone()[0]
    print(f"Memories without 'timestamp' field: {missing_timestamps}")
    
    # 4. Show sample of different timestamp formats
    print("\n📅 Sample Timestamp Values:")
    print("-" * 70)
    
    # Get a sample memory with multiple timestamp formats
    cursor.execute("""
        SELECT 
            em.id,
            MAX(CASE WHEN em.key = 'timestamp' THEN em.int_value END) as ts_int,
            MAX(CASE WHEN em.key = 'created_at' THEN em.float_value END) as created_float,
            MAX(CASE WHEN em.key = 'timestamp_str' THEN em.string_value END) as ts_str,
            SUBSTR(MAX(CASE WHEN em.key = 'chroma:document' THEN em.string_value END), 1, 50) as content
        FROM embedding_metadata em
        WHERE em.id IN (
            SELECT DISTINCT id FROM embedding_metadata 
            WHERE key IN ('timestamp', 'created_at', 'timestamp_str')
        )
        GROUP BY em.id
        HAVING COUNT(DISTINCT em.key) > 1
        LIMIT 3
    """)
    
    samples = cursor.fetchall()
    
    for i, (mem_id, ts_int, created_float, ts_str, content) in enumerate(samples, 1):
        print(f"\nMemory ID {mem_id}:")
        print(f"  Content: {content}...")
        if ts_int:
            print(f"  timestamp (int): {ts_int} = {datetime.fromtimestamp(ts_int).strftime('%Y-%m-%d %H:%M:%S')}")
        if created_float:
            print(f"  created_at (float): {created_float} = {datetime.fromtimestamp(created_float).strftime('%Y-%m-%d %H:%M:%S.%f')[:23]}")
        if ts_str:
            print(f"  timestamp_str: {ts_str}")
    
    # 5. Date range analysis
    print("\n📆 Timestamp Date Ranges:")
    print("-" * 70)
    
    # For each timestamp field, show the date range
    # Allowlist for column names to prevent SQL injection
    ALLOWED_DTYPES = {'int_value', 'float_value', 'string_value'}

    for field, dtype in [('timestamp', 'int_value'), ('created_at', 'float_value')]:
        if dtype not in ALLOWED_DTYPES:
            print(f"Error: Invalid dtype '{dtype}' not in allowed list")
            continue

        cursor.execute(f"""
            SELECT
                MIN({dtype}) as min_val,
                MAX({dtype}) as max_val,
                COUNT(DISTINCT id) as count
            FROM embedding_metadata
            WHERE key = ? AND {dtype} IS NOT NULL
        """, (field,))
        
        result = cursor.fetchone()
        if result and result[2] > 0:
            min_date = datetime.fromtimestamp(result[0]).strftime('%Y-%m-%d')
            max_date = datetime.fromtimestamp(result[1]).strftime('%Y-%m-%d')
            print(f"{field:<15} ({result[2]} memories): {min_date} to {max_date}")
    
    # 6. Summary recommendation
    print("\n💡 Recommendations:")
    print("-" * 70)
    
    if missing_timestamps > 0:
        print(f"⚠️  {missing_timestamps} memories need timestamp migration")
    
    if len(results) > 1:
        print(f"⚠️  Found {len(results)} different timestamp fields - consolidation recommended")
        print("   Run cleanup_mcp_timestamps.py to fix this issue")
    else:
        print("✅ Timestamp fields look clean!")
    
    conn.close()

if __name__ == "__main__":
    check_timestamps()
