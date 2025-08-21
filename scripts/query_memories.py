#!/usr/bin/env python3
"""Query memories from the SQLite database"""

import sqlite3
import json
import sys

def query_memories(tag_filter=None, query_text=None, limit=5):
    """Query memories from the database"""
    conn = sqlite3.connect('/home/hkr/.local/share/mcp-memory/sqlite_vec.db')
    cursor = conn.cursor()
    
    if tag_filter:
        sql = "SELECT content, tags FROM memories WHERE tags LIKE ? LIMIT ?"
        cursor.execute(sql, (f'%{tag_filter}%', limit))
    elif query_text:
        sql = "SELECT content, tags FROM memories WHERE content LIKE ? LIMIT ?"
        cursor.execute(sql, (f'%{query_text}%', limit))
    else:
        sql = "SELECT content, tags FROM memories ORDER BY created_at DESC LIMIT ?"
        cursor.execute(sql, (limit,))
    
    results = []
    for row in cursor.fetchall():
        content = row[0]
        try:
            tags = json.loads(row[1]) if row[1] else []
        except (json.JSONDecodeError, TypeError):
            # Tags might be stored differently
            tags = row[1].split(',') if row[1] and isinstance(row[1], str) else []
        results.append({
            'content': content,
            'tags': tags
        })
    
    conn.close()
    return results

if __name__ == "__main__":
    # Get memories with specific tags
    print("=== Searching for README sections ===\n")
    
    # Search for readme content
    memories = query_memories(tag_filter="readme", limit=10)
    
    for i, memory in enumerate(memories, 1):
        print(f"Memory {i}:")
        print(f"Content (first 500 chars):\n{memory['content'][:500]}")
        print(f"Tags: {', '.join(memory['tags'])}")
        print("-" * 80)
        print()
    
    # Search for specific content
    print("\n=== Searching for Installation content ===\n")
    memories = query_memories(query_text="installation", limit=5)
    
    for i, memory in enumerate(memories, 1):
        print(f"Memory {i}:")
        print(f"Content (first 500 chars):\n{memory['content'][:500]}")
        print(f"Tags: {', '.join(memory['tags'])}")
        print("-" * 80)
        print()