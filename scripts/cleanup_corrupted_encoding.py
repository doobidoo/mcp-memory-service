#!/usr/bin/env python3
"""
Clean up memories with corrupted emoji encoding from the database.
This script identifies and removes entries where emojis were incorrectly encoded.
"""

import sqlite3
import json
import re
import sys
from pathlib import Path

def detect_corrupted_encoding(text):
    """
    Detect if text contains corrupted emoji encoding patterns.
    Common patterns include:
    - üöÄ (corrupted 🚀)
    - ‚ö° (corrupted ⚡)
    - üéØ (corrupted 🎯)
    - ‚úÖ (corrupted ✅)
    - ‚û°Ô∏è (corrupted ➡️)
    - and other mangled Unicode sequences
    """
    # Pattern for corrupted emojis - looking for specific corrupted sequences
    corrupted_patterns = [
        r'[üöÄ]{2,}',  # Multiple Germanic umlauts together
        r'‚[öûú][°ÖØ]',  # Specific corrupted emoji patterns
        r'Ô∏è',  # Part of corrupted arrow emoji
        r'\uf8ff',  # Apple logo character that shouldn't be there
        r'ü[öéì][ÄØÖ]',  # Common corrupted emoji starts
        r'‚Ä[çª†]',  # Another corruption pattern
    ]
    
    for pattern in corrupted_patterns:
        if re.search(pattern, text):
            return True
    
    # Also check for suspicious character combinations
    # Real emojis are typically in ranges U+1F300-U+1F9FF, U+2600-U+27BF
    # Corrupted text often has unusual combinations of Latin extended characters
    suspicious_chars = ['ü', 'ö', 'Ä', '‚', 'Ô', '∏', 'è', '°', 'Ö', 'Ø', 'û', 'ú', 'ì', '†', 'ª', 'ç']
    char_count = sum(text.count(char) for char in suspicious_chars)
    
    # If we have multiple suspicious characters in a short span, likely corrupted
    if char_count > 3 and len(text) < 200:
        return True
    
    return False

def cleanup_corrupted_memories(db_path, dry_run=True):
    """
    Clean up memories with corrupted encoding.
    
    Args:
        db_path: Path to the SQLite database
        dry_run: If True, only show what would be deleted without actually deleting
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"{'DRY RUN - ' if dry_run else ''}Scanning for memories with corrupted encoding...")
    
    # Get all memories with potential corruption
    cursor.execute("""
        SELECT content_hash, content, tags, created_at 
        FROM memories 
        WHERE tags LIKE '%readme%' OR tags LIKE '%documentation%'
        ORDER BY created_at DESC
    """)
    
    all_memories = cursor.fetchall()
    corrupted_memories = []
    
    for content_hash, content, tags_json, created_at in all_memories:
        if detect_corrupted_encoding(content):
            try:
                tags = json.loads(tags_json) if tags_json else []
            except:
                tags = []
            
            # Skip if already marked as UTF8-fixed (these are the corrected versions)
            if 'utf8-fixed' in tags:
                continue
                
            corrupted_memories.append({
                'hash': content_hash,
                'content_preview': content[:200],
                'tags': tags,
                'created_at': created_at
            })
    
    print(f"\nFound {len(corrupted_memories)} memories with corrupted encoding")
    
    if corrupted_memories:
        print("\nCorrupted memories to be deleted:")
        print("-" * 80)
        
        for i, mem in enumerate(corrupted_memories[:10], 1):  # Show first 10
            print(f"\n{i}. Hash: {mem['hash'][:20]}...")
            print(f"   Created: {mem['created_at']}")
            print(f"   Tags: {', '.join(mem['tags'][:5])}")
            print(f"   Content preview: {mem['content_preview'][:100]}...")
        
        if len(corrupted_memories) > 10:
            print(f"\n... and {len(corrupted_memories) - 10} more")
        
        if not dry_run:
            print("\n" + "="*80)
            print("DELETING CORRUPTED MEMORIES...")
            
            # Delete from memories table
            for mem in corrupted_memories:
                cursor.execute("DELETE FROM memories WHERE content_hash = ?", (mem['hash'],))
                
                # Also delete from embeddings table if it exists
                try:
                    cursor.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (mem['hash'],))
                except:
                    pass  # Embeddings table might use different structure
            
            conn.commit()
            print(f"✅ Deleted {len(corrupted_memories)} corrupted memories")
            
            # Verify deletion
            cursor.execute("SELECT COUNT(*) FROM memories")
            remaining = cursor.fetchone()[0]
            print(f"📊 Remaining memories in database: {remaining}")
        else:
            print("\n" + "="*80)
            print("DRY RUN COMPLETE - No changes made")
            print(f"To actually delete these {len(corrupted_memories)} memories, run with --execute flag")
    else:
        print("✅ No corrupted memories found!")
    
    conn.close()

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up memories with corrupted emoji encoding')
    parser.add_argument('--db-path', type=str, 
                        default='/home/hkr/.local/share/mcp-memory/sqlite_vec.db',
                        help='Path to SQLite database')
    parser.add_argument('--execute', action='store_true',
                        help='Actually delete the corrupted memories (default is dry run)')
    
    args = parser.parse_args()
    
    if not Path(args.db_path).exists():
        print(f"❌ Database not found: {args.db_path}")
        sys.exit(1)
    
    cleanup_corrupted_memories(args.db_path, dry_run=not args.execute)

if __name__ == "__main__":
    main()