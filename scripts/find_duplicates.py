#!/usr/bin/env python3
"""
Find and remove duplicate memories from the database.
Duplicates can occur when:
1. Same content was ingested multiple times
2. Re-ingestion after encoding fixes created duplicates
3. Manual storage of similar content
"""

import sqlite3
import json
import sys
import hashlib
import urllib.request
import urllib.parse
import ssl
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def load_config():
    """Load configuration from Claude hooks config file."""
    config_path = Path.home() / '.claude' / 'hooks' / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None

def get_memories_from_api(endpoint, api_key):
    """Retrieve all memories from the API endpoint using pagination."""
    try:
        # Create SSL context that allows self-signed certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        all_memories = []
        page = 1
        page_size = 100  # Use reasonable page size
        
        while True:
            # Create request for current page
            url = f"{endpoint}/api/memories?page={page}&page_size={page_size}"
            req = urllib.request.Request(url)
            req.add_header('Authorization', f'Bearer {api_key}')
            
            # Make request
            with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
                if response.status != 200:
                    print(f"❌ API request failed: {response.status}")
                    return []
                
                data = response.read().decode('utf-8')
                api_response = json.loads(data)
            
            # Extract memories from this page
            page_memories = api_response.get('memories', [])
            total = api_response.get('total', 0)
            has_more = api_response.get('has_more', False)
            
            all_memories.extend(page_memories)
            print(f"Retrieved page {page}: {len(page_memories)} memories (total so far: {len(all_memories)}/{total})")
            
            if not has_more:
                break
                
            page += 1
        
        print(f"✅ Retrieved all {len(all_memories)} memories from API")
        
        # Convert API format to internal format
        converted_memories = []
        for mem in all_memories:
            converted_memories.append((
                mem.get('content_hash', ''),
                mem.get('content', ''),
                json.dumps(mem.get('tags', [])),
                mem.get('created_at', ''),
                json.dumps(mem.get('metadata', {}))
            ))
        
        return converted_memories
        
    except Exception as e:
        print(f"❌ Error retrieving memories from API: {e}")
        return []

def content_similarity_hash(content):
    """Create a hash for content similarity detection."""
    # Normalize content for comparison
    normalized = content.strip().lower()
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]

def find_duplicates(memories_source, similarity_threshold=0.95):
    """
    Find duplicate memories from either database or API.
    
    Args:
        memories_source: Either a database path (str) or list of memories from API
        similarity_threshold: Threshold for considering memories duplicates (0.0-1.0)
    
    Returns:
        Dict of duplicate groups
    """
    if isinstance(memories_source, str):
        # Database path provided
        conn = sqlite3.connect(memories_source)
        cursor = conn.cursor()
        
        print("Scanning for duplicate memories...")
        
        # Get all memories
        cursor.execute("""
            SELECT content_hash, content, tags, created_at, metadata
            FROM memories 
            ORDER BY created_at DESC
        """)
        
        all_memories = cursor.fetchall()
        conn.close()
    else:
        # API memories provided
        print("Analyzing memories from API...")
        all_memories = memories_source
    
    print(f"Found {len(all_memories)} total memories")
    
    # Group by content similarity
    content_groups = defaultdict(list)
    exact_content_groups = defaultdict(list)
    
    for memory in all_memories:
        content_hash, content, tags_json, created_at, metadata_json = memory
        
        # Parse tags and metadata
        try:
            tags = json.loads(tags_json) if tags_json else []
        except:
            tags = []
            
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except:
            metadata = {}
        
        # Exact content match
        exact_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        exact_content_groups[exact_hash].append({
            'hash': content_hash,
            'content': content,
            'tags': tags,
            'created_at': created_at,
            'metadata': metadata,
            'content_length': len(content)
        })
        
        # Similar content match (normalized)
        similarity_hash = content_similarity_hash(content)
        content_groups[similarity_hash].append({
            'hash': content_hash,
            'content': content,
            'tags': tags,
            'created_at': created_at,
            'metadata': metadata,
            'content_length': len(content)
        })
    
    # Find actual duplicates (groups with > 1 memory)
    exact_duplicates = {k: v for k, v in exact_content_groups.items() if len(v) > 1}
    similar_duplicates = {k: v for k, v in content_groups.items() if len(v) > 1}
    
    return {
        'exact': exact_duplicates,
        'similar': similar_duplicates,
        'total_memories': len(all_memories)
    }

def analyze_duplicate_group(group):
    """Analyze a group of duplicate memories to determine which to keep."""
    if len(group) <= 1:
        return None
        
    # Sort by creation date (newest first)
    sorted_group = sorted(group, key=lambda x: x['created_at'], reverse=True)
    
    analysis = {
        'group_size': len(group),
        'recommended_keep': None,
        'recommended_delete': [],
        'reasons': []
    }
    
    # Prefer memories with utf8-fixed tag (these are the corrected versions)
    utf8_fixed = [m for m in sorted_group if 'utf8-fixed' in m['tags']]
    if utf8_fixed:
        analysis['recommended_keep'] = utf8_fixed[0]
        analysis['recommended_delete'] = [m for m in sorted_group if m != utf8_fixed[0]]
        analysis['reasons'].append('Keeping UTF8-fixed version')
        return analysis
    
    # Prefer newer memories
    analysis['recommended_keep'] = sorted_group[0]  # Newest
    analysis['recommended_delete'] = sorted_group[1:]  # Older ones
    analysis['reasons'].append('Keeping newest version')
    
    # Check for different tag sets
    keep_tags = set(analysis['recommended_keep']['tags'])
    for delete_mem in analysis['recommended_delete']:
        delete_tags = set(delete_mem['tags'])
        if delete_tags != keep_tags:
            analysis['reasons'].append(f'Tag differences: {delete_tags - keep_tags}')
    
    return analysis

def remove_duplicates(db_path, duplicate_groups, dry_run=True):
    """
    Remove duplicate memories from the database.
    
    Args:
        db_path: Path to the SQLite database
        duplicate_groups: Dict of duplicate groups from find_duplicates()
        dry_run: If True, only show what would be deleted
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    total_to_delete = 0
    deletion_plan = []
    
    print(f"\n{'DRY RUN - ' if dry_run else ''}Analyzing duplicate groups...")
    
    # Process exact duplicates first
    print(f"\n=== EXACT DUPLICATES ===")
    for content_hash, group in duplicate_groups['exact'].items():
        analysis = analyze_duplicate_group(group)
        if analysis:
            total_to_delete += len(analysis['recommended_delete'])
            deletion_plan.extend(analysis['recommended_delete'])
            
            print(f"\nDuplicate group: {len(group)} memories")
            print(f"  Keep: {analysis['recommended_keep']['hash'][:20]}... ({analysis['recommended_keep']['created_at']})")
            print(f"  Tags: {', '.join(analysis['recommended_keep']['tags'][:3])}")
            print(f"  Delete: {len(analysis['recommended_delete'])} older versions")
            for reason in analysis['reasons']:
                print(f"  Reason: {reason}")
    
    # Process similar duplicates (but not exact)
    print(f"\n=== SIMILAR DUPLICATES ===")
    processed_exact_hashes = set()
    for group in duplicate_groups['exact'].values():
        for mem in group:
            processed_exact_hashes.add(mem['hash'])
    
    for similarity_hash, group in duplicate_groups['similar'].items():
        # Skip if these are exact duplicates we already processed
        group_hashes = {mem['hash'] for mem in group}
        if group_hashes.issubset(processed_exact_hashes):
            continue
            
        analysis = analyze_duplicate_group(group)
        if analysis:
            print(f"\nSimilar group: {len(group)} memories")
            print(f"  Keep: {analysis['recommended_keep']['hash'][:20]}... ({analysis['recommended_keep']['created_at']})")
            print(f"  Content preview: {analysis['recommended_keep']['content'][:100]}...")
            print(f"  Would delete: {len(analysis['recommended_delete'])} similar versions")
            # Don't auto-delete similar (only exact) in this version
    
    print(f"\n{'DRY RUN SUMMARY' if dry_run else 'DELETION SUMMARY'}:")
    print(f"Total exact duplicates to delete: {total_to_delete}")
    print(f"Current total memories: {duplicate_groups['total_memories']}")
    print(f"After cleanup: {duplicate_groups['total_memories'] - total_to_delete}")
    
    if not dry_run and total_to_delete > 0:
        print(f"\n{'='*50}")
        print("DELETING DUPLICATE MEMORIES...")
        
        deleted_count = 0
        for mem_to_delete in deletion_plan:
            try:
                # Delete from memories table
                cursor.execute("DELETE FROM memories WHERE content_hash = ?", (mem_to_delete['hash'],))
                
                # Also try to delete from embeddings if it exists
                try:
                    cursor.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (mem_to_delete['hash'],))
                except:
                    pass  # Embeddings table might use different structure
                    
                deleted_count += 1
                if deleted_count % 10 == 0:
                    print(f"  Deleted {deleted_count}/{total_to_delete}...")
                    
            except Exception as e:
                print(f"  Error deleting {mem_to_delete['hash'][:20]}: {e}")
        
        conn.commit()
        print(f"✅ Successfully deleted {deleted_count} duplicate memories")
        
        # Verify final count
        cursor.execute("SELECT COUNT(*) FROM memories")
        final_count = cursor.fetchone()[0]
        print(f"📊 Final memory count: {final_count}")
    
    elif dry_run and total_to_delete > 0:
        print(f"\nTo actually delete these {total_to_delete} duplicates, run with --execute flag")
    
    conn.close()
    return total_to_delete

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Find and remove duplicate memories')
    parser.add_argument('--db-path', type=str,
                        help='Path to SQLite database (if not using API)')
    parser.add_argument('--use-api', action='store_true',
                        help='Use API endpoint from config instead of database')
    parser.add_argument('--execute', action='store_true',
                        help='Actually delete the duplicates (default is dry run)')
    parser.add_argument('--similarity-threshold', type=float, default=0.95,
                        help='Similarity threshold for duplicate detection (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Try to load config first
    config = load_config()
    
    if args.use_api or (not args.db_path and config):
        if not config:
            print("❌ No configuration found. Use --db-path for local database or ensure config exists.")
            sys.exit(1)
        
        endpoint = config.get('memoryService', {}).get('endpoint')
        api_key = config.get('memoryService', {}).get('apiKey')
        
        if not endpoint or not api_key:
            print("❌ API endpoint or key not found in configuration")
            sys.exit(1)
        
        print(f"🌐 Using API endpoint: {endpoint}")
        
        # Get memories from API
        memories = get_memories_from_api(endpoint, api_key)
        if not memories:
            print("❌ Failed to retrieve memories from API")
            sys.exit(1)
        
        # Find duplicates
        duplicates = find_duplicates(memories, args.similarity_threshold)
        
        if not duplicates['exact'] and not duplicates['similar']:
            print("✅ No duplicates found!")
            return
        
        print(f"\nFound:")
        print(f"  - {len(duplicates['exact'])} exact duplicate groups")
        print(f"  - {len(duplicates['similar'])} similar content groups")
        
        if args.execute:
            print("⚠️  API-based deletion not yet implemented. Use database path for deletion.")
        else:
            # Show analysis only
            remove_duplicates(None, duplicates, dry_run=True)
            
    else:
        # Use database path
        db_path = args.db_path or '/home/hkr/.local/share/mcp-memory/sqlite_vec.db'
        
        if not Path(db_path).exists():
            print(f"❌ Database not found: {db_path}")
            print("💡 Try --use-api to use the API endpoint from config instead")
            sys.exit(1)
        
        # Find duplicates
        duplicates = find_duplicates(db_path, args.similarity_threshold)
        
        if not duplicates['exact'] and not duplicates['similar']:
            print("✅ No duplicates found!")
            return
        
        print(f"\nFound:")
        print(f"  - {len(duplicates['exact'])} exact duplicate groups")
        print(f"  - {len(duplicates['similar'])} similar content groups")
        
        # Remove duplicates
        remove_duplicates(db_path, duplicates, dry_run=not args.execute)

if __name__ == "__main__":
    main()