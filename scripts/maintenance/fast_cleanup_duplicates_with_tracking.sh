#!/bin/bash
# Fast duplicate cleanup using direct SQL with hash tracking for Cloudflare sync
set -e

# Platform-specific database path
if [[ "$OSTYPE" == "darwin"* ]]; then
    DB_PATH="$HOME/Library/Application Support/mcp-memory/sqlite_vec.db"
else
    DB_PATH="$HOME/.local/share/mcp-memory/sqlite_vec.db"
fi
HASH_FILE="$HOME/deleted_duplicates.txt"

echo "🛑 Stopping HTTP server..."
# Try to stop the HTTP server - use the actual PID method since systemd may not be available on macOS
ps aux | grep -E "uvicorn.*8889" | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null || true
sleep 2

echo "📊 Analyzing duplicates and tracking hashes..."

# Create Python script to find duplicates, save hashes, and delete
python3 << 'PYTHON_SCRIPT'
import sqlite3
from pathlib import Path
from collections import defaultdict
import hashlib
import re
import os

import platform

# Platform-specific database path
if platform.system() == "Darwin":  # macOS
    DB_PATH = Path.home() / "Library/Application Support/mcp-memory/sqlite_vec.db"
else:  # Linux/Windows
    DB_PATH = Path.home() / ".local/share/mcp-memory/sqlite_vec.db"

HASH_FILE = Path.home() / "deleted_duplicates.txt"

def normalize_content(content):
    """Normalize content by removing timestamps."""
    normalized = content
    normalized = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z', 'TIMESTAMP', normalized)
    normalized = re.sub(r'\*\*Date\*\*: \d{2,4}[./]\d{2}[./]\d{2,4}', '**Date**: DATE', normalized)
    normalized = re.sub(r'Timestamp: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', 'Timestamp: TIMESTAMP', normalized)
    return normalized.strip()

def get_normalized_hash(content):
    """Create a hash of normalized content."""
    normalized = normalize_content(content)
    return hashlib.md5(normalized.encode()).hexdigest()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("Analyzing memories...")
cursor.execute("SELECT id, content_hash, content, created_at FROM memories ORDER BY created_at DESC")
memories = cursor.fetchall()

print(f"Total memories: {len(memories)}")

# Group by normalized content
content_groups = defaultdict(list)
for mem_id, mem_hash, mem_content, created_at in memories:
    norm_hash = get_normalized_hash(mem_content)
    content_groups[norm_hash].append({
        'id': mem_id,
        'hash': mem_hash,
        'created_at': created_at
    })

# Find duplicates
duplicates = {k: v for k, v in content_groups.items() if len(v) > 1}

if not duplicates:
    print("✅ No duplicates found!")
    conn.close()
    exit(0)

print(f"Found {len(duplicates)} duplicate groups")

# Collect IDs and hashes to delete (keep newest, delete older)
ids_to_delete = []
hashes_to_delete = []

for group in duplicates.values():
    for memory in group[1:]:  # Keep first (newest), delete rest
        ids_to_delete.append(memory['id'])
        hashes_to_delete.append(memory['hash'])

print(f"Deleting {len(ids_to_delete)} duplicate memories...")

# Save hashes to file for Cloudflare cleanup
print(f"Saving {len(hashes_to_delete)} content hashes to {HASH_FILE}...")
with open(HASH_FILE, 'w') as f:
    for content_hash in hashes_to_delete:
        f.write(f"{content_hash}\n")

print(f"✅ Saved hashes to {HASH_FILE}")

# Delete from memories table
placeholders = ','.join('?' * len(ids_to_delete))
cursor.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids_to_delete)

# Note: Can't delete from virtual table without vec0 extension
# Orphaned embeddings will be cleaned up on next regeneration

conn.commit()
conn.close()

print(f"✅ Deleted {len(ids_to_delete)} duplicates from SQLite")
print(f"📝 Content hashes saved for Cloudflare cleanup")

PYTHON_SCRIPT

echo ""
echo "🚀 Restarting HTTP server..."
nohup uv run python -m uvicorn mcp_memory_service.web.app:app --host 127.0.0.1 --port 8889 > /tmp/memory_http_server.log 2>&1 &
sleep 3

echo ""
echo "✅ SQLite cleanup complete!"
echo "📋 Next steps:"
echo "   1. Review deleted hashes: cat $HASH_FILE"
echo "   2. Delete from Cloudflare: uv run python scripts/maintenance/delete_cloudflare_duplicates.py"
echo "   3. Verify counts match"
