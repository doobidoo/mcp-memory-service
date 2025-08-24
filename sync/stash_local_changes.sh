#!/bin/bash
# Stash local memory changes to staging database before sync

MAIN_DB="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec.db"
STAGING_DB="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec_staging.db"
HOSTNAME=$(hostname)

echo "$(date): Stashing local changes..."

if [ ! -f "$MAIN_DB" ]; then
    echo "$(date): No main database found at $MAIN_DB"
    exit 1
fi

if [ ! -f "$STAGING_DB" ]; then
    echo "$(date): Staging database not found. Run ./init_staging_db.sh first"
    exit 1
fi

# Get last sync timestamp from staging database
LAST_SYNC=$(sqlite3 "$STAGING_DB" "SELECT value FROM sync_status WHERE key = 'last_local_sync';" 2>/dev/null || echo "")

# If no last sync, get all memories from a reasonable time ago (7 days)
if [ -z "$LAST_SYNC" ]; then
    LAST_SYNC="datetime('now', '-7 days')"
else
    LAST_SYNC="'$LAST_SYNC'"
fi

echo "$(date): Looking for changes since: $LAST_SYNC"

# Find memories that might be new/modified locally
# Note: This assumes your sqlite_vec.db has a similar schema
# We'll need to adapt this based on your actual schema

# First, let's check the schema of the main database
echo "$(date): Analyzing main database schema..."
MAIN_SCHEMA=$(sqlite3 "$MAIN_DB" ".schema" 2>/dev/null | head -10)

if [ $? -ne 0 ] || [ -z "$MAIN_SCHEMA" ]; then
    echo "$(date): ERROR: Cannot read main database schema"
    exit 1
fi

echo "$(date): Main database schema detected"

# For now, we'll create a simple approach that looks for memories
# This will need to be customized based on your actual schema

# Query to find recent memories (adjust based on actual schema)
TEMP_QUERY_RESULT=$(mktemp)

# Try different table names that might exist in sqlite_vec databases
for TABLE in memories memory_entries memories_table memory_items; do
    if sqlite3 "$MAIN_DB" ".tables" | grep -q "^$TABLE$"; then
        echo "$(date): Found table: $TABLE"
        
        # Try to extract memories (adjust columns based on actual schema)
        sqlite3 "$MAIN_DB" "
        SELECT 
            COALESCE(id, rowid) as id,
            content,
            COALESCE(content_hash, '') as content_hash,
            COALESCE(tags, '[]') as tags,
            COALESCE(metadata, '{}') as metadata,
            COALESCE(memory_type, 'note') as memory_type,
            COALESCE(created_at, datetime('now')) as created_at
        FROM $TABLE 
        WHERE datetime(COALESCE(updated_at, created_at, datetime('now'))) > $LAST_SYNC
        LIMIT 100;
        " 2>/dev/null > "$TEMP_QUERY_RESULT"
        
        if [ -s "$TEMP_QUERY_RESULT" ]; then
            break
        fi
    fi
done

# Count changes found
CHANGE_COUNT=$(wc -l < "$TEMP_QUERY_RESULT" | tr -d ' ')

if [ "$CHANGE_COUNT" -eq 0 ]; then
    echo "$(date): No local changes found to stash"
    rm -f "$TEMP_QUERY_RESULT"
    exit 0
fi

echo "$(date): Found $CHANGE_COUNT potential local changes"

# Process each change and add to staging
while IFS='|' read -r id content content_hash tags metadata memory_type created_at; do
    # Generate content hash if missing
    if [ -z "$content_hash" ]; then
        content_hash=$(echo -n "$content" | shasum -a 256 | cut -d' ' -f1)
    fi
    
    # Escape single quotes for SQL
    content_escaped=$(echo "$content" | sed "s/'/''/g")
    tags_escaped=$(echo "$tags" | sed "s/'/''/g")
    metadata_escaped=$(echo "$metadata" | sed "s/'/''/g")
    
    # Insert into staging database
    sqlite3 "$STAGING_DB" "
    INSERT OR REPLACE INTO staged_memories (
        id, content, content_hash, tags, metadata, memory_type,
        operation, staged_at, original_created_at, source_machine
    ) VALUES (
        '$id',
        '$content_escaped',
        '$content_hash',
        '$tags_escaped',
        '$metadata_escaped',
        '$memory_type',
        'INSERT',
        datetime('now'),
        '$created_at',
        '$HOSTNAME'
    );
    "
    
    if [ $? -eq 0 ]; then
        echo "$(date): Staged change: ${content:0:50}..."
    else
        echo "$(date): ERROR: Failed to stage change for ID: $id"
    fi
    
done < "$TEMP_QUERY_RESULT"

# Update sync status
sqlite3 "$STAGING_DB" "
UPDATE sync_status 
SET value = datetime('now'), updated_at = CURRENT_TIMESTAMP 
WHERE key = 'last_local_sync';
"

# Show staging status
STAGED_COUNT=$(sqlite3 "$STAGING_DB" "SELECT value FROM sync_status WHERE key = 'total_staged_changes';" 2>/dev/null || echo "0")

echo "$(date): Stashing completed"
echo "$(date): Total staged changes: $STAGED_COUNT"
echo "$(date): New changes stashed: $CHANGE_COUNT"

# Cleanup
rm -f "$TEMP_QUERY_RESULT"