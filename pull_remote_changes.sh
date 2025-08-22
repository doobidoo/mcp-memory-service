#!/bin/bash
# Enhanced remote sync with conflict awareness
# Based on the working manual_sync.sh but with staging awareness

DB_PATH="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec.db"
STAGING_DB="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec_staging.db"
REMOTE_BASE="http://narrowbox.local:8080/mcp-memory"
BACKUP_PATH="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec.db.backup"
TEMP_DIR="/tmp/litestream_pull_$$"

echo "$(date): Starting enhanced pull from remote master..."

# Create temporary directory
mkdir -p "$TEMP_DIR"

# Check if staging database exists
if [ ! -f "$STAGING_DB" ]; then
    echo "$(date): WARNING: Staging database not found. Creating..."
    ./init_staging_db.sh
fi

# Get the latest generation ID
GENERATION=$(curl -s "$REMOTE_BASE/generations/" | grep -o 'href="[^"]*/"' | sed 's/href="//;s/\/"//g' | head -1)

if [ -z "$GENERATION" ]; then
    echo "$(date): ERROR: Could not determine generation ID"
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo "$(date): Found remote generation: $GENERATION"

# Get the latest snapshot
SNAPSHOT_URL="$REMOTE_BASE/generations/$GENERATION/snapshots/"
SNAPSHOT_FILE=$(curl -s "$SNAPSHOT_URL" | grep -o 'href="[^"]*\.snapshot\.lz4"' | sed 's/href="//;s/"//g' | tail -1)

if [ -z "$SNAPSHOT_FILE" ]; then
    echo "$(date): ERROR: Could not find snapshot file"
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo "$(date): Downloading snapshot: $SNAPSHOT_FILE"

# Download and decompress snapshot
curl -s "$SNAPSHOT_URL$SNAPSHOT_FILE" -o "$TEMP_DIR/snapshot.lz4"

if ! command -v lz4 >/dev/null 2>&1; then
    echo "$(date): ERROR: lz4 command not found. Please install: brew install lz4"
    rm -rf "$TEMP_DIR"
    exit 1
fi

lz4 -d "$TEMP_DIR/snapshot.lz4" "$TEMP_DIR/remote_database.db" 2>/dev/null

if [ ! -f "$TEMP_DIR/remote_database.db" ]; then
    echo "$(date): ERROR: Failed to decompress remote database"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Conflict detection: Check if we have staged changes that might conflict
STAGED_COUNT=0
if [ -f "$STAGING_DB" ]; then
    STAGED_COUNT=$(sqlite3 "$STAGING_DB" "SELECT COUNT(*) FROM staged_memories WHERE conflict_status = 'none';" 2>/dev/null || echo "0")
fi

if [ "$STAGED_COUNT" -gt 0 ]; then
    echo "$(date): WARNING: $STAGED_COUNT staged changes detected"
    echo "$(date): Checking for potential conflicts..."
    
    # Create a list of content hashes in staging
    sqlite3 "$STAGING_DB" "SELECT content_hash FROM staged_memories;" > "$TEMP_DIR/staged_hashes.txt"
    
    # Check if any of these hashes exist in the remote database
    # Note: This requires knowledge of the remote database schema
    # For now, we'll just warn about the existence of staged changes
    echo "$(date): Staged changes will be applied after remote pull"
fi

# Backup current database
if [ -f "$DB_PATH" ]; then
    cp "$DB_PATH" "$BACKUP_PATH"
    echo "$(date): Created backup at $BACKUP_PATH"
fi

# Replace with remote database
cp "$TEMP_DIR/remote_database.db" "$DB_PATH"

if [ $? -eq 0 ]; then
    echo "$(date): Successfully pulled database from remote master"
    
    # Update staging database with sync timestamp
    if [ -f "$STAGING_DB" ]; then
        sqlite3 "$STAGING_DB" "
        UPDATE sync_status 
        SET value = datetime('now'), updated_at = CURRENT_TIMESTAMP 
        WHERE key = 'last_remote_sync';
        "
    fi
    
    # Remove backup on success
    rm -f "$BACKUP_PATH"
    
    # Show database info
    echo "$(date): Database size: $(du -h "$DB_PATH" | cut -f1)"
    echo "$(date): Database modified: $(stat -f "%Sm" "$DB_PATH")"
    
    if [ "$STAGED_COUNT" -gt 0 ]; then
        echo "$(date): NOTE: You have $STAGED_COUNT staged changes to apply"
        echo "$(date): Run ./apply_local_changes.sh to merge them"
    fi
else
    echo "$(date): ERROR: Failed to replace database with remote version"
    # Restore backup on failure
    if [ -f "$BACKUP_PATH" ]; then
        mv "$BACKUP_PATH" "$DB_PATH"
        echo "$(date): Restored backup"
    fi
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Cleanup
rm -rf "$TEMP_DIR"
echo "$(date): Remote pull completed successfully"