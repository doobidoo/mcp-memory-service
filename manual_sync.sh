#!/bin/bash
# Manual sync using HTTP downloads (alternative to Litestream restore)

DB_PATH="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec.db"
REMOTE_BASE="http://narrowbox.local:8080/mcp-memory"
BACKUP_PATH="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec.db.backup"
TEMP_DIR="/tmp/litestream_manual_$$"

echo "$(date): Starting manual sync from remote master..."

# Create temporary directory
mkdir -p "$TEMP_DIR"

# Get the latest generation ID
GENERATION=$(curl -s "$REMOTE_BASE/generations/" | grep -o 'href="[^"]*/"' | sed 's/href="//;s/\/"//g' | head -1)

if [ -z "$GENERATION" ]; then
    echo "$(date): ERROR: Could not determine generation ID"
    exit 1
fi

echo "$(date): Found generation: $GENERATION"

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

if command -v lz4 >/dev/null 2>&1; then
    # Use lz4 if available
    lz4 -d "$TEMP_DIR/snapshot.lz4" "$TEMP_DIR/database.db"
else
    echo "$(date): ERROR: lz4 command not found. Please install: brew install lz4"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Backup current database
if [ -f "$DB_PATH" ]; then
    cp "$DB_PATH" "$BACKUP_PATH"
    echo "$(date): Created backup at $BACKUP_PATH"
fi

# Replace with new database
cp "$TEMP_DIR/database.db" "$DB_PATH"

if [ $? -eq 0 ]; then
    echo "$(date): Successfully synced database from remote master"
    # Remove backup on success
    rm -f "$BACKUP_PATH"
    
    # Show database info
    echo "$(date): Database size: $(du -h "$DB_PATH" | cut -f1)"
    echo "$(date): Database modified: $(stat -f "%Sm" "$DB_PATH")"
else
    echo "$(date): ERROR: Failed to replace database"
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
echo "$(date): Manual sync completed successfully"