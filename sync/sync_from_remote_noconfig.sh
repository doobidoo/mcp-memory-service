#!/bin/bash
# Sync script to pull latest database from remote master (no config file)

DB_PATH="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec.db"
REMOTE_URL="http://10.0.1.30:8080/mcp-memory"
BACKUP_PATH="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec.db.backup"

echo "$(date): Starting sync from remote master..."

# Create backup of current database
if [ -f "$DB_PATH" ]; then
    cp "$DB_PATH" "$BACKUP_PATH"
    echo "$(date): Created backup at $BACKUP_PATH"
fi

# Restore from remote (no config file)
litestream restore -o "$DB_PATH" "$REMOTE_URL"

if [ $? -eq 0 ]; then
    echo "$(date): Successfully synced database from remote master"
    # Remove backup on success
    rm -f "$BACKUP_PATH"
    
    # Show database info
    echo "$(date): Database size: $(du -h "$DB_PATH" | cut -f1)"
    echo "$(date): Database modified: $(stat -f "%Sm" "$DB_PATH")"
else
    echo "$(date): ERROR: Failed to sync from remote master"
    # Restore backup on failure
    if [ -f "$BACKUP_PATH" ]; then
        mv "$BACKUP_PATH" "$DB_PATH"
        echo "$(date): Restored backup"
    fi
    exit 1
fi