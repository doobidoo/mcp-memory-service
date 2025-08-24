#!/bin/bash
# Initialize staging database for offline memory changes

STAGING_DB="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec_staging.db"
INIT_SQL="$(dirname "$0")/deployment/staging_db_init.sql"

echo "$(date): Initializing staging database..."

# Create directory if it doesn't exist
mkdir -p "$(dirname "$STAGING_DB")"

# Initialize database with schema
sqlite3 "$STAGING_DB" < "$INIT_SQL"

if [ $? -eq 0 ]; then
    echo "$(date): Staging database initialized at: $STAGING_DB"
    echo "$(date): Database size: $(du -h "$STAGING_DB" | cut -f1)"
else
    echo "$(date): ERROR: Failed to initialize staging database"
    exit 1
fi

# Set permissions
chmod 644 "$STAGING_DB"

echo "$(date): Staging database ready for use"