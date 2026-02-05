#!/bin/bash
# SQLite-vec Database Backup Script
# Creates timestamped backups of the SQLite-vec database

set -e

# Configuration
MEMORY_DIR="${MCP_MEMORY_BASE_DIR:-$HOME/.local/share/mcp-memory}"
BACKUP_DIR="$MEMORY_DIR/backups"
DATABASE_FILE="$MEMORY_DIR/sqlite_vec.db"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="sqlite_backup_$TIMESTAMP"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"

# Check if database exists
if [[ ! -f "$DATABASE_FILE" ]]; then
    echo "Error: SQLite database not found at $DATABASE_FILE"
    exit 1
fi

# Create backup directory
mkdir -p "$BACKUP_PATH"

# Copy database files (main, WAL, and SHM files)
echo "Creating backup: $BACKUP_NAME"
cp "$DATABASE_FILE" "$BACKUP_PATH/" 2>/dev/null || true
cp "${DATABASE_FILE}-wal" "$BACKUP_PATH/" 2>/dev/null || true
cp "${DATABASE_FILE}-shm" "$BACKUP_PATH/" 2>/dev/null || true

# Get backup size
BACKUP_SIZE=$(du -sh "$BACKUP_PATH" | cut -f1)

# Count files backed up
FILE_COUNT=$(find "$BACKUP_PATH" -type f | wc -l)

# Create backup metadata
cat > "$BACKUP_PATH/backup_info.json" << EOF
{
  "backup_name": "$BACKUP_NAME",
  "timestamp": "$TIMESTAMP",
  "source_database": "$DATABASE_FILE",
  "backup_path": "$BACKUP_PATH",
  "backup_size": "$BACKUP_SIZE",
  "files_count": $FILE_COUNT,
  "backend": "sqlite_vec",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "Backup completed successfully:"
echo "  Name: $BACKUP_NAME"
echo "  Path: $BACKUP_PATH"
echo "  Size: $BACKUP_SIZE"
echo "  Files: $FILE_COUNT"

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "sqlite_backup_*" -type d -mtime +7 -exec rm -rf {} \; 2>/dev/null || true

exit 0
