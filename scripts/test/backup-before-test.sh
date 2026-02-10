#!/bin/bash
# Backup Production Database Before Testing
#
# CRITICAL: Run this before ANY manual testing or database modifications
#
# Usage:
#   ./scripts/test/backup-before-test.sh

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💾 MCP Memory Service - Production Backup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Determine production database location
PROD_DB_DEFAULT="$HOME/Library/Application Support/mcp-memory/sqlite_vec.db"

if [ -f "$PROD_DB_DEFAULT" ]; then
    PROD_DB="$PROD_DB_DEFAULT"
else
    # Try alternative locations
    ALTERNATIVE_PATHS=(
        "$HOME/.local/share/mcp-memory-service/sqlite_vec.db"
        "./data/sqlite_vec.db"
        "./sqlite_vec.db"
    )

    for path in "${ALTERNATIVE_PATHS[@]}"; do
        if [ -f "$path" ]; then
            PROD_DB="$path"
            break
        fi
    done
fi

if [ -z "$PROD_DB" ] || [ ! -f "$PROD_DB" ]; then
    echo "❌ ERROR: Production database not found!"
    echo ""
    echo "Searched locations:"
    echo "  - $PROD_DB_DEFAULT"
    for path in "${ALTERNATIVE_PATHS[@]}"; do
        echo "  - $path"
    done
    echo ""
    echo "Please specify database location manually:"
    echo "  export PROD_DB=/path/to/sqlite_vec.db"
    echo "  $0"
    exit 1
fi

# Create backup directory
BACKUP_DIR="$(dirname "$PROD_DB")/backups"
mkdir -p "$BACKUP_DIR"

# Generate backup filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/manual_backup_$TIMESTAMP.db"

echo "📍 Production Database:"
echo "   $PROD_DB"
echo ""

# Check database size
DB_SIZE=$(du -h "$PROD_DB" | cut -f1)
echo "📊 Database Size: $DB_SIZE"

# Count memories (if sqlite3 available)
if command -v sqlite3 &> /dev/null; then
    MEMORY_COUNT=$(sqlite3 -- "$PROD_DB" "SELECT COUNT(*) FROM memories;" 2>/dev/null || echo "N/A")
    echo "📝 Total Memories: $MEMORY_COUNT"
fi

echo ""
echo "💾 Creating backup..."

# Create backup using sqlite3 .backup for atomic, consistent copy
# (safe even if database is actively in use with WAL journal)
sqlite3 -- "$PROD_DB" ".backup \"$BACKUP_FILE\""

# Verify backup
if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ ERROR: Backup creation failed!"
    exit 1
fi

BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)

echo ""
echo "✅ Backup created successfully!"
echo ""
echo "📂 Backup Location:"
echo "   $BACKUP_FILE"
echo "   Size: $BACKUP_SIZE"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 Restore Command (if needed):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "sqlite3 -- \"$BACKUP_FILE\" ".backup \"$PROD_DB\"""
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Save restore command to file
# Uses printf '%q' for safe shell-escaping of paths and quoted heredoc
# to prevent command injection through malicious filenames
RESTORE_SCRIPT="$BACKUP_DIR/restore_$TIMESTAMP.sh"
{
    printf '#!/bin/bash\n'
    printf '# Restore from backup\n'
    printf '# Created: %s\n' "$(date)"
    printf '\nset -e\n\n'
    printf 'BACKUP_FILE=%q\n' "$BACKUP_FILE"
    printf 'PROD_DB=%q\n' "$PROD_DB"
    cat << 'EOF'

echo "Restoring production database from backup..."
echo "Backup: $BACKUP_FILE"
echo "Target: $PROD_DB"
echo ""

# Stop server if running
pkill -f "mcp_memory_service" || true
sleep 2

# Restore using sqlite3 for atomic operation
sqlite3 -- "$BACKUP_FILE" ".backup \"$PROD_DB\""

# Remove WAL files
rm -f -- "${PROD_DB}-shm" "${PROD_DB}-wal"

echo "Restore complete!"
echo ""
echo "Start server with:"
echo "  memory server --http"
EOF
} > "$RESTORE_SCRIPT"

chmod +x "$RESTORE_SCRIPT"

echo "💡 Quick restore script saved:"
echo "   $RESTORE_SCRIPT"
echo ""
