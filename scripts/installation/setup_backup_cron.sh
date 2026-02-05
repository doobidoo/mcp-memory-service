#!/bin/bash
# Setup automated backups for MCP Memory Service
# Creates cron jobs for regular SQLite-vec database backups

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_SCRIPT="$SCRIPT_DIR/backup_sqlite_vec.sh"

# Check if backup script exists
if [[ ! -f "$BACKUP_SCRIPT" ]]; then
    echo "Error: Backup script not found at $BACKUP_SCRIPT"
    exit 1
fi

# Make sure backup script is executable
chmod +x "$BACKUP_SCRIPT"

# Create cron job entry
CRON_ENTRY="0 2 * * * $BACKUP_SCRIPT > /tmp/mcp-backup.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "$BACKUP_SCRIPT"; then
    echo "Backup cron job already exists. Current crontab:"
    crontab -l | grep "$BACKUP_SCRIPT"
else
    # Add cron job
    (crontab -l 2>/dev/null || true; echo "$CRON_ENTRY") | crontab -
    echo "Added daily backup cron job:"
    echo "$CRON_ENTRY"
fi

echo ""
echo "Backup automation setup complete!"
echo "- Daily backups at 2:00 AM"
echo "- Backup script: $BACKUP_SCRIPT"
echo "- Log file: /tmp/mcp-backup.log"
echo ""
echo "To check cron jobs: crontab -l"
echo "To remove cron job: crontab -l | grep -v backup_sqlite_vec.sh | crontab -"
