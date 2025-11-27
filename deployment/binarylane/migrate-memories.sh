#!/bin/bash
# Memory Migration Script for BinaryLane Deployment
#
# Exports memories from source (local or remote Qdrant) and imports to target
#
# Usage:
#   ./migrate-memories.sh export   # Export from current source
#   ./migrate-memories.sh import   # Import to BinaryLane target
#   ./migrate-memories.sh full     # Full migration (export + transfer + import)

set -euo pipefail

# Configuration - EDIT THESE
SOURCE_TYPE="${SOURCE_TYPE:-qdrant}"  # qdrant, sqlite_vec, or cloudflare
SOURCE_QDRANT_URL="${SOURCE_QDRANT_URL:-http://localhost:6333}"
SOURCE_COLLECTION="${SOURCE_COLLECTION:-memories}"

TARGET_HOST="${TARGET_HOST:-your-vps.binarylane.cloud}"
TARGET_USER="${TARGET_USER:-root}"
TARGET_PATH="${TARGET_PATH:-/opt/mcp-memory}"

EXPORT_DIR="${EXPORT_DIR:-./migration-export}"
EXPORT_FILE="memories-$(date +%Y%m%d-%H%M%S).json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

export_from_qdrant() {
    log_info "Exporting memories from Qdrant at $SOURCE_QDRANT_URL..."

    mkdir -p "$EXPORT_DIR"

    # Use Python to export (handles pagination, embeddings, metadata)
    python3 << EOF
import json
import sys
from datetime import datetime

try:
    from qdrant_client import QdrantClient
except ImportError:
    print("Installing qdrant-client...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "qdrant-client", "-q"])
    from qdrant_client import QdrantClient

client = QdrantClient(url="${SOURCE_QDRANT_URL}")
collection = "${SOURCE_COLLECTION}"

print(f"Connecting to {collection}...")

try:
    info = client.get_collection(collection)
    total = info.points_count
    print(f"Found {total} memories to export")
except Exception as e:
    print(f"Error accessing collection: {e}")
    sys.exit(1)

# Scroll through all points
memories = []
offset = None
batch_size = 100

while True:
    results = client.scroll(
        collection_name=collection,
        limit=batch_size,
        offset=offset,
        with_payload=True,
        with_vectors=True
    )

    points, next_offset = results

    for point in points:
        memory = {
            "id": str(point.id),
            "content": point.payload.get("content", ""),
            "tags": point.payload.get("tags", []),
            "metadata": point.payload.get("metadata", {}),
            "created_at": point.payload.get("created_at"),
            "updated_at": point.payload.get("updated_at"),
            "content_hash": point.payload.get("content_hash", ""),
            "embedding": point.vector if point.vector else None
        }
        memories.append(memory)

    print(f"Exported {len(memories)}/{total} memories...", end="\r")

    if next_offset is None:
        break
    offset = next_offset

print(f"\nExported {len(memories)} memories total")

# Save to file
export_path = "${EXPORT_DIR}/${EXPORT_FILE}"
with open(export_path, 'w') as f:
    json.dump({
        "export_date": datetime.now().isoformat(),
        "source": "${SOURCE_QDRANT_URL}",
        "collection": collection,
        "count": len(memories),
        "memories": memories
    }, f, indent=2, default=str)

print(f"Saved to {export_path}")
EOF

    log_info "Export complete: $EXPORT_DIR/$EXPORT_FILE"
}

export_from_sqlite() {
    log_info "Exporting memories from SQLite-vec..."

    mkdir -p "$EXPORT_DIR"

    # Find SQLite database
    SQLITE_PATH="${SQLITE_PATH:-$HOME/.local/share/mcp-memory/memories.db}"

    if [ ! -f "$SQLITE_PATH" ]; then
        log_error "SQLite database not found at $SQLITE_PATH"
        log_info "Set SQLITE_PATH environment variable to the correct path"
        exit 1
    fi

    python3 << EOF
import json
import sqlite3
import sys
from datetime import datetime

db_path = "${SQLITE_PATH}"
print(f"Reading from {db_path}...")

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get all memories
cursor.execute("""
    SELECT content_hash, content, tags, memory_type, metadata,
           created_at, updated_at, embedding
    FROM memories
""")

memories = []
for row in cursor.fetchall():
    memory = {
        "id": row["content_hash"],
        "content": row["content"],
        "tags": json.loads(row["tags"]) if row["tags"] else [],
        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "content_hash": row["content_hash"],
        "embedding": json.loads(row["embedding"]) if row["embedding"] else None
    }
    memories.append(memory)

print(f"Found {len(memories)} memories")

conn.close()

# Save to file
export_path = "${EXPORT_DIR}/${EXPORT_FILE}"
with open(export_path, 'w') as f:
    json.dump({
        "export_date": datetime.now().isoformat(),
        "source": db_path,
        "source_type": "sqlite_vec",
        "count": len(memories),
        "memories": memories
    }, f, indent=2, default=str)

print(f"Saved to {export_path}")
EOF

    log_info "Export complete: $EXPORT_DIR/$EXPORT_FILE"
}

transfer_to_target() {
    log_info "Transferring export to BinaryLane VPS..."

    LATEST_EXPORT=$(ls -t "$EXPORT_DIR"/memories-*.json 2>/dev/null | head -1)

    if [ -z "$LATEST_EXPORT" ]; then
        log_error "No export file found in $EXPORT_DIR"
        exit 1
    fi

    log_info "Transferring $LATEST_EXPORT to $TARGET_HOST..."

    # Create target directory and transfer
    ssh "$TARGET_USER@$TARGET_HOST" "mkdir -p $TARGET_PATH/migration"
    scp "$LATEST_EXPORT" "$TARGET_USER@$TARGET_HOST:$TARGET_PATH/migration/"

    log_info "Transfer complete"
}

import_on_target() {
    log_info "Importing memories on BinaryLane VPS..."

    ssh "$TARGET_USER@$TARGET_HOST" << 'REMOTE_SCRIPT'
cd /opt/mcp-memory

# Find the latest export
EXPORT_FILE=$(ls -t migration/memories-*.json 2>/dev/null | head -1)

if [ -z "$EXPORT_FILE" ]; then
    echo "ERROR: No export file found in /opt/mcp-memory/migration/"
    exit 1
fi

echo "Importing from $EXPORT_FILE..."

# Wait for MCP service to be ready
echo "Waiting for MCP service..."
for i in {1..30}; do
    if curl -sf http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "Service is ready"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

# Import via API
python3 << PYTHON_SCRIPT
import json
import requests
import sys

with open("$EXPORT_FILE", 'r') as f:
    data = json.load(f)

memories = data.get("memories", [])
print(f"Importing {len(memories)} memories...")

success = 0
failed = 0

for i, mem in enumerate(memories):
    try:
        # Use the store API endpoint
        response = requests.post(
            "http://localhost:8000/api/memories",
            json={
                "content": mem["content"],
                "tags": mem.get("tags", []),
                "metadata": mem.get("metadata", {}),
            },
            timeout=30
        )

        if response.status_code in (200, 201):
            success += 1
        else:
            failed += 1
            if failed <= 5:
                print(f"Failed: {response.status_code} - {response.text[:100]}")

        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{len(memories)} (success: {success}, failed: {failed})")

    except Exception as e:
        failed += 1
        if failed <= 5:
            print(f"Error: {e}")

print(f"\nImport complete: {success} succeeded, {failed} failed")
PYTHON_SCRIPT

REMOTE_SCRIPT

    log_info "Import complete"
}

show_usage() {
    echo "Memory Migration Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  export    Export memories from source (local Qdrant or SQLite)"
    echo "  transfer  Transfer export file to BinaryLane VPS"
    echo "  import    Import memories on BinaryLane VPS"
    echo "  full      Run full migration (export + transfer + import)"
    echo ""
    echo "Environment variables:"
    echo "  SOURCE_TYPE        qdrant or sqlite_vec (default: qdrant)"
    echo "  SOURCE_QDRANT_URL  Qdrant URL (default: http://localhost:6333)"
    echo "  SQLITE_PATH        Path to SQLite database"
    echo "  TARGET_HOST        BinaryLane VPS hostname"
    echo "  TARGET_USER        SSH user (default: root)"
}

case "${1:-help}" in
    export)
        if [ "$SOURCE_TYPE" = "sqlite_vec" ] || [ "$SOURCE_TYPE" = "sqlite" ]; then
            export_from_sqlite
        else
            export_from_qdrant
        fi
        ;;
    transfer)
        transfer_to_target
        ;;
    import)
        import_on_target
        ;;
    full)
        log_info "Running full migration..."
        if [ "$SOURCE_TYPE" = "sqlite_vec" ] || [ "$SOURCE_TYPE" = "sqlite" ]; then
            export_from_sqlite
        else
            export_from_qdrant
        fi
        transfer_to_target
        import_on_target
        log_info "Full migration complete!"
        ;;
    *)
        show_usage
        ;;
esac
