#!/usr/bin/env python3
"""
MCP Memory Service - Sync Status Script
Compares local SQLite database with Cloudflare D1 to identify sync discrepancies.
"""

import sqlite3
import requests
import os
import sys
from datetime import datetime
from typing import Set, Dict, Tuple

# Configuration
# No defaults: credentials come from the environment only, never from source.
REQUIRED_ENV = ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_D1_DATABASE_ID", "CLOUDFLARE_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_D1_DATABASE_ID = os.environ.get("CLOUDFLARE_D1_DATABASE_ID", "")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")

# Default local DB path
if sys.platform == "win32":
    DEFAULT_LOCAL_DB = os.path.expandvars(r"%LOCALAPPDATA%\mcp-memory\sqlite_vec.db")
elif sys.platform == "darwin":
    DEFAULT_LOCAL_DB = os.path.expanduser("~/Library/Application Support/mcp-memory/sqlite_vec.db")
else:
    DEFAULT_LOCAL_DB = os.path.expanduser("~/.local/share/mcp-memory/sqlite_vec.db")

LOCAL_DB_PATH = os.environ.get("MCP_MEMORY_SQLITE_PATH", DEFAULT_LOCAL_DB)


def query_d1(sql: str) -> dict:
    """Execute a query against Cloudflare D1."""
    url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/d1/database/{CLOUDFLARE_D1_DATABASE_ID}/query"
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json={"sql": sql})
    if response.status_code != 200:
        raise Exception(f"D1 API error {response.status_code}: {response.text[:200]}")
    return response.json()


def check_d1_has_deleted_at() -> bool:
    """Check if D1 schema has deleted_at column."""
    try:
        result = query_d1("PRAGMA table_info(memories)")
        columns = [col["name"] for col in result["result"][0]["results"]]
        return "deleted_at" in columns
    except:
        return False


def get_local_hashes() -> Tuple[Set[str], Set[str]]:
    """Get all content hashes from local DB."""
    conn = sqlite3.connect(LOCAL_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content_hash, deleted_at FROM memories")
    rows = cursor.fetchall()
    conn.close()

    active = {r[0] for r in rows if not r[1]}
    deleted = {r[0] for r in rows if r[1]}
    return active, deleted


def get_d1_hashes(has_deleted_at: bool = False) -> Tuple[Set[str], Set[str]]:
    """Get all content hashes from Cloudflare D1."""
    active = set()
    deleted = set()
    offset = 0
    batch_size = 5000

    if has_deleted_at:
        sql_template = "SELECT content_hash, deleted_at FROM memories LIMIT {limit} OFFSET {offset}"
    else:
        sql_template = "SELECT content_hash FROM memories LIMIT {limit} OFFSET {offset}"

    while True:
        result = query_d1(sql_template.format(limit=batch_size, offset=offset))
        rows = result["result"][0]["results"]

        if not rows:
            break

        for row in rows:
            if has_deleted_at and row.get("deleted_at"):
                deleted.add(row["content_hash"])
            else:
                active.add(row["content_hash"])

        if len(rows) < batch_size:
            break
        offset += batch_size
        print(f"  ...fetched {offset} records from D1")

    return active, deleted


def main():
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        sys.exit(f"Missing environment variable(s): {', '.join(missing)}")

    print("=" * 70)
    print("MCP Memory Service - Sync Status Report")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    print(f"Local DB:      {LOCAL_DB_PATH}")
    print(f"Cloudflare D1: {CLOUDFLARE_D1_DATABASE_ID[:8]}...")
    print()

    if not os.path.exists(LOCAL_DB_PATH):
        print(f"ERROR: Local DB not found!")
        sys.exit(1)

    # Check D1 schema
    d1_has_deleted_at = check_d1_has_deleted_at()
    if not d1_has_deleted_at:
        print("NOTE: D1 schema differs from local (no deleted_at column)")
        print("      Tombstone comparison will be skipped.")
        print()

    # Get counts
    print("Fetching statistics...")

    conn = sqlite3.connect(LOCAL_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM memories")
    local_total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL")
    local_active = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM memories WHERE deleted_at IS NOT NULL")
    local_tombstones = cursor.fetchone()[0]
    conn.close()

    d1_result = query_d1("SELECT COUNT(*) as c FROM memories")
    d1_total = d1_result["result"][0]["results"][0]["c"]

    if d1_has_deleted_at:
        d1_result = query_d1("SELECT COUNT(*) as c FROM memories WHERE deleted_at IS NULL")
        d1_active = d1_result["result"][0]["results"][0]["c"]
        d1_result = query_d1("SELECT COUNT(*) as c FROM memories WHERE deleted_at IS NOT NULL")
        d1_tombstones = d1_result["result"][0]["results"][0]["c"]
    else:
        d1_active = d1_total
        d1_tombstones = 0

    print()
    print("-" * 70)
    print(f"{'METRIC':<25} {'LOCAL':<15} {'CLOUDFLARE':<15} {'DIFF':<10}")
    print("-" * 70)
    print(f"{'Total Records':<25} {local_total:<15} {d1_total:<15} {d1_total - local_total:<+10}")
    print(f"{'Active Memories':<25} {local_active:<15} {d1_active:<15} {d1_active - local_active:<+10}")
    print(f"{'Tombstones':<25} {local_tombstones:<15} {d1_tombstones:<15} {d1_tombstones - local_tombstones:<+10}")
    print()

    # Detailed hash comparison
    print("Fetching hashes for detailed comparison...")
    print("  Local DB...")
    local_active_h, local_deleted_h = get_local_hashes()
    print(f"    {len(local_active_h)} active, {len(local_deleted_h)} deleted")

    print("  Cloudflare D1...")
    d1_active_h, d1_deleted_h = get_d1_hashes(d1_has_deleted_at)
    print(f"    {len(d1_active_h)} active, {len(d1_deleted_h)} deleted")

    # Analysis
    local_only = local_active_h - d1_active_h - d1_deleted_h
    d1_only = d1_active_h - local_active_h - local_deleted_h
    need_local_delete = local_active_h & d1_deleted_h
    need_d1_delete = local_deleted_h & d1_active_h
    synced = local_active_h & d1_active_h

    print()
    print("-" * 70)
    print("SYNC ANALYSIS")
    print("-" * 70)
    print()
    print(f"  In sync (active both):              {len(synced)}")
    print()
    print(f"  [UPLOAD] Local -> Cloudflare:")
    print(f"    New memories to upload:           {len(local_only)}")
    print(f"    Deletions to propagate:           {len(need_d1_delete)}")
    print()
    print(f"  [DOWNLOAD] Cloudflare -> Local:")
    print(f"    New memories to download:         {len(d1_only)}")
    print(f"    Deletions to apply locally:       {len(need_local_delete)}")

    # Samples
    if local_only:
        print()
        print(f"  Sample LOCAL-ONLY memories (need upload):")
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        for h in list(local_only)[:3]:
            cursor.execute("SELECT substr(content, 1, 50) FROM memories WHERE content_hash = ?", (h,))
            row = cursor.fetchone()
            if row:
                print(f"    {h[:12]}... | {row[0][:40]}...")
        conn.close()

    if d1_only:
        print()
        print(f"  Sample CLOUDFLARE-ONLY memories (need download):")
        for h in list(d1_only)[:3]:
            try:
                result = query_d1(f"SELECT substr(content, 1, 50) as c FROM memories WHERE content_hash = '{h}'")
                if result["result"][0]["results"]:
                    c = result["result"][0]["results"][0]["c"]
                    print(f"    {h[:12]}... | {c[:40]}...")
            except:
                pass

    print()
    print("-" * 70)
    print("RECOMMENDATION")
    print("-" * 70)

    total_issues = len(local_only) + len(d1_only) + len(need_local_delete) + len(need_d1_delete)

    if total_issues == 0:
        print("  All memories are in sync!")
    else:
        print(f"  Total sync issues: {total_issues}")
        print()
        print("  To fix: Restart Claude Code/Desktop to trigger automatic sync.")
        print("  The hybrid backend should sync on startup (MCP_HYBRID_SYNC_ON_STARTUP=true)")

    print()
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
