#!/usr/bin/env python3
# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Claude command wrapper for memory sync operations.
Provides convenient commands for managing dual memory backends.
"""
import sys
import asyncio
import subprocess
from pathlib import Path

SYNC_SCRIPT = Path(__file__).parent / "sync_memory_backends.py"

def run_sync_command(args):
    """Run the sync script with given arguments."""
    cmd = [sys.executable, str(SYNC_SCRIPT)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)

    return result.returncode

def memory_sync_status():
    """Show memory sync status."""
    return run_sync_command(['--status'])

def memory_sync_backup():
    """Backup Cloudflare memories to SQLite-vec."""
    print("Backing up Cloudflare memories to SQLite-vec...")
    return run_sync_command(['--direction', 'cf-to-sqlite'])

def memory_sync_restore():
    """Restore SQLite-vec memories to Cloudflare."""
    print("Restoring SQLite-vec memories to Cloudflare...")
    return run_sync_command(['--direction', 'sqlite-to-cf'])

def memory_sync_bidirectional():
    """Perform bidirectional sync."""
    print("Performing bidirectional sync...")
    return run_sync_command(['--direction', 'bidirectional'])

def memory_sync_dry_run():
    """Show what would be synced without making changes."""
    print("Dry run - showing what would be synced:")
    return run_sync_command(['--dry-run'])

def show_usage():
    """Show usage information."""
    print("Usage: python claude_sync_commands.py <command>")
    print("Commands:")
    print("  status      - Show sync status")
    print("  backup      - Backup Cloudflare → SQLite-vec")
    print("  restore     - Restore SQLite-vec → Cloudflare")
    print("  sync        - Bidirectional sync")
    print("  dry-run     - Show what would be synced")

if __name__ == "__main__":
    # Dictionary-based command dispatch for better scalability
    commands = {
        "status": memory_sync_status,
        "backup": memory_sync_backup,
        "restore": memory_sync_restore,
        "sync": memory_sync_bidirectional,
        "dry-run": memory_sync_dry_run,
    }

    if len(sys.argv) < 2:
        show_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command in commands:
        sys.exit(commands[command]())
    else:
        print(f"Unknown command: {command}")
        show_usage()
        sys.exit(1)