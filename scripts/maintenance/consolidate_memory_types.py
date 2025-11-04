#!/usr/bin/env python3
"""
Memory Type Consolidation Script

Consolidates fragmented memory types into a standardized taxonomy.
Run with --dry-run to preview changes before executing.

⚠️ IMPORTANT SAFETY NOTES:
- Creates automatic backup before execution
- Stop HTTP server before running: systemctl --user stop mcp-memory-http.service
- Disconnect MCP clients (use /mcp in Claude Code)
- Database must not be locked or in use

Usage:
    python consolidate_memory_types.py --dry-run  # Preview changes (safe)
    python consolidate_memory_types.py            # Execute consolidation
    python consolidate_memory_types.py --config custom_mappings.json  # Use custom mappings
"""

import sqlite3
import sys
import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# Database path (platform-aware)
import platform
if platform.system() == "Darwin":  # macOS
    DB_PATH = Path.home() / "Library/Application Support/mcp-memory/sqlite_vec.db"
elif platform.system() == "Windows":
    DB_PATH = Path(os.getenv('LOCALAPPDATA')) / "mcp-memory" / "sqlite_vec.db"
else:  # Linux and other Unix-like systems
    DB_PATH = Path.home() / ".local/share/mcp-memory/sqlite_vec.db"

# Version
VERSION = "1.0.0"

# Consolidation mapping: old_type -> new_type
# Special handling for empty strings, NULL values, and pattern-based consolidation
CONSOLIDATION_MAP: Dict[str, str] = {
    # Empty type and NULL -> note
    "": "note",
    None: "note",

    # Session variants -> session
    "session-summary": "session",
    "session-checkpoint": "session",
    "session-completion": "session",
    "session-context": "session",
    "analysis-session": "session",
    "development-session": "session",
    "development_session": "session",
    "maintenance-session": "session",
    "project-session": "session",

    # Special sessions -> troubleshooting
    "troubleshooting-session": "troubleshooting",
    "diagnostic-session": "troubleshooting",
    "technical-session": "troubleshooting",

    # Milestone and completion variants -> milestone
    "project-milestone": "milestone",
    "development-milestone": "milestone",
    "major-milestone": "milestone",
    "major_milestone": "milestone",
    "documentation-milestone": "milestone",
    "release-milestone": "milestone",
    "deployment-milestone": "milestone",
    "final-milestone": "milestone",
    "progress-milestone": "milestone",
    "mission-accomplished": "milestone",

    # Completion types -> milestone
    "completion": "milestone",
    "project-completion": "milestone",
    "work-completion": "milestone",
    "completion-summary": "milestone",
    "milestone-completion": "milestone",
    "release-completion": "milestone",
    "development-completion": "milestone",
    "documentation-completion": "milestone",
    "feature-completion": "milestone",
    "final-completion": "milestone",
    "implementation-completion": "milestone",
    "merge-completion": "milestone",
    "session-completion": "milestone",
    "workflow-complete": "milestone",

    # Technical prefix removal - documentation
    "technical-documentation": "documentation",

    # Technical prefix removal - implementation
    "technical-implementation": "implementation",

    # Technical prefix removal - solution
    "technical-solution": "solution",
    "technical solution": "solution",

    # Technical prefix removal - fix
    "technical-fix": "fix",

    # Technical prefix removal - analysis
    "technical-analysis": "analysis",

    # Technical prefix removal - reference
    "technical-reference": "reference",

    # Technical prefix removal - note
    "technical-note": "note",
    "technical-notes": "note",

    # Technical prefix removal - guide
    "technical-guide": "guide",
    "technical-guidance": "guide",
    "technical-howto": "guide",

    # Technical prefix removal - other
    "technical-specification": "architecture",
    "technical-decision": "architecture",
    "technical-design": "architecture",
    "technical-knowledge": "reference",
    "technical_knowledge": "reference",
    "technical-finding": "analysis",
    "technical-pattern": "architecture",
    "technical-rule": "process",
    "technical-process": "process",
    "technical-achievement": "achievement",
    "technical_achievement": "achievement",
    "technical-data": "document",
    "technical-diagram": "document",
    "technical-enhancement": "feature",
    "technical-problem": "troubleshooting",
    "technical-setup": "configuration",
    "technical-summary": "note",
    "technical-todo": "note",

    # Project prefix removal
    "project-documentation": "documentation",
    "project-status": "status",
    "project-summary": "note",
    "project-update": "status",
    "project-management": "process",
    "project-improvement": "feature",
    "project-action": "note",
    "project-event": "note",
    "project-final-update": "status",
    "project-goals": "note",
    "project-implementation": "implementation",
    "project-outcome": "milestone",
    "project-overview": "note",
    "project-policy": "process",
    "project-requirement": "note",
    "project-planning": "process",
    "project-plan": "process",
    "project-roadmap": "process",
    "project-strategy": "process",
    "project-task": "note",
    "project-timeline": "process",
    "project-tracker": "status",
    "project-workflow": "process",
    "project-issue": "troubleshooting",
    "project-problem": "troubleshooting",
    "project-challenge": "troubleshooting",
    "project-risk": "note",
    "project-solution": "solution",
    "project-result": "milestone",
    "project-success": "achievement",
    "project-failure": "note",
    "project-learning": "reference",
    "project-lesson": "reference",
    "project-feedback": "note",
    "project-review": "analysis",
    "project-assessment": "analysis",
    "project-evaluation": "analysis",
    "project-analysis": "analysis",
    "project-report": "analysis",
    "project-metrics": "analysis",
    "project-performance": "analysis",
    "project-impact": "analysis",
    "project-outcome-analysis": "analysis",
    "project-benefit": "achievement",
    "project-achievement": "achievement",

    # System and configuration
    "system-config": "configuration",
    "system-setup": "configuration",
    "server-config": "configuration",
    "server-configuration": "configuration",
    "system-configuration": "configuration",
    "infrastructure_setup": "configuration",
    "setup": "configuration",
    "setup-guide": "guide",
    "setup-memo": "configuration",
    "configuration-guide": "guide",
    "user-preference": "configuration",

    # Infrastructure
    "infrastructure-change": "infrastructure",
    "infrastructure-analysis": "infrastructure",
    "infrastructure-report": "infrastructure",

    # Process and workflow
    "workflow": "process",
    "procedure": "process",
    "workflow-guide": "guide",
    "process-guide": "guide",
    "process-improvement": "process",
    "action-plan": "process",
    "detailed-process": "process",
    "development-plan": "process",
    "comprehensive-plan": "process",
    "cleanup": "process",
    "maintenance": "process",

    # Installation and features
    "installation-guide": "guide",
    "feature-specification": "feature",
    "feature-summary": "feature",

    # General mappings
    "summary": "note",
    "memo": "note",
    "reminder": "note",
    "clarification": "note",
    "checkpoint": "note",
    "finding": "analysis",
    "report": "analysis",
    "analysis-summary": "analysis",
    "analysis-report": "analysis",
    "financial-analysis": "analysis",
    "security-analysis": "analysis",
    "verification": "test",
    "correction": "fix",
    "enhancement": "feature",
    "improvement": "feature",
    "improvement-summary": "feature",
    "fix-summary": "fix",
    "user-feedback": "note",
    "user-identity": "note",
    "user-account": "configuration",

    # Extended mappings from database analysis
    "marketing": "note",
    "support": "note",
    "integration": "implementation",
    "strategy-integration": "implementation",
    "methodology": "process",
    "guideline": "guide",
    "critical-lesson": "reference",
    "security-reminder": "security",
    "security-recovery": "security",
    "security-resolution": "security",
    "security-update": "security",
    "workflow-rule": "process",
    "professional_story": "note",

    # Additional types found in database
    "applescript-template": "document",
    "project": "note",
    "test-document": "test",
    "documentation-summary": "documentation",
    "documentation-final": "documentation",
    "fact": "note",
    "development-summary": "note",
    "lesson-learned": "reference",
    "reference-guide": "guide",
    "task": "note",
    "update": "status",

    # German terms (found in database)
    "Bankzahlung": "note",
    "Betrugsschema": "note",
    "Finanzbeweis": "note",
    "Strafanzeige": "note",

    # Analysis and investigation types
    "analysis-finding": "analysis",
    "analysis-start": "analysis",
    "comprehensive-analysis": "analysis",
    "final-analysis": "analysis",
    "investigation": "analysis",
    "issue-identification": "analysis",
    "issue_investigation": "analysis",
    "temporal-analysis": "analysis",
    "testing-insights": "analysis",

    # Architecture types
    "architecture-decision": "architecture",
    "architecture-visualization": "architecture",
    "concept-design": "architecture",
    "design-decision": "architecture",
    "tool-decision": "architecture",

    # Status and reporting
    "backup-record": "status",
    "maintenance-report": "status",
    "maintenance-summary": "status",
    "progress": "status",
    "progress-tracking": "status",
    "system-health-report": "status",
    "system-report": "status",

    # Reference types
    "best-practice": "reference",
    "learning": "reference",
    "lesson": "reference",
    "network-info": "reference",
    "principle": "reference",

    # Fix types
    "bug": "fix",
    "critical-fix": "fix",

    # Troubleshooting types
    "configuration-issue": "troubleshooting",
    "critical-issue": "troubleshooting",
    "debugging": "troubleshooting",
    "error": "troubleshooting",
    "problem-escalation": "troubleshooting",

    # Solution types
    "final-resolution": "solution",
    "solution-complete": "solution",
    "solution-design": "solution",
    "solution-implemented": "solution",

    # Test types
    "compatibility-test": "test",
    "debug-test": "test",
    "functionality-test": "test",
    "healthcheck_test": "test",
    "post-fix-test": "test",
    "post-restart-test": "test",
    "string-format-test": "test",
    "system_test": "test",
    "test-case": "test",
    "test-result": "test",
    "testing": "test",
    "validation": "test",
    "validation-results": "test",
    "verification-test": "test",

    # Guide types
    "comprehensive-guide": "guide",
    "tutorial_resource": "guide",

    # Document types
    "comprehensive_collection": "document",
    "strategy-document": "document",

    # Achievement types
    "success": "achievement",
    "work-achievement": "achievement",

    # Note types (various)
    "concept": "note",
    "contribution": "note",
    "critical-discovery": "note",
    "design-note": "note",
    "discovery": "note",
    "important-note": "note",
    "imported": "note",
    "network-limitation": "note",
    "reflection": "note",
    "server-behavior": "note",
    "system": "note",
    "tool": "note",
    "user-input": "note",
    "user-question": "note",
    "user-request": "note",
    "issue_creation": "note",
}

def check_http_server_running() -> bool:
    """Check if HTTP server is running (Linux only)."""
    try:
        # Check systemd service
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "mcp-memory-http.service"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        # Not Linux or systemctl not available
        return False


def check_database_locked(db_path: Path) -> bool:
    """Check if database is currently locked."""
    try:
        # Try to open with a very short timeout
        conn = sqlite3.connect(db_path, timeout=0.1)
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        conn.rollback()
        conn.close()
        return False
    except sqlite3.OperationalError:
        return True


def create_backup(db_path: Path, dry_run: bool = False) -> Optional[Path]:
    """Create a timestamped backup of the database."""
    if dry_run:
        return None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}.backup-{timestamp}{db_path.suffix}"

    try:
        shutil.copy2(db_path, backup_path)

        # Verify backup
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not created: {backup_path}")

        if backup_path.stat().st_size != db_path.stat().st_size:
            raise ValueError(f"Backup size mismatch: {backup_path.stat().st_size} != {db_path.stat().st_size}")

        return backup_path
    except Exception as e:
        print(f"\n❌ Error creating backup: {e}")
        raise


def perform_safety_checks(db_path: Path, dry_run: bool = False) -> bool:
    """Perform all safety checks before consolidation."""
    print("\n" + "="*80)
    print("Safety Checks")
    print("="*80)

    all_passed = True

    # Check 1: Database exists
    if not db_path.exists():
        print("❌ Database not found at:", db_path)
        return False
    print(f"✓ Database found: {db_path}")

    # Check 2: Database is not locked
    if check_database_locked(db_path):
        print("❌ Database is currently locked (in use by another process)")
        print("   Stop HTTP server: systemctl --user stop mcp-memory-http.service")
        print("   Disconnect MCP: Use /mcp command in Claude Code")
        all_passed = False
    else:
        print("✓ Database is not locked")

    # Check 3: HTTP server status (Linux only)
    if os.name != 'nt':  # Not Windows
        if check_http_server_running():
            print("⚠️  HTTP server is running")
            print("   Recommended: systemctl --user stop mcp-memory-http.service")
            if not dry_run:
                response = input("   Continue anyway? (yes/no): ")
                if response.lower() != "yes":
                    all_passed = False
        else:
            print("✓ HTTP server is not running")

    # Check 4: Sufficient disk space
    stat = os.statvfs(db_path.parent)
    free_space = stat.f_bavail * stat.f_frsize
    db_size = db_path.stat().st_size
    if free_space < db_size * 2:  # Need at least 2x database size
        print(f"⚠️  Low disk space: {free_space / 1024**2:.1f} MB free, need {db_size * 2 / 1024**2:.1f} MB")
        all_passed = False
    else:
        print(f"✓ Sufficient disk space: {free_space / 1024**2:.1f} MB free")

    print("="*80)

    return all_passed


def analyze_database(conn: sqlite3.Connection) -> Tuple[Dict[str, int], int]:
    """Analyze current state of memory types."""
    cursor = conn.cursor()

    # Get type distribution
    cursor.execute("SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type")
    type_counts = {row[0]: row[1] for row in cursor.fetchall()}

    # Get total count
    cursor.execute("SELECT COUNT(*) FROM memories")
    total = cursor.fetchone()[0]

    return type_counts, total


def preview_consolidation(type_counts: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    """Preview what the consolidation will do."""
    # Group by target type
    consolidation_preview = defaultdict(lambda: {"old_count": 0, "sources": []})
    unchanged = {}

    for old_type, count in type_counts.items():
        if old_type in CONSOLIDATION_MAP:
            new_type = CONSOLIDATION_MAP[old_type]
            consolidation_preview[new_type]["old_count"] += count
            consolidation_preview[new_type]["sources"].append(f"{old_type} ({count})")
        else:
            unchanged[old_type] = count

    return dict(consolidation_preview), unchanged


def execute_consolidation(conn: sqlite3.Connection, dry_run: bool = True) -> Tuple[int, Dict[str, int]]:
    """Execute the consolidation."""
    cursor = conn.cursor()
    total_updated = 0
    updates_by_type = defaultdict(int)

    if dry_run:
        print("\n" + "="*80)
        print("DRY RUN MODE - No changes will be made")
        print("="*80 + "\n")

    # Process each mapping
    for old_type, new_type in CONSOLIDATION_MAP.items():
        # Handle None/NULL specially
        if old_type is None:
            if dry_run:
                cursor.execute("SELECT COUNT(*) FROM memories WHERE memory_type IS NULL")
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f"Would update {count:4d} memories: (None/NULL) → {new_type}")
                    total_updated += count
                    updates_by_type[new_type] += count
            else:
                cursor.execute("UPDATE memories SET memory_type = ? WHERE memory_type IS NULL", (new_type,))
                count = cursor.rowcount
                if count > 0:
                    print(f"Updated {count:4d} memories: (None/NULL) → {new_type}")
                    total_updated += count
                    updates_by_type[new_type] += count
        else:
            if dry_run:
                cursor.execute(
                    "SELECT COUNT(*) FROM memories WHERE memory_type = ?",
                    (old_type,)
                )
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f"Would update {count:4d} memories: {old_type!r:40s} → {new_type}")
                    total_updated += count
                    updates_by_type[new_type] += count
            else:
                cursor.execute(
                    "UPDATE memories SET memory_type = ? WHERE memory_type = ?",
                    (new_type, old_type)
                )
                count = cursor.rowcount
                if count > 0:
                    print(f"Updated {count:4d} memories: {old_type!r:40s} → {new_type}")
                    total_updated += count
                    updates_by_type[new_type] += count

    return total_updated, dict(updates_by_type)


def main():
    """Main execution."""
    dry_run = "--dry-run" in sys.argv

    print(f"\nMemory Type Consolidation Script v{VERSION}")
    print(f"Database: {DB_PATH}")
    print(f"Mode: {'DRY RUN (preview only)' if dry_run else 'LIVE EXECUTION'}")
    print("="*80)

    # Perform safety checks
    if not perform_safety_checks(DB_PATH, dry_run):
        print("\n❌ Safety checks failed. Aborting.")
        sys.exit(1)

    # Create backup (unless dry-run)
    if not dry_run:
        print("\nCreating backup...")
        try:
            backup_path = create_backup(DB_PATH, dry_run)
            if backup_path:
                print(f"✓ Backup created: {backup_path}")
                print(f"  Size: {backup_path.stat().st_size / 1024**2:.2f} MB")
        except Exception as e:
            print(f"❌ Failed to create backup: {e}")
            sys.exit(1)

    # Connect to database
    conn = sqlite3.connect(DB_PATH, timeout=30)

    try:
        # Analyze current state
        print("\nAnalyzing current state...")
        type_counts, total = analyze_database(conn)
        unique_types = len(type_counts)

        print(f"\nCurrent State:")
        print(f"  Total memories: {total:,}")
        print(f"  Unique types: {unique_types}")
        print(f"  Empty type: {type_counts.get('', 0)}")

        # Preview consolidation
        print("\nConsolidation Preview:")
        consolidation_preview, unchanged = preview_consolidation(type_counts)

        print(f"\nTypes that will be consolidated:")
        for new_type in sorted(consolidation_preview.keys()):
            info = consolidation_preview[new_type]
            print(f"\n  {new_type}: {info['old_count']} memories from {len(info['sources'])} sources")
            for source in sorted(info['sources']):
                print(f"    ← {source}")

        print(f"\nTypes that will remain unchanged: {len(unchanged)}")
        for old_type, count in sorted(unchanged.items(), key=lambda x: -x[1])[:20]:
            type_display = old_type if old_type is not None else "(None/NULL)"
            print(f"  {type_display:40s} {count:4d}")
        if len(unchanged) > 20:
            print(f"  ... and {len(unchanged) - 20} more")

        # Execute consolidation
        print("\n" + "="*80)
        if not dry_run:
            response = input("\nProceed with consolidation? (yes/no): ")
            if response.lower() != "yes":
                print("Consolidation cancelled.")
                return

        total_updated, updates_by_type = execute_consolidation(conn, dry_run)

        if not dry_run:
            conn.commit()
            print(f"\n✓ Consolidation complete!")

        print(f"\nTotal memories updated: {total_updated:,}")
        print(f"\nBreakdown by target type:")
        for new_type in sorted(updates_by_type.keys(), key=lambda x: -updates_by_type[x]):
            print(f"  {new_type:30s} +{updates_by_type[new_type]:4d}")

        # Show final state
        if not dry_run:
            print("\nAnalyzing final state...")
            final_type_counts, final_total = analyze_database(conn)
            final_unique_types = len(final_type_counts)

            print(f"\nFinal State:")
            print(f"  Total memories: {final_total:,}")
            print(f"  Unique types: {final_unique_types}")
            print(f"  Reduction: {unique_types} → {final_unique_types} types ({unique_types - final_unique_types} removed)")

            print(f"\nTop types by count:")
            for memory_type, count in sorted(final_type_counts.items(), key=lambda x: -x[1])[:25]:
                pct = (count / final_total) * 100
                print(f"  {memory_type:30s} {count:4d} ({pct:5.1f}%)")

    except Exception as e:
        print(f"\nError: {e}")
        if not dry_run:
            conn.rollback()
            print("Changes rolled back.")
        raise

    finally:
        conn.close()

    if dry_run:
        print("\n" + "="*80)
        print("DRY RUN COMPLETE - Run without --dry-run to execute")
        print("="*80)


if __name__ == "__main__":
    main()
