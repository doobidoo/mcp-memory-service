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
Intelligent Memory Type Assignment Script

Assigns appropriate types to untyped memories using:
1. Tag-based inference (highest confidence)
2. Content pattern matching (medium confidence)
3. Metadata analysis (context hints)
4. Fallback to "note" (lowest confidence)

Usage:
    python assign_memory_types.py --dry-run      # Preview assignments
    python assign_memory_types.py --verbose      # Detailed logging
    python assign_memory_types.py --show-reasoning  # Show inference logic
    python assign_memory_types.py                # Execute assignments
"""

import sys
import os
import re
import json
import sqlite3
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from mcp_memory_service.config import SQLITE_VEC_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ============================================================================
# TYPE INFERENCE RULES
# ============================================================================

# Priority 1: Tag-based inference (highest confidence)
TAG_TO_TYPE: Dict[str, str] = {
    # Activity indicators
    "session-consolidation": "session",
    "session-summary": "session",
    "session-end": "session",
    "session-start": "session",
    "development-session": "session",
    "work-session": "session",

    # Troubleshooting
    "troubleshooting": "troubleshooting",
    "debug": "troubleshooting",
    "debugging": "troubleshooting",
    "diagnostic": "troubleshooting",
    "investigation": "troubleshooting",

    # Fixes
    "bug-fix": "fix",
    "bugfix": "fix",
    "fix": "fix",
    "patch": "fix",
    "hotfix": "fix",
    "correction": "fix",

    # Releases and deployments
    "release": "release",
    "release-notes": "release",
    "version": "release",
    "deployment": "deployment",
    "deploy": "deployment",
    "production": "deployment",

    # Features
    "feature": "feature",
    "enhancement": "feature",
    "improvement": "feature",
    "new-feature": "feature",

    # Configuration
    "configuration": "configuration",
    "config": "configuration",
    "setup": "configuration",
    "settings": "configuration",
    "environment": "configuration",

    # Documentation
    "documentation": "documentation",
    "docs": "documentation",
    "readme": "documentation",
    "changelog": "documentation",

    # Guides
    "guide": "guide",
    "tutorial": "guide",
    "how-to": "guide",
    "walkthrough": "guide",
    "instructions": "guide",

    # Reference
    "reference": "reference",
    "knowledge-base": "reference",
    "cheat-sheet": "reference",
    "quick-reference": "reference",

    # Milestones
    "milestone": "milestone",
    "achievement": "achievement",
    "completion": "milestone",
    "accomplished": "achievement",

    # Analysis
    "analysis": "analysis",
    "research": "analysis",
    "findings": "analysis",
    "investigation": "analysis",
    "report": "analysis",

    # Implementation
    "implementation": "implementation",
    "development": "implementation",
    "coding": "implementation",
    "integration": "implementation",

    # Testing
    "test": "test",
    "testing": "test",
    "validation": "test",
    "qa": "test",

    # Architecture
    "architecture": "architecture",
    "design": "architecture",
    "design-pattern": "architecture",
    "technical-design": "architecture",

    # Infrastructure
    "infrastructure": "infrastructure",
    "devops": "infrastructure",
    "ci-cd": "infrastructure",
    "automation": "infrastructure",

    # Process
    "process": "process",
    "workflow": "process",
    "procedure": "process",
    "best-practices": "process",

    # Security
    "security": "security",
    "auth": "security",
    "authentication": "security",
    "authorization": "security",

    # Status
    "status": "status",
    "update": "status",
    "progress": "status",
}

# Priority 2: Content pattern matching (medium confidence)
CONTENT_PATTERNS: Dict[str, List[str]] = {
    "fix": [
        r"\bfixed\b.*\bbug\b",
        r"\bresolved\b.*\b(issue|problem)\b",
        r"\brepair(ed|ing)\b",
        r"\bhotfix\b",
        r"\bpatch(ed|ing)?\b.*\b(bug|issue)\b",
    ],
    "troubleshooting": [
        r"\berror\b.*\boccurred\b",
        r"\btroubleshooting\b",
        r"\bdiagnos(is|tic|ing)\b",
        r"\bdebugging\b",
        r"\binvestigat(ed|ing)\b.*\b(issue|problem|error)\b",
        r"\bfail(ed|ure)\b.*\banalys",
    ],
    "implementation": [
        r"\bimplemented\b",
        r"\bcreated\b.*\b(function|class|module|component)\b",
        r"\badded\b.*\b(feature|functionality)\b",
        r"\bdevelop(ed|ing)\b",
        r"\bbuilt\b.*\b(system|service|tool)\b",
    ],
    "guide": [
        r"^(How to|Step-by-step|Guide:|Tutorial:)",
        r"\binstructions?\b.*\b(follow|complete|execute)\b",
        r"\bprocedure\b",
        r"\bstep \d+",
        r"\bwalkthrough\b",
    ],
    "configuration": [
        r"\bconfigur(e|ed|ation|ing)\b",
        r"\bsetup\b",
        r"\.env\b",
        r"\bsettings?\b",
        r"\benvironment variables?\b",
        r"\binstallation\b",
    ],
    "analysis": [
        r"\banalysis\b.*\b(shows?|reveals?|indicates?)\b",
        r"\bfindings?\b",
        r"\bresults?\b.*\b(show|demonstrate|indicate)\b",
        r"\bresearch\b",
        r"\binvestigation\b.*\bresults?\b",
    ],
    "session": [
        r"\bsession\b.*(summary|recap|notes)\b",
        r"\bwork session\b",
        r"\bdevelopment session\b",
        r"\btopics? (discussed|covered)\b",
    ],
    "release": [
        r"\b(version|v)\d+\.\d+",
        r"\breleas(e|ed|ing)\b",
        r"\bchangelog\b",
        r"\brelease notes\b",
    ],
    "documentation": [
        r"\bdocument(ation|ed|ing)\b",
        r"\bREADME\b",
        r"\bAPI documentation\b",
        r"\breference (manual|guide)\b",
    ],
    "milestone": [
        r"\b(completed|finished|accomplished)\b.*\b(project|milestone|phase)\b",
        r"\bmilestone\b.*\breached\b",
        r"\bdeliverable\b",
    ],
}

# Priority 3: Metadata type hints
METADATA_TYPE_HINTS: Set[str] = {
    "session-summary",
    "troubleshooting-session",
    "feature-summary",
    "code-review",
    "release-notes",
}


# ============================================================================
# TYPE INFERENCE ENGINE
# ============================================================================

class TypeInferenceEngine:
    """Infer memory types based on tags, content, and metadata."""

    def __init__(self, show_reasoning: bool = False):
        self.show_reasoning = show_reasoning
        self.inference_stats = Counter()

    def infer_type(self, content: str, tags: List[str], metadata: Optional[Dict]) -> Tuple[str, str, int]:
        """
        Infer memory type.

        Returns:
            (inferred_type, reasoning, confidence_score)
            confidence_score: 3=high, 2=medium, 1=low
        """
        # Priority 1: Tag-based inference (confidence=3)
        for tag in tags:
            tag_clean = tag.lower().strip()
            if tag_clean in TAG_TO_TYPE:
                inferred_type = TAG_TO_TYPE[tag_clean]
                reasoning = f"Tag match: '{tag}' → '{inferred_type}'"
                self.inference_stats["tag_match"] += 1
                return (inferred_type, reasoning, 3)

        # Priority 2: Content pattern matching (confidence=2)
        for memory_type, patterns in CONTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                    reasoning = f"Content pattern: '{pattern[:30]}...' → '{memory_type}'"
                    self.inference_stats["pattern_match"] += 1
                    return (memory_type, reasoning, 2)

        # Priority 3: Metadata hints (confidence=2)
        if metadata:
            metadata_type = metadata.get("type", "")
            if metadata_type in METADATA_TYPE_HINTS:
                # Extract base type from hyphenated metadata type
                base_type = metadata_type.split("-")[0]
                if base_type in ["session", "troubleshooting", "feature", "release"]:
                    reasoning = f"Metadata hint: type='{metadata_type}' → '{base_type}'"
                    self.inference_stats["metadata_hint"] += 1
                    return (base_type, reasoning, 2)

        # Priority 4: Fallback to "note" (confidence=1)
        reasoning = "Fallback: No specific indicators → 'note'"
        self.inference_stats["fallback"] += 1
        return ("note", reasoning, 1)

    def get_stats(self) -> Dict[str, int]:
        """Get inference statistics."""
        return dict(self.inference_stats)


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def create_backup(db_path: str) -> str:
    """Create a timestamped backup of the database."""
    backup_path = f"{db_path}.backup-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(db_path, backup_path)
    logger.info(f"✅ Backup created: {backup_path}")
    return backup_path


def analyze_untyped_memories(db_path: str) -> Tuple[int, int]:
    """Count untyped memories and total memories."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM memories")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM memories WHERE memory_type = '' OR memory_type IS NULL")
    untyped = cursor.fetchone()[0]

    conn.close()
    return untyped, total


def get_untyped_memories(db_path: str) -> List[Tuple[str, str, str, str]]:
    """
    Get all untyped memories.

    Returns:
        List of (content_hash, content, tags_str, metadata_str)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT content_hash, content, tags, metadata
        FROM memories
        WHERE memory_type = '' OR memory_type IS NULL
    """)

    results = cursor.fetchall()
    conn.close()
    return results


def assign_types(db_path: str, assignments: Dict[str, str], dry_run: bool = False) -> int:
    """
    Assign types to memories.

    Args:
        db_path: Database path
        assignments: {content_hash: inferred_type}
        dry_run: If True, don't actually update

    Returns:
        Number of memories updated
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would update {len(assignments)} memories")
        return len(assignments)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    updated = 0
    for content_hash, memory_type in assignments.items():
        cursor.execute(
            "UPDATE memories SET memory_type = ? WHERE content_hash = ?",
            (memory_type, content_hash)
        )
        updated += cursor.rowcount

    conn.commit()
    conn.close()

    logger.info(f"✅ Updated {updated} memories")
    return updated


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Intelligently assign types to untyped memories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview assignments
  python assign_memory_types.py --dry-run

  # Show detailed reasoning
  python assign_memory_types.py --dry-run --show-reasoning

  # Execute assignments
  python assign_memory_types.py

  # Verbose logging
  python assign_memory_types.py --verbose
"""
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview assignments without modifying database'
    )
    parser.add_argument(
        '--show-reasoning',
        action='store_true',
        help='Show inference reasoning for each memory'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=SQLITE_VEC_PATH,
        help=f'Path to SQLite database (default: {SQLITE_VEC_PATH})'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check database exists
    if not os.path.exists(args.db_path):
        logger.error(f"❌ Database not found: {args.db_path}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("🤖 Intelligent Memory Type Assignment")
    logger.info("=" * 80)
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Mode: {'DRY RUN (preview only)' if args.dry_run else 'EXECUTE (will modify)'}")
    logger.info("")

    # Analyze current state
    logger.info("📊 Analyzing database...")
    untyped_count, total_count = analyze_untyped_memories(args.db_path)

    logger.info(f"Total memories: {total_count}")
    logger.info(f"Untyped memories: {untyped_count} ({untyped_count/total_count*100:.1f}%)")
    logger.info("")

    if untyped_count == 0:
        logger.info("✅ No untyped memories found! Database is clean.")
        return

    # Initialize inference engine
    engine = TypeInferenceEngine(show_reasoning=args.show_reasoning)

    # Get untyped memories
    logger.info("🔍 Retrieving untyped memories...")
    untyped_memories = get_untyped_memories(args.db_path)
    logger.info(f"Retrieved {len(untyped_memories)} untyped memories")
    logger.info("")

    # Infer types
    logger.info("🧠 Inferring types...")
    assignments = {}
    type_distribution = Counter()
    confidence_distribution = Counter()

    for content_hash, content, tags_str, metadata_str in untyped_memories:
        # Parse tags and metadata
        tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
        metadata = None
        if metadata_str:
            try:
                metadata = json.loads(metadata_str)
            except json.JSONDecodeError:
                pass

        # Infer type
        inferred_type, reasoning, confidence = engine.infer_type(content, tags, metadata)

        # Store assignment
        assignments[content_hash] = inferred_type
        type_distribution[inferred_type] += 1
        confidence_distribution[confidence] += 1

        # Show reasoning if requested
        if args.show_reasoning:
            logger.info(f"{content_hash[:8]}... → {inferred_type} (conf={confidence})")
            logger.info(f"  Reason: {reasoning}")
            logger.info(f"  Tags: {tags[:3]}{'...' if len(tags) > 3 else ''}")
            logger.info(f"  Preview: {content[:100]}...")
            logger.info("")

    # Display statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("📈 Inference Statistics")
    logger.info("=" * 80)

    logger.info("\nInference Methods:")
    for method, count in engine.get_stats().items():
        logger.info(f"  {method}: {count}")

    logger.info("\nConfidence Distribution:")
    logger.info(f"  High (tag match): {confidence_distribution[3]}")
    logger.info(f"  Medium (pattern/metadata): {confidence_distribution[2]}")
    logger.info(f"  Low (fallback): {confidence_distribution[1]}")

    logger.info("\nType Distribution:")
    for memory_type, count in type_distribution.most_common():
        logger.info(f"  {memory_type}: {count}")

    logger.info("")
    logger.info("=" * 80)

    # Create backup and execute if not dry-run
    if not args.dry_run:
        logger.info("")
        logger.info("💾 Creating backup...")
        backup_path = create_backup(args.db_path)

        logger.info("")
        logger.info("✍️  Assigning types...")
        updated = assign_types(args.db_path, assignments, dry_run=False)

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ Type assignment completed successfully!")
        logger.info(f"   Backup saved to: {backup_path}")
        logger.info("=" * 80)
    else:
        logger.info("")
        logger.info("⚠️  This was a DRY RUN - no changes were made")
        logger.info("   Run without --dry-run to apply assignments")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
