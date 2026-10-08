"""Schema migration runner with version tracking and registry."""

import hashlib
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Pattern: NNN_description.sql (e.g., 008_add_graph_table.sql)
MIGRATION_FILE_PATTERN = re.compile(r"^(\d{3})_(.+)\.sql$")


class MigrationRunner:
    """Schema migration runner with version tracking and registry."""

    def __init__(self, migrations_dir: Path):
        self.migrations_dir = migrations_dir

    def _discover_migrations(self) -> list[tuple[int, str, Path]]:
        """Discover migration files sorted by version number.
        Returns: [(version, name, path), ...]
        """
        migrations = []
        if not self.migrations_dir.exists():
            return migrations
        for f in self.migrations_dir.iterdir():
            m = MIGRATION_FILE_PATTERN.match(f.name)
            if m:
                version = int(m.group(1))
                name = m.group(2)
                migrations.append((version, name, f))
        return sorted(migrations, key=lambda x: x[0])

    def _ensure_registry_table(self, conn):
        """Create migration_registry table if it doesn't exist."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS migration_registry (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                filename TEXT NOT NULL,
                applied_at TEXT NOT NULL,
                checksum TEXT
            )
        """)

    def _get_applied_versions(self, conn) -> set[int]:
        """Read migration_registry to find already-applied versions."""
        try:
            cursor = conn.execute("SELECT version FROM migration_registry")
            return {row[0] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            return set()

    def _get_registry(self, conn) -> list[dict]:
        """Read full migration registry."""
        try:
            cursor = conn.execute(
                "SELECT version, name, filename, applied_at, checksum FROM migration_registry ORDER BY version"
            )
            return [
                {"version": r[0], "name": r[1], "filename": r[2], "applied_at": r[3], "checksum": r[4]}
                for r in cursor.fetchall()
            ]
        except sqlite3.OperationalError:
            return []

    def _get_current_version(self, conn) -> int:
        """Read schema_version from metadata. Default 7 if missing."""
        try:
            cursor = conn.execute("SELECT value FROM metadata WHERE key='schema_version'")
            row = cursor.fetchone()
            return int(row[0]) if row else 7
        except sqlite3.OperationalError:
            return 7

    def _checksum(self, path: Path) -> str:
        """SHA256 checksum of migration file content."""
        return hashlib.sha256(path.read_text().encode()).hexdigest()

    def _stamp_baseline(self, conn) -> list[int]:
        """Stamp pre-existing migrations as applied without re-executing.

        On existing databases, migrations 008-011 were applied by the old
        idempotent runner (which swallowed duplicate-column errors). We detect
        their artifacts and register them so run_pending() skips them.

        Returns list of stamped versions.
        """
        applied = set(self._get_applied_versions(conn))

        # Delta-sync recovery (15/16): probe these INDEPENDENTLY of the legacy baseline below,
        # and only when absent from the registry. A partial manual rollback can leave v15
        # registered while v16's row was removed but its columns retained; without this, the
        # forward migration would re-run the unconditional ADD COLUMN and fail duplicate-column
        # (Greptile). This runs even when the registry is non-empty, but is a strict no-op unless
        # the artifact exists AND its registry row is missing — so it never changes the legacy
        # 8-11 baseline semantics handled further down.
        delta_probes = {
            15: "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sync_events'",
            16: "SELECT 1 FROM pragma_table_info('sync_events') WHERE name='hlc_physical'",
        }
        recovered = self._stamp_probes(
            conn, {v: sql for v, sql in delta_probes.items() if v not in applied}
        )

        # Legacy baseline (008-011): only on a fresh existing DB with an EMPTY registry, detect
        # artifacts left by the old idempotent runner and stamp them so run_pending skips them.
        # Preserved exactly as before (early-return when the registry is already populated).
        if applied:
            return recovered

        probes = {
            8: "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_graph'",
            9: "SELECT 1 FROM pragma_table_info('memory_graph') WHERE name='relationship_type'",
            10: "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_graph'",  # graph indexes
            11: "SELECT 1 FROM pragma_table_info('memories') WHERE name='version'",
        }
        stamped = recovered + self._stamp_probes(conn, probes)
        if stamped:
            max_version = max(stamped)
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', ?)",
                (str(max_version),),
            )
            conn.commit()
            logger.info(f"Baseline stamp: registered migrations {stamped} (existing DB detected)")

        return stamped

    def _stamp_probes(self, conn, probes: dict) -> list[int]:
        """Stamp each probed version whose artifact exists but whose registry row is missing.

        Shared by the delta-sync recovery and the legacy 8-11 baseline. Does NOT touch
        schema_version (callers decide); commits only the registry inserts it performs.

        Before stamping, repairs any SECONDARY artifact a partial rollback may have dropped
        while the primary artifact (the probed column/table) was retained — otherwise the
        migration is marked applied forever but a dropped index/trigger never comes back.
        Repairs are idempotent (IF NOT EXISTS).

        Returns the list of versions stamped.
        """
        if not probes:
            return []

        # Idempotent repairs for secondary artifacts, keyed by migration version. The primary
        # artifact is detected by the probe; these restore the rest (e.g. the HLC index that a
        # rollback dropped alongside the retained hlc columns — Greptile).
        artifact_repairs = {
            16: "CREATE INDEX IF NOT EXISTS idx_sync_events_hlc "
                "ON sync_events(hlc_physical, hlc_logical, agent_id, event_id)",
        }

        stamped = []
        now = datetime.now(timezone.utc).isoformat()
        migrations = self._discover_migrations()
        migrations_by_version = {v: (n, p) for v, n, p in migrations}

        for version, probe_sql in probes.items():
            try:
                cursor = conn.execute(probe_sql)
                if cursor.fetchone():
                    # Secondary-artifact repair (idempotent) before stamping.
                    repair_sql = artifact_repairs.get(version)
                    if repair_sql:
                        conn.execute(repair_sql)
                    # Artifact exists — stamp as applied
                    name, path = migrations_by_version.get(version, (f"legacy_{version}", None))
                    checksum = self._checksum(path) if path else ""
                    filename = path.name if path else f"{version:03d}_unknown.sql"
                    conn.execute(
                        "INSERT OR IGNORE INTO migration_registry (version, name, filename, applied_at, checksum) VALUES (?, ?, ?, ?, ?)",
                        (version, name, filename, now, checksum),
                    )
                    stamped.append(version)
            except sqlite3.OperationalError:
                continue

        if stamped:
            conn.commit()
        return stamped

    def run_pending(self, conn, dry_run=False) -> dict:
        """Run all pending migrations in order.

        Returns: {applied: [...], skipped: [...], error: None|str}
        """
        result = {"applied": [], "skipped": [], "error": None}
        try:
            if not dry_run:
                self._ensure_registry_table(conn)
                conn.commit()  # Ensure no open transaction before we begin
                # Stamp pre-existing migrations on upgrade from old runner
                self._stamp_baseline(conn)
            applied_versions = self._get_applied_versions(conn)
            migrations = self._discover_migrations()

            for version, name, path in migrations:
                if version in applied_versions:
                    result["skipped"].append({"version": version, "name": name})
                    continue

                if dry_run:
                    result["applied"].append({"version": version, "name": name})
                    continue

                sql = path.read_text()
                checksum = self._checksum(path)
                now = datetime.now(timezone.utc).isoformat()

                try:
                    conn.execute("BEGIN")
                    # Execute each statement individually (not executescript which auto-commits)
                    for statement in self._split_sql(sql):
                        conn.execute(statement)
                    conn.execute(
                        "INSERT INTO migration_registry (version, name, filename, applied_at, checksum) VALUES (?, ?, ?, ?, ?)",
                        (version, name, path.name, now, checksum),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', ?)",
                        (str(version),),
                    )
                    conn.execute("COMMIT")
                    result["applied"].append({"version": version, "name": name})
                    logger.info(f"Migration {version:03d}_{name} applied successfully")
                except Exception as e:
                    conn.execute("ROLLBACK")
                    result["error"] = f"Migration {version:03d}_{name} failed: {e}"
                    logger.error(result["error"])
                    break

        except Exception as e:
            result["error"] = str(e)
        return result

    def check(self, conn) -> dict:
        """Diagnosis only (safe for read-only/locked DB).

        Returns: {current_version, pending: [...], registry: [...], healthy: bool, checksum_mismatches: [...]}
        """
        current_version = self._get_current_version(conn)
        registry = self._get_registry(conn)
        applied_versions = {r["version"] for r in registry}
        migrations = self._discover_migrations()

        pending = [
            {"version": v, "name": n, "filename": p.name}
            for v, n, p in migrations
            if v not in applied_versions
        ]

        # Check for checksum mismatches
        checksum_mismatches = []
        registry_by_version = {r["version"]: r for r in registry}
        for version, name, path in migrations:
            if version in registry_by_version:
                stored = registry_by_version[version].get("checksum")
                if stored and stored != self._checksum(path):
                    checksum_mismatches.append({"version": version, "name": name, "filename": path.name})

        healthy = len(pending) == 0 and len(checksum_mismatches) == 0

        return {
            "current_version": current_version,
            "pending": pending,
            "registry": registry,
            "healthy": healthy,
            "checksum_mismatches": checksum_mismatches,
        }

    @staticmethod
    def _split_sql(sql: str) -> list[str]:
        """Split SQL text into individual statements, skipping empty/transaction control ones."""
        # Transaction control is handled by the runner, skip them in migration files
        skip_keywords = {"begin", "begin transaction", "commit", "rollback"}
        statements = []
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            # Remove leading comment lines
            lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
            clean = "\n".join(lines).strip()
            if clean and clean.lower() not in skip_keywords:
                statements.append(stmt)
        return statements
