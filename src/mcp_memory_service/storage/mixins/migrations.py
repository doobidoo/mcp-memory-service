"""MigrationsMixin: schema migrations, FTS5, table creation, initialize."""

import sqlite3
import logging
import re
import traceback
import os
import asyncio
from pathlib import Path

try:
    import sqlite_vec
    from sqlite_vec import serialize_float32
except ImportError:
    pass

from ..migration_runner import MigrationRunner

logger = logging.getLogger(__name__)


class MigrationsMixin:
    """Mixin providing database initialization, migrations, and schema setup."""

    def _run_schema_migrations(self):
        """Execute all pending schema migrations using versioned MigrationRunner."""
        try:
            migrations_dir = Path(__file__).parent.parent / "migrations"
            if not migrations_dir.exists():
                logger.debug("Migrations directory not found, skipping schema migrations")
                return
            runner = MigrationRunner(migrations_dir)
            result = runner.run_pending(self.conn)
            if result["error"]:
                logger.warning(f"Schema migration warning: {result['error']}")
            else:
                version = runner._get_current_version(self.conn)
                applied_count = len(result["applied"])
                if applied_count > 0:
                    logger.info(f"Schema at v{version}, {applied_count} migrations applied")
                else:
                    logger.debug(f"Schema at v{version}, no pending migrations")
        except Exception as e:
            logger.warning(f"Failed to run schema migrations (non-fatal): {e}")

    def _ensure_fts5_initialized(self):
        """Ensure FTS5 virtual table exists for BM25 keyword search (v10.8.0+)."""
        try:
            cursor = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE name='memory_content_fts'"
            )
            if cursor.fetchone() is not None:
                return

            logger.info("Creating FTS5 table for hybrid BM25 search...")
            self.conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_content_fts USING fts5(
                    content,
                    content='memories',
                    content_rowid='id',
                    tokenize='trigram'
                )
            ''')

            self.conn.execute('''
                CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories
                BEGIN
                    INSERT INTO memory_content_fts(rowid, content)
                    VALUES (new.id, new.content);
                END;
            ''')
            self.conn.execute('''
                CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories
                BEGIN
                    DELETE FROM memory_content_fts WHERE rowid = old.id;
                    INSERT INTO memory_content_fts(rowid, content)
                    VALUES (new.id, new.content);
                END;
            ''')
            self.conn.execute('''
                CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories
                BEGIN
                    DELETE FROM memory_content_fts WHERE rowid = old.id;
                END;
            ''')

            logger.info("Rebuilding FTS5 trigram index from memories table...")
            self.conn.execute(
                "INSERT INTO memory_content_fts(memory_content_fts) VALUES('rebuild')"
            )

            self.conn.execute("""
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES ('fts5_enabled', 'true')
            """)
            self.conn.commit()
            logger.info("FTS5 initialization complete")
        except Exception as e:
            logger.warning(f"FTS5 initialization failed (non-fatal): {e}")

    async def initialize(self, strict_dimension_check: bool = True):
        """Initialize the SQLite database with vec0 extension.

        Args:
            strict_dimension_check: When True (default), a mismatch between the
                loaded embedding model's dimension and an existing database's vec0
                table raises a RuntimeError (fail fast, #143). Read-only diagnostics
                (e.g. `memory status`) pass False so they can open a mismatched
                database and report the problem instead of crashing.
        """
        if self._initialized:
            return

        try:
            self._check_dependencies()

            extension_supported, support_message = self._check_extension_support()
            if not extension_supported:
                self._handle_extension_loading_failure()

            await self._run_in_thread(self._connect_and_load_extension)

            try:
                def _check_tables_exist():
                    c1 = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
                    mem_exists = c1.fetchone() is not None
                    c2 = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'")
                    emb_exists = c2.fetchone() is not None
                    return mem_exists, emb_exists

                memories_table_exists, embeddings_table_exists = await self._execute_with_retry(_check_tables_exist)

                if memories_table_exists and embeddings_table_exists:
                    logger.info("Database already initialized, checking for schema migrations...")

                    try:
                        def _migrate_deleted_at_existing():
                            cursor = self.conn.execute("PRAGMA table_info(memories)")
                            columns = [row[1] for row in cursor.fetchall()]
                            if 'deleted_at' not in columns:
                                self.conn.execute('ALTER TABLE memories ADD COLUMN deleted_at REAL DEFAULT NULL')
                                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_deleted_at ON memories(deleted_at)')
                                self.conn.commit()
                                return True
                            return False

                        migrated = await self._execute_with_retry(_migrate_deleted_at_existing)
                        if migrated:
                            logger.info("Migration complete: deleted_at column added")
                        else:
                            logger.debug("Migration check: deleted_at column already exists")
                    except Exception as e:
                        logger.warning(f"Migration check for deleted_at (non-fatal): {e}")

                    # Multi-store migration: add store partition key to vec0 and store column to memories
                    try:
                        def _migrate_multi_store():
                            # Check if vec0 table already has partition key
                            cursor = self.conn.execute(
                                "SELECT sql FROM sqlite_master WHERE name='memory_embeddings'"
                            )
                            row = cursor.fetchone()
                            if row and 'partition' in (row[0] or '').lower():
                                # Already migrated. Recover from an interrupted prior
                                # migration: if a durable backup table survives, it means
                                # a previous run dropped/recreated the vec0 table but
                                # crashed before removing the backup. Restore any rows
                                # that are missing from the (possibly empty) new table
                                # instead of silently accepting the loss (issue #134).
                                backup_exists = self.conn.execute(
                                    "SELECT name FROM sqlite_master WHERE type='table' "
                                    "AND name='memory_embeddings_migration_backup'"
                                ).fetchone() is not None
                                if backup_exists:
                                    backup_count = self.conn.execute(
                                        "SELECT COUNT(*) FROM memory_embeddings_migration_backup"
                                    ).fetchone()[0]
                                    current_count = self.conn.execute(
                                        "SELECT COUNT(*) FROM memory_embeddings"
                                    ).fetchone()[0]
                                    if current_count < backup_count:
                                        logger.warning(
                                            "Detected interrupted multi-store migration: vec0 table "
                                            "has %s rows but backup has %s; restoring missing embeddings "
                                            "from memory_embeddings_migration_backup...",
                                            current_count, backup_count
                                        )
                                        # vec0 virtual tables do NOT honor INSERT OR IGNORE:
                                        # a rowid conflict raises a UNIQUE constraint error
                                        # regardless. Skip rows already present by checking
                                        # existing rowids first, then plain-INSERT the rest,
                                        # so a partial (mid-reinsert) restore does not crash.
                                        existing_rowids = {
                                            r[0] for r in self.conn.execute(
                                                "SELECT rowid FROM memory_embeddings"
                                            ).fetchall()
                                        }
                                        for bk_row in self.conn.execute(
                                            "SELECT rowid, content_embedding "
                                            "FROM memory_embeddings_migration_backup"
                                        ).fetchall():
                                            if bk_row[0] in existing_rowids:
                                                continue
                                            self.conn.execute(
                                                "INSERT INTO memory_embeddings "
                                                "(rowid, content_embedding, store) VALUES (?, ?, ?)",
                                                (bk_row[0], bk_row[1], 'default')
                                            )
                                        restored_count = self.conn.execute(
                                            "SELECT COUNT(*) FROM memory_embeddings"
                                        ).fetchone()[0]
                                        logger.info(
                                            "Recovery complete: vec0 table now has %s embeddings",
                                            restored_count
                                        )
                                    self.conn.execute(
                                        "DROP TABLE IF EXISTS memory_embeddings_migration_backup"
                                    )
                                    self.conn.commit()
                                return False  # Already migrated

                            logger.info("Migrating memory_embeddings to add store partition key...")
                            # Derive the embedding dimension from the EXISTING vec0 table's
                            # DDL, NOT from self.embedding_dimension. The embedding model has
                            # not been initialized yet at this point, so that attribute still
                            # holds the hardcoded constructor default (384) and would silently
                            # destroy the embeddings of any database built with a different
                            # dimension (issue #134). Refuse to drop anything if we cannot
                            # determine the dimension.
                            dim_match = re.search(r'FLOAT\[(\d+)\]', row[0] or '')
                            if not dim_match:
                                raise RuntimeError(
                                    "Cannot determine embedding dimension from existing "
                                    "memory_embeddings table; refusing to drop it. "
                                    f"DDL was: {row[0]!r}"
                                )
                            embedding_dim_val = int(dim_match.group(1))

                            # Snapshot existing embeddings into memory. The `memories`
                            # table is the source of truth and embeddings are derivable,
                            # so re-inserting from this snapshot is a safe migration path.
                            existing = self.conn.execute(
                                "SELECT rowid, content_embedding FROM memory_embeddings"
                            ).fetchall()

                            # Crash-safety: persist the snapshot to a real (non-virtual)
                            # table and COMMIT it BEFORE the destructive DROP. Under sqlite3
                            # legacy isolation, DDL autocommits, so the rollback below cannot
                            # restore a dropped vec0 table; this durable backup lets the next
                            # startup recover the data instead of losing it (issue #134).
                            self.conn.execute(
                                "DROP TABLE IF EXISTS memory_embeddings_migration_backup"
                            )
                            self.conn.execute(
                                "CREATE TABLE memory_embeddings_migration_backup ("
                                "rowid INTEGER, content_embedding BLOB)"
                            )
                            for bk_row in existing:
                                self.conn.execute(
                                    "INSERT INTO memory_embeddings_migration_backup "
                                    "(rowid, content_embedding) VALUES (?, ?)",
                                    (bk_row[0], bk_row[1])
                                )
                            self.conn.commit()

                            try:
                                # Drop-and-recreate instead of rename. `memory_embeddings`
                                # is a vec0 virtual table; ALTER TABLE ... RENAME does NOT
                                # carry its shadow tables (memory_embeddings_info/_chunks/
                                # _rowids/...), so the rename-based migration collided with
                                # 'memory_embeddings_info already exists' on existing DBs
                                # (issue #68). vec0's DROP removes the shadow tables cleanly.
                                self.conn.execute("DROP TABLE IF EXISTS memory_embeddings")
                                self.conn.execute("DROP TABLE IF EXISTS memory_embeddings_old")
                                # Defensively clear any orphaned vec0 shadow tables left by an
                                # earlier aborted run (parent virtual table already gone).
                                orphans = self.conn.execute(
                                    "SELECT name FROM sqlite_master WHERE type='table' "
                                    "AND name LIKE 'memory_embeddings\\_%' ESCAPE '\\' "
                                    "AND name != 'memory_embeddings_migration_backup'"
                                ).fetchall()
                                for (orphan_name,) in orphans:
                                    self.conn.execute(f'DROP TABLE IF EXISTS "{orphan_name}"')

                                # Create the partitioned table fresh (no rename -> no collision).
                                self.conn.execute(f'''
                                    CREATE VIRTUAL TABLE memory_embeddings USING vec0(
                                        content_embedding FLOAT[{embedding_dim_val}] distance_metric=cosine,
                                        store TEXT partition key
                                    )
                                ''')
                                # Re-insert with store='default'
                                for row in existing:
                                    self.conn.execute(
                                        "INSERT INTO memory_embeddings (rowid, content_embedding, store) VALUES (?, ?, ?)",
                                        (row[0], row[1], 'default')
                                    )
                                # Verify row count
                                new_count = self.conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0]
                                if new_count != len(existing):
                                    raise RuntimeError(
                                        f"Migration verification failed: expected {len(existing)}, got {new_count}"
                                    )
                                self.conn.commit()
                                logger.info("Multi-store migration complete: %s embeddings migrated", new_count)
                                # Verification passed and committed: the durable backup is
                                # no longer needed, so drop it and commit the cleanup.
                                self.conn.execute(
                                    "DROP TABLE IF EXISTS memory_embeddings_migration_backup"
                                )
                                self.conn.commit()
                            except Exception as e:
                                self.conn.rollback()
                                logger.error("Multi-store migration failed: %s", e)
                                raise
                            return True

                        migrated = await self._execute_with_retry(_migrate_multi_store)
                        if migrated:
                            logger.info("Migration complete: store partition key added to memory_embeddings")
                    except Exception as e:
                        raise RuntimeError(f"Multi-store vec0 migration failed (aborting): {e}") from e

                    # Add store column to memories table
                    try:
                        def _add_store_column():
                            try:
                                self.conn.execute("ALTER TABLE memories ADD COLUMN store TEXT DEFAULT 'default'")
                                self.conn.commit()
                                return True
                            except Exception:
                                return False

                        await self._execute_with_retry(_add_store_column)
                    except Exception as e:
                        logger.warning(f"Add store column (non-fatal): {e}")

                    await self._run_in_thread(self._run_schema_migrations)
                    await self._run_in_thread(self._ensure_fts5_initialized)

                    await self._initialize_embedding_model()
                    # Guard (#143): fail loudly if the loaded model's dimension does not
                    # match the existing vec0 table. Without this, the mismatch first
                    # surfaces as a hard INSERT error far from its cause (it reads as data
                    # corruption rather than a config problem). The hash fallback adapts its
                    # dimension to the existing DB, so this only trips for a fixed-dimension
                    # model (ONNX / SentenceTransformer) configured against a DB built at a
                    # different dimension.
                    existing_dim = await self._run_in_thread(self._get_existing_db_embedding_dimension)
                    if existing_dim is not None and existing_dim != self.embedding_dimension:
                        mismatch_msg = (
                            f"Embedding dimension mismatch: the existing database's "
                            f"memory_embeddings table is FLOAT[{existing_dim}], but the configured "
                            f"embedding model '{self.embedding_model_name}' produces dimension "
                            f"{self.embedding_dimension}. All store and search operations would fail. "
                            f"Use an embedding model that outputs {existing_dim}-dim vectors, or "
                            f"re-embed the database to {self.embedding_dimension} dimensions."
                        )
                        if strict_dimension_check:
                            raise RuntimeError(mismatch_msg)
                        logger.warning("%s (continuing: strict dimension check disabled)", mismatch_msg)
                    self._initialized = True
                    logger.info(f"SQLite-vec storage initialized successfully (existing database) with embedding dimension: {self.embedding_dimension}")
                    return
            except sqlite3.Error as e:
                logger.debug(f"Could not check existing tables (will attempt full initialization): {e}")

            default_pragmas = {
                "journal_mode": "WAL",
                "busy_timeout": "5000",
                "synchronous": "NORMAL",
                "cache_size": "10000",
                "temp_store": "MEMORY"
            }

            custom_pragmas = os.environ.get("MCP_MEMORY_SQLITE_PRAGMAS", "")
            if custom_pragmas:
                for pragma_pair in custom_pragmas.split(","):
                    pragma_pair = pragma_pair.strip()
                    if "=" in pragma_pair:
                        pragma_name, pragma_value = pragma_pair.split("=", 1)
                        default_pragmas[pragma_name.strip()] = pragma_value.strip()
                        logger.info(f"Custom pragma from env: {pragma_name}={pragma_value}")

            def _apply_pragmas_and_create_tables(dp=default_pragmas):
                applied = []
                for pragma_name, pragma_value in dp.items():
                    try:
                        self.conn.execute(f"PRAGMA {pragma_name}={pragma_value}")
                        applied.append(f"{pragma_name}={pragma_value}")
                    except sqlite3.Error as e:
                        logger.warning(f"Failed to set pragma {pragma_name}={pragma_value}: {e}")

                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                ''')

                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content_hash TEXT UNIQUE NOT NULL,
                        content TEXT NOT NULL,
                        tags TEXT,
                        memory_type TEXT,
                        metadata TEXT,
                        created_at REAL,
                        updated_at REAL,
                        created_at_iso TEXT,
                        updated_at_iso TEXT,
                        deleted_at REAL DEFAULT NULL
                    )
                ''')
                return applied

            applied_pragmas = await self._execute_with_retry(_apply_pragmas_and_create_tables)
            logger.info(f"SQLite pragmas applied: {', '.join(applied_pragmas)}")

            try:
                def _migrate_deleted_at_new():
                    cursor = self.conn.execute("PRAGMA table_info(memories)")
                    columns = [row[1] for row in cursor.fetchall()]
                    if 'deleted_at' not in columns:
                        self.conn.execute('ALTER TABLE memories ADD COLUMN deleted_at REAL DEFAULT NULL')
                        self.conn.commit()
                        return True
                    return False

                if await self._execute_with_retry(_migrate_deleted_at_new):
                    logger.info("Migration complete: deleted_at column added")
            except Exception as e:
                logger.warning(f"Migration check for deleted_at (non-fatal): {e}")

            await self._initialize_embedding_model()

            try:
                def _check_distance_migration():
                    cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
                    metadata_exists = cursor.fetchone() is not None
                    if not metadata_exists:
                        return False
                    cursor = self.conn.execute("SELECT value FROM metadata WHERE key='distance_metric'")
                    current_metric = cursor.fetchone()
                    return not current_metric or current_metric[0] != 'cosine'

                needs_migration = await self._execute_with_retry(_check_distance_migration)

                if needs_migration:
                    logger.info("Migrating embeddings table from L2 to cosine distance...")
                    logger.info("This is a one-time operation - embeddings will be regenerated automatically")

                    max_retries = 3
                    retry_delay = 1.0

                    for attempt in range(max_retries):
                        try:
                            def _drop_embeddings():
                                self.conn.execute("DROP TABLE IF EXISTS memory_embeddings")

                            await self._execute_with_retry(_drop_embeddings)
                            logger.info("Successfully dropped old embeddings table")
                            break
                        except sqlite3.OperationalError as drop_error:
                            if "database is locked" in str(drop_error):
                                if attempt < max_retries - 1:
                                    logger.warning(f"Database locked during migration (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                                    await asyncio.sleep(retry_delay)
                                    retry_delay *= 2
                                else:
                                    def _check_emb_exists():
                                        cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'")
                                        return cursor.fetchone() is not None

                                    if not await self._execute_with_retry(_check_emb_exists):
                                        logger.info("Embeddings table doesn't exist - migration likely completed by another process")
                                        break
                                    else:
                                        logger.error("Failed to drop embeddings table after retries - will attempt to continue")
                                        break
                            else:
                                raise
                else:
                    logger.debug("Fresh database or cosine distance already configured, no migration needed")
            except Exception as e:
                logger.warning(f"Migration check warning (non-fatal): {e}")

            embedding_dim = self.embedding_dimension

            def _create_virtual_table_and_indexes():
                self.conn.execute(f'''
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
                        content_embedding FLOAT[{embedding_dim}] distance_metric=cosine,
                        store TEXT partition key
                    )
                ''')
                self.conn.execute("""
                    INSERT OR REPLACE INTO metadata (key, value) VALUES ('distance_metric', 'cosine')
                """)
                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash)')
                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)')
                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)')
                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_deleted_at ON memories(deleted_at)')

                # Add store column to memories table
                try:
                    self.conn.execute("ALTER TABLE memories ADD COLUMN store TEXT DEFAULT 'default'")
                except Exception:
                    pass  # Column already exists

            await self._execute_with_retry(_create_virtual_table_and_indexes)

            await self._run_in_thread(self._ensure_fts5_initialized)
            await self._run_in_thread(self._run_schema_migrations)

            self._initialized = True

            logger.info(f"SQLite-vec storage initialized successfully with embedding dimension: {self.embedding_dimension}")

        except Exception as e:
            error_msg = f"Failed to initialize SQLite-vec storage: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
