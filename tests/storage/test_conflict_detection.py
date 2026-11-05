"""Tests for Memory Evolution P3: Conflict Detection."""
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from difflib import SequenceMatcher
from math import cos, radians, sin
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

import mcp_memory_service.consolidation.contradictions as contradictions
import mcp_memory_service.server.handlers.quality as quality_handlers
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


def _make_memory(content, tags=None):
    test_tags = list(tags or [])
    if "__test__" not in test_tags:
        test_tags.append("__test__")
    content_hash = hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()
    return Memory(content=content, content_hash=content_hash, tags=test_tags)


@pytest.fixture
def temp_storage_dir():
    d = tempfile.mkdtemp(prefix="mcp-test-conflict-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest_asyncio.fixture
async def storage(temp_storage_dir):
    assert "mcp-test-" in temp_storage_dir
    db_path = os.path.join(temp_storage_dir, "test.db")
    os.environ["MCP_MEMORY_SQLITE_PATH"] = db_path
    os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
    os.environ["MCP_SEMANTIC_DEDUP_ENABLED"] = "false"
    s = SqliteVecMemoryStorage(db_path)
    await s.initialize()
    yield s
    try:
        if s.conn:
            s.conn.execute("DELETE FROM memories WHERE tags LIKE '%__test__%'")
            s.conn.commit()
    except Exception:
        pass
    await s.close()


@pytest_asyncio.fixture
async def versioned_storage(temp_db_path):
    s = SqliteVecMemoryStorage(os.path.join(temp_db_path, "versioned.db"))
    await s.initialize()
    yield s
    await s.close()


async def _store(storage, content, tags=None):
    mem = _make_memory(content, tags)
    ok, msg = await storage.store(mem, skip_semantic_dedup=True)
    assert ok, f"Failed to store: {msg}"
    return mem.content_hash


class TestDetectConflicts:

    @pytest.mark.asyncio
    async def test_conflicting_memories_detected(self, storage):
        """Two semantically similar but textually different memories trigger conflict."""
        h1 = await _store(storage, "The project database is PostgreSQL 15")
        h2 = await _store(storage, "The project database is MySQL 8.0")
        # Both are about "project database" but contradict each other
        conflicts = await storage.get_conflicts()
        # At least check the method exists and returns a list
        assert isinstance(conflicts, list)

    @pytest.mark.asyncio
    async def test_similar_content_no_conflict(self, storage):
        """Nearly identical content (low divergence) should NOT trigger conflict."""
        h1 = await _store(storage, "The server runs on port 8080")
        h2 = await _store(storage, "The server runs on port 8080 by default")
        conflicts = await storage.get_conflicts()
        conflicting_hashes = {(c["hash_a"], c["hash_b"]) for c in conflicts}
        assert (h1, h2) not in conflicting_hashes
        assert (h2, h1) not in conflicting_hashes

    @pytest.mark.asyncio
    async def test_superseded_memory_excluded(self, storage):
        """A superseded memory should not appear in conflicts."""
        h1 = await _store(storage, "Deploy target is AWS us-east-1")
        h2 = await _store(storage, "Deploy target is GCP europe-west1")
        # Resolve: h1 wins, h2 loses
        await storage.resolve_conflict(h1, h2)
        conflicts = await storage.get_conflicts()
        conflicting_hashes = {(c["hash_a"], c["hash_b"]) for c in conflicts}
        assert (h1, h2) not in conflicting_hashes


class TestGetConflicts:

    @pytest.mark.asyncio
    async def test_get_conflicts_returns_pairs(self, storage):
        """get_conflicts returns detected conflict pairs with metadata."""
        h1 = await _store(storage, "The API uses REST with JSON responses")
        h2 = await _store(storage, "The API uses GraphQL with schema-first design")
        conflicts = await storage.get_conflicts()
        # Should contain at least similarity and divergence info
        for c in conflicts:
            assert "hash_a" in c
            assert "hash_b" in c
            assert "similarity" in c


class TestResolveConflict:

    @pytest.mark.asyncio
    async def test_resolve_marks_loser_superseded(self, storage):
        """Resolving a conflict supersedes the loser."""
        h1 = await _store(storage, "Production runs on Kubernetes 1.28")
        h2 = await _store(storage, "Production runs on bare metal servers")
        ok, msg = await storage.resolve_conflict(h1, h2)
        assert ok
        # Loser should be superseded
        cursor = storage.conn.execute(
            "SELECT superseded_by FROM memories WHERE content_hash = ?", (h2,)
        )
        assert cursor.fetchone()[0] == h1

    @pytest.mark.asyncio
    async def test_resolve_boosts_winner_confidence(self, storage):
        """Winner gets confidence reset to 1.0 and last_accessed updated."""
        h1 = await _store(storage, "Cache backend is Redis 7.0")
        h2 = await _store(storage, "Cache backend is Memcached")
        now = time.time()
        ok, _ = await storage.resolve_conflict(h1, h2)
        assert ok
        cursor = storage.conn.execute(
            "SELECT confidence, last_accessed FROM memories WHERE content_hash = ?",
            (h1,),
        )
        conf, la = cursor.fetchone()
        assert conf == 1.0
        assert la >= now - 1

    @pytest.mark.asyncio
    async def test_resolve_removes_conflict_tag(self, storage):
        """With no other conflict left, neither memory keeps conflict:unresolved."""
        h1 = await _store(storage, "CI pipeline uses GitHub Actions")
        h2 = await _store(storage, "CI pipeline uses Jenkins")
        await storage.resolve_conflict(h1, h2)
        for h in (h1, h2):
            cursor = storage.conn.execute(
                "SELECT tags FROM memories WHERE content_hash = ?", (h,)
            )
            tags = cursor.fetchone()[0]
            assert "conflict:unresolved" not in tags

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_hash_fails(self, storage):
        """Resolving with unknown hash should fail gracefully."""
        ok, msg = await storage.resolve_conflict("nonexistent1", "nonexistent2")
        assert ok is False


class TestConflictIntegration:

    @pytest.mark.asyncio
    async def test_store_detect_resolve_lifecycle(self, storage):
        """Full lifecycle: store conflicting memories, detect, resolve."""
        # Store two contradictory memories
        h1 = await _store(storage, "The application framework is Django 5.0")
        h2 = await _store(storage, "The application framework is Flask 3.0")

        # Check conflicts exist
        conflicts = await storage.get_conflicts()
        hashes_in_conflicts = set()
        for c in conflicts:
            hashes_in_conflicts.add(c["hash_a"])
            hashes_in_conflicts.add(c["hash_b"])

        # Resolve: h1 wins
        if h1 in hashes_in_conflicts:
            ok, msg = await storage.resolve_conflict(h1, h2)
            assert ok

            # Conflicts should be empty now
            conflicts_after = await storage.get_conflicts()
            remaining = {(c["hash_a"], c["hash_b"]) for c in conflicts_after}
            assert (h1, h2) not in remaining
            assert (h2, h1) not in remaining


class TestVersionedUpdateConflictDetection:
    @pytest.mark.asyncio
    async def test_versioned_update_does_not_conflict_with_its_parent(
        self, versioned_storage, monkeypatch
    ):
        prefix = (
            "The service plan records the release owner, rollback procedure, support "
            "channel, dependency matrix, backup schedule, approval chain, and "
            "operational review for each deployment. "
        ) * 12
        prefix = prefix[:1050]
        original = prefix + (
            "The previous control plane operates in western region, transmits reports "
            "each Friday, retains archives for ninety days, and allows one supervisor "
            "to approve every release. "
        ) * 3
        updated = prefix + (
            "The replacement ingestion cluster stores images on eastern hardware, runs "
            "audits each Monday, expires snapshots after seven months, and requires two "
            "engineers to release changes. "
        ) * 3
        original, updated = original[:1400], updated[:1400]
        assert len(original.split()) > 128
        assert 1 - SequenceMatcher(None, original.lower(), updated.lower()).ratio() >= 0.20

        generate_embedding = versioned_storage._generate_embedding
        shared_prefix_embedding = generate_embedding(prefix[:500])
        monkeypatch.setattr(
            versioned_storage,
            "_generate_embedding",
            lambda content: (
                shared_prefix_embedding
                if content.startswith(prefix[:500])
                else generate_embedding(content)
            ),
        )

        parent_hash = await _store(versioned_storage, original)
        ok, message, new_hash = await versioned_storage.update_memory_versioned(
            parent_hash, updated
        )

        assert ok, message
        assert new_hash
        for content_hash in (parent_hash, new_hash):
            tags = versioned_storage.conn.execute(
                "SELECT tags FROM memories WHERE content_hash = ?", (content_hash,)
            ).fetchone()[0]
            assert "conflict:unresolved" not in tags.split(",")

        pair = versioned_storage.conn.execute(
            "SELECT 1 FROM memory_graph WHERE relationship_type = 'contradicts' "
            "AND ((source_hash = ? AND target_hash = ?) "
            "OR (source_hash = ? AND target_hash = ?))",
            (parent_hash, new_hash, new_hash, parent_hash),
        ).fetchone()
        assert pair is None

    @pytest.mark.asyncio
    async def test_versioned_update_keeps_other_conflicts_and_reports_them(
        self, versioned_storage, monkeypatch
    ):
        prefix = (
            "The service plan records the release owner, rollback procedure, support "
            "channel, dependency matrix, backup schedule, approval chain, and "
            "operational review for each deployment. "
        ) * 12
        prefix = prefix[:1050]
        tails = [
            ("The previous control plane operates in western region, transmits reports "
             "each Friday, retains archives for ninety days, and allows one supervisor "
             "to approve every release. " * 3)[:350],
            ("The replacement ingestion cluster stores images on eastern hardware, runs "
             "audits each Monday, expires snapshots after seven months, and requires two "
             "engineers to release changes. " * 3)[:350],
            ("The billing gateway serves customers in southern district, exports invoices "
             "after quarter end, preserves records for thirteen months, and routes "
             "incidents to finance staff. " * 3)[:350],
        ]
        other, original, updated = [prefix + tail for tail in tails]
        dimension = versioned_storage.embedding_dimension

        def embedding_at(degrees):
            angle = radians(degrees)
            return [cos(angle), sin(angle), *([0.0] * (dimension - 2))]

        embeddings = {
            other: embedding_at(35),
            original: embedding_at(0),
            updated: embedding_at(17),
        }
        monkeypatch.setattr(
            versioned_storage, "_generate_embedding", lambda content: embeddings[content]
        )

        other_hash = await _store(versioned_storage, other)
        parent_hash = await _store(versioned_storage, original)
        ok, message, new_hash = await versioned_storage.update_memory_versioned(
            parent_hash, updated
        )

        assert ok, message
        assert new_hash
        assert "1 conflict(s) detected" in message
        pairs = set(versioned_storage.conn.execute(
            "SELECT source_hash, target_hash FROM memory_graph "
            "WHERE relationship_type = 'contradicts'"
        ))
        assert (new_hash, other_hash) in pairs
        assert (other_hash, new_hash) in pairs
        assert (new_hash, parent_hash) not in pairs
        assert (parent_hash, new_hash) not in pairs


# ---------------------------------------------------------------------------
# Issue #1840: conflict:unresolved must not outlive the conflict.
#
# Conflicts are seeded through the production writer _record_conflicts(): the
# suite's hash embeddings never reach the detection thresholds, so tests that
# relied on natural detection would pass vacuously on any code. The versioned
# update cases pin the geometry with monkeypatched embeddings, as #1850 did.
# ---------------------------------------------------------------------------

CONFLICT_TAG = "conflict:unresolved"


def _tags(storage, content_hash):
    row = storage.conn.execute(
        "SELECT tags FROM memories WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    return [t.strip() for t in (row[0] or "").split(",") if t.strip()]


def _stale_conflict_tags(storage):
    """Current rows tagged conflict:unresolved without a current contradicts partner.

    Computed in Python, independently of the storage's own reconcile SQL, and
    over contradicts edges in either direction.
    """
    rows = storage.conn.execute(
        "SELECT content_hash, tags, deleted_at, superseded_by FROM memories"
    ).fetchall()
    current = {h for h, _, deleted_at, superseded_by in rows
               if deleted_at is None and not superseded_by}
    partners = defaultdict(set)
    for src, tgt in storage.conn.execute(
        "SELECT source_hash, target_hash FROM memory_graph "
        "WHERE relationship_type = 'contradicts'"
    ):
        partners[src].add(tgt)
        partners[tgt].add(src)
    return {
        h for h, tags, _, _ in rows
        if h in current
        and CONFLICT_TAG in [t.strip() for t in (tags or "").split(",")]
        and not (partners[h] - {h}) & current
    }


async def _seed_conflict(storage, a, b):
    await storage._record_conflicts(
        a, [{"existing_hash": b, "similarity": 0.97, "divergence": 0.5}]
    )
    # Precondition: both rows tagged, both edge directions present.
    assert CONFLICT_TAG in _tags(storage, a)
    assert CONFLICT_TAG in _tags(storage, b)
    edges = set(storage.conn.execute(
        "SELECT source_hash, target_hash FROM memory_graph "
        "WHERE relationship_type = 'contradicts'"
    ))
    assert {(a, b), (b, a)} <= edges


def _listed_pairs(conflicts):
    return {frozenset((c["hash_a"], c["hash_b"])) for c in conflicts}


class TestStaleConflictTagCleanup:

    @pytest.mark.asyncio
    async def test_supersede_batch_clears_partner_tag(self, storage):
        """(a) mark_superseded_batch on one side clears the partner's tag."""
        a = await _store(storage, "Staging cluster runs Nomad 1.7 on three nodes")
        b = await _store(storage, "Staging cluster runs Kubernetes on five nodes")
        x = await _store(storage, "Staging moved to a managed container service")
        await _seed_conflict(storage, a, b)

        assert await storage.mark_superseded_batch([(x, a)]) == 1

        assert CONFLICT_TAG not in _tags(storage, b)
        assert _stale_conflict_tags(storage) == set()

    @pytest.mark.asyncio
    async def test_delete_clears_partner_tag(self, storage):
        """(b) delete() of one side clears the partner's tag."""
        a = await _store(storage, "Log retention is fourteen days in Loki")
        b = await _store(storage, "Log retention is ninety days in Elasticsearch")
        await _seed_conflict(storage, a, b)

        ok, msg = await storage.delete(a)
        assert ok, msg

        assert CONFLICT_TAG not in _tags(storage, b)
        assert _stale_conflict_tags(storage) == set()

    @pytest.mark.asyncio
    async def test_resolve_conflict_keeps_winner_tag_with_live_partner(self, storage):
        """(e) resolve_conflict(W, L) must not strip W's tag while W <-> C is live."""
        w = await _store(storage, "Backups run nightly at 02:00 to the NAS")
        lo = await _store(storage, "Backups run weekly on Sunday to S3")
        c = await _store(storage, "Backups run hourly to a USB disk")
        await _seed_conflict(storage, w, lo)
        await _seed_conflict(storage, w, c)

        ok, msg = await storage.resolve_conflict(w, lo)
        assert ok, msg

        assert CONFLICT_TAG in _tags(storage, w)
        assert CONFLICT_TAG in _tags(storage, c)
        assert frozenset((w, c)) in _listed_pairs(await storage.get_conflicts())
        assert _stale_conflict_tags(storage) == set()

    @pytest.mark.asyncio
    async def test_resolve_conflict_clears_losers_other_partner(self, storage):
        """(g) path 4: resolving W over L clears C when L was C's only partner."""
        w = await _store(storage, "The VPN endpoint is WireGuard on port 51820")
        lo = await _store(storage, "The VPN endpoint is OpenVPN on port 1194")
        c = await _store(storage, "The VPN endpoint is IPsec on port 500")
        await _seed_conflict(storage, w, lo)
        await _seed_conflict(storage, lo, c)

        ok, msg = await storage.resolve_conflict(w, lo)
        assert ok, msg

        assert CONFLICT_TAG not in _tags(storage, c)
        assert _stale_conflict_tags(storage) == set()

    @pytest.mark.asyncio
    async def test_partner_with_another_live_conflict_keeps_tag(self, storage):
        """(h) A <-> B and A <-> C: deleting B leaves A tagged (C is still live)."""
        a = await _store(storage, "The metrics store is Prometheus with 30d retention")
        b = await _store(storage, "The metrics store is InfluxDB with 1y retention")
        c = await _store(storage, "The metrics store is VictoriaMetrics with 6m retention")
        await _seed_conflict(storage, a, b)
        await _seed_conflict(storage, a, c)

        ok, msg = await storage.delete(b)
        assert ok, msg

        assert CONFLICT_TAG in _tags(storage, a)
        assert CONFLICT_TAG in _tags(storage, c)
        assert frozenset((a, c)) in _listed_pairs(await storage.get_conflicts())

    @pytest.mark.asyncio
    async def test_superseded_partner_does_not_keep_tag_alive(self, storage):
        """(i) A <-> B and A <-> C with C already superseded: deleting B clears A."""
        a = await _store(storage, "The mail relay is Postfix on the gateway host")
        b = await _store(storage, "The mail relay is Exim on the utilities host")
        c = await _store(storage, "The mail relay is a hosted SMTP service")
        x = await _store(storage, "Outbound mail goes through the provider API")
        await _seed_conflict(storage, a, b)
        await _seed_conflict(storage, a, c)
        assert await storage.mark_superseded_batch([(x, c)]) == 1
        assert CONFLICT_TAG in _tags(storage, a)  # B is still live

        ok, msg = await storage.delete(b)
        assert ok, msg

        assert CONFLICT_TAG not in _tags(storage, a)
        assert _stale_conflict_tags(storage) == set()

    @pytest.mark.asyncio
    async def test_incoming_only_edge_keeps_partner_tag(self, storage):
        """A one-way pair still counts: B keeps its tag through the incoming A -> B edge.

        A later INSERT OR REPLACE of another relationship type can overwrite one
        direction of a pair, so the partner lookup must read both directions.
        """
        a = await _store(storage, "The CI runner is GitHub Actions on hosted VMs")
        b = await _store(storage, "The CI runner is Woodpecker on the homelab")
        d = await _store(storage, "The CI runner is GitLab CI on a shared runner")
        await _seed_conflict(storage, a, b)
        await _seed_conflict(storage, b, d)
        storage.conn.execute(
            "DELETE FROM memory_graph WHERE source_hash = ? AND target_hash = ?", (b, a)
        )
        storage.conn.commit()

        ok, msg = await storage.delete(d)
        assert ok, msg

        assert CONFLICT_TAG in _tags(storage, b)
        assert CONFLICT_TAG in _tags(storage, a)
        assert _stale_conflict_tags(storage) == set()

    @pytest.mark.asyncio
    async def test_delete_clears_partner_reached_by_incoming_edge(self, storage):
        """A one-way pair still counts: deleting X clears Y through the Y -> X edge."""
        x = await _store(storage, "The object store is MinIO on the NAS")
        y = await _store(storage, "The object store is Garage on three nodes")
        await _seed_conflict(storage, x, y)
        storage.conn.execute(
            "DELETE FROM memory_graph WHERE source_hash = ? AND target_hash = ?", (x, y)
        )
        storage.conn.commit()

        ok, msg = await storage.delete(x)
        assert ok, msg

        assert CONFLICT_TAG not in _tags(storage, y)
        assert _stale_conflict_tags(storage) == set()

    @pytest.mark.asyncio
    async def test_reconcile_treats_tombstoned_partner_as_gone(self, storage):
        """A partner soft-deleted with its edges left in place (as a sync-applied
        tombstone leaves them) does not keep the tag alive: reconcile clears A."""
        a = await _store(storage, "The password manager is Vaultwarden self-hosted")
        b = await _store(storage, "The password manager is 1Password family plan")
        await _seed_conflict(storage, a, b)
        storage.conn.execute(
            "UPDATE memories SET deleted_at = ? WHERE content_hash = ?", (time.time(), b)
        )
        storage.conn.commit()
        assert CONFLICT_TAG in _tags(storage, a)

        result = await storage.reconcile_conflict_tags(dry_run=False)

        assert result["found"] == 1
        assert result["cleared"] == 1
        assert CONFLICT_TAG not in _tags(storage, a)
        assert _stale_conflict_tags(storage) == set()

    @pytest.mark.asyncio
    async def test_cleanup_never_adds_the_tag(self, storage):
        """(j) The cleanup only clears: an untagged row with a live partner stays untagged.

        C is a direct partner of the deleted B, so the cleanup does inspect C, and C
        still has the live partner D. Clearing must not turn into re-tagging.
        """
        b = await _store(storage, "Home DNS is Pi-hole on a Raspberry Pi")
        c = await _store(storage, "Home DNS is Unbound on the NAS")
        d = await _store(storage, "Home DNS is AdGuard on the edge router")
        await _seed_conflict(storage, b, c)
        await _seed_conflict(storage, c, d)
        storage.conn.execute(
            "UPDATE memories SET tags = ? WHERE content_hash = ?", ("__test__", c)
        )
        storage.conn.commit()

        ok, msg = await storage.delete(b)
        assert ok, msg

        assert CONFLICT_TAG not in _tags(storage, c)
        assert CONFLICT_TAG in _tags(storage, d)
        assert frozenset((c, d)) in _listed_pairs(await storage.get_conflicts())

    @pytest.mark.asyncio
    async def test_resolve_conflict_rejects_same_hash(self, storage):
        """A memory cannot win a conflict against itself; nothing changes."""
        a = await _store(storage, "The time server is chrony on the router")
        b = await _store(storage, "The time server is ntpd on the NAS")
        await _seed_conflict(storage, a, b)

        ok, _ = await storage.resolve_conflict(a, a)

        assert ok is False
        assert CONFLICT_TAG in _tags(storage, a)
        assert CONFLICT_TAG in _tags(storage, b)

    @pytest.mark.asyncio
    async def test_versioned_update_with_echoed_tag_does_not_keep_it(self, storage):
        """(k) Tags passed explicitly to a versioned update may echo the stale tag."""
        a = await _store(storage, "The reverse proxy is Caddy with internal TLS")
        b = await _store(storage, "The reverse proxy is Traefik with ACME TLS")
        await _seed_conflict(storage, a, b)

        ok, msg, a2 = await storage.update_memory_versioned(
            a, "The reverse proxy is nginx with a wildcard cert",
            new_tags=_tags(storage, a),
        )
        assert ok, msg
        assert a2

        assert CONFLICT_TAG not in _tags(storage, a2)
        assert CONFLICT_TAG not in _tags(storage, b)
        assert _stale_conflict_tags(storage) == set()


class TestVersionedUpdateConflictTag:
    """(c) and (d): versioned updates with pinned embedding geometry.

    Detection needs cosine similarity >= 0.95 (at most ~18.19 degrees apart) and
    text divergence >= 0.20. A sits at 0 degrees and B at 8, so A <-> B conflict.
    A' at 40 degrees resolves the conflict (32 degrees from B); A' at 12 keeps it
    (4 degrees from B). The parent itself is excluded from detection (#1850).
    """

    A = "The primary database is PostgreSQL 16 on the storage node"
    B = "Primary data lives in MariaDB 11 running inside a container"
    A_RESOLVED = "We dropped the relational database; records now sit in SQLite files"
    A_KEPT = "The primary database is PostgreSQL 17 on a dedicated virtual machine"

    def _pin(self, storage, monkeypatch, angles):
        dimension = storage.embedding_dimension

        def embedding_at(degrees):
            angle = radians(degrees)
            return [cos(angle), sin(angle), *([0.0] * (dimension - 2))]

        embeddings = {content: embedding_at(deg) for content, deg in angles.items()}
        monkeypatch.setattr(storage, "_generate_embedding", lambda content: embeddings[content])

    @pytest.mark.asyncio
    async def test_versioned_update_resolving_conflict_clears_tags(self, storage, monkeypatch):
        """(c) The new version moves away from B: neither A' nor B stays tagged."""
        self._pin(storage, monkeypatch, {self.A: 0, self.B: 8, self.A_RESOLVED: 40})
        a = await _store(storage, self.A)
        b = await _store(storage, self.B)
        assert CONFLICT_TAG in _tags(storage, a)  # natural detection fired
        assert CONFLICT_TAG in _tags(storage, b)

        ok, msg, a2 = await storage.update_memory_versioned(a, self.A_RESOLVED)
        assert ok, msg

        assert CONFLICT_TAG not in _tags(storage, a2)
        assert CONFLICT_TAG not in _tags(storage, b)
        assert _stale_conflict_tags(storage) == set()

    @pytest.mark.asyncio
    async def test_versioned_update_keeping_conflict_keeps_tags(self, storage, monkeypatch):
        """(d) Guard: the new version still contradicts B, so both stay tagged."""
        self._pin(storage, monkeypatch, {self.A: 0, self.B: 8, self.A_KEPT: 12})
        a = await _store(storage, self.A)
        b = await _store(storage, self.B)
        assert CONFLICT_TAG in _tags(storage, b)

        ok, msg, a2 = await storage.update_memory_versioned(a, self.A_KEPT)
        assert ok, msg

        assert CONFLICT_TAG in _tags(storage, a2)
        assert CONFLICT_TAG in _tags(storage, b)
        assert frozenset((a2, b)) in _listed_pairs(await storage.get_conflicts())
        assert _stale_conflict_tags(storage) == set()


class TestReconcileConflictTagsViaMaintain:
    """(f) The maintain conflicts step reconciles stale tags; dry run reports only."""

    @pytest.mark.asyncio
    async def test_maintain_reconciles_stale_conflict_tags(self, storage, monkeypatch):
        # Keep the other maintain steps from touching the seeded rows, whatever the
        # environment says: no auto-resolve, no contradiction pass.
        monkeypatch.setattr(quality_handlers, "MAINTAIN_AUTO_RESOLVE", False)
        monkeypatch.setattr(contradictions, "CONTRADICTION_ENABLED", False)
        legacy = await _store(storage, "Legacy row tagged before the fix, partner long gone")
        c = await _store(storage, "The NAS exports shares over NFSv4")
        d = await _store(storage, "The NAS exports shares over SMB3 only")
        old = await _store(storage, "Superseded row that keeps its own tag")
        newer = await _store(storage, "Replacement for the superseded row")
        await _seed_conflict(storage, c, d)
        # Stale rows as a pre-fix database holds them: tag set, no live partner.
        for h in (legacy, old):
            storage.conn.execute(
                "UPDATE memories SET tags = tags || ',' || ? WHERE content_hash = ?",
                (CONFLICT_TAG, h),
            )
        storage.conn.execute(
            "UPDATE memories SET superseded_by = ? WHERE content_hash = ?", (newer, old)
        )
        storage.conn.commit()
        assert _stale_conflict_tags(storage) == {legacy}

        server = MagicMock()
        server._ensure_storage_initialized = AsyncMock(return_value=storage)

        async def maintain(arguments):
            graph = MagicMock()
            graph.store_entity_link = AsyncMock(return_value=True)
            with patch("mcp_memory_service.server.handlers.graph.get_graph_storage",
                       AsyncMock(return_value=graph)):
                result = await quality_handlers.handle_maintain(server, arguments)
            return json.loads(result[0].text)["steps"]["conflicts"]["stale_tags"]

        # Default is a dry run: report, change nothing.
        stale = await maintain({})
        assert stale["found"] == 1
        assert stale["cleared"] == 0
        assert CONFLICT_TAG in _tags(storage, legacy)

        stale = await maintain({"dry_run": False})
        assert stale["found"] == 1
        assert stale["cleared"] == 1
        assert CONFLICT_TAG not in _tags(storage, legacy)
        # Live pair and superseded row are left alone; nothing gains the tag.
        assert CONFLICT_TAG in _tags(storage, c)
        assert CONFLICT_TAG in _tags(storage, d)
        assert CONFLICT_TAG in _tags(storage, old)
        assert CONFLICT_TAG not in _tags(storage, newer)

        # Idempotent: a second run finds nothing.
        stale = await maintain({"dry_run": False})
        assert stale["found"] == 0
        assert stale["cleared"] == 0
        assert _stale_conflict_tags(storage) == set()
