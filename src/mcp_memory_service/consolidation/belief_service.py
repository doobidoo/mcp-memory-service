"""Belief derivation service — orchestrates observation-to-belief pipeline.

Queries observations from storage, groups by semantic similarity,
derives confidence via pure math (belief.py), and persists results
to the beliefs table. No LLM dependency.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import List, Optional

from .belief import (
    CONFIDENCE_FLOOR,
    LAMBDA,
    derive_confidence,
    should_promote,
    should_supersede,
)
from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)

_NOISE_PREFIXES = (
    "Association between",
    "[CHECKPOINT]",
    "harvested_sessions:",
    "Cluster of",
    "[SESSÃO]",
    "[MINED:",
    "[GAP]",      # auto decision-gap statistic (not a claim) — investigação 31/out
    "[PATTERN]",  # auto tag co-occurrence statistic — investigação 31/out
    "[TREND]",    # auto trend statistic — investigação 31/out
    "# Session Summary",
    "## Session Summary",
    "Session: ",  # commit_session_legacy entries (#121 follow-up)
)

_NOISE_PATTERNS = [
    re.compile(r"^#{1,3}\s+Session\s+Summary", re.IGNORECASE),
    re.compile(r"^User:\s", re.MULTILINE),
    re.compile(r"^Assistant:\s", re.MULTILINE),
    re.compile(r"<task-notification>", re.IGNORECASE),
    re.compile(r"<ide_opened_file>", re.IGNORECASE),
    re.compile(r"^Topics Discussed:", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Session:\s+[0-9a-f]{8}-", re.IGNORECASE),  # UUID session IDs (#121)
    # Meta-statistics that escape the bracket prefixes (the bracket-less
    # "[PATTERN]" body): "N memories share tags [...], indicating a recurring
    # theme." is a co-occurrence count, not knowledge. Anchored to the trailing
    # "indicating a recurring theme"; the list body is [^\n]* (NOT [^\]], because
    # a tag name can itself contain "]" e.g. "[project]" — Greptile P2
    # r4431605443). No DOTALL, so legit prose across paragraphs is not caught.
    # Anchored to the exact shape _detect_patterns emits in insights.py:
    # "... share tags [<sorted list>], indicating a recurring theme." The list
    # is unbounded (every shared tag is printed), so match the bracketed-list
    # STRUCTURE — "share tags [" ... "], indicating a recurring theme" — never a
    # char budget (Greptile P2 on #1727: two long tags exceed any fixed limit).
    # No DOTALL and the list body excludes line breaks, so legitimate prose that
    # mentions both phrases across paragraphs is not caught.
    re.compile(
        r"memories share tags \[[^\n]*\],?\s*indicating a recurring theme",
        re.IGNORECASE,
    ),
]


def _is_noise(content: str) -> bool:
    """Filter out non-useful observations using structural heuristics."""
    text = content.lstrip()

    # Prefix check (fast path)
    if any(text.startswith(prefix) for prefix in _NOISE_PREFIXES):
        return True

    # Pattern check (regex)
    if any(pat.search(text) for pat in _NOISE_PATTERNS):
        return True

    # Length check: a long observation is noise only when it is actually a DUMP —
    # low token diversity (repetitive) or degenerate (a few giant tokens). Length
    # alone is a bad proxy: measured against the live corpus, a blind `len > 500`
    # discarded 538 actionable decision/learning memories, 537 of them substantive
    # (entropy >= 0.49). Use _content_entropy (information density) to keep those.
    if len(text) > 500:
        words = text.split()
        if _content_entropy(text) < 0.4 or len(words) < 10:
            return True

    # Bullet-list ratio: >60% lines starting with "- " = likely a list dump
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) > 3:
        bullet_ratio = sum(1 for l in lines if l.lstrip().startswith("- ")) / len(lines)
        if bullet_ratio > 0.6:
            return True

    return False


# Markers that are pure consolidation meta-statistics, not knowledge. Subset of
# _NOISE_PREFIXES used at READ time (get_beliefs): excluding stored noise must not
# reapply the full _is_noise dump heuristics (entropy/bullet/length) to beliefs that
# are merely long or repetitive — those are legitimate and were accepted at
# derivation. Only the auto-insight markers are retroactively excluded.
_META_STAT_PREFIXES = ("[GAP]", "[PATTERN]", "[TREND]", "[MINED:")
_META_STAT_PATTERN = re.compile(
    r"memories share tags \[[^\n]*\],?\s*indicating a recurring theme",
    re.IGNORECASE,
)


def _is_meta_statistic(content: str) -> bool:
    """True if content is a consolidation auto-statistic ([GAP]/[PATTERN]/[TREND]/
    [MINED:] or the bracket-less 'N memories share tags [...] recurring theme').

    Narrow by design for read-time exclusion (Greptile P1 on #1727): beliefs
    stored before the derivation filter existed stay in the table, and
    derive_beliefs no longer reaches them to supersede. This keeps existing
    stores clean without a data migration, WITHOUT reapplying the dump
    heuristics that would hide legitimate long/repetitive beliefs.
    """
    text = (content or "").lstrip()
    if any(text.startswith(p) for p in _META_STAT_PREFIXES):
        return True
    return bool(_META_STAT_PATTERN.search(text))


def _cosine(a, b) -> float:
    """Cosine similarity of two equal-length float vectors. 0.0 on degenerate input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    import math  # inline import: only needed for cosine evaluation
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _desc_str(s: str) -> tuple:
    """Sort key that orders strings DESCENDING inside an ascending sorted():
    invert each code point. Used for updated_at (most-recent-first) as a tie-break.
    Correct only for fixed-width ISO-8601 timestamps (the belief updated_at shape);
    not a general-purpose descending key."""
    return tuple(-ord(c) for c in (s or ""))


def _content_entropy(content: str) -> float:
    """Unique-token ratio as proxy for information density.

    Returns 0.0-1.0. Low values (<0.3) indicate repetitive/boilerplate content.
    """
    words = content.lower().split()
    if not words:
        return 0.0
    unique = set(words)
    return len(unique) / len(words)


def _hash_content(content: str) -> str:
    """Deterministic hash for belief content."""
    return hashlib.sha256(content.encode()).hexdigest()


def _sum_numeric_counters(target: dict, source: dict) -> None:
    """Add source's numeric counters into target in place (bool excluded).

    A non-numeric value already held by ``target`` for a key is NEVER overwritten
    by a numeric value from ``source`` (the canonical's own value wins).
    """
    for k, v in source.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            base = target.get(k, 0)
            if isinstance(base, (int, float)) and not isinstance(base, bool):
                target[k] = base + v


class BeliefService:
    # Default belief-grouping similarity. Decoupled from the store() dedup
    # threshold (MCP_SEMANTIC_DEDUP_THRESHOLD) so the two — which serve opposite
    # intents: dedup *rejects* a write, grouping *needs* the write — can be tuned
    # independently (issue #1216). Overridable via MCP_BELIEF_SIMILARITY_THRESHOLD.
    SIMILARITY_THRESHOLD = 0.85
    """Derives and manages beliefs from observations."""

    def __init__(self, storage, graph=None):
        self.storage = storage
        # Graph handle for contradiction lookups. It lives beside the memory
        # storage, not on it, so callers that derive beliefs pass it in.
        self.graph = graph
        # Parsed with the repo's fallback pattern: an invalid value logs an
        # error and keeps the default rather than raising out of __init__ and
        # disabling belief derivation.
        from ..config import safe_get_float_env
        self.SIMILARITY_THRESHOLD = safe_get_float_env(
            "MCP_BELIEF_SIMILARITY_THRESHOLD", self.SIMILARITY_THRESHOLD,
            min_value=0.0, max_value=1.0,
        )

    async def derive_beliefs(self) -> dict:
        """Run belief derivation cycle. Called by scheduler.

        1. Get all observation-type memories
        2. Group by semantic similarity (existing embeddings)
        3. For each group, derive/update a belief
        4. Promote candidates meeting provenance floor
        5. Supersede beliefs below confidence floor
        """
        stats = {"created": 0, "updated": 0, "promoted": 0, "superseded": 0, "errors": []}

        try:
            observations = await self._load_derivation_observations()
            if not observations:
                return stats

            # Group observations by similarity using storage search
            groups = await self._group_observations(observations)

            now = datetime.now(timezone.utc)

            for group_content, group_obs in groups.items():
                try:
                    await self._derive_group_belief(group_content, group_obs, now, stats)

                except Exception as e:
                    stats["errors"].append(str(e))
                    logger.warning("[belief] Error deriving belief: %s", _sanitize_log_value(str(e)))

        except Exception as e:
            stats["errors"].append(str(e))
            logger.error("[belief] Derivation cycle failed: %s", _sanitize_log_value(str(e)), exc_info=True)

        logger.info(
            "[belief] Derivation complete: created=%d updated=%d promoted=%d superseded=%d",
            stats["created"], stats["updated"], stats["promoted"], stats["superseded"],
        )
        return stats

    async def _load_derivation_observations(self):
        """Load observations eligible for grouping, excluding noise and boilerplate."""
        observations = await self.storage.get_all_memories(memory_type="observation")
        if not observations:
            return []

        # Filter noise before grouping
        observations = [
            obs for obs in observations
            if not _is_noise(obs.content if hasattr(obs, "content") else obs.get("content", ""))
        ]
        if not observations:
            return []

        # Filter low-entropy content (repetitive boilerplate)
        observations = [
            obs for obs in observations
            if _content_entropy(
                obs.content if hasattr(obs, "content") else obs.get("content", "")
            ) > 0.25
        ]
        if not observations:
            return []
        return observations

    async def _derive_group_belief(self, group_content, group_obs, now, stats):
        """Derive one group, preserving partial progress if a later step fails."""
        obs_dicts = [self._obs_to_dict(o) for o in group_obs]
        # All observations in a group are supporting; contradictions come from
        # memories explicitly marked via contradiction detection
        contradicting = await self._find_contradictions(group_obs)

        supporting_hashes = [o.content_hash for o in group_obs]
        contradicting_hashes = [c.get("content_hash", "") for c in contradicting]

        confidence = derive_confidence(
            supporting=obs_dicts,
            contradicting=contradicting,
            current_time=now,
            lambda_weight=LAMBDA,
        )

        belief_hash = _hash_content(group_content)
        existing = await self._get_belief(belief_hash)

        if existing:
            await self._update_belief(
                belief_hash, confidence, supporting_hashes, contradicting_hashes, now
            )
            stats["updated"] += 1

            # Promote or supersede
            status = existing["status"]
            if status == "candidate" and should_promote(confidence, len(supporting_hashes)):
                stats["promoted"] += await self._set_belief_status(
                    belief_hash, "active", now, expected_status=status
                )
            elif status == "active" and should_supersede(confidence):
                stats["superseded"] += await self._set_belief_status(
                    belief_hash, "superseded", now, expected_status=status
                )
        else:
            # Create new candidate
            status = "active" if should_promote(confidence, len(supporting_hashes)) else "candidate"
            await self._create_belief(
                belief_hash, group_content, confidence, status,
                supporting_hashes, contradicting_hashes, now
            )
            stats["created"] += 1
            if status == "active":
                stats["promoted"] += 1

    async def challenge_belief(self, belief_hash: str) -> dict:
        """Force re-derivation of a specific belief."""
        belief = await self._get_belief(belief_hash)
        if not belief:
            return {"error": "Belief not found"}

        now = datetime.now(timezone.utc)

        # Mark as disputed
        await self._set_belief_status(belief_hash, "disputed", now)

        # Re-derive from current observations
        derived_from = json.loads(belief.get("derived_from", "[]"))
        contradicted_by = json.loads(belief.get("contradicted_by", "[]"))

        supporting = await self._get_observations_by_hashes(derived_from)
        contradicting = await self._get_observations_by_hashes(contradicted_by)

        confidence = derive_confidence(
            supporting=supporting,
            contradicting=contradicting,
            current_time=now,
            lambda_weight=LAMBDA,
        )

        # Reinstate or supersede based on new confidence
        if should_promote(confidence, len(supporting)):
            new_status = "active"
        elif should_supersede(confidence):
            new_status = "superseded"
        else:
            new_status = "candidate"

        await self._update_belief(
            belief_hash, confidence,
            [s.get("content_hash", "") for s in supporting],
            [c.get("content_hash", "") for c in contradicting],
            now,
        )
        await self._set_belief_status(belief_hash, new_status, now)

        return {
            "belief_hash": belief_hash,
            "new_status": new_status,
            "confidence": round(confidence, 4),
            "supporting_count": len(supporting),
            "contradicting_count": len(contradicting),
        }

    async def get_beliefs(
        self,
        status: str = "active",
        min_confidence: float = CONFIDENCE_FLOOR,
        store: str | None = None,
    ) -> List[dict]:
        """Get beliefs without waiting for the connection lock on the event loop."""
        return await asyncio.to_thread(self._get_beliefs_sync, status, min_confidence, store)

    def _get_beliefs_sync(self, status, min_confidence, store):
        """Read and decode beliefs under the shared connection lock in a worker."""
        try:
            with self._conn_lock_cm():
                conn = self.storage.conn
                sql = (
                    "SELECT belief_hash, content, confidence, status, created_at, updated_at, "
                    "derived_from, contradicted_by, metadata FROM beliefs "
                    "WHERE status = ? AND confidence >= ?"
                )
                params: list[str | float] = [status, min_confidence]
                if store is not None:
                    sql += """
                        AND json_array_length(derived_from) > 0
                        AND NOT EXISTS (
                            SELECT 1 FROM json_each(beliefs.derived_from) source
                            WHERE NOT EXISTS (
                                SELECT 1 FROM memories m
                                WHERE m.content_hash = source.value AND m.store = ?
                                  AND m.deleted_at IS NULL
                            )
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM json_each(beliefs.contradicted_by) contrary
                            WHERE NOT EXISTS (
                                SELECT 1 FROM memories m
                                WHERE m.content_hash = contrary.value AND m.store = ?
                                  AND m.deleted_at IS NULL
                            )
                        )
                    """
                    params.extend((store, store))
                sql += " ORDER BY confidence DESC"
                cursor = conn.execute(sql, tuple(params))
                rows = cursor.fetchall()
            return [
                {
                    "belief_hash": r[0],
                    "content": r[1],
                    "confidence": r[2],
                    "status": r[3],
                    "created_at": r[4],
                    "updated_at": r[5],
                    "derived_from": json.loads(r[6]),
                    "contradicted_by": json.loads(r[7]),
                    "metadata": json.loads(r[8] or "{}"),
                    "result_type": "belief",
                }
                for r in rows
                if not _is_meta_statistic(r[1] or "")
            ]
        except Exception as e:
            logger.error("[belief] get_beliefs failed: %s", _sanitize_log_value(str(e)))
            return []

    @staticmethod
    def _env_flag_on(name: str, default: str = "false") -> bool:
        """Parse a boolean env flag with the repo's tolerant pattern (#391).

        True only for the explicit tokens true/1/yes (case/space-insensitive);
        every other value — including typos and the unset default — is OFF.
        """
        return os.environ.get(name, default).strip().lower() in ("true", "1", "yes")

    @staticmethod
    def _dedup_config() -> tuple:
        """Read (threshold, cap) for dedup from env with float/int fallbacks."""
        try:
            threshold = float(os.environ.get("MCP_BELIEF_DEDUP_THRESHOLD", "0.90"))
        except (ValueError, TypeError):
            threshold = 0.90
        try:
            cap_str = os.environ.get("MCP_BELIEF_DEDUP_CAP", "1000")
            cap = int(cap_str) if cap_str else 1000
        except (ValueError, TypeError):
            cap = 1000
        return threshold, cap

    def _load_dedup_candidates(self, report: dict) -> list:
        """SELECT dedupable beliefs and deserialize their embeddings.

        Returns a list of ``(hash, confidence, updated_at, n_sources, vec)``.
        NULL/undeserializable embeddings are counted in report['skipped'].
        ORDER BY belief_hash keeps the downstream greedy grouping STABLE across
        runs (SQLite has no implicit row order — tuvok P2).
        """
        with self._conn_lock_cm():
            conn = self.storage.conn
            rows = conn.execute(
                "SELECT belief_hash, content, confidence, status, updated_at, "
                "derived_from, embedding FROM beliefs WHERE status IN ('active', 'candidate') "
                "ORDER BY belief_hash"
            ).fetchall()

        items = []
        for bhash, content, confidence, status, updated_at, derived_from, emb in rows:
            report["evaluated"] += 1
            if emb is None:
                report["skipped"] += 1
                continue
            vec = self._deserialize_embedding(emb)
            if vec is None:
                report["skipped"] += 1
                continue
            try:
                n_src = len(json.loads(derived_from) if derived_from else [])
            except (ValueError, TypeError):
                n_src = 0
            items.append((bhash, float(confidence or 0.0), updated_at or "", n_src, vec))
        return items

    @staticmethod
    def _members_from_seed(items: list, seed_idx: int, used: set, threshold: float) -> list:
        """Members of the seed's group: the seed plus every later unused item with
        cosine >= threshold against it (single-linkage from the seed)."""
        seed = items[seed_idx]
        members = [seed]
        for j in range(seed_idx + 1, len(items)):
            if items[j][0] in used:
                continue
            if _cosine(seed[4], items[j][4]) >= threshold:
                members.append(items[j])
        return members

    @staticmethod
    def _group_by_cosine(items: list, threshold: float, cap: int | None = None) -> list:
        """Greedy single-linkage grouping by cosine >= threshold.

        Each later belief is tested against the seed only (approximation, NOT
        exact clustering): a chain A~B, B~C with A≁C groups {A,B} and leaves C a
        singleton. Stable because ``items`` arrives ORDER BY belief_hash. Returns
        a list of member-lists, each with >= 2 members (singletons dropped).

        ``cap`` stops the scan once that many groups are found, so a capped call
        does not compare the whole remaining corpus for groups it will discard
        (Greptile P2 #1733). None = no cap.
        """
        used = set()
        groups = []
        for i in range(len(items)):
            if items[i][0] in used:
                continue
            members = BeliefService._members_from_seed(items, i, used, threshold)
            if len(members) < 2:
                continue
            used.update(m[0] for m in members)
            groups.append(members)
            if cap is not None and len(groups) >= cap:
                break
        return groups

    @staticmethod
    def _elect_canonical(members: list) -> tuple:
        """R4 deterministic election: confidence desc, |derived_from| desc,
        updated_at desc, belief_hash asc (NO utility_prior).

        Returns ``(canonical_hash, [loser_hashes])``. Numeric fields are negated
        for descending; updated_at desc via _desc_str; belief_hash ascending is
        the final deterministic tie-break.
        """
        members_sorted = sorted(
            members,
            key=lambda m: (-m[1], -m[3], _desc_str(m[2]), m[0]),
        )
        canonical = members_sorted[0][0]
        losers = [m[0] for m in members_sorted[1:]]
        return canonical, losers

    def dedup_beliefs(self, dry_run: bool = True) -> dict:
        """Semantic dedup of beliefs by embedding cosine (ARCO SÍNTESE B.2a/B.2b).

        Opt-in behind MCP_BELIEF_DEDUP. Groups active/candidate beliefs whose
        cosine similarity over the persisted ``beliefs.embedding`` is >=
        MCP_BELIEF_DEDUP_THRESHOLD (default 0.90), elects a deterministic
        canonical per group (R4), and reports the groups. dry_run=True (B.2a)
        never writes. dry_run=False (B.2b) merges losers into the canonical and
        supersedes them under the connection lock (per-group transaction).

        Report dict: {"evaluated", "groups", "merged", "skipped", "capped",
        "group_details": [{"canonical": hash, "superseded": [hashes]}], "errors"}.
        """
        report = {"evaluated": 0, "groups": 0, "merged": 0, "skipped": 0,
                  "capped": False, "group_details": [], "errors": []}

        if not self._env_flag_on("MCP_BELIEF_DEDUP"):
            return report

        threshold, cap = self._dedup_config()

        # A cap of 0 (or negative) means "produce no groups": return before
        # grouping, mirroring backfill_belief_embeddings (cap <= 0 -> return).
        # Without this, _group_by_cosine appends the first group BEFORE its
        # len(groups) >= cap check, so dedup_beliefs(dry_run=False) would merge
        # that one group before the line-544 cap check could stop it
        # (doobidoo review #1733).
        if cap <= 0:
            report["capped"] = True
            return report

        try:
            now = datetime.now(timezone.utc)
            items = self._load_dedup_candidates(report)
            # Pass the cap so grouping STOPS at the cap instead of scanning the
            # whole corpus for groups that will be discarded (Greptile P2 #1733:
            # the refactor had lost the early exit — quadratic work on large corpora).
            groups = self._group_by_cosine(items, threshold, cap)

            for members in groups:
                canonical, losers = self._elect_canonical(members)
                report["group_details"].append(
                    {"canonical": canonical, "superseded": losers}
                )
                report["groups"] += 1
                # B.2b apply: merge losers into canonical + supersede, per-group
                # transaction (A4). dry_run=True only reports. A failing group is
                # rolled back and does not stop the others (per-group try/except,
                # like sweep_degenerate_beliefs).
                if not dry_run:
                    try:
                        self._merge_into_canonical(canonical, losers, now)
                        report["merged"] += len(losers)
                    except Exception as e:
                        report["errors"].append(f"merge group {canonical}: {e}")
                if report["groups"] >= cap:
                    report["capped"] = True
                    break

            logger.info(
                "[belief] dedup (dry_run=%s): evaluated=%d groups=%d skipped=%d capped=%s",
                dry_run, report["evaluated"], report["groups"],
                report["skipped"], report["capped"],
            )
        except Exception as e:
            report["errors"].append(str(e))
            logger.error("[belief] dedup failed: %s", _sanitize_log_value(str(e)))

        return report

    @staticmethod
    def _deserialize_embedding(blob):
        """Deserialize a sqlite_vec float32 BLOB into a list of floats, or None."""
        if blob is None:
            return None
        try:
            import struct  # inline import: only needed when deserializing a blob
            n = len(blob) // 4
            return list(struct.unpack(f"<{n}f", blob))  # little-endian: matches serialize_float32
        except Exception:
            return None

    def _conn_lock_cm(self):
        """Context manager that holds the storage connection lock if present.

        All belief reads and writes share this lock. The merge transaction
        mutates the shared sqlite connection
        (check_same_thread=False); holding ``storage._conn_lock`` for the whole
        read+BEGIN+UPDATE+commit serializes it against the storage's own worker
        thread so a concurrent writer cannot interleave and lose an update
        (#562). Backends/mocks without the lock (e.g. the test storage exposing
        only ``.conn``) fall back to a no-op context — correctness is unchanged
        single-threaded, only the cross-thread guard is absent.

        Keep the critical section synchronous: release it before awaiting
        storage operations or embedding generation, which may take this lock.
        """
        lock = getattr(self.storage, "_conn_lock", None)
        if lock is not None:
            return lock
        from contextlib import nullcontext
        return nullcontext()

    def _fetch_merge_rows(self, conn, canonical: str, losers: list) -> dict:
        """SELECT (status, derived_from, metadata) for the canonical + losers as a dict."""
        placeholders = ",".join("?" * (len(losers) + 1))
        return {
            r[0]: (r[1], json.loads(r[2] or "[]"), json.loads(r[3] or "{}"))
            for r in conn.execute(
                "SELECT belief_hash, status, derived_from, metadata FROM beliefs "
                "WHERE belief_hash IN (%s)" % placeholders,
                (canonical, *losers),
            ).fetchall()
        }

    @staticmethod
    def _fold_losers_into_canonical(rows: dict, canonical: str, losers: list) -> tuple:
        """Fold losers' provenance+counters into the canonical (pure, no I/O).

        Returns ``(merged_df, merged_meta)``. Unions derived_from (A2, set-dedup),
        sums numeric metadata counters (A3; a non-numeric canonical value wins —
        never overwritten by a loser number), and records every loser-sourced
        hash under ``metadata['merged_from_hashes']`` so a later re-derivation can
        preserve them (#566).

        Idempotency (#1771, Greptile P1): a loser ALREADY ``superseded`` — by an
        overlapping dedup run on the shared storage — is skipped entirely, so its
        counters are never summed into the canonical twice.
        """
        _canon_status, canon_df, canon_meta = rows[canonical]
        merged_df = list(canon_df)
        seen = set(merged_df)
        merged_meta = dict(canon_meta)
        merged_from = list(merged_meta.get("merged_from_hashes", []))
        merged_from_seen = set(merged_from)

        for lh in losers:
            if lh not in rows:
                continue
            l_status, l_df, l_meta = rows[lh]
            # Skip a loser another worker already superseded: re-folding it would
            # double-count its counters and re-add its provenance on the canonical.
            if l_status == "superseded":
                continue
            for h in l_df:
                if h not in seen:
                    merged_df.append(h)
                    seen.add(h)
                if h not in merged_from_seen:
                    merged_from.append(h)
                    merged_from_seen.add(h)
            _sum_numeric_counters(merged_meta, l_meta)

        if merged_from:
            merged_meta["merged_from_hashes"] = merged_from
        return merged_df, merged_meta

    @staticmethod
    def _persist_merge(conn, canonical, merged_df, merged_meta, losers, now_iso) -> None:
        """Write the canonical + supersede losers in one transaction (A1/A4)."""
        try:
            # Guard against a stale open transaction on the shared connection
            # (check_same_thread=False): a dangling BEGIN would raise "cannot
            # start a transaction within a transaction". Same defense as
            # storage/sqlite_vec store.py before an explicit BEGIN (tuvok MEDIUM).
            if getattr(conn, "in_transaction", False):
                conn.rollback()
            conn.execute("BEGIN")
            conn.execute(
                "UPDATE beliefs SET derived_from = ?, metadata = ?, updated_at = ? "
                "WHERE belief_hash = ?",
                (json.dumps(merged_df), json.dumps(merged_meta), now_iso, canonical),
            )
            for lh in losers:
                conn.execute(
                    "UPDATE beliefs SET status = 'superseded', updated_at = ? "
                    "WHERE belief_hash = ? AND status != 'superseded'",
                    (now_iso, lh),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _merge_into_canonical(self, canonical: str, losers: list, now) -> None:
        """Merge losers into the canonical belief and supersede them, atomically.

        One transaction per group (B.2b, EARS A1-A4): unions the losers'
        ``derived_from`` into the canonical (set, no dup — A2), sums the numeric
        feedback counters in ``metadata`` (e.g. injected_used/unused — A3), writes
        the canonical, and marks every loser ``superseded`` (A1, never deleted). On
        any error the whole group rolls back (A4) and the exception propagates so
        the caller records it and moves on. Confidence/content of the canonical are
        left untouched (dedup is identity consolidation, not re-derivation).

        Concurrency (#562): the ENTIRE read+merge+write runs while holding the
        storage connection lock (``storage._conn_lock`` via ``_conn_lock_cm``),
        so a concurrent storage write on the shared connection cannot interleave
        between the SELECT and the commit and silently drop one side's update.

        Provenance preservation (#566): see ``_fold_losers_into_canonical``.
        """
        with self._conn_lock_cm():
            # Read fresh under the lock: the A4-internal test swaps storage.conn
            # to a failing proxy right before the call, so resolve conn here.
            conn = self.storage.conn
            rows = self._fetch_merge_rows(conn, canonical, losers)
            if canonical not in rows:
                raise ValueError(f"canonical {canonical} not found")
            merged_df, merged_meta = self._fold_losers_into_canonical(
                rows, canonical, losers
            )
            self._persist_merge(conn, canonical, merged_df, merged_meta, losers, now.isoformat())

    async def _invoke_embedding(self, gen, content):
        """Call the storage embedding fn, handling sync vs async (#422).

        Coroutine functions (cloudflare) are awaited; synchronous ones
        (sqlite_vec/milvus) run via ``storage._run_in_thread`` so they execute
        off the event loop under the connection lock — falling back to an inline
        call when the backend/mock has no ``_run_in_thread``.
        """
        import inspect as _inspect  # inline import: inspect optional embedding providers
        if _inspect.iscoroutinefunction(gen):
            return await gen(content)
        runner = getattr(self.storage, "_run_in_thread", None)
        vec = await runner(lambda: gen(content)) if runner is not None else gen(content)
        # Defensive: a sync method could still hand back a coroutine (wrapped
        # elsewhere); await it rather than serialize a coroutine.
        if _inspect.isawaitable(vec):
            vec = await vec
        return vec

    async def _compute_belief_embedding(self, content: str):
        """Compute a belief's embedding BLOB (serialize_float32), or None (#422).

        Embedding is OPTIONAL enrichment: this NEVER raises. Returns None when
        there is no model, no embedding method on the storage, or any failure.
        sync/async dispatch lives in ``_invoke_embedding``.
        """
        if not content:
            return None
        gen = getattr(self.storage, "_generate_embedding", None)
        if gen is None:
            return None
        # A model must actually be loaded; sqlite_vec raises if embedding_model
        # is None, so short-circuit on the common "no model" case.
        if getattr(self.storage, "embedding_model", "unset") is None:
            return None
        try:
            vec = await self._invoke_embedding(gen, content)
            if not vec:
                return None
            from sqlite_vec import serialize_float32
            return serialize_float32([float(x) for x in vec])
        except Exception as e:
            logger.debug("[belief] embedding skipped: %s", _sanitize_log_value(str(e)))
            return None

    def _select_null_embedding_rows(self, cap: int, report: dict) -> list:
        """Fetch up to ``cap`` beliefs with a NULL embedding (sets report['capped'])."""
        with self._conn_lock_cm():
            conn = self.storage.conn
            rows = conn.execute(
                "SELECT belief_hash, content FROM beliefs "
                "WHERE embedding IS NULL ORDER BY belief_hash LIMIT ?",
                (cap + 1,),
            ).fetchall()
        if len(rows) > cap:
            report["capped"] = True
            rows = rows[:cap]
        return rows

    def _write_backfill_embedding(self, bhash: str, blob) -> None:
        """Persist one backfilled embedding under the connection lock."""
        with self._conn_lock_cm():
            conn = self.storage.conn
            conn.execute(
                "UPDATE beliefs SET embedding = ? WHERE belief_hash = ? "
                "AND embedding IS NULL",
                (blob, bhash),
            )
            conn.commit()

    async def backfill_belief_embeddings(self, cap: int = 1000) -> dict:
        """Opt-in backfill of NULL belief embeddings (#422).

        Behind MCP_BELIEF_DEDUP (same flag that gates dedup): with no producer
        wired for pre-existing rows, migration 020 leaves their embedding NULL
        and they are never dedupable. This fills up to ``cap`` NULL-embedding
        beliefs in a single pass. Each row is computed and written under the
        connection lock; a row whose embedding can't be computed is left NULL and
        counted in ``skipped`` (never raises for one bad row).

        Report: {"evaluated", "filled", "skipped", "capped", "errors"}.
        """
        report = {"evaluated": 0, "filled": 0, "skipped": 0, "capped": False, "errors": []}
        if not self._env_flag_on("MCP_BELIEF_DEDUP"):
            return report
        try:
            cap = int(cap)
        except (ValueError, TypeError):
            cap = 1000
        if cap <= 0:
            return report

        try:
            rows = await asyncio.to_thread(self._select_null_embedding_rows, cap, report)
        except Exception as e:
            report["errors"].append(str(e))
            logger.error("[belief] backfill select failed: %s", _sanitize_log_value(str(e)))
            return report

        for bhash, content in rows:
            report["evaluated"] += 1
            blob = await self._compute_belief_embedding(content)
            if blob is None:
                report["skipped"] += 1
                continue
            try:
                await asyncio.to_thread(self._write_backfill_embedding, bhash, blob)
                report["filled"] += 1
            except Exception as e:
                report["errors"].append(f"belief {bhash}: {e}")
                report["skipped"] += 1

        logger.info(
            "[belief] backfill: evaluated=%d filled=%d skipped=%d capped=%s",
            report["evaluated"], report["filled"], report["skipped"], report["capped"],
        )
        return report

    # --- Internal helpers ---

    async def _group_observations(self, observations) -> dict:
        """Group observations by semantic similarity using existing embeddings.

        Uses storage KNN search to find similar observations (threshold 0.85).
        Greedy single-pass: first observation becomes cluster representative,
        subsequent observations join if similarity > threshold.
        Falls back to exact-content matching when embeddings are unavailable.
        """
        groups = {}  # representative content -> [observations]
        assigned = set()  # track which observations have been assigned
        content_hashes_seen = set()  # dedup near-identical clusters

        for i, obs in enumerate(observations):
            content = obs.content if hasattr(obs, "content") else obs.get("content", "")
            obs_hash = obs.content_hash if hasattr(obs, "content_hash") else obs.get("content_hash", str(i))

            if obs_hash in assigned or not content:
                continue

            # This observation becomes a cluster representative
            cluster = [obs]
            assigned.add(obs_hash)

            # Fast path: exact content match (always works, no embeddings needed)
            for other_obs in observations:
                other_content = other_obs.content if hasattr(other_obs, "content") else other_obs.get("content", "")
                other_hash = other_obs.content_hash if hasattr(other_obs, "content_hash") else other_obs.get("content_hash", "")
                if other_hash not in assigned and other_content == content:
                    cluster.append(other_obs)
                    assigned.add(other_hash)

            # Find similar observations using existing embeddings
            try:
                similar = await self.storage.retrieve(content, n_results=50)
                for result in similar:
                    r_hash = result.memory.content_hash if hasattr(result, "memory") else result.get("content_hash", "")
                    r_score = result.relevance_score if hasattr(result, "relevance_score") else result.get("relevance_score", 0)

                    if r_hash in assigned:
                        continue
                    if r_score < self.SIMILARITY_THRESHOLD:
                        continue

                    # Check if this result is in our observations list
                    for other_obs in observations:
                        other_hash = other_obs.content_hash if hasattr(other_obs, "content_hash") else other_obs.get("content_hash", "")
                        if other_hash == r_hash and other_hash not in assigned:
                            cluster.append(other_obs)
                            assigned.add(other_hash)
                            break
            except Exception:
                pass  # If search fails, rely on exact-match grouping above

            # Dedup: normalize content, hash it. Skip if we already have near-identical representative.
            normalized = " ".join(content.lower().split())
            content_digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
            if content_digest in content_hashes_seen:
                continue  # Skip this cluster — near-identical to an existing one
            content_hashes_seen.add(content_digest)

            groups[content] = cluster

        return groups

    async def _find_contradictions(self, observations) -> List[dict]:
        """Find observations linked to the group by a ``contradicts`` edge.

        Contradiction detection stores ``contradicts`` edges; following them in
        both directions finds the other side whichever memory came first.
        """
        if not self.graph:
            return []

        contradicting_hashes = []
        for obs in observations:
            content_hash = obs.content_hash if hasattr(obs, "content_hash") else obs.get("content_hash", "")
            if not content_hash:
                continue
            connected = await self.graph.find_connected(
                content_hash, max_hops=1, relationship_type="contradicts"
            )
            for other_hash, _distance in connected:
                if other_hash not in contradicting_hashes:
                    contradicting_hashes.append(other_hash)

        return await self._get_observations_by_hashes(contradicting_hashes)

    def _obs_to_dict(self, obs) -> dict:
        """Convert Memory object to dict for derive_confidence."""
        if isinstance(obs, dict):
            return obs
        return {
            "content_hash": obs.content_hash,
            "content": obs.content,
            "created_at_iso": (
                datetime.fromtimestamp(obs.created_at, tz=timezone.utc).isoformat()
                if isinstance(obs.created_at, (int, float)) and obs.created_at
                else str(obs.created_at) if obs.created_at else ""
            ),
            "metadata": obs.metadata if obs.metadata else {},
        }

    async def _get_observations_by_hashes(self, hashes: List[str]) -> List[dict]:
        """Retrieve observations by their content hashes."""
        results = []
        for h in hashes:
            if not h:
                continue
            try:
                mem = await self.storage.get_by_hash(h)
                if mem:
                    results.append(self._obs_to_dict(mem))
            except Exception:
                continue
        return results

    async def _get_belief(self, belief_hash: str) -> Optional[dict]:
        """Get a belief by hash without blocking the event loop."""
        return await asyncio.to_thread(self._get_belief_sync, belief_hash)

    def _get_belief_sync(self, belief_hash):
        """Read one belief under the shared connection lock in a worker."""
        try:
            with self._conn_lock_cm():
                conn = self.storage.conn
                cursor = conn.execute(
                    "SELECT belief_hash, content, confidence, status, created_at, updated_at, "
                    "derived_from, contradicted_by, metadata FROM beliefs WHERE belief_hash = ?",
                    (belief_hash,),
                )
                row = cursor.fetchone()
            if not row:
                return None
            return {
                "belief_hash": row[0],
                "content": row[1],
                "confidence": row[2],
                "status": row[3],
                "created_at": row[4],
                "updated_at": row[5],
                "derived_from": row[6],
                "contradicted_by": row[7],
                "metadata": row[8],
            }
        except Exception:
            return None

    async def _create_belief(
        self, belief_hash, content, confidence, status, supporting_hashes, contradicting_hashes, now
    ):
        """Insert a new belief, stamping its embedding at creation (#422).

        The embedding is computed up front and written with a separate guarded
        UPDATE so the row is created even on schemas that predate migration 020
        (the beliefs.embedding column): embedding is optional enrichment and must
        never block belief creation. A None embedding (no model, or failure)
        simply skips the UPDATE — behaviour identical to before this change.
        """
        await asyncio.to_thread(
            self._insert_belief, belief_hash, content, confidence, status,
            supporting_hashes, contradicting_hashes, now,
        )

        embedding = await self._compute_belief_embedding(content)
        if embedding is not None:
            try:
                # Persist the embedding UNDER the connection lock (Greptile P1
                # #1733): committing it off-lock could end a concurrent storage
                # worker's open transaction early. _write_backfill_embedding holds
                # _conn_lock_cm and only fills a still-NULL row (the one just
                # inserted), so it is the right primitive here too.
                await asyncio.to_thread(self._write_backfill_embedding, belief_hash, embedding)
            except Exception as e:
                # Column absent (pre-020 schema) or any write issue: the belief
                # is already persisted; embedding is optional, so swallow.
                logger.debug(
                    "[belief] embedding not stored for %s: %s",
                    _sanitize_log_value(belief_hash), _sanitize_log_value(str(e)),
                )

    def _insert_belief(
        self, belief_hash, content, confidence, status, supporting_hashes, contradicting_hashes, now
    ):
        """Insert and commit a belief in a worker under the connection lock."""
        with self._conn_lock_cm():
            conn = self.storage.conn
            now_iso = now.isoformat()
            conn.execute(
                "INSERT OR IGNORE INTO beliefs "
                "(belief_hash, content, confidence, status, created_at, updated_at, derived_from, contradicted_by) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    belief_hash, content, confidence, status, now_iso, now_iso,
                    json.dumps(supporting_hashes), json.dumps(contradicting_hashes),
                ),
            )
            conn.commit()

    async def _update_belief(self, belief_hash, confidence, supporting_hashes, contradicting_hashes, now):
        """Update an existing belief's confidence and provenance.

        Preserves merged provenance (#566): if a prior dedup recorded
        ``metadata['merged_from_hashes']`` on this belief, those hashes are
        UNIONED into derived_from instead of being overwritten by the current
        group's supporting hashes. Without this, re-deriving a canonical would
        silently drop the sources inherited from the losers it absorbed.
        """
        await asyncio.to_thread(
            self._update_belief_sync, belief_hash, confidence,
            supporting_hashes, contradicting_hashes, now,
        )

    def _update_belief_sync(self, belief_hash, confidence, supporting_hashes, contradicting_hashes, now):
        """Update provenance and commit under one worker-held connection lock."""
        # Keep the metadata read, provenance union and commit in one critical
        # section: dedup may add merged sources between any two SQL statements.
        with self._conn_lock_cm():
            conn = self.storage.conn
            merged_df = list(supporting_hashes)
            try:
                row = conn.execute(
                    "SELECT metadata FROM beliefs WHERE belief_hash = ?", (belief_hash,)
                ).fetchone()
                meta = json.loads(row[0] or "{}") if row and row[0] else {}
                merged_from = meta.get("merged_from_hashes", [])
                if merged_from:
                    seen = set(merged_df)
                    for h in merged_from:
                        if h not in seen:
                            merged_df.append(h)
                            seen.add(h)
            except (ValueError, TypeError):
                # Malformed metadata -> fall back to the current group's hashes only.
                merged_df = list(supporting_hashes)
            conn.execute(
                "UPDATE beliefs SET confidence = ?, derived_from = ?, contradicted_by = ?, updated_at = ? "
                "WHERE belief_hash = ?",
                (confidence, json.dumps(merged_df), json.dumps(contradicting_hashes), now.isoformat(), belief_hash),
            )
            conn.commit()

    async def _set_belief_status(
        self, belief_hash: str, status: str, now: datetime,
        expected_status: Optional[str] = None,
    ) -> int:
        """Update status, optionally guarding a derivation's earlier snapshot.

        A dedup worker may have superseded this belief since derivation read
        it. Return the affected row count so a stale promotion is neither
        applied nor counted. Explicit challenge operations omit the guard.
        """
        return await asyncio.to_thread(
            self._set_belief_status_sync, belief_hash, status, now, expected_status,
        )

    def _set_belief_status_sync(self, belief_hash, status, now, expected_status):
        """Apply the guarded status change under the connection lock in a worker."""
        with self._conn_lock_cm():
            conn = self.storage.conn
            cursor = conn.execute(
                "UPDATE beliefs SET status = ?, updated_at = ? "
                "WHERE belief_hash = ? AND (? IS NULL OR status = ?)",
                (status, now.isoformat(), belief_hash, expected_status, expected_status),
            )
            conn.commit()
            return cursor.rowcount
