"""Belief derivation service — orchestrates observation-to-belief pipeline.

Queries observations from storage, groups by semantic similarity,
derives confidence via pure math (belief.py), and persists results
to the beliefs table. No LLM dependency.
"""

import hashlib
import json
import logging
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

logger = logging.getLogger(__name__)

_NOISE_PREFIXES = (
    "Association between",
    "[CHECKPOINT]",
    "harvested_sessions:",
    "Cluster of",
    "[SESSÃO]",
    "[MINED:",
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

    # Length check: very long observations are usually dumps, not insights
    if len(text) > 500:
        return True

    # Bullet-list ratio: >60% lines starting with "- " = likely a list dump
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) > 3:
        bullet_ratio = sum(1 for l in lines if l.lstrip().startswith("- ")) / len(lines)
        if bullet_ratio > 0.6:
            return True

    return False


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


class BeliefService:
    SIMILARITY_THRESHOLD = 0.85
    """Derives and manages beliefs from observations."""

    def __init__(self, storage):
        self.storage = storage

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
            observations = await self.storage.get_all_memories(memory_type="observation")
            if not observations:
                return stats

            # Filter noise before grouping
            observations = [
                obs for obs in observations
                if not _is_noise(obs.content if hasattr(obs, "content") else obs.get("content", ""))
            ]
            if not observations:
                return stats

            # Filter low-entropy content (repetitive boilerplate)
            observations = [
                obs for obs in observations
                if _content_entropy(
                    obs.content if hasattr(obs, "content") else obs.get("content", "")
                ) > 0.25
            ]
            if not observations:
                return stats

            # Group observations by similarity using storage search
            groups = await self._group_observations(observations)

            now = datetime.now(timezone.utc)

            for group_content, group_obs in groups.items():
                try:
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
                            await self._set_belief_status(belief_hash, "active", now)
                            stats["promoted"] += 1
                        elif status == "active" and should_supersede(confidence):
                            await self._set_belief_status(belief_hash, "superseded", now)
                            stats["superseded"] += 1
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

                except Exception as e:
                    stats["errors"].append(str(e))
                    logger.warning(f"[belief] Error deriving belief: {e}")

        except Exception as e:
            stats["errors"].append(str(e))
            logger.error(f"[belief] Derivation cycle failed: {e}", exc_info=True)

        logger.info(
            "[belief] Derivation complete: created=%d updated=%d promoted=%d superseded=%d",
            stats["created"], stats["updated"], stats["promoted"], stats["superseded"],
        )
        return stats

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
        self, status: str = "active", min_confidence: float = CONFIDENCE_FLOOR
    ) -> List[dict]:
        """Get beliefs filtered by status and confidence."""
        try:
            conn = self.storage.conn
            cursor = conn.execute(
                "SELECT belief_hash, content, confidence, status, created_at, updated_at, "
                "derived_from, contradicted_by, metadata FROM beliefs "
                "WHERE status = ? AND confidence >= ? ORDER BY confidence DESC",
                (status, min_confidence),
            )
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
            ]
        except Exception as e:
            logger.error(f"[belief] get_beliefs failed: {e}")
            return []

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
        """Find contradicting observations using existing graph edges."""
        contradicting = []
        if not hasattr(self.storage, "graph") or not self.storage.graph:
            return contradicting

        for obs in observations:
            content_hash = obs.content_hash if hasattr(obs, "content_hash") else obs.get("content_hash", "")
            if not content_hash:
                continue
            try:
                edges = await self.storage.graph.get_edges(content_hash, "CONTRADICTED_BY")
                for edge in edges:
                    target_hash = edge.get("target") or edge.get("to_hash", "")
                    if target_hash:
                        mems = await self._get_observations_by_hashes([target_hash])
                        contradicting.extend(mems)
            except Exception:
                continue

        return contradicting

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
                mem = await self.storage.get_memory_by_hash(h)
                if mem:
                    results.append(self._obs_to_dict(mem))
            except Exception:
                continue
        return results

    async def _get_belief(self, belief_hash: str) -> Optional[dict]:
        """Get a belief by hash."""
        try:
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
        """Insert a new belief."""
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
        """Update an existing belief's confidence and provenance."""
        conn = self.storage.conn
        conn.execute(
            "UPDATE beliefs SET confidence = ?, derived_from = ?, contradicted_by = ?, updated_at = ? "
            "WHERE belief_hash = ?",
            (confidence, json.dumps(supporting_hashes), json.dumps(contradicting_hashes), now.isoformat(), belief_hash),
        )
        conn.commit()

    async def _set_belief_status(self, belief_hash: str, status: str, now: datetime):
        """Update belief status."""
        conn = self.storage.conn
        conn.execute(
            "UPDATE beliefs SET status = ?, updated_at = ? WHERE belief_hash = ?",
            (status, now.isoformat(), belief_hash),
        )
        conn.commit()
