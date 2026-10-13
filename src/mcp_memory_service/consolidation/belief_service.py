"""Belief derivation service — orchestrates observation-to-belief pipeline.

Queries observations from storage, groups by semantic similarity,
derives confidence via pure math (belief.py), and persists results
to the beliefs table. No LLM dependency.
"""

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

    # Minimum-length check: fragments too short to carry a useful insight
    # (e.g. "TYPE:", "Reg", "convention", "usar `") must not become beliefs.
    # Require at least ~15 chars and 3 words of substance.
    stripped = text.strip()
    if len(stripped) < 15 or len(stripped.split()) < 3:
        return True

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
    # Default belief-grouping similarity. Decoupled from the store() dedup
    # threshold (MCP_SEMANTIC_DEDUP_THRESHOLD) so the two — which serve opposite
    # intents: dedup *rejects* a write, grouping *needs* the write — can be tuned
    # independently (issue #1216). Overridable via MCP_BELIEF_SIMILARITY_THRESHOLD.
    SIMILARITY_THRESHOLD = 0.85
    """Derives and manages beliefs from observations."""

    def __init__(self, storage):
        self.storage = storage
        # Parsed with the repo's fallback pattern: an invalid value logs an
        # error and keeps the default rather than raising out of __init__ and
        # disabling belief derivation.
        from ..config import safe_get_float_env
        self.SIMILARITY_THRESHOLD = safe_get_float_env(
            "MCP_BELIEF_SIMILARITY_THRESHOLD", self.SIMILARITY_THRESHOLD,
            min_value=0.0, max_value=1.0,
        )

    # --- Dreaming-loop Inc 1: singular beliefs (rfc-dreaming-loop-inc1-distill-specific.md) ---
    # ADDITIVE, ISOLATED mechanism validated empirically (11/out): a single-source
    # actionable memory (decision/learning/error) that never reaches PROVENANCE_FLOOR=2
    # becomes an injectable "singular" belief with a fixed modest confidence, WITHOUT
    # re-deriving/regrouping existing beliefs (multi-type intake re-derivation was proven
    # to DESTROY the calibration of the derived beliefs — see memory f6126f64).
    _SINGULAR_INTAKE_TYPES = ("decision", "learning", "error")
    _SINGULAR_DEFAULT_CONFIDENCE = 0.55

    @staticmethod
    def _singular_enabled() -> bool:
        """Opt-in flag MCP_BELIEF_SINGULAR (default OFF — byte-reversible, R6)."""
        raw = os.environ.get("MCP_BELIEF_SINGULAR")
        return raw is not None and raw.strip().lower() in {"true", "1", "yes"}

    @staticmethod
    def _singular_confidence() -> float:
        """Fixed modest confidence for singular beliefs (env-overridable). It does NOT
        pass through derive_confidence (Σ support − λ·contradiction), so the derived-belief
        calibration is never affected (R3)."""
        from ..config import safe_get_float_env
        return safe_get_float_env(
            "MCP_SINGULAR_BELIEF_CONFIDENCE", BeliefService._SINGULAR_DEFAULT_CONFIDENCE,
            min_value=0.0, max_value=1.0,
        )

    async def create_singular_belief(self, memory) -> bool:
        """Create ONE additive singular belief for a single actionable memory.

        This is the on-demand trigger (the CORRECT one, per the empirical finding
        that batch creation floods the corpus with 1639 beliefs — memory f02ac3dd):
        call this when an actionable memory is stored, so TODAY's fix becomes
        injectable same-day without resurrecting the whole backlog.

        Returns True if a singular belief was created. No-op when flag off, when the
        content is noise, or when a belief for that content already exists.
        """
        if not self._singular_enabled():
            return False
        content = memory.content if hasattr(memory, "content") else memory.get("content", "")
        chash = memory.content_hash if hasattr(memory, "content_hash") else memory.get("content_hash", "")
        if not content or not chash or _is_noise(content):
            return False
        belief_hash = _hash_content(content)
        if await self._get_belief(belief_hash):
            return False
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        try:
            await self._create_belief(belief_hash, content, self._singular_confidence(),
                                      "active", [chash], [], now)
            try:
                import json as _json
                self.storage.conn.execute(
                    "UPDATE beliefs SET metadata=? WHERE belief_hash=?",
                    (_json.dumps({"origin": "singular"}), belief_hash),
                )
                self.storage.conn.commit()
            except Exception:
                pass
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("[belief] create_singular_belief failed: %s", _sanitize_log_value(str(e)))
            return False

    # --- Frente D Inc 1: passive utility_prior (SEPARATE from confidence) ---
    # Zero-discipline attribution (insight 11/out, memory b34c007e): the service infers
    # usefulness from behavior it ALREADY sees (injected-then-used / injected-never-used),
    # NOT from the agent remembering to report. The signal lives in a SEPARATE
    # utility_prior (beliefs.metadata), never mutating `confidence` — decoupling
    # "useful to inject?" from "is it true?" AND disarming the A×B×C double-count
    # (frente A reads utility_prior; it must not re-penalize confidence).
    _UTILITY_PRIOR_FLOOR = 0.0
    _UTILITY_NEUTRAL = 0.5
    # Beta prior strength: higher = more episodes needed to move the prior (stability).
    _UTILITY_PRIOR_STRENGTH = 8.0
    # D2 — 3-signal refinement: asymmetric weighting and episode clamping
    _UTILITY_NEG_WEIGHT = 3.0  # negative signal weighs more than positive (clear asymmetry)
    _UTILITY_MAX_STEP = 0.1  # max change per episode (burst protection, one episode nudges not jumps)

    @staticmethod
    def _utility_prior_enabled() -> bool:
        """Opt-in flag MCP_BELIEF_UTILITY_PRIOR (default OFF)."""
        raw = os.environ.get("MCP_BELIEF_UTILITY_PRIOR")
        return raw is not None and raw.strip().lower() in {"true", "1", "yes"}

    @staticmethod
    def _beta_utility(positive: int, negative: int, strength: float = None) -> float:
        """D2: Beta-Bernoulli posterior with 3-signal refinement and asymmetric weighting.

        Signature: (positive, negative, strength)
        - positive = injected_then_used (boosts utility_prior)
        - negative = contradictions/re-query (penalizes, weighted by W_NEG > 1)
        - never_used is NEUTRAL — it is simply NOT passed here (not counted at all),
          so an injected-but-unused belief stays neutral (never penalized). This is the
          core D2 correction: absence-of-use is not evidence of uselessness.
        Formula: alpha = neutral*s + positive; beta = (1-neutral)*s + negative*W_NEG
        """

        s = strength if strength is not None else BeliefService._UTILITY_PRIOR_STRENGTH
        neutral = BeliefService._UTILITY_NEUTRAL
        w_neg = BeliefService._UTILITY_NEG_WEIGHT
        
        alpha = neutral * s + float(positive)
        beta = (1.0 - neutral) * s + float(negative) * w_neg
        
        if alpha + beta <= 0:
            return neutral
        return alpha / (alpha + beta)

    async def recompute_utility_priors(self, _signals_override: Optional[dict] = None) -> dict:
        """D2: Recompute per-belief utility_prior from 3-signal refinement (used/never_used/negative).

        Opt-in (MCP_BELIEF_UTILITY_PRIOR). Maps 3 signals:
        - used = injected_then_used (positive)
        - never_used = injected but never reused (neutral, ignored)
        - negative = contradictions/re-query (negative with asymmetric weight)
        
        Backward compatibility: accepts old {used, unused} format, treating unused as never_used.
        Episode clamping prevents wild swings from burst activity.
        """
        stats = {"updated": 0, "errors": []}
        if not self._utility_prior_enabled():
            return stats
        try:
            import json as _json
            if _signals_override is not None:
                signals = _signals_override
            else:
                signals = await self._collect_utility_signals()
            for belief_hash, sig in (signals or {}).items():
                # D2: Map to 3-signal interface with backward compatibility
                positive = int(sig.get("used", 0))
                negative = int(sig.get("negative", 0))
                never_used = int(sig.get("never_used", 0))
                
                # Backward compatibility strategy:
                # - If new format keys exist (never_used/negative), use new behavior
                # - If only old format exists (unused), maintain old behavior for compatibility
                has_new_format = "never_used" in sig or "negative" in sig
                
                if not has_new_format and "unused" in sig:
                    # Old format compatibility: map unused -> never_used (NEUTRAL, not negative)
                    # This fixes the conceptual bug where never_used was treated as penalty
                    unused_val = int(sig.get("unused", 0))
                    never_used = unused_val  # Correct semantics: unused is neutral
                elif "unused" in sig and has_new_format:
                    # Mixed format: 'unused' present but new keys also exist -> treat unused as never_used
                    never_used = int(sig.get("unused", 0))
                
                # Skip if no signals (never_used is neutral and doesn't count)
                has_any_signal = positive > 0 or negative > 0 or never_used > 0
                if not has_any_signal:
                    continue
                
                # Get current utility_prior for clamping
                row = self.storage.conn.execute(
                    "SELECT metadata FROM beliefs WHERE belief_hash=?", (belief_hash,)
                ).fetchone()
                if row is None:
                    continue
                    
                try:
                    meta = _json.loads(row[0]) if row[0] else {}
                except Exception:
                    meta = {}
                
                current_up = meta.get("utility_prior", self._UTILITY_NEUTRAL)
                
                # Compute new utility_prior using 3-signal formula (never_used ignored)
                new_up = self._beta_utility(positive, negative)
                
                # D2: Episode clamping - limit change per recompute call
                delta = new_up - current_up
                if abs(delta) > self._UTILITY_MAX_STEP:
                    clamped_delta = self._UTILITY_MAX_STEP if delta > 0 else -self._UTILITY_MAX_STEP
                    new_up = current_up + clamped_delta
                
                meta["utility_prior"] = round(new_up, 4)
                self.storage.conn.execute(
                    "UPDATE beliefs SET metadata=? WHERE belief_hash=?",
                    (_json.dumps(meta), belief_hash),
                )
                stats["updated"] += 1
            self.storage.conn.commit()
            logger.info("[belief] utility_prior recompute: updated=%d", stats["updated"])
        except Exception as e:  # noqa: BLE001
            stats["errors"].append(str(e))
            logger.warning("[belief] utility_prior recompute failed: %s", _sanitize_log_value(str(e)))
        return stats

    async def _collect_utility_signals(self) -> dict:
        """D2: Collect passive telemetry into 3-signal format {belief_hash: {used, never_used, negative}}.

        - used = injected_then_used count (positive signal)
        - never_used = injected_never_used count (neutral signal, ignored in calculation)
        - negative = contradictions/re-query count (negative signal, will come from _classify_requeries in D3)

        For now, negative=0 since telemetry doesn't yet classify contradictions.
        """
        out: dict = {}
        try:
            from ..storage.usage_telemetry import _derive_injected_then_used_map, _derive_injected_never_used, _derive_injected_then_requeried_map
            # Build source_hash → [belief_hash] mapping for positive signal correlation
            src_to_beliefs: dict = {}
            try:
                import json as _json
                for row in self.storage.conn.execute(
                    "SELECT belief_hash, derived_from FROM beliefs WHERE status='active'"
                ).fetchall():
                    bh = row[0]
                    try:
                        for sh in (_json.loads(row[1] or "[]") or []):
                            src_to_beliefs.setdefault(sh, []).append(bh)
                    except (ValueError, TypeError):
                        continue
            except Exception:  # noqa: BLE001
                src_to_beliefs = {}

            used_map = await _derive_injected_then_used_map(self.storage)   # {source_hash: count}
            unused_map = await _derive_injected_never_used(self.storage)    # {"agent:belief_hash": {...}}
            requeried_map = await _derive_injected_then_requeried_map(self.storage)  # {source_hash: count}
            
            
            # D2: Initialize all signals to 3-signal format
            for sh, cnt in (used_map or {}).items():
                targets = src_to_beliefs.get(sh, [sh])
                for bh in targets:
                    out.setdefault(bh, {"used": 0, "never_used": 0, "negative": 0})
                    out[bh]["used"] += int(cnt or 0)
            
            # Map never_used signal (neutral)
            for k, v in (unused_map or {}).items():
                bh = v.get("belief_hash") or k.split(":", 1)[-1]
                out.setdefault(bh, {"used": 0, "never_used": 0, "negative": 0})
                out[bh]["never_used"] += int(v.get("injections_without_use", 1) or 1)
            
            # D3: Implement negative signal from re-query detection
            for sh, cnt in (requeried_map or {}).items():
                targets = src_to_beliefs.get(sh, [sh])
                for bh in targets:
                    out.setdefault(bh, {"used": 0, "never_used": 0, "negative": 0})
                    out[bh]["negative"] += int(cnt or 0)
            
        except Exception as e:  # noqa: BLE001
            logger.debug("utility signal collection failed (non-fatal): %s", _sanitize_log_value(str(e)))
        return out

    async def derive_singular_beliefs(self, since_hours: Optional[float] = None,
                                      max_create: int = 50) -> dict:
        """Create ADDITIVE singular beliefs for actionable single-source memories.

        Opt-in (MCP_BELIEF_SINGULAR). BOUNDED by design (empirical finding: an
        unbounded batch created 1639 beliefs and flooded the corpus — memory
        f02ac3dd). Only memories newer than `since_hours` (default
        MCP_SINGULAR_SINCE_HOURS or 24h) are considered, and at most `max_create`
        singular beliefs are created per run. NEVER regroups or re-derives existing
        beliefs — proven byte-safe for the derived corpus.
        """
        stats = {"singular_created": 0, "errors": []}
        if not self._singular_enabled():
            return stats
        if since_hours is None:
            from ..config import safe_get_float_env
            since_hours = safe_get_float_env("MCP_SINGULAR_SINCE_HOURS", 24.0, min_value=0.0, max_value=8760.0)
        try:
            from datetime import datetime, timezone, timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).timestamp()
            for mtype in self._SINGULAR_INTAKE_TYPES:
                if stats["singular_created"] >= max_create:
                    break
                mems = await self.storage.get_all_memories(memory_type=mtype)
                for mem in mems or []:
                    if stats["singular_created"] >= max_create:
                        break
                    # Recency gate: only recent actionable memories (the "specific
                    # recent fix" the RFC targets), not the whole backlog.
                    created = getattr(mem, "created_at", None)
                    if created is None and isinstance(mem, dict):
                        created = mem.get("created_at")
                    try:
                        if created is not None and float(created) < cutoff:
                            continue
                    except (TypeError, ValueError):
                        pass
                    if await self.create_singular_belief(mem):
                        stats["singular_created"] += 1
            logger.info("[belief] singular batch: created=%d (<=%d, since %sh)",
                        stats["singular_created"], max_create, since_hours)
        except Exception as e:  # noqa: BLE001
            stats["errors"].append(str(e))
            logger.warning("[belief] singular derivation failed: %s", _sanitize_log_value(str(e)))
        return stats

    async def derive_beliefs(self) -> dict:
        """Run belief derivation cycle. Called by scheduler.

        1. Get all observation-type memories (or ontology base types if filtering enabled)
        2. Group by semantic similarity (existing embeddings)
        3. For each group, derive/update a belief
        4. Promote candidates meeting provenance floor
        5. Supersede beliefs below confidence floor
        """
        stats = {"created": 0, "updated": 0, "promoted": 0, "superseded": 0, "errors": []}

        try:
            # Check if ontology-based filtering is enabled
            ontology_filtering_enabled = self._ontology_filtering_enabled()
            
            if ontology_filtering_enabled:
                # New behavior: get all memories from default store and filter by base types
                observations = await self._get_memories_by_ontology()
            else:
                # Old behavior: get only observation type memories (no store filter)
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
                        # Apply negative use-signal penalty before updating (learning-loop frente B)
                        confidence = await self._apply_unused_penalty(belief_hash, confidence, existing)
                        
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
                    logger.warning("[belief] Error deriving belief: %s", _sanitize_log_value(str(e)))

        except Exception as e:
            stats["errors"].append(str(e))
            logger.error("[belief] Derivation cycle failed: %s", _sanitize_log_value(str(e)), exc_info=True)

        logger.info(
            "[belief] Derivation complete: created=%d updated=%d promoted=%d superseded=%d",
            stats["created"], stats["updated"], stats["promoted"], stats["superseded"],
        )
        return stats

    @staticmethod
    def _ontology_filtering_enabled() -> bool:
        """Check if ontology-based filtering is enabled via environment flag."""
        raw = os.environ.get("MCP_BELIEF_ONTOLOGY_FILTERING")
        return raw is not None and raw.strip().lower() in {"true", "1", "yes"}
    
    @staticmethod
    def _get_batch_cap() -> int:
        """Get batch processing cap from environment to prevent flooding."""
        try:
            raw = os.environ.get("MCP_BELIEF_BATCH_CAP")
            if raw:
                cap = int(raw.strip())
                return cap if cap > 0 else 1000  # Default cap if invalid
            return 1000  # Default cap
        except (ValueError, TypeError):
            return 1000  # Default cap on error
    
    async def _get_memories_by_ontology(self) -> list:
        """Get memories using ontology base types and store filtering.
        
        Returns memories from store='default' that match the curated belief-eligible
        base types (decision, learning, error, observation) or their subtypes.
        Restricts to the 4-type design scope, not all 12 ontology base types.
        """
        from ..models.ontology import get_parent_type
        
        # Curated set for belief derivation - restrict to the 4 designed types
        # not all 12 ontology base types to avoid scope creep
        BELIEF_ELIGIBLE_BASE_TYPES = {"decision", "learning", "error", "observation"}
        
        # Get all memories from default store only
        all_memories = await self.storage.get_all_memories(store="default")
        
        # Apply batch cap to prevent flooding
        batch_cap = self._get_batch_cap()
        if len(all_memories) > batch_cap:
            all_memories = all_memories[:batch_cap]
        
        # Filter memories to those with eligible base types or subtypes that resolve to eligible types
        filtered_memories = []
        for memory in all_memories:
            memory_type = memory.memory_type if hasattr(memory, "memory_type") else memory.get("memory_type", "")
            
            # Check if it's already an eligible base type
            if memory_type in BELIEF_ELIGIBLE_BASE_TYPES:
                filtered_memories.append(memory)
            else:
                # Check if it's a subtype that resolves to an eligible base type
                parent_type = get_parent_type(memory_type)
                if parent_type and parent_type in BELIEF_ELIGIBLE_BASE_TYPES:
                    filtered_memories.append(memory)
        
        return filtered_memories

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
            try:
                cursor = conn.execute(
                    "SELECT belief_hash, content, confidence, status, created_at, updated_at, "
                    "derived_from, contradicted_by, metadata, embedding FROM beliefs "
                    "WHERE status = ? AND confidence >= ? ORDER BY confidence DESC",
                    (status, min_confidence),
                )
                rows = cursor.fetchall()
                has_embedding = True
            except Exception as e:
                if "no such column: embedding" in str(e) or "has no column" in str(e):
                    cursor = conn.execute(
                        "SELECT belief_hash, content, confidence, status, created_at, updated_at, "
                        "derived_from, contradicted_by, metadata FROM beliefs "
                        "WHERE status = ? AND confidence >= ? ORDER BY confidence DESC",
                        (status, min_confidence),
                    )
                    rows = cursor.fetchall()
                    has_embedding = False
                else:
                    raise
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
                    "embedding": (r[9] if has_embedding else None),
                    "result_type": "belief",
                }
                for r in rows
            ]
        except Exception as e:
            logger.error("[belief] get_beliefs failed: %s", _sanitize_log_value(str(e)))
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
                mem = await self.storage.get_by_hash(h)
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
        """Insert a new belief. Also stores a semantic embedding of the content (R4) when
        an embedding model is available; NULL otherwise (re-rank falls back to lexical)."""
        conn = self.storage.conn
        now_iso = now.isoformat()
        embedding_blob = self._compute_belief_embedding(content)
        try:
            conn.execute(
                "INSERT OR IGNORE INTO beliefs "
                "(belief_hash, content, confidence, status, created_at, updated_at, derived_from, contradicted_by, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    belief_hash, content, confidence, status, now_iso, now_iso,
                    json.dumps(supporting_hashes), json.dumps(contradicting_hashes), embedding_blob,
                ),
            )
        except Exception as e:
            # Backward compat: belief table predates migration 021 (no embedding column).
            if "no column named embedding" in str(e) or "has no column" in str(e):
                conn.execute(
                    "INSERT OR IGNORE INTO beliefs "
                    "(belief_hash, content, confidence, status, created_at, updated_at, derived_from, contradicted_by) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        belief_hash, content, confidence, status, now_iso, now_iso,
                        json.dumps(supporting_hashes), json.dumps(contradicting_hashes),
                    ),
                )
            else:
                raise
        conn.commit()

    def _compute_belief_embedding(self, content: str):
        """Return a serialized embedding BLOB for the belief content, or None if no model.

        Never raises: embedding is an optional enrichment (R4). A failure degrades the
        belief to lexical-only relevance in the re-rank, never blocks belief creation.
        """
        try:
            if getattr(self.storage, "embedding_model", None) is None:
                return None
            from sqlite_vec import serialize_float32
            vec = self.storage._generate_embedding(content)
            return serialize_float32(vec)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("belief embedding skipped (%s): %s", _sanitize_log_value(type(e).__name__), _sanitize_log_value(e))
            return None

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

    def _get_max_unused_count_for_belief(self, belief_hash: str, unused_stats: dict) -> tuple:
        """Extract maximum unused count and relevant agent for a belief across all agents.
        
        Returns (max_count, agent_id). Takes conservative max when belief is
        used by multiple agents (unused-by-one penalizes-globally).
        This is a product decision: confidence is a single global column, so we
        cannot apply different penalties per-agent. Ideally would penalize only
        per the agent(s) that inject without use.
        """
        max_count = 0
        relevant_agent = None
        
        for composite_key, stats in unused_stats.items():
            if stats.get("belief_hash") == belief_hash:
                count = stats.get("injections_without_use", 0)
                if count > max_count:
                    max_count = count
                    relevant_agent = stats.get("agent_id")
        
        return max_count, relevant_agent

    def _update_belief_metadata_counter(self, belief_hash: str, existing: dict, count: int) -> None:
        """Store unused injection counter in beliefs.metadata JSON.
        
        If count is 0, removes the counter (reset). This enables visibility
        of the negative signal state and facilitates debugging.
        """
        metadata = json.loads(existing.get("metadata") or "{}")
        
        if count == 0:
            metadata.pop("injections_without_use", None)
        else:
            metadata["injections_without_use"] = count
        
        conn = self.storage.conn
        conn.execute(
            "UPDATE beliefs SET metadata = ? WHERE belief_hash = ?",
            (json.dumps(metadata), belief_hash)
        )
        conn.commit()

    async def _apply_unused_penalty(self, belief_hash: str, confidence: float, existing: dict) -> float:
        """Apply negative use-signal penalty to belief confidence (learning-loop frente B).
        
        Checks if belief has been injected ≥N times without use and applies penalty.
        Stores unused counter in beliefs.metadata JSON to enable counter reset on use.
        
        NOTE: Uses max() across agents when a belief is used by multiple agents
        (conservative penalty — unused-by-one penalizes-globally). See 
        _get_max_unused_count_for_belief docstring for rationale.
        """
        # Import _sanitize_log_value for consistent log sanitization
        from .contradictions import _sanitize_log_value
        
        # Check opt-in flag
        use_feedback_enabled = os.getenv("MCP_BELIEF_USE_FEEDBACK", "false").lower() == "true"
        if not use_feedback_enabled:
            return confidence
        
        try:
            # Import here to avoid circular dependency
            from ..storage.usage_telemetry import _derive_injected_never_used
            
            # Get unused stats for all beliefs (keyed by "agent_id:belief_hash")
            unused_stats = await _derive_injected_never_used(self.storage)
            
            # Find maximum unused count for this belief across all agents
            max_unused_count, relevant_agent = self._get_max_unused_count_for_belief(
                belief_hash, unused_stats
            )
            
            if max_unused_count == 0:
                # No penalty - belief is being used or has no unused injections
                # Reset counter in metadata if it exists (R4: use resets)
                self._update_belief_metadata_counter(belief_hash, existing, 0)
                return confidence
            
            # Check if belief meets the threshold for penalty
            threshold = int(os.getenv("MCP_BELIEF_UNUSED_THRESHOLD", "3"))
            
            if max_unused_count < threshold:
                # Below threshold - no penalty but store counter for visibility
                self._update_belief_metadata_counter(belief_hash, existing, max_unused_count)
                return confidence
            
            # Belief meets unused threshold - apply penalty (R2)
            self._update_belief_metadata_counter(belief_hash, existing, max_unused_count)
            
            # Apply penalty factor
            penalty_factor = float(os.getenv("MCP_BELIEF_UNUSED_PENALTY", "0.20"))
            penalized_confidence = confidence * (1 - penalty_factor)
            
            # Log with sanitization for consistency (L2 fix)
            logger.info(
                "[belief] Applying unused penalty to %s: %d unused injections, "
                "confidence %.3f → %.3f",
                _sanitize_log_value(belief_hash[:8]), 
                max_unused_count, 
                confidence, 
                penalized_confidence
            )
            
            return penalized_confidence
            
        except Exception as e:
            logger.warning(
                "[belief] Failed to apply unused penalty (non-fatal): %s", 
                _sanitize_log_value(str(e))
            )
            return confidence
