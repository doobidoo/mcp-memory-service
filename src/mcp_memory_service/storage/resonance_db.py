"""ResonanceDB — Wave-Based Semantic Memory (Phase-Aware Retrieval)

Implements the wave-based memory model from:
"Wave-Based Semantic Memory with Resonance-Based Retrieval"
arXiv:2509.09691 (submitted Aug 2025)

Core idea: Instead of storing memories as static real-valued vectors and
comparing with cosine similarity, store them as complex waveforms:

    ψ(x) = A(x) · exp(i · φ(x))

where:
    A(x) = semantic amplitude (magnitude of meaning)
    φ(x) = contextual phase (semantic polarity, negation, modality)

Similarity is measured via **constructive/destructive interference** instead
of distance in a Euclidean space.

Key property that solves the "happy" vs "not happy" problem:
    Two waves that are in-phase amplify (resonate).
    Two waves shifted by π cancel (destructive interference).
    Negation naturally produces a phase shift → anti-resonance.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Complex waveform math
# ---------------------------------------------------------------------------

def _complex_dot_product(a: List[complex], b: List[complex]) -> complex:
    """Compute <a|b> = Σ a*(x) · b(x) for two complex waveforms."""
    result = complex(0, 0)
    for ax, bx in zip(a, b):
        result += ax.conjugate() * bx
    return result


def _wave_interference_energy(psi_a: List[complex], psi_b: List[complex]) -> float:
    """
    Compute constructive interference energy between two waveforms.
    
    Maximal when in-phase (aligned semantics), minimal when out-of-phase
    (contrasting/negated semantics).
    
    Formula from paper:
        E = |Σ A_a(x)·A_b(x)·cos(φ_a(x) - φ_b(x))|
    
    Returns value in [0, 1] after normalization.
    """
    if not psi_a or not psi_b:
        return 0.0
    
    n = min(len(psi_a), len(psi_b))
    
    # Extract amplitude and phase from each waveform
    real_part = 0.0
    for x in range(n):
        ax = psi_a[x]
        bx = psi_b[x]
        # Amplitude product × phase difference cosine
        real_part += ax.real * bx.real + ax.imag * bx.imag
    
    # Normalize by maximum possible energy (sum of |ψ(x)|²)
    energy_a = sum(abs(a) ** 2 for a in psi_a[:n])
    energy_b = sum(abs(b) ** 2 for b in psi_b[:n])
    
    if energy_a == 0 or energy_b == 0:
        return 0.0
    
    max_energy = math.sqrt(energy_a * energy_b)
    if max_energy == 0:
        return 0.0
    
    resonance = real_part / max_energy
    
    # Clamp to [0, 1] for score
    return max(0.0, min(1.0, resonance))


def _phase_shift_score(psi_a: List[complex], psi_b: List[complex]) -> float:
    """
    Detect phase-shift patterns (negation, modality inversion).
    
    Returns the mean phase difference normalized to [0, 1]:
    0 = in phase (aligned), 1 = shifted by π (contrast/negation).
    """
    if not psi_a or not psi_b:
        return 0.0
    
    n = min(len(psi_a), len(psi_b))
    if n == 0:
        return 0.0
    
    # Compute mean phase difference
    phase_diffs = []
    for x in range(n):
        phase_a = math.atan2(psi_a[x].imag, psi_a[x].real)
        phase_b = math.atan2(psi_b[x].imag, psi_b[x].real)
        diff = phase_a - phase_b
        # Wrap to [-π, π]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        phase_diffs.append(abs(diff))
    
    mean_phase_diff = sum(phase_diffs) / len(phase_diffs) if phase_diffs else 0.0
    
    # 0 → aligned, π → negation; callers rank by (1 - this value)
    return abs(mean_phase_diff) / math.pi


def _constructive_interference(psi_a: List[complex], psi_b: List[complex]) -> float:
    """
    Full resonance score: constructive interference.
    
    When aligned → waves amplify (high score).
    When opposed → waves cancel (low score).
    This is the CORE similarity metric from the paper.
    """
    return _wave_interference_energy(psi_a, psi_b)


def _destructive_interference(psi_a: List[complex], psi_b: List[complex]) -> float:
    """
    Anti-resonance score.
    
    When opposed (e.g. negation) → high destructive score.
    When aligned → low destructive score.
    """
    return _phase_shift_score(psi_a, psi_b)


# ---------------------------------------------------------------------------
# Phase-aware embedding: real vector → complex waveform
# ---------------------------------------------------------------------------

def _embed_to_waveform(embedding: List[float], dimension: int) -> List[complex]:
    """
    Convert a real-valued embedding (e.g. from all-MiniLM) to a complex waveform.
    
    Strategy: Split embedding into amplitude (magnitude) and phase (semantics).
    
    Phase encoding uses a PCA-like approach:
    - Positive semantics → positive phase angles (0 to π/2)
    - Negative semantics → negative phase angles (-π/2 to 0)
    - Negation patterns create phase shifts across multiple dimensions
    
    This is a lightweight approximation of the paper's approach without
    needing to retrain an embedding model.
    """
    n = min(len(embedding), dimension)
    waveform = [complex(0, 0)] * n
    
    # Pre-compute global statistics for better phase encoding
    mean_val = sum(embedding[:n]) / n if n > 0 else 0.0
    std_val = math.sqrt(sum((e - mean_val) ** 2 for e in embedding[:n]) / n) if n > 0 else 1.0
    if std_val < 1e-10:
        std_val = 1.0
    
    for i in range(n):
        val = embedding[i] if i < len(embedding) else 0.0
        
        # Amplitude: relative magnitude (how strong is this semantic dimension?)
        relative_val = (val - mean_val) / std_val
        amplitude = math.sqrt(abs(relative_val)) + 1e-10
        
        # Phase: encode semantic polarity
        # Use the sign and magnitude of the deviation from mean
        # This captures whether a dimension expresses "positive" or "negative" semantics
        sign = 1.0 if relative_val > 0 else -1.0
        # Scale phase angle by magnitude of deviation
        phase_magnitude = min(abs(relative_val) / 2.0, 1.0) * (math.pi / 2)
        phase = sign * phase_magnitude
        
        # Construct waveform: ψ(x) = A(x) · exp(i · φ(x))
        waveform[i] = complex(
            amplitude * math.cos(phase),
            amplitude * math.sin(phase)
        )
    
    return waveform


# ---------------------------------------------------------------------------
# ResonanceDB class
# ---------------------------------------------------------------------------

@dataclass
class ResonanceEntry:
    """A memory stored as a complex waveform in ResonanceDB."""
    content_hash: str
    embedding: List[float]           # Original real embedding (from sentence-transformers)
    waveform: List[complex]           # Complex waveform ψ(x)
    
    @classmethod
    def from_embedding(cls, content_hash: str, embedding: List[float], dimension: int = 384) -> "ResonanceEntry":
        """Create a ResonanceEntry from a real-valued embedding."""
        waveform = _embed_to_waveform(embedding, dimension)
        return cls(
            content_hash=content_hash,
            embedding=embedding,
            waveform=waveform,
        )


class ResonanceDB:
    """
    Phase-aware semantic memory storage and retrieval.
    
    Wraps existing real-valued embeddings and adds a wave-based similarity
    layer on top. Can be used as a drop-in replacement for cosine similarity
    in search pipelines.
    
    Architecture:
    1. Real embeddings stored as before (compatible with existing database)
    2. Waveform conversion happens at query time (no schema change needed)
    3. Resonance-ordered results returned to caller
    
    Usage:
        db = ResonanceDB(dimension=384)
        db.add("hash1", embedding_vector)
        results = db.search("query embedding", top_k=5)
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.entries: Dict[str, ResonanceEntry] = {}
        
        # Resonance tuning parameters (from paper)
        self.resonance_weight = 0.7      # Weight for constructive interference
        self.phase_weight = 0.3          # Weight for phase-shift detection
    
    def add(self, content_hash: str, embedding: List[float]) -> ResonanceEntry:
        """Store a memory's embedding as a complex waveform."""
        entry = ResonanceEntry.from_embedding(content_hash, embedding, self.dimension)
        self.entries[content_hash] = entry
        return entry
    
    def remove(self, content_hash: str) -> bool:
        """Remove a memory from ResonanceDB."""
        if content_hash in self.entries:
            del self.entries[content_hash]
            return True
        return False
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        mode: str = "resonance",
    ) -> List[Dict[str, Any]]:
        """
        Search memories using wave-based resonance retrieval.
        
        Args:
            query_embedding: Real-valued embedding for the query
            top_k: Number of results to return
            mode: 
                - "resonance": Pure constructive interference (default)
                - "hybrid": Combines resonance + phase-shift scores
                - "phase": Emphasizes phase-shift detection
        
        Returns:
            List of {"hash": str, "score": float, "details": {...}}
        """
        # Convert query to waveform
        query_waveform = _embed_to_waveform(query_embedding, self.dimension)
        
        # Score all stored entries
        scored: List[Tuple[str, float, Dict[str, float]]] = []
        
        for content_hash, entry in self.entries.items():
            # Skip entries with empty/None waveform
            if not entry.waveform or all(c == complex(0, 0) for c in entry.waveform):
                continue
            
            if mode == "resonance":
                # Pure constructive interference
                score = _constructive_interference(entry.waveform, query_waveform)
                details = {
                    "resonance": score,
                    "phase_shift": 0.0,
                }
            elif mode == "phase":
                # Emphasize phase-shift (negation detection)
                phase_score = _destructive_interference(entry.waveform, query_waveform)
                score = 1.0 - phase_score  # Invert: low phase shift = high relevance
                details = {
                    "resonance": 0.0,
                    "phase_shift": phase_score,
                }
            else:  # hybrid (default)
                resonance = _constructive_interference(entry.waveform, query_waveform)
                phase_shift = _destructive_interference(entry.waveform, query_waveform)
                # Weighted combination
                score = self.resonance_weight * resonance + self.phase_weight * (1.0 - phase_shift)
                details = {
                    "resonance": resonance,
                    "phase_shift": phase_shift,
                }
            
            scored.append((content_hash, score, details))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        results = []
        for content_hash, score, details in scored[:top_k]:
            results.append({
                "content_hash": content_hash,
                "score": round(score, 6),
                "details": {k: round(v, 6) for k, v in details.items()},
            })
        
        return results
    
    def compare_pair(self, hash_a: str, hash_b: str) -> Dict[str, Any]:
        """
        Compare two stored memories using wave-based resonance.
        
        Useful for debugging and understanding how ResonanceDB differs from
        cosine similarity on specific examples.
        """
        if hash_a not in self.entries or hash_b not in self.entries:
            return {"error": "One or both hashes not found"}
        
        entry_a = self.entries[hash_a]
        entry_b = self.entries[hash_b]
        
        # Resonance score
        resonance = _constructive_interference(entry_a.waveform, entry_b.waveform)
        
        # Phase-shift score
        phase_shift = _destructive_interference(entry_a.waveform, entry_b.waveform)
        
        # Hybrid score
        hybrid = self.resonance_weight * resonance + self.phase_weight * (1.0 - phase_shift)
        
        # Cosine similarity (baseline for comparison)
        cosine = _cosine_similarity(entry_a.embedding, entry_b.embedding)
        
        return {
            "hash_a": hash_a,
            "hash_b": hash_b,
            "resonance_score": round(resonance, 6),
            "phase_shift_score": round(phase_shift, 6),
            "hybrid_score": round(hybrid, 6),
            "cosine_similarity": round(cosine, 6),
            "waveform_length": len(entry_a.waveform),
        }
    
    def batch_add(self, memories: List[Tuple[str, List[float]]]) -> int:
        """Add multiple memories at once."""
        count = 0
        for content_hash, embedding in memories:
            self.add(content_hash, embedding)
            count += 1
        return count
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __bool__(self) -> bool:
        return bool(self.entries)


# ---------------------------------------------------------------------------
# Cosine similarity helper (for comparison)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Standard cosine similarity for comparison with resonance scores."""
    if not a or not b:
        return 0.0
    
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    
    dot = sum(ax * bx for ax, bx in zip(a, b))
    mag_a = math.sqrt(sum(ax ** 2 for ax in a))
    mag_b = math.sqrt(sum(bx ** 2 for bx in b))
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return dot / (mag_a * mag_b)
