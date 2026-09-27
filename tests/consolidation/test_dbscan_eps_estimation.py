"""
Regression tests for DBSCAN eps estimation (chaining fix, upstream issue
observed in production: 989/1104 memories collapsed into a single cluster
with coherence 0.461).

The old `_dbscan_clustering` derived eps purely from dataset size
(`0.5 - n/10000`, clamped to [0.2, 0.7]) with no reference to the actual
similarity distribution of the embeddings being clustered. Some
sentence-embedding models (e.g. the all-MiniLM-L6-v2 default) produce a
narrow, elevated baseline cosine similarity across unrelated content, so that
size-only eps let DBSCAN's density-reachability chain unrelated memories
into one giant cluster.

These tests exercise `_estimate_eps` directly. It uses
`sklearn.neighbors.NearestNeighbors` internally for the k-distance lookup
(replacing an earlier full n x n distance matrix, which was O(n^2) memory
and time -- flagged in review), so scikit-learn is required to run these
tests. Chaining verification itself uses a from-scratch connected-components
check rather than DBSCAN, so no additional dependency is needed for that
part.
"""

import numpy as np
import pytest

# _estimate_eps uses sklearn.neighbors.NearestNeighbors (see module docstring).
# The lean [sqlite] install has no scikit-learn; the ml-extras CI job does.
pytest.importorskip("sklearn", reason="scikit-learn is required for the eps estimation tests")

from mcp_memory_service.consolidation.clustering import SemanticClusteringEngine


def _old_formula_eps(n_samples: int) -> float:
    """Reproduce the pre-fix size-only eps formula for comparison."""
    return max(0.2, min(0.7, 0.5 - (n_samples / 10000) * 0.1))


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return 1.0 - (normalized @ normalized.T)


def _largest_connected_component(distance: np.ndarray, eps: float) -> int:
    """BFS connected-components size at a given eps -- what DBSCAN's
    density-reachability effectively produces for a fully-dense neighborhood
    graph (a superset of true DBSCAN clusters, but sufficient to prove
    whether an eps value structurally permits one component to swallow
    (almost) everything)."""
    n = distance.shape[0]
    adjacency = distance <= eps
    np.fill_diagonal(adjacency, False)
    visited = np.zeros(n, dtype=bool)
    largest = 0
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = np.where(adjacency[node] & ~visited)[0]
            visited[neighbors] = True
            stack.extend(neighbors.tolist())
        largest = max(largest, size)
    return largest


@pytest.mark.unit
class TestEstimateEps:
    def test_falls_back_to_default_for_tiny_datasets(self):
        embeddings = np.random.default_rng(0).normal(size=(3, 8))
        eps = SemanticClusteringEngine._estimate_eps(embeddings, min_samples=5)
        assert eps == 0.3

    def test_returns_a_sane_positive_value(self):
        rng = np.random.default_rng(1)
        embeddings = rng.normal(size=(50, 16))
        eps = SemanticClusteringEngine._estimate_eps(embeddings, min_samples=3)
        assert 0.05 <= eps <= 2.0

    def test_separates_two_tight_clusters_from_each_other(self):
        """Two well-separated dense blobs: the knee-based eps must sit below
        the inter-cluster gap, so each blob stays its own component rather
        than merging into one."""
        rng = np.random.default_rng(2)
        dim = 16
        base_a = rng.normal(size=dim)
        base_b = -base_a  # maximally dissimilar direction under cosine
        cluster_a = base_a + rng.normal(scale=0.02, size=(25, dim))
        cluster_b = base_b + rng.normal(scale=0.02, size=(25, dim))
        embeddings = np.vstack([cluster_a, cluster_b])

        eps = SemanticClusteringEngine._estimate_eps(embeddings, min_samples=3)
        distance = _cosine_distance_matrix(embeddings)
        largest = _largest_connected_component(distance, eps)

        # Each blob has 25 points; a correct eps keeps them apart.
        assert largest <= 26, (
            f"eps={eps} merged the two separated clusters into one "
            f"component of size {largest}"
        )

    def test_reproduces_less_chaining_than_the_old_size_only_formula(self):
        """Reproduce the production chaining symptom: a slow random-walk
        manifold in embedding space, which mimics an embedding model with a
        narrow/elevated baseline similarity (adjacent points are close, but
        the old size-only eps was loose enough to chain distant, unrelated
        points together transitively). The new eps must produce a
        meaningfully smaller largest component than the old formula on the
        same data.
        """
        rng = np.random.default_rng(3)
        n, dim = 200, 16
        steps = rng.normal(scale=0.05, size=(n, dim))
        walk = np.cumsum(steps, axis=0)
        embeddings = walk + rng.normal(scale=0.01, size=(n, dim))

        distance = _cosine_distance_matrix(embeddings)
        min_samples = 3

        new_eps = SemanticClusteringEngine._estimate_eps(embeddings, min_samples)
        old_eps = _old_formula_eps(n)

        new_largest = _largest_connected_component(distance, new_eps)
        old_largest = _largest_connected_component(distance, old_eps)

        assert new_largest < old_largest, (
            f"new eps={new_eps:.4f} (largest component {new_largest}) did not "
            f"reduce chaining vs. old eps={old_eps:.4f} "
            f"(largest component {old_largest})"
        )
        # The old formula is expected to reproduce the production symptom:
        # most of the dataset ends up in one chained component.
        assert old_largest > n * 0.5
