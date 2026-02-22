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

"""Semantic clustering system for memory organization."""

import uuid
import numpy as np
from typing import List, Dict
from datetime import datetime, timezone
from collections import Counter
import re

try:
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .base import ConsolidationBase, ConsolidationConfig, MemoryCluster
from ..models.memory import Memory

class SemanticClusteringEngine(ConsolidationBase):
    """
    Creates semantic clusters of related memories for organization and compression.
    
    Uses embedding-based clustering algorithms (DBSCAN, Hierarchical) to group
    semantically similar memories, enabling efficient compression and retrieval.
    """
    
    def __init__(self, config: ConsolidationConfig):
        super().__init__(config)
        self.min_cluster_size = config.min_cluster_size
        self.algorithm = config.clustering_algorithm
        
        if not SKLEARN_AVAILABLE:
            self.logger.warning("sklearn not available, using simple clustering fallback")
            self.algorithm = 'simple'
    
    async def process(self, memories: List[Memory], **kwargs) -> List[MemoryCluster]:
        """Create semantic clusters from memories."""
        if not self._validate_memories(memories) or len(memories) < self.min_cluster_size:
            return []
        
        # Filter memories with embeddings
        memories_with_embeddings = [m for m in memories if m.embedding]
        
        if len(memories_with_embeddings) < self.min_cluster_size:
            self.logger.warning(f"Only {len(memories_with_embeddings)} memories have embeddings, need at least {self.min_cluster_size}")
            return []
        
        # Extract embeddings matrix
        embeddings = np.array([m.embedding for m in memories_with_embeddings])
        
        # Perform clustering
        if self.algorithm == 'dbscan':
            cluster_labels = await self._dbscan_clustering(embeddings)
        elif self.algorithm == 'hierarchical':
            cluster_labels = await self._hierarchical_clustering(embeddings)
        else:
            cluster_labels = await self._simple_clustering(embeddings)
        
        # Create cluster objects
        clusters = await self._create_clusters(memories_with_embeddings, cluster_labels, embeddings)
        
        # Filter by minimum cluster size
        valid_clusters = [c for c in clusters if len(c.memory_hashes) >= self.min_cluster_size]
        
        self.logger.info(f"Created {len(valid_clusters)} valid clusters from {len(memories_with_embeddings)} memories")
        return valid_clusters
    
    async def _dbscan_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering on embeddings."""
        if not SKLEARN_AVAILABLE:
            return await self._simple_clustering(embeddings)
        
        # Adaptive epsilon based on data size and dimensionality
        n_samples, n_features = embeddings.shape
        eps = 0.5 - (n_samples / 10000) * 0.1  # Decrease eps for larger datasets
        eps = max(0.2, min(0.7, eps))  # Clamp between 0.2 and 0.7
        
        min_samples = max(2, self.min_cluster_size // 2)
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        self.logger.debug(f"DBSCAN: eps={eps}, min_samples={min_samples}, found {len(set(labels))} clusters")
        return labels
    
    async def _hierarchical_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform hierarchical clustering on embeddings."""
        if not SKLEARN_AVAILABLE:
            return await self._simple_clustering(embeddings)
        
        # Estimate number of clusters (heuristic: sqrt of samples / 2)
        n_samples = embeddings.shape[0]
        n_clusters = max(2, min(n_samples // self.min_cluster_size, int(np.sqrt(n_samples) / 2)))
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)
        
        self.logger.debug(f"Hierarchical: n_clusters={n_clusters}, found {len(set(labels))} clusters")
        return labels
    
    async def _simple_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Simple fallback clustering using cosine similarity threshold."""
        n_samples = embeddings.shape[0]
        labels = np.full(n_samples, -1)  # Start with all as noise
        current_cluster = 0
        
        similarity_threshold = 0.7  # Threshold for grouping
        
        for i in range(n_samples):
            if labels[i] != -1:  # Already assigned
                continue
            
            # Start new cluster
            cluster_members = [i]
            labels[i] = current_cluster
            
            # Find similar memories
            for j in range(i + 1, n_samples):
                if labels[j] != -1:  # Already assigned
                    continue
                
                # Calculate cosine similarity
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                
                if similarity >= similarity_threshold:
                    labels[j] = current_cluster
                    cluster_members.append(j)
            
            # Only keep cluster if it meets minimum size
            if len(cluster_members) >= self.min_cluster_size:
                current_cluster += 1
            else:
                # Mark as noise
                for member in cluster_members:
                    labels[member] = -1
        
        self.logger.debug(f"Simple clustering: threshold={similarity_threshold}, found {current_cluster} clusters")
        return labels
    
    async def _create_clusters(
        self,
        memories: List[Memory],
        labels: np.ndarray,
        embeddings: np.ndarray
    ) -> List[MemoryCluster]:
        """Create MemoryCluster objects from clustering results."""
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            
            # Get memories in this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_memories = [memories[i] for i in cluster_indices]
            cluster_embeddings = embeddings[cluster_indices]
            
            if len(cluster_memories) < self.min_cluster_size:
                continue
            
            # Calculate centroid embedding
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Calculate coherence score (average cosine similarity to centroid)
            coherence_scores = []
            for embedding in cluster_embeddings:
                similarity = np.dot(embedding, centroid) / (
                    np.linalg.norm(embedding) * np.linalg.norm(centroid)
                )
                coherence_scores.append(similarity)
            
            coherence_score = np.mean(coherence_scores)
            
            # Extract theme keywords
            theme_keywords = await self._extract_theme_keywords(cluster_memories)
            
            # Create cluster
            cluster = MemoryCluster(
                cluster_id=str(uuid.uuid4()),
                memory_hashes=[m.content_hash for m in cluster_memories],
                centroid_embedding=centroid.tolist(),
                coherence_score=float(coherence_score),
                created_at=datetime.now(),
                theme_keywords=theme_keywords,
                metadata={
                    'algorithm': self.algorithm,
                    'cluster_size': len(cluster_memories),
                    'average_memory_age': self._calculate_average_age(cluster_memories),
                    'tag_distribution': self._analyze_tag_distribution(cluster_memories)
                }
            )
            
            clusters.append(cluster)
        
        return clusters
    
    async def _extract_theme_keywords(self, memories: List[Memory]) -> List[str]:
        """Extract theme keywords that represent the cluster."""
        # Combine all content
        all_text = ' '.join([m.content for m in memories])
        
        # Collect all tags
        all_tags = []
        for memory in memories:
            all_tags.extend(memory.tags)
        
        # Count tag frequency
        tag_counts = Counter(all_tags)
        
        # Extract frequent words from content (simple approach)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
        word_counts = Counter(words)
        
        # Remove common stop words
        stop_words = {
            'this', 'that', 'with', 'have', 'will', 'from', 'they', 'know',
            'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when',
            'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over',
            'such', 'take', 'than', 'them', 'well', 'were', 'what', 'work',
            'your', 'could', 'should', 'would', 'there', 'their', 'these',
            'about', 'after', 'again', 'before', 'being', 'between', 'during',
            'under', 'where', 'while', 'other', 'through', 'against'
        }
        
        # Filter and get top words
        filtered_words = {word: count for word, count in word_counts.items() 
                         if word not in stop_words and count > 1}
        
        # Combine tags and words, prioritize tags
        theme_keywords = []
        
        # Add top tags (weight by frequency)
        for tag, count in tag_counts.most_common(5):
            if count > 1:  # Tag appears in multiple memories
                theme_keywords.append(tag)
        
        # Add top words
        for word, count in sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:10]:
            if word not in theme_keywords:
                theme_keywords.append(word)
        
        return theme_keywords[:10]  # Limit to top 10
    
    def _calculate_average_age(self, memories: List[Memory]) -> float:
        """Calculate average age of memories in days."""
        now = datetime.now(timezone.utc)
        ages = []
        
        for memory in memories:
            if memory.created_at:
                created_dt = datetime.fromtimestamp(memory.created_at, tz=timezone.utc)
                age_days = (now - created_dt).days
                ages.append(age_days)
            elif memory.timestamp:
                ts = memory.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_days = (now - ts).days
                ages.append(age_days)
        
        return sum(ages) / len(ages) if ages else 0.0
    
    def _analyze_tag_distribution(self, memories: List[Memory]) -> Dict[str, int]:
        """Analyze tag distribution within the cluster."""
        all_tags = []
        for memory in memories:
            all_tags.extend(memory.tags)
        
        return dict(Counter(all_tags))
    
    async def merge_similar_clusters(
        self,
        clusters: List[MemoryCluster],
        similarity_threshold: float = 0.8
    ) -> List[MemoryCluster]:
        """Merge clusters that are very similar to each other."""
        if len(clusters) <= 1:
            return clusters
        
        # Calculate pairwise similarities between cluster centroids
        centroids = np.array([cluster.centroid_embedding for cluster in clusters])
        
        merged = [False] * len(clusters)
        result_clusters = []
        
        for i, cluster1 in enumerate(clusters):
            if merged[i]:
                continue
            
            # Start with current cluster
            merge_group = [i]
            merged[i] = True
            
            # Find similar clusters to merge
            for j in range(i + 1, len(clusters)):
                if merged[j]:
                    continue
                
                # Calculate cosine similarity between centroids
                similarity = np.dot(centroids[i], centroids[j]) / (
                    np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j])
                )
                
                if similarity >= similarity_threshold:
                    merge_group.append(j)
                    merged[j] = True
            
            # Create merged cluster
            if len(merge_group) == 1:
                # No merging needed
                result_clusters.append(clusters[i])
            else:
                # Merge clusters
                merged_cluster = await self._merge_cluster_group(
                    [clusters[idx] for idx in merge_group]
                )
                result_clusters.append(merged_cluster)
        
        self.logger.info(f"Merged {len(clusters)} clusters into {len(result_clusters)}")
        return result_clusters
    
    async def _merge_cluster_group(self, clusters: List[MemoryCluster]) -> MemoryCluster:
        """Merge a group of similar clusters into one."""
        # Combine all memory hashes
        all_memory_hashes = []
        for cluster in clusters:
            all_memory_hashes.extend(cluster.memory_hashes)
        
        # Calculate new centroid (average of all centroids weighted by cluster size)
        total_size = sum(len(cluster.memory_hashes) for cluster in clusters)
        weighted_centroid = np.zeros(len(clusters[0].centroid_embedding))
        
        for cluster in clusters:
            weight = len(cluster.memory_hashes) / total_size
            centroid = np.array(cluster.centroid_embedding)
            weighted_centroid += weight * centroid
        
        # Combine theme keywords
        all_keywords = []
        for cluster in clusters:
            all_keywords.extend(cluster.theme_keywords)
        
        keyword_counts = Counter(all_keywords)
        merged_keywords = [kw for kw, count in keyword_counts.most_common(10)]
        
        # Calculate average coherence score
        total_memories = sum(len(cluster.memory_hashes) for cluster in clusters)
        weighted_coherence = sum(
            cluster.coherence_score * len(cluster.memory_hashes) / total_memories
            for cluster in clusters
        )
        
        return MemoryCluster(
            cluster_id=str(uuid.uuid4()),
            memory_hashes=all_memory_hashes,
            centroid_embedding=weighted_centroid.tolist(),
            coherence_score=weighted_coherence,
            created_at=datetime.now(),
            theme_keywords=merged_keywords,
            metadata={
                'algorithm': f"{self.algorithm}_merged",
                'cluster_size': len(all_memory_hashes),
                'merged_from': [cluster.cluster_id for cluster in clusters],
                'merge_timestamp': datetime.now().isoformat()
            }
        )