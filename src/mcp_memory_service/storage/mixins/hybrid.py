"""HybridMixin: retrieve_hybrid, BM25 search, score fusion (weighted + RRF)."""

import re
import logging
import traceback
import asyncio
from typing import List, Optional, Tuple

from ...compat import _sanitize_log_value
from ...models.memory import MemoryQueryResult

logger = logging.getLogger(__name__)


class HybridMixin:
    """Mixin providing hybrid BM25 + vector search."""

    def _normalize_bm25_score(self, bm25_rank: float) -> float:
        """Convert BM25's negative ranking to 0-1 scale."""
        return max(0.0, min(1.0, 1.0 + bm25_rank / 10.0))

    async def _search_bm25(
        self,
        query: str,
        n_results: int = 5,
        sanitize_query: bool = True,
        include_superseded: bool = False
    ) -> List[Tuple[str, float]]:
        """Perform BM25 keyword search using FTS5."""
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return []

            if sanitize_query:
                query_clean = re.sub(r'[^\w\s-]', '', query)
            else:
                query_clean = query

            if not query_clean.strip():
                logger.warning("Query is empty after sanitization")
                return []

            # Match any of the query's words, not the query as one phrase: a
            # single quoted string is an FTS5 phrase query, which only matches
            # the words adjacent and in order. Each word is quoted on its own
            # so words like AND/OR/NOT/NEAR stay words instead of operators;
            # bm25() still ranks rows matching more of them higher.
            #
            # The FTS5 trigram tokenizer cannot index terms shorter than three
            # characters, so a query made entirely of short words (e.g. "go up")
            # would match nothing as an OR of quoted terms. Fall back to the
            # whole-query phrase in that case, which still searches something.
            terms = [t for t in query_clean.split() if len(t) >= 3]
            if terms:
                fts_query = " OR ".join(
                    '"{}"'.format(term.replace('"', '""')) for term in terms
                )
            else:
                fts_query = '"{}"'.format(query_clean.replace('"', '""'))

            superseded_filter = (
                "" if include_superseded
                else " AND (m.superseded_by IS NULL OR m.superseded_by = '')"
            )

            def search_fts():
                cursor = self.conn.execute(f'''
                    SELECT m.content_hash, bm25(memory_content_fts) as rank
                    FROM memory_content_fts f
                    JOIN memories m ON f.rowid = m.id
                    WHERE memory_content_fts MATCH ? AND m.deleted_at IS NULL{superseded_filter}
                    ORDER BY rank
                    LIMIT ?
                ''', (fts_query, n_results))
                return cursor.fetchall()

            results = await self._execute_with_retry(search_fts)

            logger.debug("BM25 search found %d results for query: %s",
                        len(results), _sanitize_log_value(query_clean))
            return results

        except Exception as e:
            logger.error("BM25 search failed: %s", _sanitize_log_value(e))
            logger.error(traceback.format_exc())
            return []

    def _fuse_scores(
        self,
        keyword_score: float,
        semantic_score: float,
        keyword_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None
    ) -> float:
        """Combine keyword and semantic scores using weighted average."""
        from ...config import MCP_HYBRID_KEYWORD_WEIGHT, MCP_HYBRID_SEMANTIC_WEIGHT

        kw_weight = keyword_weight if keyword_weight is not None else MCP_HYBRID_KEYWORD_WEIGHT
        sem_weight = semantic_weight if semantic_weight is not None else MCP_HYBRID_SEMANTIC_WEIGHT

        return (keyword_score * kw_weight) + (semantic_score * sem_weight)

    async def _fuse_rrf(
        self,
        bm25_results,
        vector_results,
        n_results: int
    ):
        """Fuse results using Reciprocal Rank Fusion (RRF)."""
        from ...config import MCP_HYBRID_RRF_K, MCP_HYBRID_RRF_CONSENSUS_BOOST

        k = MCP_HYBRID_RRF_K
        boost = MCP_HYBRID_RRF_CONSENSUS_BOOST

        bm25_hashes = [ch for ch, _ in bm25_results]
        vector_hashes = [r.memory.content_hash for r in vector_results]
        bm25_set = set(bm25_hashes)
        vector_set = set(vector_hashes)
        consensus = bm25_set & vector_set

        scores = {}
        for rank, ch in enumerate(vector_hashes, start=1):
            scores[ch] = scores.get(ch, 0.0) + 1.0 / (k + rank)
        for rank, ch in enumerate(bm25_hashes, start=1):
            scores[ch] = scores.get(ch, 0.0) + 1.0 / (k + rank)
        for ch in consensus:
            scores[ch] += boost

        vector_memories = {r.memory.content_hash: r.memory for r in vector_results}

        bm25_only = [h for h in scores if h not in vector_memories]
        fetched = {}
        if bm25_only:
            try:
                for i in range(0, len(bm25_only), 999):
                    batch = bm25_only[i:i+999]
                    ph = ",".join("?" for _ in batch)
                    def fetch_batch(ph=ph, b=batch):
                        cur = self.conn.execute(
                            f"SELECT content_hash, content, tags, memory_type, metadata, "
                            f"created_at, updated_at, created_at_iso, updated_at_iso "
                            f"FROM memories WHERE content_hash IN ({ph}) AND deleted_at IS NULL", b)
                        return cur.fetchall()
                    rows = await self._execute_with_retry(fetch_batch)
                    for row in rows:
                        m = self._row_to_memory(row)
                        if m:
                            fetched[m.content_hash] = m
            except Exception as e:
                logger.warning("RRF batch fetch failed: %s", _sanitize_log_value(e))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for ch, rrf_score in ranked[:n_results]:
            memory = vector_memories.get(ch) or fetched.get(ch)
            if memory:
                results.append(MemoryQueryResult(
                    memory=memory,
                    relevance_score=rrf_score,
                    debug_info={
                        "rrf_score": rrf_score,
                        "in_semantic": ch in vector_set,
                        "in_keyword": ch in bm25_set,
                        "consensus": ch in consensus,
                        "backend": "hybrid-rrf",
                    },
                ))

        logger.info("RRF hybrid: %d results (BM25: %d, Vec: %d, Consensus: %d)",
                    len(results), len(bm25_results), len(vector_results), len(consensus))
        return results

    async def retrieve_hybrid(
        self,
        query: str,
        n_results: int = 5,
        keyword_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        include_superseded: bool = False
    ) -> List[MemoryQueryResult]:
        """Hybrid search combining BM25 keyword matching and vector similarity."""
        try:
            bm25_task = asyncio.create_task(self._search_bm25(query, n_results * 2, include_superseded=include_superseded))
            vector_task = asyncio.create_task(self.retrieve(query, n_results * 2, include_superseded=include_superseded))

            bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)

            from ...config import MCP_HYBRID_FUSION_METHOD
            if MCP_HYBRID_FUSION_METHOD == 'rrf':
                return await self._fuse_rrf(bm25_results, vector_results, n_results)

            bm25_scores = {}
            for content_hash, bm25_rank in bm25_results:
                bm25_scores[content_hash] = self._normalize_bm25_score(bm25_rank)

            semantic_scores = {}
            vector_memories = {}
            for result in vector_results:
                semantic_scores[result.memory.content_hash] = result.relevance_score
                vector_memories[result.memory.content_hash] = result.memory

            all_hashes = set(bm25_scores.keys()) | set(semantic_scores.keys())

            bm25_only_hashes = [h for h in all_hashes if h not in vector_memories]
            fetched_memories = {}
            if bm25_only_hashes:
                try:
                    for batch_start in range(0, len(bm25_only_hashes), 999):
                        batch = bm25_only_hashes[batch_start : batch_start + 999]
                        placeholders = ",".join("?" for _ in batch)

                        superseded_filter = "" if include_superseded else " AND (m.superseded_by IS NULL OR m.superseded_by = '')"

                        def fetch_batch(ph=placeholders, b=batch, sf=superseded_filter):
                            cursor = self.conn.execute(
                                f"SELECT content_hash, content, tags, memory_type, metadata, "
                                f"created_at, updated_at, created_at_iso, updated_at_iso "
                                f"FROM memories m WHERE content_hash IN ({ph}) AND deleted_at IS NULL{sf}",
                                b,
                            )
                            return cursor.fetchall()

                        rows = await self._execute_with_retry(fetch_batch)
                        for row in rows:
                            memory = self._row_to_memory(row)
                            if memory:
                                fetched_memories[memory.content_hash] = memory
                except Exception as e:
                    logger.warning(
                        "Batch fetch for BM25-only hashes failed, some results may be missing: %s",
                        _sanitize_log_value(e),
                    )

            merged_results = []
            for content_hash in all_hashes:
                keyword_score = bm25_scores.get(content_hash, 0.0)
                semantic_score = semantic_scores.get(content_hash, 0.0)

                final_score = self._fuse_scores(
                    keyword_score,
                    semantic_score,
                    keyword_weight,
                    semantic_weight
                )

                memory = vector_memories.get(content_hash) or fetched_memories.get(
                    content_hash
                )

                if memory:
                    merged_results.append(MemoryQueryResult(
                        memory=memory,
                        relevance_score=final_score,
                        debug_info={
                            "keyword_score": keyword_score,
                            "semantic_score": semantic_score,
                            "backend": "hybrid-bm25-vector"
                        }
                    ))

            merged_results.sort(key=lambda r: r.relevance_score, reverse=True)
            results = merged_results[:n_results]

            logger.info("Hybrid search found %d results (BM25: %d, Vector: %d)",
                        len(results), len(bm25_results), len(vector_results))

            return results

        except Exception as e:
            logger.error("Hybrid search failed: %s", _sanitize_log_value(e))
            logger.error(traceback.format_exc())
            return []
