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
        """Convert an FTS5 bm25() rank to a 0-1 keyword score.

        FTS5 returns bm25() negated: the better the match, the more negative
        the rank, and 0 means no keyword evidence. x / (1 + x) on x = -rank
        keeps that order, stays below 1 without a fixed cut-off for the
        corpus-dependent magnitude, and gives a non-match 0, the same score
        the fusion uses for rows BM25 did not return.
        """
        strength = max(0.0, -bm25_rank)
        return strength / (1.0 + strength)

    async def _search_bm25(
        self,
        query: str,
        n_results: int = 5,
        sanitize_query: bool = True,
        include_superseded: bool = False,
        as_of: Optional[float] = None,
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

            # Issue #1724 follow-up: when as_of is set, do NOT pre-exclude
            # superseded rows here — a row superseded after as_of was still
            # current at as_of. The historical bound is applied post-fusion
            # instead (no-op when as_of is None).
            superseded_filter = (
                ""
                if (include_superseded or as_of is not None)
                else " AND (m.superseded_by IS NULL OR m.superseded_by = '')"
            )

            def search_fts():
                cursor = self.conn.execute(f'''
                    SELECT m.content_hash, bm25(memory_content_fts) as rank
                    FROM memory_content_fts f
                    JOIN memories m ON f.rowid = m.id
                    WHERE memory_content_fts MATCH ? AND m.deleted_at IS NULL AND (m.embedding_pending IS NULL OR m.embedding_pending = 0){superseded_filter}
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
        n_results: int,
        as_of: Optional[float] = None,
    ):
        """Fuse results using Reciprocal Rank Fusion (RRF).

        Issue #1724: supersession lineage is carried through the fusion so the
        point-in-time filter can be applied before truncating to n_results.
        """
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
        # Issue #1724: supersession lineage for BM25-only rows, mirroring the
        # weighted branch fetch (successor lookups are live-row only).
        rrf_supersession = {}
        if bm25_only:
            try:
                for i in range(0, len(bm25_only), 999):
                    batch = bm25_only[i:i+999]
                    ph = ",".join("?" for _ in batch)
                    def fetch_batch(ph=ph, b=batch):
                        cur = self.conn.execute(
                            f"SELECT content_hash, content, tags, memory_type, metadata, "
                            f"created_at, updated_at, created_at_iso, updated_at_iso, "
                            f"m.superseded_by, "
                            f"COALESCE(m.superseded_at, (SELECT s.created_at FROM memories s WHERE s.content_hash = m.superseded_by AND s.deleted_at IS NULL)) "
                            f"FROM memories m WHERE content_hash IN ({ph}) AND deleted_at IS NULL AND (embedding_pending IS NULL OR embedding_pending = 0)", b)
                        return cur.fetchall()
                    rows = await self._execute_with_retry(fetch_batch)
                    for row in rows:
                        m = self._row_to_memory(row)
                        if m:
                            fetched[m.content_hash] = m
                            rrf_supersession[m.content_hash] = (
                                row[9] or None, row[10],
                            )
            except Exception as e:
                logger.warning("RRF batch fetch failed: %s", _sanitize_log_value(e))

        vector_results_by_hash = {r.memory.content_hash: r for r in vector_results}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for ch, rrf_score in ranked:
            memory = vector_memories.get(ch) or fetched.get(ch)
            if memory:
                # Issue #1724: carry supersession lineage so the as_of
                # post-filter below works. Vector hits already carry it from
                # retrieve().
                vresult = vector_results_by_hash.get(ch)
                if vresult is not None:
                    sup_by, sup_at = vresult.superseded_by, vresult.superseded_at
                else:
                    sup_by, sup_at = rrf_supersession.get(ch, (None, None))
                results.append(MemoryQueryResult(
                    memory=memory,
                    relevance_score=rrf_score,
                    superseded_by=sup_by,
                    superseded_at=sup_at,
                    debug_info={
                        "rrf_score": rrf_score,
                        "in_semantic": ch in vector_set,
                        "in_keyword": ch in bm25_set,
                        "consensus": ch in consensus,
                        "backend": "hybrid-rrf",
                    },
                ))

        # Issue #1724: apply the historical bound BEFORE truncating, so
        # dropped rows don't consume result slots (no-op when as_of is None).
        if as_of is not None:
            results = self._post_filter_as_of(results, as_of)
        results = results[:n_results]

        logger.info("RRF hybrid: %d results (BM25: %d, Vec: %d, Consensus: %d)",
                    len(results), len(bm25_results), len(vector_results), len(consensus))
        return results

    def _only_in_store(self, results: List[MemoryQueryResult], store: Optional[str]) -> List[MemoryQueryResult]:
        """Store isolation (#1466): drop results whose row is not in ``store``.

        BM25 is not store-aware, so both fusion paths run their results through
        this before truncating. ``store=None`` means every store.
        """
        if store is None or not results:
            return results
        hashes = [r.memory.content_hash for r in results]
        in_store = set()
        try:
            for batch_start in range(0, len(hashes), 999):
                batch = hashes[batch_start:batch_start + 999]
                ph = ",".join("?" for _ in batch)
                cur = self.conn.execute(
                    f"SELECT content_hash FROM memories WHERE content_hash IN ({ph}) AND store = ? AND deleted_at IS NULL",
                    (*batch, store),
                )
                in_store.update(row[0] for row in cur.fetchall())
        except Exception as e:
            # Fail CLOSED: a store-scoped search must never fall back to unfiltered
            # results (BM25 spans every store), so on error return nothing rather
            # than leaking other partitions.
            logger.warning("Hybrid store filter failed, returning no results (fail-closed): %s", _sanitize_log_value(e))
            return []
        return [r for r in results if r.memory.content_hash in in_store]

    async def retrieve_hybrid(
        self,
        query: str,
        n_results: int = 5,
        keyword_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        include_superseded: bool = False,
        store: Optional[str] = "default",
        as_of: Optional[float] = None,
    ) -> List[MemoryQueryResult]:
        """Hybrid search combining BM25 keyword matching and vector similarity.

        When ``store`` is given, results are scoped to that partition: the vector
        half already filters by store, and a final store filter is applied over the
        merged set so the (store-unaware) BM25 half cannot leak other partitions.
        """
        try:
            bm25_task = asyncio.create_task(self._search_bm25(query, n_results * 2, include_superseded=include_superseded, as_of=as_of))
            vector_task = asyncio.create_task(self.retrieve(query, n_results * 2, include_superseded=include_superseded, store=store, as_of=as_of))

            bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)

            from ...config import MCP_HYBRID_FUSION_METHOD
            if MCP_HYBRID_FUSION_METHOD == 'rrf':
                # Fuse everything, then scope to the store before truncating, as the
                # weighted path below does: BM25 hits can come from any store.
                # Issue #1724: as_of is applied inside _fuse_rrf (pre-truncation).
                fused = await self._fuse_rrf(bm25_results, vector_results, len(bm25_results) + len(vector_results), as_of=as_of)
                return self._only_in_store(fused, store)[:n_results]

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
            # Issue #1724: (superseded_by, eff_superseded_at) per BM25-only hash.
            bm25_supersession = {}
            if bm25_only_hashes:
                try:
                    for batch_start in range(0, len(bm25_only_hashes), 999):
                        batch = bm25_only_hashes[batch_start : batch_start + 999]
                        placeholders = ",".join("?" for _ in batch)

                        # Issue #1724: keep supersession lineage for the as_of
                        # post-filter below. When as_of is set, superseded rows
                        # stay in the candidate pool (they may have been current
                        # at as_of); the historical bound is applied post-fusion.
                        # Successor lookups are live-row only (deleted_at).
                        superseded_filter = (
                            "" if (include_superseded or as_of is not None)
                            else " AND (m.superseded_by IS NULL OR m.superseded_by = '')"
                        )

                        def fetch_batch(ph=placeholders, b=batch, sf=superseded_filter):
                            cursor = self.conn.execute(
                                f"SELECT content_hash, content, tags, memory_type, metadata, "
                                f"created_at, updated_at, created_at_iso, updated_at_iso, "
                                f"m.superseded_by, "
                                f"COALESCE(m.superseded_at, (SELECT s.created_at FROM memories s WHERE s.content_hash = m.superseded_by AND s.deleted_at IS NULL)) "
                                f"FROM memories m WHERE content_hash IN ({ph}) AND deleted_at IS NULL AND (m.embedding_pending IS NULL OR m.embedding_pending = 0){sf}",
                                b,
                            )
                            return cursor.fetchall()

                        rows = await self._execute_with_retry(fetch_batch)
                        for row in rows:
                            memory = self._row_to_memory(row)
                            if memory:
                                fetched_memories[memory.content_hash] = memory
                                # Issue #1724: keep supersession lineage for the
                                # as_of post-filter below (_row_to_memory ignores
                                # the trailing columns).
                                bm25_supersession[memory.content_hash] = (
                                    row[9] or None, row[10],
                                )
                except Exception as e:
                    logger.warning(
                        "Batch fetch for BM25-only hashes failed, some results may be missing: %s",
                        _sanitize_log_value(e),
                    )

            merged_results = []
            vector_results_by_hash = {r.memory.content_hash: r for r in vector_results}
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
                    # Issue #1724: carry supersession lineage so the as_of
                    # post-filter below works for BM25-only hits too. Vector
                    # hits already carry it from retrieve().
                    vresult = vector_results_by_hash.get(content_hash)
                    if vresult is not None:
                        sup_by, sup_at = vresult.superseded_by, vresult.superseded_at
                    else:
                        sup_by, sup_at = bm25_supersession.get(content_hash, (None, None))
                    merged_results.append(MemoryQueryResult(
                        memory=memory,
                        relevance_score=final_score,
                        superseded_by=sup_by,
                        superseded_at=sup_at,
                        debug_info={
                            "keyword_score": keyword_score,
                            "semantic_score": semantic_score,
                            "backend": "hybrid-bm25-vector"
                        }
                    ))

            merged_results.sort(key=lambda r: r.relevance_score, reverse=True)

            # Issue #1724: the BM25 half has no SQL-level as_of support; filter
            # the fused set here (no-op when as_of is None).
            if as_of is not None:
                merged_results = self._post_filter_as_of(merged_results, as_of)

            results = self._only_in_store(merged_results, store)[:n_results]

            logger.info("Hybrid search found %d results (BM25: %d, Vector: %d)",
                        len(results), len(bm25_results), len(vector_results))

            return results

        except Exception as e:
            logger.error("Hybrid search failed: %s", _sanitize_log_value(e))
            logger.error(traceback.format_exc())
            return []
