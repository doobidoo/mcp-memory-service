#!/usr/bin/env python3
"""
Rerank Benchmark Harness for mcp-memory-service (R1 - Frente A)

BENCHMARK SCOPE:
Measures the quality of belief ranking (frente A) in memory_context() using
real beliefs from the production database. Metrics: MRR@5, nDCG@5, Hit@3, 
off-topic@3. Deterministic, offline, no LLM-judge, for baseline vs variants 
comparison.

REQUIREMENTS:
- Database: ~/local-data/mcp/sqlite_vec.db (READ mode)
- Environment: VIRTUAL_ENV=.venv PYTHONPATH=src MCP_MEMORY_USE_ONNX=1 
  MCP_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2 
  MCP_INJECT_RERANK_A=true MCP_BELIEF_UTILITY_PRIOR=true

IMPORTANT: The utility_prior should be the SAME in both sides of an A/B test
(snapshot the DB to ensure deterministic comparison across baseline vs variant).

Author: reg (Gate 4 subagent)
"""

import asyncio
import json
import logging
import math
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Suppress debug logging for cleaner output
logging.getLogger().setLevel(logging.WARNING)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.storage.context_injection import memory_context


# INLINE DATASET: Real beliefs from production DB ~/local-data/mcp/sqlite_vec.db
# Each query has expected gold belief_hashes that should rank highly for that theme
BENCHMARK_QUERIES = [
    {
        # CANÁRIO do rich-get-richer: query de tema NICHO cujo gold existe e tem util
        # moderado; o forbidden é um belief GENÉRICO de util 1.0 de OUTRO tema (RFC/spec).
        # ALVO: o genérico-util-alto NÃO pode afogar o nicho-relevante (off-topic@3=0).
        # (O caso original 'feedback passivo' foi removido: não existe belief desse tema —
        #  é memória atômica, não belief derivado; gold era inválido. Ver mem 12699e27.)
        "query": "geoserver réplica alta disponibilidade qa NFS",
        "gold": [
            "d41bf9f22647516483f0642e6d2f10d970cf3eda2f6a7320451ef8620df04217",  # geoserver=1 réplica qa (nicho real)
        ],
        "category": "canario",
        "off_topic_forbidden": [
            "465448edaee19970a0232a466354438f9a8d58f423b8fd7e09384305b7a46ec7"  # Specs implemented/planned — genérico util 1.0, OUTRO tema
        ]
    },
    {
        "query": "git workflow branch desenvolvimento",
        "gold": [
            "42cfed0569ebbf65023cc7c7bb5c67b0b8731f74674ab6dc837dae3a3a5fe3d6",  # F008 branch workflow
            "47ef5d2657625774b50d6390c5f4aa68b3623c87ef088cf9d11e3014f1c6b0f6",  # GitHub HTTPS tokens
        ],
        "category": "tematica"
    },
    {
        "query": "teste CPF MIR staging",
        "gold": [
            "49d3d240fb72ae9f011b8fc07f9bc9dc0f250536eebe51d2031eddabffdaf537",  # CPF teste MIR staging
        ],
        "category": "exata"
    },
    {
        "query": "kubernetes deploy RER namespace",
        "gold": [
            "01d865667129d961b4af2841fe643b49a282bf47cd8ea284894dbf5a01bd1319",  # RER K8s DEV nodegroup
            "be1911ef404ab9b6d3a2819121729ac2d079eaffbdf63d512d88331c5ad31131",  # RER K8s manifestos DEV
        ],
        "category": "tematica"
    },
    {
        "query": "EIP NAT Gateway HCSO rede",
        "gold": [
            "df79ced5fc209c24946ddbde90af1df4a4ca752b3d3668dc7db5f619a3907d1d",  # EIP preso no NAT 
            "1fe15a35c1df14f3518eefc2d66bd466a2ef134d1f3a0dd9fcb57a30cb130ac4",  # EIP duplicate entry
        ],
        "category": "tematica"
    },
    {
        "query": "RFC spec documentação padrão",
        "gold": [
            "bb842499d6dffaee8496fa56911b4aa81c89fc6ea4eab34236b1552bc9e0bc13",  # RFC fork-only docs
            "465448edaee19970a0232a466354438f9a8d58f423b8fd7e09384305b7a46ec7",  # specs implemented/planned
        ],
        "category": "tematica"
    },
    {
        "query": "posts externos comentários issues tradução",
        "gold": [
            "d5fca59ec485e705a8d494228c1e0d67c86ef9945d338555acf493aa7a0f2e81",  # posts externos EN+PT-BR
            "6bb74d800581e8059eb31f955a80750d9f1120e9a20bb4ecc477b1c5f4b85abe",  # posts Henry issues
        ],
        "category": "tematica"
    },
    {
        "query": "OpenAPI SQL datasource segurança",
        "gold": [
            "6b13591ebfa380b145b46d54660d05be0ddd2505714572574faecbf3aedfde57",  # OpenAPI nunca incluir sql_text
            "01f3445a7e8429ff83afe0c5375e3fe45c0e717cb438ee23ddd9b95fc755b5be",  # OpenAPI R3 duplicate
        ],
        "category": "tematica"
    },
    {
        "query": "memory service configuração ONNX embedding",
        "gold": [
            "a84ea4febc2bd3b129cafb04f0fb580801a5756ae7ab4ee7a9ea93d43553a734",  # Memory metadata bolsa
        ],
        "category": "tematica"
    },
    {
        "query": "geoserver réplicas alta disponibilidade",
        "gold": [
            "d41bf9f22647516483f0642e6d2f10d970cf3eda2f6a7320451ef8620df04217",  # geoserver 1 replica qa
        ],
        "category": "exata"
    },
    {
        "query": "receita feature documentação STATUS",
        "gold": [
            "a93a2326dbfec3afece67258973341cb62438ac69b2aa3d36c48041019683f21",  # receita feature doc acompanhamento
            "178be3c5bd54652b716c96212f10f182510e4ed05a16bf800a688367d0024488",  # TYPE convention feature doc
        ],
        "category": "tematica"
    },
    {
        "query": "Spring Boot classe main dataplane",
        "gold": [
            "7fd977bc790771ea6c87375dacd006e60e80d6d57bffe097fbc28b8e111e16de",  # Spring Boot main dataplane
        ],
        "category": "exata"
    }
]


def calculate_mrr_at_k(gold_hashes: List[str], retrieved_hashes: List[str], k: int = 5) -> float:
    """Calculate Mean Reciprocal Rank@K: 1/rank of first gold item in top-K, 0 if absent."""
    if not gold_hashes or not retrieved_hashes:
        return 0.0
    
    topk = retrieved_hashes[:k]
    for i, h in enumerate(topk):
        if h in gold_hashes:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg_at_k(gold_hashes: List[str], retrieved_hashes: List[str], k: int = 5) -> float:
    """Calculate Normalized Discounted Cumulative Gain@K."""
    if not gold_hashes or not retrieved_hashes:
        return 0.0
    
    # DCG: sum of relevance[i] / log2(i+2) for i in [0,k)
    dcg = 0.0
    topk = retrieved_hashes[:k]
    for i, h in enumerate(topk):
        if h in gold_hashes:
            relevance = 1.0  # binary relevance
            dcg += relevance / math.log2(i + 2)
    
    # IDCG: best possible DCG (all golds at the top)
    idcg = 0.0
    num_relevant = min(len(gold_hashes), k)
    for i in range(num_relevant):
        idcg += 1.0 / math.log2(i + 2)
    
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def calculate_hit_at_k(gold_hashes: List[str], retrieved_hashes: List[str], k: int = 3) -> float:
    """Calculate Hit@K: 1.0 if any gold in top-K, 0.0 otherwise."""
    if not gold_hashes or not retrieved_hashes:
        return 0.0
    
    topk = retrieved_hashes[:k]
    return 1.0 if any(h in gold_hashes for h in topk) else 0.0


def calculate_offtopic_at_k(forbidden_hashes: List[str], retrieved_hashes: List[str], k: int = 3) -> float:
    """Calculate off-topic@K: 1.0 if any forbidden in top-K, 0.0 otherwise. We want 0.0."""
    if not forbidden_hashes or not retrieved_hashes:
        return 0.0
    
    topk = retrieved_hashes[:k]
    return 1.0 if any(h in forbidden_hashes for h in topk) else 0.0


async def run_benchmark() -> Dict[str, Any]:
    """Run the complete rerank benchmark and return results."""
    
    # Initialize storage with production database
    db_path = Path.home() / "local-data" / "mcp" / "sqlite_vec.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    storage = SqliteVecMemoryStorage(str(db_path))
    
    # Initialize the storage properly - this ensures the connection and embedding model are ready
    try:
        # Initialize connection directly using the internal method
        storage._connect_and_load_extension()
        # Initialize embedding model if available
        if hasattr(storage, '_initialize_embedding_model'):
            await storage._initialize_embedding_model()
        else:
            # Fallback: set basic attributes that memory_context might need
            storage.embedding_model = None  # Will trigger embedding fallback in context_injection
    except Exception as e:
        print(f"Storage initialization failed: {e}")
        return {"error": f"Storage initialization failed: {e}"}
    
    results = []
    metrics_by_category = {"tematica": [], "canario": [], "exata": []}
    
    print("Running rerank benchmark...")
    print(f"Database: {db_path}")
    print(f"Queries: {len(BENCHMARK_QUERIES)}")
    print()
    
    for i, query_data in enumerate(BENCHMARK_QUERIES):
        query = query_data["query"]
        gold = query_data["gold"]
        category = query_data["category"]
        forbidden = query_data.get("off_topic_forbidden", [])
        
        print(f"Query {i+1:2d}: {query[:60]}...")
        
        # Call the REAL production memory_context function
        context_result = await memory_context(storage, task=query, limit=10)
        
        # Extract belief_hashes from the result
        belief_hashes = context_result.get("belief_hashes", [])
        
        # Calculate metrics
        mrr5 = calculate_mrr_at_k(gold, belief_hashes, k=5)
        ndcg5 = calculate_ndcg_at_k(gold, belief_hashes, k=5)
        hit3 = calculate_hit_at_k(gold, belief_hashes, k=3)
        offtopic3 = calculate_offtopic_at_k(forbidden, belief_hashes, k=3)
        
        result = {
            "query": query,
            "category": category,
            "gold_count": len(gold),
            "retrieved_count": len(belief_hashes),
            "mrr_at_5": mrr5,
            "ndcg_at_5": ndcg5,
            "hit_at_3": hit3,
            "offtopic_at_3": offtopic3,
            "belief_hashes": belief_hashes,
            "gold_hashes": gold,
            "forbidden_hashes": forbidden,
        }
        
        results.append(result)
        metrics_by_category[category].append(result)
        
        # Print per-query results
        print(f"         MRR@5: {mrr5:.3f}, nDCG@5: {ndcg5:.3f}, Hit@3: {hit3:.1f}, Off-topic@3: {offtopic3:.1f}")
        if belief_hashes:
            print(f"         Top-3: {belief_hashes[:3]}")
        print()
    
    # Calculate aggregate metrics
    all_metrics = {
        "mrr_at_5": sum(r["mrr_at_5"] for r in results) / len(results),
        "ndcg_at_5": sum(r["ndcg_at_5"] for r in results) / len(results),
        "hit_at_3": sum(r["hit_at_3"] for r in results) / len(results),
        "offtopic_at_3": sum(r["offtopic_at_3"] for r in results) / len(results),
    }
    
    # Calculate by category
    category_metrics = {}
    for cat, cat_results in metrics_by_category.items():
        if cat_results:
            category_metrics[cat] = {
                "count": len(cat_results),
                "mrr_at_5": sum(r["mrr_at_5"] for r in cat_results) / len(cat_results),
                "ndcg_at_5": sum(r["ndcg_at_5"] for r in cat_results) / len(cat_results),
                "hit_at_3": sum(r["hit_at_3"] for r in cat_results) / len(cat_results),
                "offtopic_at_3": sum(r["offtopic_at_3"] for r in cat_results) / len(cat_results),
            }
    
    return {
        "queries": results,
        "aggregate": all_metrics,
        "by_category": category_metrics,
        "metadata": {
            "total_queries": len(BENCHMARK_QUERIES),
            "database_path": str(db_path),
            "rerank_a_enabled": True,  # MCP_INJECT_RERANK_A=true in env
            "utility_prior_enabled": True,  # MCP_BELIEF_UTILITY_PRIOR=true in env
        }
    }


def print_results_table(results: Dict[str, Any]):
    """Print results in a human-readable table format."""
    
    print("="*80)
    print("RERANK BENCHMARK RESULTS")
    print("="*80)
    
    # Per-query table
    print("\nPER-QUERY RESULTS:")
    print("-" * 90)
    print(f"{'#':>2} {'Category':<8} {'MRR@5':>7} {'nDCG@5':>8} {'Hit@3':>7} {'Off@3':>7} {'Query':<40}")
    print("-" * 90)
    
    for i, query in enumerate(results["queries"]):
        print(f"{i+1:>2} {query['category']:<8} "
              f"{query['mrr_at_5']:>7.3f} {query['ndcg_at_5']:>8.3f} "
              f"{query['hit_at_3']:>7.1f} {query['offtopic_at_3']:>7.1f} "
              f"{query['query'][:40]:<40}")
    
    # Aggregate results
    print("-" * 90)
    agg = results["aggregate"]
    print(f"{'AVG':<11} "
          f"{agg['mrr_at_5']:>7.3f} {agg['ndcg_at_5']:>8.3f} "
          f"{agg['hit_at_3']:>7.1f} {agg['offtopic_at_3']:>7.1f}")
    
    # By category
    print("\nBY CATEGORY:")
    print("-" * 60)
    print(f"{'Category':<10} {'Count':>5} {'MRR@5':>7} {'nDCG@5':>8} {'Hit@3':>7} {'Off@3':>7}")
    print("-" * 60)
    
    for cat, metrics in results["by_category"].items():
        print(f"{cat:<10} {metrics['count']:>5} "
              f"{metrics['mrr_at_5']:>7.3f} {metrics['ndcg_at_5']:>8.3f} "
              f"{metrics['hit_at_3']:>7.1f} {metrics['offtopic_at_3']:>7.1f}")
    
    print("\nMETADATA:")
    meta = results["metadata"]
    print(f"- Total queries: {meta['total_queries']}")
    print(f"- Database: {meta['database_path']}")
    print(f"- Rerank A enabled: {meta['rerank_a_enabled']}")
    print(f"- Utility prior enabled: {meta['utility_prior_enabled']}")


def main():
    """Main entry point."""
    try:
        results = asyncio.run(run_benchmark())
        
        # Print human-readable results
        print_results_table(results)
        
        # Print JSON block for automated comparison
        print("\n" + "="*80)
        print("JSON BASELINE (for A/B comparison):")
        print("="*80)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())