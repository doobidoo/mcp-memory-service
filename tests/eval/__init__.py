"""
Hybrid Search Evaluation Harness

Provides retrieval quality metrics (Hit Rate@10, MRR, NDCG@10) for
tuning hybrid search parameters. Uses ranx library for standard IR
evaluation.

Run with:
    pytest tests/eval/ -v              # Run all eval tests
    python tests/eval/sweep_alpha.py   # Alpha grid search
"""
