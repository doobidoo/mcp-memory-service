"""E2E provenance tests — exercise REAL LLM providers (no mocks).

Opt-in: skipped unless MCP_E2E_LLM=1 and the provider env vars are set.
Run with the service env sourced:

    set -a; source ~/dtp/ai-configs/services/env/memory-service.env; set +a
    MCP_E2E_LLM=1 PYTHONPATH=src pytest tests/test_harvest_provenance_e2e.py -v

Validates that the rewriter really hits each provider and stamps the correct
provider/model on the RewriteResult (RFC-harvest-provenance R2/R3, end-to-end).
"""
import os
import pytest

from mcp_memory_service.harvest.rewriter import HarvestRewriter

E2E = os.getenv("MCP_E2E_LLM") == "1"
pytestmark = pytest.mark.skipif(not E2E, reason="set MCP_E2E_LLM=1 to run real-LLM E2E")

SAMPLE = "Decided to use asyncpg pool (min=2, max=10) for the database layer because it survives connection drops."


def _rewrite_with(provider: str):
    """Force a single provider and run a real rewrite. Returns RewriteResult|None."""
    old = os.environ.get("HARVEST_LLM_PROVIDERS")
    os.environ["HARVEST_LLM_PROVIDERS"] = provider
    try:
        return HarvestRewriter().rewrite_sync(SAMPLE, "decision")
    finally:
        if old is None:
            os.environ.pop("HARVEST_LLM_PROVIDERS", None)
        else:
            os.environ["HARVEST_LLM_PROVIDERS"] = old


@pytest.mark.parametrize("provider", ["deepseek", "groq"])
def test_provider_rewrites_and_stamps_provenance(provider):
    """Each configured provider really rewrites and stamps provider/model."""
    if not os.getenv(f"HARVEST_LLM_{provider.upper()}_API_KEY"):
        pytest.skip(f"{provider} not configured")
    res = _rewrite_with(provider)
    assert res is not None, f"{provider} returned None (SKIP/empty) — check key/model"
    assert res.provider == provider, f"provenance provider mismatch: {res.provider}"
    assert res.model, f"{provider}: model not stamped"
    assert len(res.content) > 10, "rewrite produced no meaningful content"


def test_fallback_chain_reaches_a_working_provider():
    """With the full chain, at least one provider must produce a stamped result."""
    if not os.getenv("HARVEST_LLM_PROVIDERS"):
        pytest.skip("HARVEST_LLM_PROVIDERS not set")
    res = HarvestRewriter().rewrite_sync(SAMPLE, "decision")
    assert res is not None, "entire provider chain failed — no rewrite produced"
    assert res.provider and res.model, "chain produced result without provenance stamp"
