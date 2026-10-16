"""RED tests for ARCO SÍNTESE — Slice 4: LLM SYNTHESIS of the cluster (the MOTOR).

Architecture decision (spec arco-sintese-slice4-G3-rok.md):
  - When MCP_BELIEF_SYNTHESIS is ON *and* a usable LLM provider is configured
    (HARVEST_LLM_PROVIDERS=lia, deepseek-v4-flash), ``derive_beliefs`` sends the CLUSTER
    of observations to the LLM — REUSING the existing client in harvest/rewriter.py
    (``load_llm_providers`` + ``HarvestRewriter._call_openai_compatible``) — asking for a
    generalized, actionable CLAIM. The belief content becomes that SYNTHESIZED claim, not
    a verbatim copy of any source observation. This is the enhancement exp-002 proved moves
    Recall@pool 0.417 → 0.917.
  - Fallback is gracious: provider unavailable OR the call raises → fall back to the
    extractive/verbatim path (Slice 2), NEVER crash ``derive_beliefs``.
  - Cost instrumentation: each synthesis emits a usage_event(event_type="belief_synthesis")
    whose metadata JSON carries prompt_tokens, completion_tokens and model. Today
    ``_call_openai_compatible`` DISCARDS resp.json()["usage"] (returns only content, L511),
    so this hook does not exist yet.
  - Flag OFF (default) = byte-identical to Slice 2 / current: NO LLM call, content is the
    extractive-cleaned or verbatim representative.
  - Anti-hallucination guard: a synthesized claim that DROPS the cluster's shared thematic
    terms is DISCARDED (the LLM wandered off); fall back to extractive.

────────────────────────────────────────────────────────────────────────────────────────
MOCK SEAM — read before judging the tests.

No network. The LLM provider is mocked at the lowest stable seam the implementation is
contracted to reuse: ``HarvestRewriter._call_openai_compatible`` (the actual
OpenAI-compatible HTTP call, rewriter.py:485). The LIA provider is made *usable* by setting
the documented env (HARVEST_LLM_PROVIDERS=lia, HARVEST_LLM_LIA_BASE_URL/MODEL/API_KEY), so
``load_llm_providers()`` returns exactly one usable provider. The tests therefore exercise
the REAL provider-resolution + REAL (mocked) call path, not a hand-rolled stub.

CLUSTER SHAPE — read before judging RED.

Like the Slice 2 tests, the cluster is built from N identical-content observations
(perturbed only in their content_hash via a salt) so they group by EXACT-content match in
``_group_observations`` — deterministic, independent of whether a real embedding model is
present in the test environment. The single source string is known, so "synthesis != source
verbatim" is a crisp assertion.
────────────────────────────────────────────────────────────────────────────────────────

RED / GUARD status on current main (branch feat/arco-sintese-slice0-identity):
  T1 (RED)   — flag ON + mocked LLM returns a synthesized claim → belief content == that
               claim and != the source verbatim. FAILS today: no synthesis step exists, so
               the belief content is the source copied verbatim.
  T2 (RED)   — flag ON → a usage_event event_type='belief_synthesis' exists with
               prompt_tokens/completion_tokens/model in metadata. FAILS today: no such
               event is ever emitted (and _call_openai_compatible discards usage).
  T5 (RED)   — flag ON + mocked LLM returns a claim that DROPS the cluster's shared terms →
               guard discards it, belief falls back to the extractive claim (which keeps
               the terms) and is NOT the hallucinated claim. FAILS today: with the flag ON
               the belief content is just the verbatim source, so the "content == the
               hallucinated claim" check below is what pins the missing synthesis+guard.
  T3 (GUARD) — flag ON but provider UNUSABLE / call raises → gracious fallback: derive does
               not crash, exactly one belief is created from the extractive/verbatim path,
               and NO belief_synthesis event is emitted. Passes today (no synthesis at all);
               must STAY green after the fix (fallback must behave exactly like today).
  T4 (GUARD) — flag OFF (default) → NO LLM call at all (mock asserts it is never invoked),
               content is the current extractive/verbatim representative. Passes today;
               must STAY green after the fix (OFF = byte-identical contract).

HONESTY NOTE (per the SDET rules): T3 and T4 are declared GUARDS, not RED — their core
behavior (no crash / no LLM when OFF / verbatim fallback) is ALREADY the current behavior,
so claiming them as RED would be dishonest. The genuine RED that pins the missing MOTOR is
T1, T2 and T5. T5 is written to assert the GOAL (belief becomes the synthesized/guarded
claim), which fails today precisely because synthesis does not run.
"""

import json
import os
import shutil
import tempfile

import pytest
import pytest_asyncio

# Skip the whole module if sqlite-vec is unavailable (same guard as slice0/1/2 tests).
try:
    import sqlite_vec  # noqa: F401
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.models.memory import Memory
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.consolidation.belief_service import BeliefService
    from mcp_memory_service.utils.hashing import generate_content_hash
    from mcp_memory_service.harvest.rewriter import HarvestRewriter

pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")


# ── Cluster content: an actionable observation with clear thematic terms ──

SOURCE = (
    "Configurei cache Redis nas respostas da API para reduzir a latencia de leitura "
    "no endpoint de consulta de imoveis."
)
# Terms that are shared across the (identical) cluster and must be RETAINED by any valid
# synthesized claim (anti-hallucination guard, T5) — and that trivially survive extraction.
THEME_TERMS = ["cache", "Redis", "respostas", "API", "latencia"]

# A VALID synthesized claim: generalized/reworded (NOT the source verbatim) but still
# carrying the cluster's thematic terms. This is what a good LLM would return.
SYNTH_CLAIM = (
    "Aplicar cache Redis nas respostas da API reduz a latencia de leitura — "
    "use como padrao em endpoints de consulta."
)

# A HALLUCINATED claim: fluent but DROPS every shared thematic term of the cluster. The
# anti-hallucination guard must discard this and fall back to extraction.
HALLUCINATED_CLAIM = "Prefira sempre escrever testes antes de implementar qualquer feature."


# ── Fixtures (real sqlite-vec storage, identical pattern to slice0/1/2) ──

@pytest_asyncio.fixture
async def storage():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_slice4.db")
    st = SqliteVecMemoryStorage(db_path)
    await st.initialize()
    yield st
    if st.conn:
        st.conn.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest_asyncio.fixture
async def svc(storage):
    return BeliefService(storage)


# ── Helpers ──

async def _store_cluster(storage, content: str = SOURCE, n: int = 3) -> None:
    """Store N identical-content observations (distinct content_hash via salt) so they
    group by exact-content match into one cluster."""
    for i in range(n):
        chash = generate_content_hash(content + "#" + str(i))
        mem = Memory(
            content=content,
            content_hash=chash,
            tags=["observation"],
            memory_type="observation",
        )
        await storage.store(mem)


def _non_superseded_beliefs(storage):
    cur = storage.conn.execute(
        "SELECT belief_hash, content, metadata, status FROM beliefs"
    )
    return [
        {"belief_hash": r[0], "content": r[1],
         "metadata": json.loads(r[2] or "{}"), "status": r[3]}
        for r in cur.fetchall()
        if r[3] != "superseded"
    ]


def _synthesis_events(storage):
    cur = storage.conn.execute(
        "SELECT metadata FROM usage_events WHERE event_type = 'belief_synthesis'"
    )
    out = []
    for (meta,) in cur.fetchall():
        try:
            out.append(json.loads(meta) if meta else {})
        except (ValueError, TypeError):
            out.append({})
    return out


def _lia_env(mp):
    """Configure the documented LIA provider so load_llm_providers() returns it usable."""
    mp.setenv("HARVEST_LLM_PROVIDERS", "lia")
    mp.setenv("HARVEST_LLM_LIA_BASE_URL", "https://lia.oci.dataprev.gov.br/api")
    mp.setenv("HARVEST_LLM_LIA_MODEL", "deepseek-v4-flash")
    mp.setenv("HARVEST_LLM_LIA_API_KEY", "test-key")


def _mock_llm_returning(claim: str, usage: dict | None = None):
    """Build an async replacement for HarvestRewriter._call_openai_compatible_with_usage
    that returns ``(claim, usage)`` and records that it was called. The real method's
    signature is (self, base_url, model, api_key, prompt, timeout, max_tokens=200)
    -> tuple[str, dict]. When ``usage`` is None an empty dict is returned (provider omitted
    the usage block → implementation falls back to word-count)."""
    state = {"calls": 0, "last_prompt": None}
    usage_block = {} if usage is None else usage

    async def _mock(self, base_url, model, api_key, prompt, timeout, max_tokens=200):
        state["calls"] += 1
        state["last_prompt"] = prompt
        return claim, usage_block

    return _mock, state


# ── T1 (RED): synthesized belief content differs from the source verbatim ──

@pytest.mark.asyncio
async def test_synthesis_on_belief_is_claim_not_source_verbatim(storage, svc):
    """T1: flag ON + usable LLM (mocked) → the derived belief content is the SYNTHESIZED
    claim, not a verbatim copy of the source observation.

    RED today: Slice 4 does not exist, so derive_beliefs never calls the LLM; the belief
    content is the source copied verbatim. 'content == SYNTH_CLAIM' fails (it equals
    SOURCE), and 'content != SOURCE' fails.
    """
    await _store_cluster(storage)
    mock, state = _mock_llm_returning(SYNTH_CLAIM)

    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_SYNTHESIS", "true")
        _lia_env(mp)
        mp.setattr(HarvestRewriter, "_call_openai_compatible_with_usage", mock, raising=True)
        await svc.derive_beliefs()

    beliefs = _non_superseded_beliefs(storage)
    assert len(beliefs) == 1, (
        f"Expected exactly one belief from the single cluster, found {len(beliefs)}: "
        f"{[b['content'] for b in beliefs]}"
    )
    content = beliefs[0]["content"]

    assert state["calls"] >= 1, (
        "Synthesis must invoke the LLM provider (mocked _call_openai_compatible) at least "
        f"once when the flag is ON and a usable provider is configured; calls={state['calls']}"
    )
    assert content == SYNTH_CLAIM, (
        f"Belief content must be the SYNTHESIZED claim; expected {SYNTH_CLAIM!r}, "
        f"got {content!r}"
    )
    assert content != SOURCE, (
        f"Belief content must NOT be the source observation verbatim; got {content!r}"
    )


# ── T2 (RED): cost instrumentation — a belief_synthesis usage_event with token metadata ──

@pytest.mark.asyncio
async def test_synthesis_emits_usage_event_with_token_metadata(storage, svc):
    """T2: after a synthesis, exactly one usage_event(event_type='belief_synthesis') must
    exist, carrying the REAL provider token counts (prompt_tokens / completion_tokens /
    total_tokens) and model in its metadata JSON. The counts must be the values the
    provider's ``usage`` block reported — NOT a word-count estimate.

    RED today: no belief_synthesis event is ever emitted (synthesis does not exist, and
    _call_openai_compatible discards resp.json()['usage']). _synthesis_events(storage) is
    empty, so the first assertion fails. Hardened (tuvok): the mock returns a KNOWN usage
    block and the test asserts the stored metadata equals those exact values, so an
    implementation that falls back to word-count estimation (prompt has many words, not 123)
    would FAIL — proving the real-cost wiring, not merely the presence of the keys.
    """
    await _store_cluster(storage)
    # Provider ``usage`` block with values that do NOT match any word-count of the prompt or
    # claim, so a word-count estimate would produce different numbers and fail the asserts.
    real_usage = {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168}
    mock, _ = _mock_llm_returning(SYNTH_CLAIM, usage=real_usage)

    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_SYNTHESIS", "true")
        _lia_env(mp)
        mp.setattr(HarvestRewriter, "_call_openai_compatible_with_usage", mock, raising=True)
        await svc.derive_beliefs()

    events = _synthesis_events(storage)
    assert len(events) >= 1, (
        "A usage_event with event_type='belief_synthesis' must be emitted after synthesis; "
        f"found none. all_event_types="
        f"{[r[0] for r in storage.conn.execute('SELECT DISTINCT event_type FROM usage_events').fetchall()]}"
    )
    meta = events[0]
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "model"):
        assert key in meta, (
            f"belief_synthesis metadata must carry {key!r} (cost instrumentation); "
            f"got metadata keys {sorted(meta.keys())}"
        )
    # REAL provider token counts — not a word-count estimate (tuvok's hardened assertion).
    assert meta["prompt_tokens"] == 123, (
        f"belief_synthesis must record the REAL provider prompt_tokens (123 from the usage "
        f"block), not a word-count estimate; got {meta.get('prompt_tokens')!r}"
    )
    assert meta["completion_tokens"] == 45, (
        f"belief_synthesis must record the REAL provider completion_tokens (45 from the "
        f"usage block), not a word-count estimate; got {meta.get('completion_tokens')!r}"
    )
    assert meta["total_tokens"] == 168, (
        f"belief_synthesis must record the REAL provider total_tokens (168 from the usage "
        f"block); got {meta.get('total_tokens')!r}"
    )
    assert meta["model"] == "deepseek-v4-flash", (
        f"belief_synthesis must record the provider model; got {meta.get('model')!r}"
    )


# ── T5 (RED): anti-hallucination guard — claim that drops shared terms is discarded ──

@pytest.mark.asyncio
async def test_synthesis_discards_hallucinated_claim_falls_back_to_extractive(storage, svc):
    """T5: flag ON + mocked LLM returns a claim that DROPS every shared thematic term of
    the cluster → the anti-hallucination guard must DISCARD it and fall back to the
    extractive/verbatim claim (which retains the terms). The belief must NOT be the
    hallucinated claim.

    RED today: with the flag ON there is no synthesis and no guard; the belief content is
    the verbatim source. The assertion that the content retains the thematic terms AND is
    not the hallucinated claim pins the GOAL — but the positive 'the synthesis path ran and
    the guard rejected it' can only hold once Slice 4 exists. Today the belief is the
    verbatim source (keeps terms, != hallucination), so this test is written to fail on the
    guard's observable contract: when a hallucinated claim is returned, the belief content
    must equal the EXTRACTIVE claim produced by the guard's fallback, which today is never
    exercised. We assert that the LLM was consulted (guard path) AND the content kept the
    terms; 'calls >= 1' fails today.
    """
    await _store_cluster(storage)
    mock, state = _mock_llm_returning(HALLUCINATED_CLAIM)

    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_SYNTHESIS", "true")
        _lia_env(mp)
        mp.setattr(HarvestRewriter, "_call_openai_compatible_with_usage", mock, raising=True)
        await svc.derive_beliefs()

    beliefs = _non_superseded_beliefs(storage)
    assert len(beliefs) == 1, (
        f"Expected one belief, found {len(beliefs)}: {[b['content'] for b in beliefs]}"
    )
    content = beliefs[0]["content"]

    # The synthesis path must have been attempted (so the guard could run).
    assert state["calls"] >= 1, (
        "Anti-hallucination guard can only act if synthesis ran; the LLM (mock) was never "
        f"called → Slice 4 does not exist. calls={state['calls']}"
    )
    # The hallucinated claim must be DISCARDED.
    assert content != HALLUCINATED_CLAIM, (
        f"Hallucinated claim (drops all shared terms) must be discarded; got {content!r}"
    )
    # The fallback belief must retain the cluster's thematic terms.
    lowered = content.lower()
    for term in THEME_TERMS:
        assert term.lower() in lowered, (
            f"Fallback-after-guard belief must retain thematic term {term!r}; got {content!r}"
        )


# ── T3 (GUARD): provider unusable / call raises → gracious fallback, no crash, no event ──

@pytest.mark.asyncio
async def test_synthesis_fallback_is_gracious_when_provider_fails(storage, svc):
    """T3 (GUARD, not RED): flag ON but the LLM call RAISES → derive_beliefs must not crash;
    exactly one belief is created from the extractive/verbatim path; NO belief_synthesis
    cost event is emitted (the call failed before producing usage).

    Passes today (no synthesis exists → no crash, verbatim belief, no event). Must STAY
    green after the fix: the gracious-fallback contract (errors never break derive).
    """
    await _store_cluster(storage)

    async def _boom(self, base_url, model, api_key, prompt, timeout, max_tokens=200):
        raise RuntimeError("LIA unreachable")

    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_SYNTHESIS", "true")
        _lia_env(mp)
        mp.setattr(HarvestRewriter, "_call_openai_compatible_with_usage", _boom, raising=True)
        stats = await svc.derive_beliefs()

    assert not stats.get("errors"), (
        f"Gracious fallback: a provider error must NOT surface as a derive error; "
        f"stats={stats}"
    )
    beliefs = _non_superseded_beliefs(storage)
    assert len(beliefs) == 1, (
        f"A belief must still be created via the extractive/verbatim fallback; "
        f"found {len(beliefs)}: {[b['content'] for b in beliefs]}"
    )
    # No successful-synthesis cost event when the call failed.
    assert _synthesis_events(storage) == [], (
        "No belief_synthesis success event should be emitted when the LLM call failed"
    )


# ── T4 (GUARD): flag OFF → no LLM call, content = current extractive/verbatim ──

@pytest.mark.asyncio
async def test_synthesis_off_never_calls_llm_byte_identical(storage, svc):
    """T4 (GUARD, not RED): with MCP_BELIEF_SYNTHESIS unset (default OFF), derive_beliefs
    must NEVER call the LLM and the belief content is the current extractive/verbatim
    representative (byte-identical contract). A mock that fails loudly if called proves the
    OFF path touches no provider.

    Passes today; must STAY green after the fix.
    """
    await _store_cluster(storage)

    async def _must_not_call(self, base_url, model, api_key, prompt, timeout, max_tokens=200):
        raise AssertionError("OFF path must never call the LLM provider")

    with pytest.MonkeyPatch().context() as mp:
        mp.delenv("MCP_BELIEF_SYNTHESIS", raising=False)
        mp.delenv("MCP_BELIEF_EXTRACTIVE", raising=False)
        mp.delenv("MCP_BELIEF_BEST_REPRESENTATIVE", raising=False)
        mp.delenv("MCP_BELIEF_STABLE_IDENTITY", raising=False)
        mp.setattr(HarvestRewriter, "_call_openai_compatible_with_usage", _must_not_call, raising=True)
        await svc.derive_beliefs()

    beliefs = _non_superseded_beliefs(storage)
    assert len(beliefs) == 1, (
        f"Expected one belief, found {len(beliefs)}: {[b['content'] for b in beliefs]}"
    )
    # OFF path: content is exactly the verbatim source (no synthesis, no cleaning).
    assert beliefs[0]["content"] == SOURCE, (
        f"OFF path must store the representative VERBATIM (byte-identical); "
        f"expected {SOURCE!r}, got {beliefs[0]['content']!r}"
    )
    # And no synthesis event at all.
    assert _synthesis_events(storage) == [], (
        "OFF path must emit no belief_synthesis events"
    )
