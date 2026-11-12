"""The 4-way judgment in the #1460 harness: pure decide() rules and the
score report contract.

decide() is a pure function of the two directional verdicts and the ends
signal, so these tests need no model. The score-contract tests drive the
real script with --backend heuristic — the 4-way block is only emitted for
the transformers backend, and the suite must run everywhere, so the contract
tested here is the transformer-less half (explicit refusal), plus decide()'s
mapping itself. The real-model end-to-end (both directions on the labeled
40-pair prototype set) lives in test_nli_onnx_model.py's cache-gated suite.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "nli" / "export_band_pairs.py"

_spec = importlib.util.spec_from_file_location("export_band_pairs", SCRIPT)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


class TestFourwayDecide:
    def test_double_entailment_is_repeats(self):
        assert harness.fourway_decide("a", "b", "entailment", "entailment") == "repeats"

    def test_reverse_entailment_alone_is_repeats(self):
        # paraphrases and supersets share the shape; dedup is the safe action
        assert harness.fourway_decide("a", "b", "neutral", "entailment") == "repeats"

    def test_forward_entailment_alone_is_unrelated(self):
        # the older statement is the general one; nothing retires
        assert harness.fourway_decide("a", "b", "entailment", "neutral") == "unrelated"

    def test_double_contradiction_is_changes(self):
        assert harness.fourway_decide("a", "b", "contradiction", "contradiction") == "changes"

    def test_double_contradiction_with_ends_word_is_ends(self):
        assert harness.fourway_decide(
            "a", "Project Atlas is closed", "contradiction", "contradiction") == "ends"

    def test_single_contradiction_without_ends_word_is_unrelated(self):
        # inert by design: one-direction contradiction caused 3 of the 5
        # unrelated false positives in the prototype's v1 rules
        assert harness.fourway_decide("a", "b", "contradiction", "neutral") == "unrelated"
        assert harness.fourway_decide("a", "b", "neutral", "contradiction") == "unrelated"

    def test_single_contradiction_with_ends_word_is_ends(self):
        assert harness.fourway_decide(
            "a", "the job was cancelled", "contradiction", "neutral") == "ends"
        assert harness.fourway_decide(
            "a", "the tool is deprecated and removed", "neutral", "contradiction") == "ends"

    def test_excluded_end_words_do_not_fire(self):
        # config change / version bump phrasing stays out of ends
        assert harness.fourway_decide(
            "a", "dark mode is disabled by default", "contradiction", "contradiction") == "changes"
        assert harness.fourway_decide(
            "a", "2.3.1 is no longer live", "contradiction", "contradiction") == "changes"

    def test_neutral_neutral_is_unrelated(self):
        assert harness.fourway_decide("a", "b", "neutral", "neutral") == "unrelated"


def _score(tmp_path, rows, *extra):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return subprocess.run(
        [sys.executable, str(SCRIPT), "score", "--pairs", str(pairs), *extra],
        capture_output=True, text=True, cwd=str(SCRIPT.parents[2]),
    )


ROWS = [
    {"older": "The service version is 3.11.", "newer": "The service version is 3.12.",
     "label": "updates"},
    {"older": "Tags help organize memories.", "newer": "Tag filters support wildcards.",
     "label": "neutral"},
]


def test_fourway_block_refused_for_heuristic_backend(tmp_path):
    proc = _score(tmp_path, ROWS, "--backend", "heuristic")
    assert proc.returncode == 0, proc.stderr[-400:]
    report = json.loads(proc.stdout)
    assert report["fourway"] is None
    assert "transformers" in report["fourway_note"]


# ── Real-model end-to-end over the 40-pair prototype set ────────────────────
# Same set as the #1460 prototype comment (shapes from timkjr's labeled run,
# #1954's reproductions, #1460's framing). Skips without the model, exactly
# like test_nli_onnx_model.py.

FOURWAY_SET = [
    ("changes", "The cron job for backups runs every night at 2am.", "The backup cron moved to 4am after the maintenance window changed."),
    ("changes", "The API service is deployed at version 2.3.1.", "We shipped API service 2.4.0 this morning; 2.3.1 is no longer live."),
    ("changes", "The embedding service version is 3.11.", "The embedding service version is 3.12."),
    ("changes", "The cache TTL is 5 minutes.", "The cache TTL is 5 hours now."),
    ("changes", "The service runs on port 8000.", "The service runs on port 3000 since the refactor."),
    ("changes", "Dark mode is enabled by default.", "Dark mode is disabled by default."),
    ("changes", "The rate limit is 100 requests per minute.", "The rate limit was raised to 1000 requests per minute."),
    ("changes", "The model is loaded lazily on first use.", "The model is loaded on startup as of last week's change."),
    ("changes", "Memory records go to the default store.", "Memory records now go to the project store by default."),
    ("changes", "The earlier diagnosis was a disk permissions issue.", "On closer inspection the failures come from a failing SATA cable, not disk permissions."),
    ("ends", "Project Atlas is active and we meet weekly.", "Project Atlas is closed; the team was reassigned last month."),
    ("ends", "The nightly sync job runs at midnight.", "The nightly sync job was cancelled when we moved to streaming."),
    ("ends", "We use the legacy import tool for migrations.", "The legacy import tool is deprecated and removed from the repo."),
    ("ends", "OAuth tokens are cached for a week.", "We stopped caching OAuth tokens; the cache was dropped entirely."),
    ("ends", "The staging cluster is running.", "The staging cluster was shut down in March."),
    ("ends", "Feature flags are read from the config service.", "The config service is retired; feature flags are gone with it."),
    ("ends", "We maintain a mirror of the wiki.", "The wiki mirror is no longer maintained."),
    ("ends", "The metrics dashboard is available at port 9090.", "The metrics dashboard was taken offline."),
    ("ends", "Backups are sent to the offsite box.", "Offsite backups ended; the box was returned."),
    ("ends", "The experimental search backend is enabled for all users.", "The experimental search backend was disabled for everyone."),
    ("repeats", "The API rate limit is 100 requests per minute.", "Requests on the API are limited to 100 per minute."),
    ("repeats", "Embeddings use all-MiniLM-L6-v2 with 384 dimensions.", "The embedding model is all-MiniLM-L6-v2, producing 384-dim vectors."),
    ("repeats", "Backups run daily.", "A backup happens every day."),
    ("repeats", "The service requires an API key for write access.", "Writes need an API key; that has been the rule all along."),
    ("repeats", "The default port is 8000.", "By default the service listens on port 8000."),
    ("repeats", "Tags help organize memories.", "Tags are useful for organizing memories."),
    ("repeats", "Consolidation runs every six hours.", "Every six hours, consolidation runs."),
    ("repeats", "OAuth tokens expire after one hour.", "OAuth tokens are valid for sixty minutes."),
    ("repeats", "The store lives in a single sqlite file.", "All memories are kept in one sqlite database file."),
    ("repeats", "Dark mode follows the system preference.", "The theme switches with the OS setting."),
    ("unrelated", "Never commit directly to main.", "The cache was never warmed after deploy."),
    ("unrelated", "The deploy pipeline runs on Fridays.", "The cache layer was tuned for read-heavy workloads."),
    ("unrelated", "Errors are retried three times.", "Error messages are sanitized before logging."),
    ("unrelated", "Tags help organize memories.", "Tag filters support wildcards."),
    ("unrelated", "The vector store uses cosine distance.", "Vector embeddings are cached in memory."),
    ("unrelated", "Repo A lint rule: never use print statements.", "Repo B lint rule: never use print statements."),
    ("unrelated", "The dashboard shows sync status.", "The dashboard language can be switched."),
    ("unrelated", "We migrated the auth service to OAuth 2.1.", "The auth service logs were moved to a separate store."),
    ("unrelated", "The search index was rebuilt last night.", "Search latency improved after the rebuild."),
    ("unrelated", "The homelab VPN uses wireguard on port 51820.", "The homelab wiki runs in a container behind traefik."),
]


def _model_cached() -> bool:
    import os
    try:
        from huggingface_hub import try_to_load_from_cache
        repo = os.environ.get("MCP_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
        onnx_file = os.environ.get("MCP_NLI_ONNX_FILE", "onnx/model.onnx")
        for fname in (onnx_file, "tokenizer.json", "config.json"):
            if not isinstance(try_to_load_from_cache(repo, fname), str):
                return False
        return True
    except Exception:
        return False


@pytest.mark.asyncio
@pytest.mark.skipif(not _model_cached(), reason="model weights not in local HF cache")
async def test_fourway_end_to_end_on_prototype_set():
    import asyncio
    from mcp_memory_service.reasoning.nli import NLIClassifier

    clf = NLIClassifier(backend="transformers")
    pairs = [(o, n) for _, o, n in FOURWAY_SET]
    fwd = await clf.classify_batch(pairs)
    rev = await clf.classify_batch([(n, o) for _, o, n in FOURWAY_SET])
    preds = [harness.fourway_decide(o, n, f.label, r.label)
             for (_, o, n), f, r in zip(FOURWAY_SET, fwd, rev)]

    correct = sum(1 for (gold, _, _), p in zip(FOURWAY_SET, preds) if gold == p)
    retired_wrong = sum(
        1 for (gold, _, _), p in zip(FOURWAY_SET, preds)
        if gold in ("unrelated", "repeats") and p in ("changes", "ends"))
    assert clf.degraded is False
    # prototype reference: 29/40 classes, 2 wrong retirements (both the known
    # blind spots); thresholds leave room for model-version drift
    assert correct >= 26, f"class accuracy fell below 26/40: {correct}"
    assert retired_wrong <= 3, f"too many non-changes retired: {retired_wrong}"
