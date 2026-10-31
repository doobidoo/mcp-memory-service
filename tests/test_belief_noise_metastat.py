"""RED tests: _is_noise must reject META-STATISTICS beliefs (#belief-corpus-metastat).

Investigation (31/out) measured ~230 active/candidate beliefs in the live DB that are
META-NOISE escaping `_is_noise`: automatic tag statistics and decision gaps, NOT
actionable knowledge — they occupy top-k slots in search.

These tests call the REAL `_is_noise` imported from
`mcp_memory_service.consolidation.belief_service` (NO mocks, NO stubs of internals),
so they exercise the real filter contract. They FAIL today (RED) because the four
meta-noise shapes below currently return False, and go GREEN when the filter learns to
reject them. The three legit cases guard against over-reach (must stay False).
"""

from mcp_memory_service.consolidation.belief_service import _is_noise


# ── Real cases measured in the live corpus that MUST be rejected (True) ──

# [GAP]: automatic "decision gap" statistic — meta, not a claim.
GAP_CASE = (
    "[GAP] Decision gap: 'shared_tags'\n\n"
    "Tag 'shared_tags' has 189 memories but no decisions recorded. "
    "Consider documenting key decisions for this area."
)

# [PATTERN]: automatic "recurring theme" tag co-occurrence statistic.
PATTERN_CASE = (
    "[PATTERN] Recurring pattern: daia + sa3\n\n"
    "6 memories share tags ['daia', 'sa3'], indicating a recurring theme."
)

# [TREND]: any belief opening with the [TREND] marker is an auto trend statistic.
TREND_CASE = "[TREND] Growing focus on consolidation over the last 30 days."

# Bracket-less variant: the "N memories share tags ... indicating a recurring theme"
# sentence is meta-statistics even without the [PATTERN] bracket.
NOBRACKET_CASE = (
    "6 memories share tags ['injecao', 'learning-loop'], indicating a recurring theme."
)

# Greptile P2 (#1727): _detect_patterns prints the FULL sorted tag list with no
# length cap, so a handful of long tags puts >80 chars inside the brackets. The
# pattern must match the list STRUCTURE, not a char budget.
LONGLIST_CASE = (
    "12 memories share tags "
    + str([f"tag-de-nome-bem-longo-{i:02d}" for i in range(8)])
    + ", indicating a recurring theme."
)


# ── Legit knowledge that MUST keep passing the filter (False = not noise) ──

LEGIT_PR_COLLISION = (
    "Antes de implementar um fix, verificar se já existe PR aberto para o mesmo "
    "problema (colisão de PRs)."
)

LEGIT_ONTOLOGY = (
    'Filtrar memory_type="observation" literal ignora a hierarquia da ontologia '
    "(models/ontology.py); decisões, aprendizados e erros nunca entram."
)

# Merely MENTIONS "padrão" (pattern) in a legit sentence — must not be caught by mistake.
LEGIT_MENTIONS_PATTERN = "O padrão de retry deve usar backoff exponencial."


class TestMetaStatNoiseRejected:
    """The four meta-statistics shapes must be rejected as noise (True)."""

    def test_gap_decision_statistic_is_noise(self):
        # [GAP] auto decision-gap statistic. FAILS today: returns False.
        assert _is_noise(GAP_CASE) is True

    def test_pattern_recurring_theme_is_noise(self):
        # [PATTERN] auto tag co-occurrence statistic. FAILS today: returns False.
        assert _is_noise(PATTERN_CASE) is True

    def test_trend_marker_is_noise(self):
        # [TREND] auto trend statistic. FAILS today: returns False.
        assert _is_noise(TREND_CASE) is True

    def test_bracketless_share_tags_is_noise(self):
        # "N memories share tags ... recurring theme" sans bracket. FAILS today: False.
        assert _is_noise(NOBRACKET_CASE) is True

    def test_long_tag_list_is_noise(self):
        # Greptile P2 #1727: unbounded tag list (>80 chars inside brackets) must
        # still be rejected — the pattern matches list structure, not length.
        assert _is_noise(LONGLIST_CASE) is True


class TestLegitBeliefsNotNoise:
    """Anti-regression: legitimate, actionable beliefs stay un-filtered (False)."""

    def test_pr_collision_lesson_not_noise(self):
        assert _is_noise(LEGIT_PR_COLLISION) is False

    def test_ontology_filter_lesson_not_noise(self):
        assert _is_noise(LEGIT_ONTOLOGY) is False

    def test_sentence_mentioning_pattern_not_noise(self):
        assert _is_noise(LEGIT_MENTIONS_PATTERN) is False


class TestOverReachBoundaries:
    """G5 (tuvok) over-reach probes: pin down the accepted trade-offs.

    The bracket prefixes match on `startswith`, so a belief that *opens* with
    `[PATTERN]`/`[GAP]`/`[TREND]` is rejected by design — those markers are
    reserved for the consolidation engine's auto-insights, so a genuine belief
    has no reason to lead with them. We assert that contract here (True = noise)
    rather than leave it undocumented. A mid-text mention of the same marker is
    NOT at the start, so it stays a valid belief (False).
    """

    def test_belief_opening_with_bracket_is_rejected_by_design(self):
        # Accepted trade-off: leading [PATTERN]/[GAP]/[TREND] == consolidation
        # marker, so rejected. Documented, not accidental.
        assert _is_noise("[PATTERN] matching deve usar regex ancorado.") is True
        assert _is_noise("[GAP] analysis is a technique we should adopt.") is True

    def test_marker_mentioned_mid_text_is_not_noise(self):
        # startswith is literal: a marker cited mid-sentence is real knowledge.
        assert _is_noise(
            "Ao derivar beliefs, o marker [GAP] que o consolidation gera não "
            "deve virar conhecimento — filtrar no _is_noise."
        ) is False

    def test_prose_mentioning_share_tags_across_paragraphs_not_noise(self):
        # The bounded, non-DOTALL pattern must NOT catch legit prose that only
        # mentions both expressions far apart (tuvok P2 regression guard).
        assert _is_noise(
            "Beliefs onde memories share tags demais são suspeitos de over-tagging.\n\n"
            "Isso pode indicar um recurring theme de disciplina de tags fraca."
        ) is False


class TestGetBeliefsExcludesStoredNoise:
    """Greptile P1 (#1727): beliefs stored before the filter existed must not
    leak through get_beliefs — exclude meta-statistics at read time too."""

    @staticmethod
    def _make_storage():
        import sqlite3
        from pathlib import Path
        from unittest.mock import MagicMock
        conn = sqlite3.connect(":memory:")
        migration = (
            Path(__file__).parent.parent / "src" / "mcp_memory_service"
            / "storage" / "migrations" / "012_add_belief_store.sql"
        )
        conn.executescript(migration.read_text())
        storage = MagicMock()
        storage.conn = conn
        return storage

    @staticmethod
    def _store_raw_belief(storage, content, confidence=0.9):
        import json, hashlib, time
        h = hashlib.sha256(content.encode()).hexdigest()
        now = time.time()
        storage.conn.execute(
            "INSERT INTO beliefs (belief_hash, content, confidence, status, "
            "created_at, updated_at, derived_from, contradicted_by, metadata) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (h, content, confidence, "active", now, now, json.dumps(["x"]),
             json.dumps([]), json.dumps({})),
        )
        storage.conn.commit()
        return h

    async def test_stored_meta_noise_not_returned(self):
        from mcp_memory_service.consolidation.belief_service import BeliefService
        storage = self._make_storage()
        svc = BeliefService(storage)
        noise = "[GAP] Decision gap: 'legacy'\n\nTag 'legacy' has 50 memories but no decisions recorded."
        legit = "Antes de abrir PR, rodar o gate G5 com o reviewer independente."
        self._store_raw_belief(storage, noise)
        self._store_raw_belief(storage, legit)
        out = await svc.get_beliefs(status="active", min_confidence=0.0)
        contents = [b["content"] for b in out]
        assert legit in contents
        assert noise not in contents
