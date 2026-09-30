"""The runtime retention_periods must key on the real memory_type ontology.

`server_impl.py` and `web/app.py` build their ConsolidationConfig from
`CONSOLIDATION_CONFIG`, and passing `retention_periods` as a keyword replaces
the dataclass default wholesale. Keyed only by the legacy names, every
ontology-typed memory missed in `_calculate_memory_relevance` and fell back
to the 30-day default (#1355).

The tests below deliberately read `CONSOLIDATION_CONFIG` instead of building
their own dict: `tests/consolidation/conftest.py` constructs a config with
the ontology keys by hand, which is exactly why the runtime mismatch went
unnoticed.
"""

import importlib
import math
from datetime import datetime, timedelta

import pytest

from mcp_memory_service.config import consolidation as config_mod
from mcp_memory_service.consolidation.base import ConsolidationConfig
from mcp_memory_service.consolidation.decay import ExponentialDecayCalculator
from mcp_memory_service.models.memory import Memory

# Phase 0 Ontology Foundation types and the retention days each is documented
# to get (ConsolidationConfig's dataclass default in consolidation/base.py).
ONTOLOGY_RETENTION = {
    'decision': 365,
    'learning': 180,
    'pattern': 90,
    'error': 30,
    'observation': 30,
}

# Legacy names kept for memories stored before the ontology existed
# (critical -> decision, reference -> learning, standard/temporary -> observation).
LEGACY_RETENTION = {
    'critical': 365,
    'reference': 180,
    'standard': 30,
    'temporary': 7,
}

# CONSOLIDATION_CONFIG reads the environment at import time, so tests that
# assert defaults or a single override must reload it with the other
# retention variables cleared (a developer's shell export must not flip
# these assertions).
RETENTION_ENV_VARS = [
    'MCP_RETENTION_DECISION',
    'MCP_RETENTION_LEARNING',
    'MCP_RETENTION_PATTERN',
    'MCP_RETENTION_ERROR',
    'MCP_RETENTION_OBSERVATION',
    'MCP_RETENTION_CRITICAL',
    'MCP_RETENTION_REFERENCE',
    'MCP_RETENTION_STANDARD',
    'MCP_RETENTION_TEMPORARY',
]


def _retention_with_env(monkeypatch, **setenv):
    """Retention periods read under a controlled environment.

    CONSOLIDATION_CONFIG reads the environment at import time (same reload
    pattern as test_clustering_algorithm_selection.py), so tests reload it
    with every MCP_RETENTION_* variable cleared first: a developer's shell
    export must not flip these assertions.
    """
    for var in RETENTION_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for var, value in setenv.items():
        monkeypatch.setenv(var, value)
    try:
        reloaded = importlib.reload(config_mod)
        return dict(reloaded.CONSOLIDATION_CONFIG['retention_periods'])
    finally:
        monkeypatch.undo()
        importlib.reload(config_mod)


def _clean_config(monkeypatch, **setenv):
    """A ConsolidationConfig whose retention periods came from a clean env."""
    return ConsolidationConfig(
        retention_periods=_retention_with_env(monkeypatch, **setenv)
    )


def _memory(memory_type, age_days, now):
    created = now - timedelta(days=age_days)
    return Memory(
        content=f"{memory_type} memory",
        content_hash=f"hash-{memory_type}",
        tags=[memory_type],
        memory_type=memory_type,
        embedding=[0.1] * 320,
        created_at=created.timestamp(),
        created_at_iso=created.isoformat() + 'Z',
        updated_at=created.timestamp(),
        updated_at_iso=created.isoformat() + 'Z',
    )


class TestRuntimeRetentionKeys:
    def test_ontology_types_get_their_documented_retention(self, monkeypatch):
        """Every real memory_type value must hit a retention key, not the fallback."""
        periods = _retention_with_env(monkeypatch)
        for memory_type, days in ONTOLOGY_RETENTION.items():
            assert periods.get(memory_type) == days, memory_type

    def test_legacy_keys_still_honored(self, monkeypatch):
        """Memories typed with the legacy names keep their periods."""
        periods = _retention_with_env(monkeypatch)
        for memory_type, days in LEGACY_RETENTION.items():
            assert periods.get(memory_type) == days, memory_type

    def test_ontology_types_have_env_overrides(self, monkeypatch):
        """MCP_RETENTION_<TYPE> must override the ontology periods alone."""
        periods = _retention_with_env(monkeypatch, MCP_RETENTION_DECISION='540')
        assert periods['decision'] == 540
        assert periods['learning'] == 180  # only the override moved


class TestRelevanceUsesOntologyRetention:
    @pytest.mark.asyncio
    async def test_decision_memory_decays_on_365_days_not_the_30_fallback(self, monkeypatch):
        """A 100-day-old decision must decay per its 365-day period (#1355)."""
        calc = ExponentialDecayCalculator(_clean_config(monkeypatch))
        now = datetime.now()

        score = await calc._calculate_memory_relevance(
            _memory('decision', age_days=100, now=now), now, {}, {}
        )

        assert score.metadata['memory_type'] == 'decision'
        assert score.metadata['retention_period'] == 365
        assert score.decay_factor == pytest.approx(
            math.exp(-score.metadata['age_days'] / 365)
        )

    @pytest.mark.asyncio
    async def test_learning_outlives_error_at_the_same_age(self, monkeypatch):
        """Types must get different periods through the runtime config."""
        calc = ExponentialDecayCalculator(_clean_config(monkeypatch))
        now = datetime.now()

        scores = {}
        for memory_type in ('learning', 'error'):
            score = await calc._calculate_memory_relevance(
                _memory(memory_type, age_days=90, now=now), now, {}, {}
            )
            scores[memory_type] = score

        assert scores['learning'].metadata['retention_period'] == 180
        assert scores['error'].metadata['retention_period'] == 30
        assert scores['learning'].decay_factor > scores['error'].decay_factor


class TestSubtypeMemoriesInheritBaseRetention:
    @pytest.mark.asyncio
    async def test_subtypes_decay_on_their_base_type_period(self, monkeypatch):
        """Stored subtypes resolve to the base period, not the 30-day fallback.

        Taxonomy subtypes like 'insight' (learning) and 'architecture'
        (decision) are real stored memory_type values; without parent
        resolution they missed every retention key (#1355).
        """
        calc = ExponentialDecayCalculator(_clean_config(monkeypatch))
        now = datetime.now()

        for subtype, base_days in (('insight', 180), ('architecture', 365)):
            score = await calc._calculate_memory_relevance(
                _memory(subtype, age_days=100, now=now), now, {}, {}
            )
            assert score.metadata['retention_period'] == base_days, subtype
            # The parent only selects the period; the score keeps the stored
            # subtype, which forgetting.py archives with the memory.
            assert score.metadata['memory_type'] == subtype

    @pytest.mark.asyncio
    async def test_env_override_reaches_subtype_memories(self, monkeypatch):
        """MCP_RETENTION_LEARNING must govern 'insight' memories too."""
        calc = ExponentialDecayCalculator(
            _clean_config(monkeypatch, MCP_RETENTION_LEARNING='540')
        )
        now = datetime.now()

        score = await calc._calculate_memory_relevance(
            _memory('insight', age_days=100, now=now), now, {}, {}
        )

        assert score.metadata['retention_period'] == 540


class TestConfigApiListsRetentionVars:
    def test_retention_vars_in_descriptions_and_category(self):
        """The five new overrides must be visible in the configuration API."""
        from mcp_memory_service.web.api import configuration as config_api

        for var in RETENTION_ENV_VARS:
            assert var in config_api.PARAM_DESCRIPTIONS, var

        listed = {param[0] for param in config_api.ENV_CATEGORIES['consolidation']['params']}
        for var in RETENTION_ENV_VARS:
            assert var in listed, var
