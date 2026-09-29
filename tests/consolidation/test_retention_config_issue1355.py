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


def _runtime_config():
    """The config the running server actually gets."""
    return ConsolidationConfig(**config_mod.CONSOLIDATION_CONFIG)


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
    def test_ontology_types_get_their_documented_retention(self):
        """Every real memory_type value must hit a retention key, not the fallback."""
        periods = _runtime_config().retention_periods
        for memory_type, days in ONTOLOGY_RETENTION.items():
            assert periods.get(memory_type) == days, memory_type

    def test_legacy_keys_still_honored(self):
        """Memories typed with the legacy names keep their periods."""
        periods = _runtime_config().retention_periods
        for memory_type, days in LEGACY_RETENTION.items():
            assert periods.get(memory_type) == days, memory_type

    def test_ontology_types_have_env_overrides(self, monkeypatch):
        """MCP_RETENTION_<TYPE> must override the ontology periods.

        Reloaded deliberately: CONSOLIDATION_CONFIG reads the environment
        while being imported (same pattern as
        test_clustering_algorithm_selection.py).
        """
        monkeypatch.setenv('MCP_RETENTION_DECISION', '540')
        reloaded = importlib.reload(config_mod)
        try:
            assert reloaded.CONSOLIDATION_CONFIG['retention_periods']['decision'] == 540
        finally:
            monkeypatch.undo()
            importlib.reload(config_mod)


class TestRelevanceUsesOntologyRetention:
    @pytest.mark.asyncio
    async def test_decision_memory_decays_on_365_days_not_the_30_fallback(self):
        """A 100-day-old decision must decay per its 365-day period.

        On the broken config the lookup missed and decayed it with the
        30-day fallback (exp(-100/30) ~ 0.036 instead of exp(-100/365) ~ 0.76).
        """
        calc = ExponentialDecayCalculator(_runtime_config())
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
    async def test_learning_outlives_error_at_the_same_age(self):
        """Types must get different periods through the runtime config."""
        calc = ExponentialDecayCalculator(_runtime_config())
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
