"""MCP_LOCALE is the single switch for harvest, the rewriter and the Kiro bootstrap."""

import pytest

from mcp_memory_service.bootstrap.formatter import get_formatter
from mcp_memory_service.config.locale import get_active_locales
from mcp_memory_service.harvest.extractor import PatternExtractor
from mcp_memory_service.harvest.harvester import SessionHarvester
from mcp_memory_service.harvest.rewriter import HarvestRewriter


@pytest.fixture(autouse=True)
def clean_locale_env(monkeypatch):
    monkeypatch.delenv("MCP_LOCALE", raising=False)
    monkeypatch.delenv("HARVEST_LOCALE", raising=False)
    get_active_locales.cache_clear()
    yield
    get_active_locales.cache_clear()


def _set_locale(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    get_active_locales.cache_clear()


@pytest.mark.parametrize("var", ["MCP_LOCALE", "HARVEST_LOCALE"])
def test_extractor_reads_the_active_locale(monkeypatch, var):
    _set_locale(monkeypatch, var, "en,pt_BR")
    assert PatternExtractor()._locale == "en,pt_BR"


@pytest.mark.parametrize("var", ["MCP_LOCALE", "HARVEST_LOCALE"])
def test_harvester_loads_the_filters_for_the_active_locale(monkeypatch, tmp_path, var):
    _set_locale(monkeypatch, var, "pt_BR")
    harvester = SessionHarvester(tmp_path)
    assert harvester._is_meta_or_temporal("O prompt mais rigoroso vai filtrar nas próximas colheitas")


def test_harvester_stays_english_without_a_locale(tmp_path):
    harvester = SessionHarvester(tmp_path)
    assert not harvester._is_meta_or_temporal("O prompt mais rigoroso vai filtrar nas próximas colheitas")


@pytest.mark.parametrize("var", ["MCP_LOCALE", "HARVEST_LOCALE"])
def test_rewriter_builds_its_instruction_from_the_active_locale(monkeypatch, var):
    _set_locale(monkeypatch, var, "pt_BR")
    assert HarvestRewriter()._locale == "pt_BR"


@pytest.mark.parametrize("var", ["MCP_LOCALE", "HARVEST_LOCALE"])
def test_kiro_bootstrap_follows_the_locale_set_after_import(monkeypatch, var):
    _set_locale(monkeypatch, var, "pt_BR")
    assert "SEMPRE" in get_formatter("kiro").format([], [], ["usar WAL"], {})


def test_kiro_bootstrap_is_english_by_default():
    assert "ALWAYS" in get_formatter("kiro").format([], [], ["use WAL"], {})


def test_mcp_locale_wins_over_harvest_locale(monkeypatch):
    monkeypatch.setenv("HARVEST_LOCALE", "en")
    _set_locale(monkeypatch, "MCP_LOCALE", "pt_BR")
    assert PatternExtractor()._locale == "pt_BR"
    assert HarvestRewriter()._locale == "pt_BR"
    assert "SEMPRE" in get_formatter("kiro").format([], [], ["usar WAL"], {})


def test_kiro_bootstrap_ignores_a_stale_locale_cache(monkeypatch):
    # Something else (NLI at import) already resolved the locale as English.
    # Deliberately no cache_clear after setting MCP_LOCALE.
    assert get_active_locales() == ["en"]
    monkeypatch.setenv("MCP_LOCALE", "pt_BR")
    assert get_active_locales() == ["pt_BR"]
    assert "SEMPRE" in get_formatter("kiro").format([], [], ["usar WAL"], {})
