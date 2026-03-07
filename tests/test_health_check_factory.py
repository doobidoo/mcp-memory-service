"""Tests for health check strategy selection."""

from mcp_memory_service.utils.health_check import (
    HealthCheckFactory,
    HybridHealthChecker,
    SqliteHealthChecker,
    UnknownStorageChecker,
)


class SqliteVecMemoryStorage:
    pass


class DelegatedHybridStorage:
    def __init__(self):
        self.primary = object()
        self.secondary = object()


class UnknownStorage:
    pass


def test_factory_selects_sqlite_checker_by_class_name():
    checker = HealthCheckFactory.create(SqliteVecMemoryStorage())
    assert isinstance(checker, SqliteHealthChecker)


def test_factory_selects_hybrid_checker_for_structural_hybrid_storage():
    checker = HealthCheckFactory.create(DelegatedHybridStorage())
    assert isinstance(checker, HybridHealthChecker)


def test_factory_selects_unknown_checker_for_unrecognized_storage():
    checker = HealthCheckFactory.create(UnknownStorage())
    assert isinstance(checker, UnknownStorageChecker)
