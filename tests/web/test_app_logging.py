#!/usr/bin/env python3
"""
Tests for the uvicorn access-log filter that suppresses noise from
healthcheck and dashboard polling endpoints.

The filter is installed at module import time on the 'uvicorn.access'
logger; these tests exercise the filter class directly so they don't
depend on a running server.
"""

import logging

import pytest

from mcp_memory_service.web.app import _HealthAccessLogFilter


def _access_record(path: str, status: int = 200) -> logging.LogRecord:
    """Build a LogRecord that matches uvicorn's access-log format."""
    msg = '%s:%d - "%s %s HTTP/1.1" %d OK'
    args = ("172.21.0.7", 33636, "GET", path, status)
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=msg,
        args=args,
        exc_info=None,
    )


class TestHealthAccessLogFilter:
    """Filter must drop polling-noise paths and let real traffic through."""

    @pytest.fixture
    def filt(self):
        return _HealthAccessLogFilter()

    def test_drops_health(self, filt):
        assert filt.filter(_access_record("/api/health")) is False

    def test_drops_health_detailed(self, filt):
        assert filt.filter(_access_record("/api/health/detailed")) is False

    def test_drops_sync_status(self, filt):
        """Dashboard polls /api/sync/status continuously."""
        assert filt.filter(_access_record("/api/sync/status")) is False

    def test_drops_regardless_of_status_code(self, filt):
        """A 401 on a noisy path is still noise — don't let auth failures
        sneak through and make logs misleading."""
        assert filt.filter(_access_record("/api/health", status=401)) is False
        assert filt.filter(_access_record("/api/sync/status", status=500)) is False

    def test_keeps_memories_endpoint(self, filt):
        assert filt.filter(_access_record("/api/memories")) is True

    def test_keeps_documents_upload(self, filt):
        assert filt.filter(_access_record("/api/documents/upload")) is True

    def test_keeps_paths_that_only_share_a_prefix(self, filt):
        """Don't accidentally swallow paths that start with the same chars
        as a noise path. The filter looks for '<path> HTTP/' which already
        prevents this — guard it explicitly."""
        # /api/healthful is not /api/health
        assert filt.filter(_access_record("/api/healthful")) is True
        # /api/sync/statuses is not /api/sync/status
        assert filt.filter(_access_record("/api/sync/statuses")) is True

    def test_keeps_root_path(self, filt):
        assert filt.filter(_access_record("/")) is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
