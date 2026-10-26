"""Tests for storage/shared.py utility functions."""

from mcp_memory_service.storage.shared import (
    _embedding_cache_get,
    _embedding_cache_put,
    _embedding_cache_size,
    _escape_like,
    _safe_json_loads,
    _sanitize_log_value,
    _string_to_tags,
    _tags_to_string,
    _EMBEDDING_CACHE,
    _EMBEDDING_CACHE_LOCK,
)


def _clear_cache():
    with _EMBEDDING_CACHE_LOCK:
        _EMBEDDING_CACHE.clear()


class TestTagsToStringAndBack:
    def test_roundtrip(self):
        tags = ["python", "web", "api"]
        encoded = _tags_to_string(tags)
        assert encoded == ",python,web,api,"
        decoded = _string_to_tags(encoded)
        assert decoded == tags

    def test_empty(self):
        assert _tags_to_string(None) == ""
        assert _tags_to_string([]) == ""
        assert _string_to_tags(None) == []
        assert _string_to_tags("") == []

    def test_strips_whitespace(self):
        tags = [" foo ", "bar "]
        encoded = _tags_to_string(tags)
        decoded = _string_to_tags(encoded)
        assert decoded == ["foo", "bar"]


class TestEscapeLikeSpecialChars:
    """Wildcards are escaped, not dropped (#1684).

    Dropping ``_`` turned the needle ``upload_id:abc`` into ``uploadid:abc``,
    so a tag with an underscore could never be matched on Milvus.
    """

    def test_escapes_percent(self):
        assert _escape_like("a%b") == "a\\%b"

    def test_escapes_underscore(self):
        assert _escape_like("upload_id:abc") == "upload\\_id:abc"

    def test_leaves_backslash_alone(self):
        # Milvus server versions disagree on a backslash before a non-wildcard
        # inside a like pattern (milvus-io/milvus#43864); doubling it broke the
        # #1107 trailing-backslash tag match on Milvus Lite.
        assert _escape_like("foo\\") == "foo\\"

    def test_clean_value_unchanged(self):
        assert _escape_like("hello") == "hello"


class TestSafeJsonLoads:
    def test_valid_json(self):
        result = _safe_json_loads('{"key": "value"}', "test")
        assert result == {"key": "value"}

    def test_invalid_json(self):
        result = _safe_json_loads("not json", "test")
        assert result == {}

    def test_empty_string(self):
        assert _safe_json_loads("", "test") == {}

    def test_non_dict_json(self):
        assert _safe_json_loads("[1,2,3]", "test") == {}


class TestSanitizeLogValue:
    def test_truncates_newlines(self):
        result = _sanitize_log_value("line1\nline2\rline3")
        assert "\n" not in result
        assert "\r" not in result
        assert "\\n" in result
        assert "\\r" in result

    def test_escapes_ansi(self):
        result = _sanitize_log_value("\x1b[31mred\x1b[0m")
        assert "\x1b" not in result
        assert "\\x1b" in result


class TestEmbeddingCache:
    def setup_method(self):
        _clear_cache()

    def test_put_get(self):
        _embedding_cache_put("key1", [1.0, 2.0, 3.0])
        assert _embedding_cache_get("key1") == [1.0, 2.0, 3.0]

    def test_miss_returns_none(self):
        assert _embedding_cache_get("nonexistent") is None

    def test_size(self):
        _embedding_cache_put("a", [1.0])
        _embedding_cache_put("b", [2.0])
        assert _embedding_cache_size() == 2
