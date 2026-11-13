import pytest

from mcp_memory_service.utils.content_splitter import split_content


@pytest.mark.parametrize(
    ("content", "window_end", "token"),
    [
        (
            "More context. https://example.com/path remains intact after splitting.",
            "com",
            "https://example.com/path",
        ),
        (
            "More context. Version 1.2.3 remains supported after splitting.",
            "3 remains",
            "1.2.3",
        ),
        (
            "More context. Value 3.14 remains valid after splitting.",
            "14 remains",
            "3.14",
        ),
    ],
    ids=["url", "version", "decimal"],
)
def test_window_edge_period_inside_token_does_not_split_token(
    content: str, window_end: str, token: str
) -> None:
    """An artificial window end is not a sentence boundary inside a token."""
    max_length = content.index(window_end)
    assert content[max_length - 1] == "."
    assert not content[max_length].isspace()

    chunks = split_content(content, max_length=max_length, overlap=0)

    assert chunks[0].endswith("More context.")
    assert any(token in chunk for chunk in chunks)


def test_sentence_at_window_edge_uses_following_whitespace() -> None:
    """A sentence ending at max_length remains a valid boundary."""
    content = "More context. Another sentence continues past the limit."
    max_length = len("More context.")

    chunks = split_content(content, max_length=max_length, overlap=0)

    assert chunks[0] == "More context."
    assert chunks[1].startswith("Another")


@pytest.mark.parametrize(
    ("content", "max_length", "token"),
    [
        (
            "Docs live at https://example.com/guide and the version is 3.14159 today",
            29,
            "https://example.com/guide",
        ),
        (
            "Pi is 3.14159. The release is v2.10.4 and the host is api.example.com.",
            18,
            "v2.10.4",
        ),
        (
            "Pi is 3.14159. The release is v2.10.4 and the host is api.example.com.",
            21,
            "v2.10.4",
        ),
    ],
    ids=["url-before-word-boundary", "version-at-window-18", "version-at-window-21"],
)
def test_window_edge_period_falls_back_to_word_boundary(
    content: str, max_length: int, token: str
) -> None:
    """Without a prior sentence boundary, split before a fitting token."""
    chunks = split_content(content, max_length=max_length, overlap=0)

    assert all(len(chunk) <= max_length for chunk in chunks)
    assert any(token in chunk for chunk in chunks)
