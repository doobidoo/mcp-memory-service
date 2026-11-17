"""Character-based chunking must not emit a trailing chunk that duplicates the previous one."""

import pytest

from mcp_memory_service.ingestion.chunker import ChunkingStrategy, TextChunker


def _char_chunker() -> TextChunker:
    return TextChunker(
        ChunkingStrategy(
            chunk_size=1000,
            chunk_overlap=200,
            respect_paragraph_boundaries=False,
            respect_sentence_boundaries=False,
        )
    )


def _unique_text(length: int) -> str:
    return " ".join(f"w{i:05d}" for i in range(length))[:length]


@pytest.mark.parametrize("length", [1700, 2500, 3400])
def test_final_chunk_is_not_contained_in_previous_chunk(length):
    chunks = [text for text, _ in _char_chunker().chunk_text(_unique_text(length))]

    assert len(chunks) >= 2
    for previous, current in zip(chunks, chunks[1:]):
        assert current not in previous


@pytest.mark.parametrize("length", [1100, 1500, 1700, 2500])
def test_chunks_cover_the_whole_text_and_metadata_is_consistent(length):
    text = _unique_text(length)
    result = _char_chunker().chunk_text(text)

    assert text.rstrip().split()[-1] in result[-1][0]
    assert [meta["chunk_index"] for _, meta in result] == list(range(len(result)))
    assert all(meta["total_chunks"] == len(result) for _, meta in result)
