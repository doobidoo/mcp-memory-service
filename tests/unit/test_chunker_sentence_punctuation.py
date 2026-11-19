"""Tests verifying that sentence-aware chunking preserves punctuation and sentence boundaries."""

import pytest
from mcp_memory_service.ingestion.chunker import ChunkingStrategy, TextChunker
def test_sentence_aware_chunking_preserves_sentence_ending_punctuation():
    """Sentence chunking must preserve '.', '!', and '?' on intermediate sentences."""
    strategy = ChunkingStrategy(
        chunk_size=80,
        chunk_overlap=0,
        respect_paragraph_boundaries=False,
        respect_sentence_boundaries=True,
        min_chunk_size=10,
    )
    chunker = TextChunker(strategy)
    text = (
        "First sentence here. Second sentence here! Third sentence here? "
        "Fourth sentence here."
    )
    chunks = [chunk for chunk, _ in chunker.chunk_text(text)]
    assert chunks == [
        "First sentence here. Second sentence here! Third sentence here?",
        "Fourth sentence here.",
    ]
def test_sentence_aware_overlap_preserves_punctuation_and_sentence_boundary():
    """Overlap extraction must start at a complete sentence boundary with punctuation."""
    strategy = ChunkingStrategy(
        chunk_size=80,
        chunk_overlap=35,
        respect_paragraph_boundaries=False,
        respect_sentence_boundaries=True,
        min_chunk_size=10,
    )
    chunker = TextChunker(strategy)
    text = (
        "Hello world! How are you doing? I am doing great... "
        "Thanks for asking. Have a wonderful day!"
    )
    chunks = [chunk for chunk, _ in chunker.chunk_text(text)]
    assert chunks[0] == (
        "Hello world! How are you doing? I am doing great... Thanks for asking."
    )
    assert chunks[1] == "Thanks for asking. Have a wonderful day!"
def test_sentence_aware_chunking_multiple_punctuation_marks():
    """Sentences with ellipses and question-exclamation marks must be cleanly split."""
    strategy = ChunkingStrategy(
        chunk_size=35,
        chunk_overlap=0,
        respect_paragraph_boundaries=False,
        respect_sentence_boundaries=True,
        min_chunk_size=10,
    )
    chunker = TextChunker(strategy)
    text = "Wait... What is this?! I see now. Let us proceed."
    chunks = [chunk for chunk, _ in chunker.chunk_text(text)]
    assert chunks[0] == "Wait... What is this?! I see now."
    assert chunks[1] == "Let us proceed."