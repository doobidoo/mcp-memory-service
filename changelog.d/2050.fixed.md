- **Character-based chunking no longer emits a duplicate trailing chunk.**
  `TextChunker._chunk_by_characters` (used by the CSV and JSON loaders) stepped back by the overlap even after a chunk had reached the end of the text, so it appended one extra chunk that was entirely contained in the previous one and stored the same tail twice. The loop now stops once a chunk reaches the end of the text.
