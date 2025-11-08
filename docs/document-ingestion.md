# Document Ingestion (v7.6.0+)

Enhanced document parsing with optional semtools integration for superior quality extraction.

## Supported Formats

| Format | Native Parser | With Semtools | Quality |
|--------|--------------|---------------|---------|
| PDF | PyPDF2/pdfplumber | LlamaParse | Excellent (OCR, tables) |
| DOCX | Not supported | LlamaParse | Excellent |
| PPTX | Not supported | LlamaParse | Excellent |
| TXT/MD | Built-in | N/A | Perfect |

## Semtools Integration (Optional)

Install [semtools](https://github.com/run-llama/semtools) for enhanced document parsing:

```bash
# Install via npm (recommended)
npm i -g @llamaindex/semtools

# Or via cargo
cargo install semtools

# Optional: Configure LlamaParse API key for best quality
export LLAMAPARSE_API_KEY="your-api-key"
```

## Configuration

```bash
# Document chunking settings
export MCP_DOCUMENT_CHUNK_SIZE=1000          # Characters per chunk
export MCP_DOCUMENT_CHUNK_OVERLAP=200        # Overlap between chunks

# LlamaParse API key (optional, improves quality)
export LLAMAPARSE_API_KEY="llx-..."
```

## Usage Examples

```bash
# Ingest a single document
claude /memory-ingest document.pdf --tags documentation

# Ingest directory
claude /memory-ingest-dir ./docs --tags knowledge-base

# Via Python
from mcp_memory_service.ingestion import get_loader_for_file

loader = get_loader_for_file(Path("document.pdf"))
async for chunk in loader.extract_chunks(Path("document.pdf")):
    await store_memory(chunk.content, tags=["doc"])
```

## Features

- **Automatic format detection** - Selects best loader for each file
- **Intelligent chunking** - Respects paragraph/sentence boundaries
- **Metadata enrichment** - Preserves file info, extraction method, page numbers
- **Graceful fallback** - Uses native parsers if semtools unavailable
- **Progress tracking** - Reports chunks processed during ingestion

## Performance Considerations

- LlamaParse provides superior quality but requires API key and internet connection
- Native parsers work offline but may have lower extraction quality for complex documents
- Chunk size affects retrieval granularity vs context completeness
- Larger overlap improves continuity but increases storage
