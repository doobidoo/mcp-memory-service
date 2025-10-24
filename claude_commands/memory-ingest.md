# memory-ingest

Ingest a document file into the MCP Memory Service database.

## Usage

```
claude /memory-ingest <file_path> [--tags TAG1,TAG2] [--chunk-size SIZE] [--chunk-overlap OVERLAP] [--memory-type TYPE]
```

## Parameters

- `file_path`: Path to the document file to ingest (required)
- `--tags`: Comma-separated list of tags to apply to all memories created from this document
- `--chunk-size`: Target size for text chunks in characters (default: 1000)
- `--chunk-overlap`: Characters to overlap between chunks (default: 200)
- `--memory-type`: Type label for created memories (default: "document")

## Supported Formats

- PDF files (.pdf)
- Text files (.txt, .md, .markdown, .rst)
- JSON files (.json)

## Implementation

I need to upload the document to the MCP Memory Service HTTP API endpoint and monitor the progress.

First, let me check if the service is running and get the correct endpoint:

```bash
# Check if the service is running on default port
curl -s http://localhost:8080/api/health || echo "Service not running on 8080"

# Or check common alternative ports
curl -s http://localhost:8443/api/health || echo "Service not running on 8443"
```

Assuming the service is running (adjust the URL as needed), I'll upload the document:

```bash
# Upload the document with specified parameters
curl -X POST "http://localhost:8080/api/documents/upload" \\
  -F "file=@$FILE_PATH" \\
  -F "tags=$TAGS" \\
  -F "chunk_size=$CHUNK_SIZE" \\
  -F "chunk_overlap=$CHUNK_OVERLAP" \\
  -F "memory_type=$MEMORY_TYPE" \\
  
```

Then I'll monitor the upload progress:

```bash
# Monitor progress (replace UPLOAD_ID with the ID from the upload response)
curl -s "http://localhost:8080/api/documents/status/UPLOAD_ID"
```

## Examples

```
# Ingest a PDF with tags
claude /memory-ingest manual.pdf --tags documentation,reference

# Ingest a markdown file with custom chunking
claude /memory-ingest README.md --chunk-size 1500 --chunk-overlap 300 --tags project,readme

# Ingest a document as reference material
claude /memory-ingest api-docs.json --tags api,reference --memory-type reference
```

## Actual Execution Steps

When you run this command, I will:

1. **Validate the file exists** and check if it's a supported format
2. **Determine the service endpoint** (try localhost:8080, then 8443)
3. **Upload the file** using the documents API endpoint with your specified parameters
4. **Monitor progress** and show real-time updates
5. **Report results** including chunks created and any errors

The document will be automatically parsed, chunked, and stored as searchable memories in your MCP Memory Service database.

## Notes

- The document will be automatically parsed and chunked for optimal retrieval
- Each chunk becomes a separate memory entry with semantic embeddings
- Progress will be displayed during ingestion
- Failed chunks will be reported but won't stop the overall process
