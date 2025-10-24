# memory-ingest-dir

Batch ingest all supported documents from a directory into the MCP Memory Service database.

## Usage

```
claude /memory-ingest-dir <directory_path> [--tags TAG1,TAG2] [--recursive] [--file-extensions EXT1,EXT2] [--chunk-size SIZE] [--chunk-overlap SIZE] [--max-files COUNT]
```

## Parameters

- `directory_path`: Path to the directory containing documents to ingest (required)
- `--tags`: Comma-separated list of tags to apply to all memories created
- `--recursive`: Process subdirectories recursively (default: true)
- `--file-extensions`: Comma-separated list of file extensions to process (default: pdf,txt,md,json)
- `--chunk-size`: Target size for text chunks in characters (default: 1000)
- `--chunk-overlap`: Characters to overlap between chunks (default: 200)
- `--max-files`: Maximum number of files to process (default: 100)

## Supported Formats

- PDF files (.pdf)
- Text files (.txt, .md, .markdown, .rst)
- JSON files (.json)

## Implementation

I need to upload multiple documents from a directory to the MCP Memory Service HTTP API endpoint.

First, let me check if the service is running and find all supported files in the directory:

```bash
# Check if the service is running
curl -s http://localhost:8080/api/health || curl -s http://localhost:8443/api/health || echo "Service not running"

# Find supported files in the directory
find "$DIRECTORY_PATH" -type f \( -iname "*.pdf" -o -iname "*.txt" -o -iname "*.md" -o -iname "*.json" \) | head -n $MAX_FILES
```

Then I'll upload the files in batch:

```bash
# Create a temporary script to upload files
UPLOAD_SCRIPT=$(mktemp)
cat > "$UPLOAD_SCRIPT" << 'EOF'
#!/bin/bash
TAGS="$1"
CHUNK_SIZE="$2"
CHUNK_OVERLAP="$3"
MAX_FILES="$4"
shift 4
FILES=("$@")

for file in "${FILES[@]}"; do
  echo "Uploading: $file"
  curl -X POST "http://localhost:8080/api/documents/upload" \
    -F "file=@$file" \
    -F "tags=$TAGS" \
    -F "chunk_size=$CHUNK_SIZE" \
    -F "chunk_overlap=$CHUNK_OVERLAP" \
    -F "memory_type=document"
  echo ""
done
EOF

chmod +x "$UPLOAD_SCRIPT"
```

## Examples

```
# Ingest all PDFs from a directory
claude /memory-ingest-dir ./docs --file-extensions pdf --tags documentation

# Recursively ingest from knowledge base
claude /memory-ingest-dir ./knowledge-base --recursive --tags knowledge,reference

# Limit processing to specific formats
claude /memory-ingest-dir ./articles --file-extensions md,txt --max-files 50 --tags articles
```

## Actual Execution Steps

When you run this command, I will:

1. **Scan the directory** for supported file types (PDF, TXT, MD, JSON)
2. **Apply filtering** based on file extensions and max files limit
3. **Validate the service** is running and accessible
4. **Upload files in batch** using the documents API endpoint
5. **Monitor progress** for each file and show real-time updates
6. **Report results** including total chunks created and any errors

All documents will be processed with consistent tagging and chunking parameters.

## Notes

- Files are processed in parallel for efficiency
- Progress is displayed with file counts and chunk statistics
- Each document gets processed independently - failures in one don't stop others
- Automatic tagging includes source directory and file type information
- Large directories may take time - consider using --max-files for testing
