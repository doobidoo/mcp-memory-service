# External Embedding API Support

MCP Memory Service can use external OpenAI-compatible embedding APIs instead of running embedding models locally. This is useful for:

- **Shared infrastructure**: Run a single embedding service for multiple MCP instances
- **Resource efficiency**: Offload GPU/CPU intensive embedding to dedicated servers
- **Model flexibility**: Use embedding models not available in SentenceTransformers
- **Hosted services**: Use OpenAI, Cohere, or other embedding APIs

## ⚠️ Storage Backend Compatibility

**External embedding APIs are currently only supported with the `sqlite_vec` storage backend.**

If you're using `hybrid` or `cloudflare` backends, the external API will NOT be used:
- **Hybrid**: SQLite-vec will fall back to local models, Cloudflare uses Workers AI
- **Cloudflare**: Always uses Workers AI (`@cf/baai/bge-base-en-v1.5`)

To use external embedding APIs, configure:
```bash
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_EXTERNAL_EMBEDDING_URL=http://localhost:8890/v1/embeddings
```

Support for external APIs with Hybrid/Cloudflare backends is planned for a future release.

## Configuration

Set these environment variables to enable external embeddings:

```bash
# Required: API endpoint URL
export MCP_EXTERNAL_EMBEDDING_URL=http://localhost:8890/v1/embeddings

# Optional: Model name (default: nomic-embed-text)
export MCP_EXTERNAL_EMBEDDING_MODEL=nomic-embed-text

# Optional: API key for authenticated endpoints
export MCP_EXTERNAL_EMBEDDING_API_KEY=sk-xxx
```

## Supported Backends

### vLLM

[vLLM](https://docs.vllm.ai/) provides high-performance inference with OpenAI-compatible API.

```bash
# Start vLLM with an embedding model
vllm serve nomic-ai/nomic-embed-text-v1.5 --port 8890

# Configure MCP Memory Service
export MCP_EXTERNAL_EMBEDDING_URL=http://localhost:8890/v1/embeddings
export MCP_EXTERNAL_EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
```

### Ollama

[Ollama](https://ollama.ai/) provides easy local model deployment.

```bash
# Pull and run embedding model
ollama pull nomic-embed-text

# Configure MCP Memory Service
export MCP_EXTERNAL_EMBEDDING_URL=http://localhost:11434/v1/embeddings
export MCP_EXTERNAL_EMBEDDING_MODEL=nomic-embed-text
```

### Text Embeddings Inference (TEI)

[TEI](https://github.com/huggingface/text-embeddings-inference) is HuggingFace's optimized embedding server.

```bash
# Start TEI
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id nomic-ai/nomic-embed-text-v1.5

# Configure MCP Memory Service
export MCP_EXTERNAL_EMBEDDING_URL=http://localhost:8080/v1/embeddings
export MCP_EXTERNAL_EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
```

### OpenAI

```bash
export MCP_EXTERNAL_EMBEDDING_URL=https://api.openai.com/v1/embeddings
export MCP_EXTERNAL_EMBEDDING_MODEL=text-embedding-3-small
export MCP_EXTERNAL_EMBEDDING_API_KEY=sk-xxx
```

## Embedding Dimension Compatibility

⚠️ **Important**: The embedding dimension must match your database schema.

| Model | Dimensions |
|-------|------------|
| nomic-embed-text | 768 |
| text-embedding-3-small | 1536 |
| text-embedding-3-large | 3072 |
| all-MiniLM-L6-v2 | 384 |
| all-mpnet-base-v2 | 768 |

If you're migrating from a local model to an external API (or vice versa), ensure the dimensions match or you'll need to re-embed your memories.

## Fallback Behavior

If the external API is unavailable at startup, MCP Memory Service will fall back to local embedding models (ONNX → SentenceTransformer → Hash embeddings).

To require external embeddings without fallback, you can set:

```bash
export MCP_MEMORY_USE_ONNX=false  # Disable ONNX fallback
# Don't install sentence-transformers # Disable ST fallback
```

## Performance Considerations

- **Batching**: The adapter batches requests (default 32 sentences) for efficiency
- **Caching**: Embedding models are cached per API URL + model combination
- **Timeout**: Default 30 second timeout per request (configurable)
- **Retry**: Currently no automatic retry; failures fall back to local models

## Troubleshooting

### Connection refused
```
ConnectionError: Cannot connect to external embedding API at http://localhost:8890/v1/embeddings
```
- Verify the embedding service is running
- Check the port is correct
- Ensure no firewall blocking

### Dimension mismatch
```
RuntimeError: Dimension mismatch for inserted vector. Expected 768 dimensions but received 384.
```
- The external model produces different dimensions than your database
- Either change the model or migrate your database

### Authentication error
```
ConnectionError: API returned status 401: Unauthorized
```
- Set `MCP_EXTERNAL_EMBEDDING_API_KEY` environment variable
- Verify the API key is valid

## API Compatibility

The adapter expects OpenAI-compatible `/v1/embeddings` endpoint:

**Request:**
```json
{
  "input": ["text to embed", "another text"],
  "model": "model-name"
}
```

**Response:**
```json
{
  "data": [
    {"index": 0, "embedding": [0.1, 0.2, ...]},
    {"index": 1, "embedding": [0.3, 0.4, ...]}
  ]
}
```
