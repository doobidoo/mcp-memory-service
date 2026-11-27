# Docker Setup for MCP Memory Service

## Tags

| Tag | Description | Size delta vs `:latest` |
|-----|-------------|------------------------|
| `:latest` | Standard image, full feature set | baseline |
| `:slim` | CPU-only, no PyTorch/CUDA | ~90% smaller |

There is **no published `:quality-cpu` tag.** Its last publish was
`10-quality-cpu` on 2026-05-29; the build job lived in the GitHub workflow
removed in `9c7f8b89` and was not ported. It is not coming back as a per-release
build: the ONNX quality models are version-independent, so rebuilding them on
every patch release is waste, and that job was historically the most expensive
and flakiest one in the matrix. `tools/docker/Dockerfile.quality-cpu` stays in
the tree as a supported **build-it-yourself** option (issue #171, from #170).

## Local quality scoring in a container

The standard and `:slim` images ship `onnxruntime` but neither the exported ONNX
models nor the `torch`/`transformers` needed to export them, so
`MCP_QUALITY_AI_PROVIDER=local` needs the models supplied from outside. Three
supported paths, cheapest first.

### 1. Export once, mount the directory (recommended)

On any machine that has `torch` and `transformers`:

```bash
python scripts/quality/export_deberta_onnx.py
# writes to ~/.cache/mcp_memory/onnx_models/<model_name>/
```

Then mount that directory and point the service at it:

```bash
docker run --rm \
  -e MCP_QUALITY_BOOST_ENABLED=true \
  -e MCP_QUALITY_AI_PROVIDER=local \
  -e MCP_QUALITY_LOCAL_MODEL=nvidia-quality-classifier-deberta \
  -e MCP_QUALITY_ONNX_MODEL_DIR=/models/onnx_models \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v ~/.cache/mcp_memory/onnx_models:/models/onnx_models:ro \
  doobidoo/mcp-memory-service:slim
```

`MCP_QUALITY_ONNX_MODEL_DIR` is the **parent** directory; the loader appends
`/<model_name>`, so one setting serves every model. It defaults to
`~/.cache/mcp_memory/onnx_models`. Setting it is what makes the mount target
explicit — without it the path depends on `Path.home()`, which resolved to
`/root` only because these images run as root and set neither `USER` nor `HOME`.

Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so a missing artifact fails
loudly instead of silently attempting a download.

#### Kubernetes, non-root with a read-only root filesystem

Verified in @tecnobrat's deployment (UID 1000, `readOnlyRootFilesystem: true`),
with the models delivered as an image volume — a `FROM scratch` stage holding
nothing but the exported artifacts, versioned independently of the service:

```yaml
env:
  - name: MCP_QUALITY_ONNX_MODEL_DIR
    value: /models/onnx_models
volumeMounts:
  - name: quality-model
    mountPath: /models/onnx_models/nvidia-quality-classifier-deberta
    subPath: opt/onnx_models_baked/nvidia-quality-classifier-deberta
    readOnly: true
volumes:
  - name: quality-model
    image:
      reference: <registry>/mcp-memory-service-quality-cpu:latest
```

On releases before `MCP_QUALITY_ONNX_MODEL_DIR` existed, the same thing was done
by redirecting `HOME` at a writable volume and mounting the models inside it:

```yaml
env:
  - name: HOME
    value: /home/mcp-memory-service
volumeMounts:
  - name: home
    mountPath: /home/mcp-memory-service        # emptyDir, so $HOME is writable
  - name: quality-model
    mountPath: /home/mcp-memory-service/.cache/mcp_memory/onnx_models/nvidia-quality-classifier-deberta
    subPath: opt/onnx_models_baked/nvidia-quality-classifier-deberta
    readOnly: true
```

That indirection is no longer necessary.

### 2. Build `Dockerfile.quality-cpu` yourself

```bash
docker build -t my-mcp-memory:quality-cpu -f tools/docker/Dockerfile.quality-cpu .

# Verify both models load from the baked cache (no export triggered):
docker run --rm my-mcp-memory:quality-cpu \
  python -c "
from mcp_memory_service.quality.onnx_ranker import get_onnx_ranker_model
print(get_onnx_ranker_model('ms-marco-MiniLM-L-6-v2'))
print(get_onnx_ranker_model('nvidia-quality-classifier-deberta'))
print('Both quality models loaded from baked ONNX cache')
"
```

The image sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so no live model
download can occur.

**Expect roughly 736 MB for the deberta artifacts, not the ~600 MB the tags
table used to claim.** The published `10-quality-cpu` was 1.26 GB compressed
against 164 MB for `10-slim`. See the quantization note below for why.

### 3. Point at an endpoint you already run

No models to manage:

```bash
MCP_QUALITY_AI_PROVIDER=openai-compatible
MCP_QUALITY_AI_BASE_URL=http://localhost:11434/v1
MCP_QUALITY_AI_MODEL=qwen2.5:7b-instruct
```

#### Build-time quantization

The `nvidia-quality-classifier-deberta` ONNX model ships ~702 MB of fp32 external
weights, which dominates the image size. The build pipeline runs
`tools/docker/scripts/quantize_quality_models.py` after export to:

1. Convert fp32 → fp16 (via `onnxconverter-common`)
2. Apply dynamic int8 to MatMul + Gather (via `onnxruntime.quantization`)
3. Benchmark each variant: file size, mean inference latency, and Pearson
   correlation against the fp32 baseline on 100 sample texts
4. Replace fp32 with the smallest variant whose correlation is ≥ 0.98
5. Fall back to fp32 (no failure) if no variant meets the threshold

**In practice, step 5 is what happens for this model.** @tecnobrat measured int8
at 0.28-0.39 correlation against fp32 for `nvidia-quality-classifier-deberta`, so
the 0.98 gate rejects it — correctly — and the build keeps fp32. Do not plan
capacity around a quantization win here; assume the fp32 size.

Override the strategy at build time:

```bash
# Force fp16 only (safer on ARM)
docker build --build-arg QUANTIZE_MODE=fp16 -f tools/docker/Dockerfile.quality-cpu .

# Tighten the correlation gate
docker build --build-arg QUANTIZE_MIN_CORR=0.99 -f tools/docker/Dockerfile.quality-cpu .

# Skip quantization entirely (set mode to fp16 with an unreachable corr; falls back to fp32)
docker build --build-arg QUANTIZE_MIN_CORR=1.01 -f tools/docker/Dockerfile.quality-cpu .
```

`ms-marco-MiniLM-L-6-v2` is intentionally not quantized — it's already ~80 MB
and not worth the engineering risk.

## 🚀 Quick Start

Choose your mode:

### MCP Protocol Mode (for Claude Desktop, VS Code)
```bash
docker-compose up -d
```

### HTTP API Mode (for REST API, Web Dashboard)
```bash
echo "MCP_API_KEY=$(openssl rand -hex 32)" > .env
docker-compose -f docker-compose.http.yml up -d
```

`MCP_API_KEY` has no default; compose refuses to start without it. The API is
published on `127.0.0.1` only. Set `HTTP_BIND=0.0.0.0` in `.env` to reach it from
other machines.

## 📝 What's New (v5.0.4)

Thanks to feedback from Joe Esposito, we've completely simplified the Docker setup:

### ✅ Fixed Issues
- **PYTHONPATH** now correctly set to `/app/src`
- **run_server.py** properly copied for HTTP mode
- **Embedding models** pre-downloaded during build (no runtime failures)

### 🎯 Simplified Structure
- **2 clear modes** instead of 4 confusing variants
- **Unified entrypoint** that auto-detects mode
- **Single Dockerfile** for all configurations

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_MODE` | Operation mode: `mcp` or `http` | `mcp` |
| `MCP_API_KEY` | API key for HTTP mode | none, required |
| `HTTP_BIND` | Host address the HTTP port is published on | `127.0.0.1` |
| `HTTP_PORT` | Host port for HTTP mode | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Volume Mounts

All data is stored in a single `./data` directory:
- SQLite database: `./data/sqlite_vec.db`
- Backups: `./data/backups/`

## 🧪 Testing

Run the test script to verify both modes work:
```bash
./test-docker-modes.sh
```

## 📊 HTTP Mode Endpoints

When running in HTTP mode:
- **Dashboard**: http://localhost:8000/
- **API Docs**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/api/health

## 🔄 Migration from Old Setup

If you were using the old Docker files:

| Old File | New Alternative |
|----------|-----------------|
| `docker-compose.standalone.yml` | Use `docker-compose.http.yml` |
| `docker-compose.uv.yml` | UV is now built-in |
| `docker-compose.pythonpath.yml` | Fixed in main Dockerfile |

See [DEPRECATED.md](./DEPRECATED.md) for details.

## 🐛 Troubleshooting

### Container exits immediately
- For HTTP mode: Check logs with `docker-compose -f docker-compose.http.yml logs`
- Ensure `MCP_MODE=http` is set in environment

### Cannot connect to HTTP endpoints
- Verify container is running: `docker ps`
- Check port mapping: `docker port <container_name>`
- Test health: `curl http://localhost:8000/api/health`

### Embedding model errors
- Models are pre-downloaded during build
- If issues persist, rebuild: `docker-compose build --no-cache`

## 🙏 Credits

Special thanks to **Joe Esposito** for identifying and helping fix the Docker setup issues!
