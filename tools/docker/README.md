# Docker Setup for MCP Memory Service

## Tags

| Tag | Description | Size delta vs `:latest` |
|-----|-------------|------------------------|
| `:latest` | Standard image, full feature set | baseline |
| `:slim` | CPU-only, no PyTorch/CUDA | ~90% smaller |
| `:quality-cpu` | Slim + pre-exported ONNX quality models, no runtime PyTorch | ~250-300 MB larger than `:slim` |

### `:quality-cpu` — local quality scoring without runtime PyTorch

Pull this tag to get local ONNX quality scoring (`ms-marco-MiniLM-L-6-v2` and
`nvidia-quality-classifier-deberta`) out-of-the-box, without managing the ONNX
export yourself and without shipping `torch`/`transformers` in your deployment
container.

```bash
docker pull doobidoo/mcp-memory-service:quality-cpu

# Verify both quality models load from baked cache (no export triggered):
docker run --rm doobidoo/mcp-memory-service:quality-cpu \
  python -c "
from mcp_memory_service.quality.onnx_ranker import get_onnx_ranker_model
print(get_onnx_ranker_model('ms-marco-MiniLM-L-6-v2'))
print(get_onnx_ranker_model('nvidia-quality-classifier-deberta'))
print('Both quality models loaded from baked ONNX cache')
"
```

The image sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` at runtime, so
no live model download can occur. Quality scoring uses only the pre-baked
artifacts at `/root/.cache/mcp_memory/onnx_models/`.

**Homelab use case:** ideal for Raspberry Pi, NAS, or any always-on box where
you want quality scoring without pulling a 2 GB PyTorch wheel every restart.

## 🚀 Quick Start

Choose your mode:

### MCP Protocol Mode (for Claude Desktop, VS Code)
```bash
docker-compose up -d
```

### HTTP API Mode (for REST API, Web Dashboard)
```bash
docker-compose -f docker-compose.http.yml up -d
```

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
| `MCP_API_KEY` | API key for HTTP mode | `your-secure-api-key-here` |
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
