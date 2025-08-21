# Docker Setup for MCP Memory Service

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