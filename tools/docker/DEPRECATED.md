# Deprecated Docker Files

The following Docker files are deprecated as of v5.0.4 and will be removed in v6.0.0:

## Deprecated Files

### 1. `docker-compose.standalone.yml`
- **Replaced by**: `docker-compose.http.yml`
- **Reason**: Confusing name, mixed ChromaDB/SQLite configs, incorrect entrypoint for HTTP mode
- **Migration**: Use `docker-compose.http.yml` for HTTP/API access

### 2. `docker-compose.uv.yml`
- **Replaced by**: UV is now built into the main Dockerfile
- **Reason**: UV support should be in the image, not a separate compose file
- **Migration**: UV is automatically available in all configurations

### 3. `docker-compose.pythonpath.yml`
- **Replaced by**: Fixed PYTHONPATH in main Dockerfile
- **Reason**: PYTHONPATH fix belongs in Dockerfile, not compose variant
- **Migration**: All compose files now have correct PYTHONPATH=/app/src

### 4. `docker-entrypoint-persistent.sh`
- **Replaced by**: `docker-entrypoint-unified.sh`
- **Reason**: Overcomplicated, doesn't support HTTP mode, named pipes unnecessary
- **Migration**: Use unified entrypoint with MCP_MODE environment variable

## New Simplified Structure

Use one of these two configurations:

1. **MCP Protocol Mode** (for Claude Desktop, VS Code):
   ```bash
   docker-compose up -d
   ```

2. **HTTP/API Mode** (for web access, REST API):
   ```bash
   docker-compose -f docker-compose.http.yml up -d
   ```

## Timeline

- **v5.0.4**: Files marked as deprecated, new structure introduced
- **v5.1.0**: Warning messages added when using deprecated files
- **v6.0.0**: Deprecated files removed

## Credits

Thanks to Joe Esposito for identifying the Docker setup issues that led to this simplification.