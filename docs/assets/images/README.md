# MCP Memory Service Screenshots

This directory contains screenshots and visual assets for the MCP Memory Service documentation.

## Dashboard Screenshots

### v3.3.0 Dashboard Features

The modern dashboard includes:

![Dashboard Layout](dashboard-v3.3.0-preview.png)

**Key Features:**
- **Modern Design**: Gradient backgrounds with professional card layout
- **Live Statistics**: Real-time server metrics including:
  - Total memories count
  - Embedding model information
  - Server health status
  - Response time metrics
- **Interactive Endpoints**: Organized API documentation with hover effects
- **Tech Stack Badges**: Visual representation of FastAPI, SQLite-vec, PyTorch, etc.
- **Responsive Layout**: Works on desktop and mobile devices
- **Auto-Refresh**: Stats update every 30 seconds

## Access URLs

- **Dashboard**: http://localhost:8000
- **mDNS**: http://mcp-memory-service.local:8000
- **API Docs**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

## Screenshot Instructions

To capture the dashboard for documentation:

1. Ensure the HTTP server is running: `python scripts/run_http_server.py`
2. Wait for stats to load (shows actual memory count)
3. Take full-page screenshot at 1920x1080 resolution
4. Save as `dashboard-v3.3.0.png` in this directory
5. Update README.md references accordingly

## File Naming Convention

- `dashboard-v{version}.png` - Main dashboard screenshots
- `api-docs-v{version}.png` - API documentation screenshots
- `mobile-v{version}.png` - Mobile responsive views
- `feature-{name}-v{version}.png` - Specific feature screenshots
