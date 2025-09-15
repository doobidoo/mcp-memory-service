@echo off
REM MCP Memory Service HTTP Debug Mode Startup Script
REM This script starts the MCP Memory Service in HTTP mode for debugging and testing

echo ========================================
echo MCP Memory Service HTTP Debug Mode
echo Using uv for dependency management
echo ========================================

REM Check if uv is available
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: uv is not installed or not in PATH
    echo Please install uv: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

echo uv version:
uv --version

REM Install dependencies using uv sync (recommended)
echo.
echo Installing dependencies...
echo This may take a few minutes on first run...
echo Installing core dependencies...
uv sync --active

REM Check if installation was successful
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    echo Please check the error messages above
    pause
    exit /b 1
)

echo Dependencies installed successfully!

REM Verify Python can import the service
echo.
echo Verifying installation...
python -c "import sys; sys.path.insert(0, 'src'); import mcp_memory_service; print('✓ MCP Memory Service imported successfully')"
if %errorlevel% neq 0 (
    echo ERROR: Failed to import MCP Memory Service
    echo Please check the error messages above
    pause
    exit /b 1
)

REM Set environment variables for HTTP mode
set MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
set MCP_HTTP_ENABLED=true
set MCP_HTTP_PORT=8000
set MCP_HTTPS_ENABLED=false
set MCP_MDNS_ENABLED=true
set MCP_MDNS_SERVICE_NAME=MCP-Memory-Service-Debug
set MCP_MEMORY_USE_ONNX=true

REM Fix Transformers cache warning
set HF_HOME=%USERPROFILE%\.cache\huggingface
set TRANSFORMERS_CACHE=%USERPROFILE%\.cache\huggingface\transformers

REM Optional: Set API key for security (change this!)
set MCP_API_KEY=debug-api-key-12345

REM Optional: Enable debug logging
set MCP_DEBUG=true
set LOG_LEVEL=DEBUG

echo Configuration:
echo   Storage Backend: %MCP_MEMORY_STORAGE_BACKEND%
echo   HTTP Port: %MCP_HTTP_PORT%
echo   HTTPS Enabled: %MCP_HTTPS_ENABLED%
echo   mDNS Enabled: %MCP_MDNS_ENABLED%
echo   Service Name: %MCP_MDNS_SERVICE_NAME%
echo   ONNX Embeddings: %MCP_MEMORY_USE_ONNX%
echo   API Key Set: Yes
echo   Debug Mode: %MCP_DEBUG%
echo   Log Level: %LOG_LEVEL%
echo.

echo Starting MCP Memory Service...
echo.
echo Service will be available at:
echo   HTTP: http://localhost:%MCP_HTTP_PORT%
echo   API: http://localhost:%MCP_HTTP_PORT%/api
echo   Health: http://localhost:%MCP_HTTP_PORT%/api/health
echo   Dashboard: http://localhost:%MCP_HTTP_PORT%/dashboard
echo.
echo Press Ctrl+C to stop the service
echo.
echo ========================================
echo Starting MCP Memory Service...
echo ========================================

REM Start the service using Python directly (required for HTTP mode)
echo Starting service with Python...
echo Note: Using Python directly for HTTP server mode
uv run python run_server.py
