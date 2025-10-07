@echo off
REM Start the MCP Memory Service HTTP server in the background on Windows

echo Starting MCP Memory Service HTTP server...

REM Check if server is already running
uv run python scripts\server\check_http_server.py -q
if %errorlevel% == 0 (
    echo HTTP server is already running!
    uv run python scripts\server\check_http_server.py -v
    exit /b 0
)

REM Start the server in a new window
start "MCP Memory HTTP Server" uv run python scripts\server\run_http_server.py

REM Wait a moment for server to start
timeout /t 3 /nobreak >nul

REM Check if it started successfully
uv run python scripts\server\check_http_server.py -v
if %errorlevel% == 0 (
    echo.
    echo [OK] HTTP server started successfully!
) else (
    echo.
    echo [WARN] Server may still be starting... Check the server window.
)
