@echo off
REM Start the MCP Memory Service HTTP server in the background on Windows

echo Starting MCP Memory Service HTTP server...

REM Check if server is already running
uv run python scripts\server\check_http_server.py -q
if %errorlevel% == 0 (
    echo HTTP server is already running!
    uv run python scripts\server\check_http_server.py
    exit /b 0
)

REM Start the server in a new window
start "MCP Memory HTTP Server" uv run python scripts\server\run_http_server.py

REM Wait up to 5 seconds for the server to start
FOR /L %%i IN (1,1,5) DO (
    timeout /t 1 /nobreak >nul
    uv run python scripts\server\check_http_server.py -q
    if %errorlevel% == 0 (
        echo.
        echo [OK] HTTP server started successfully!
        uv run python scripts\server\check_http_server.py
        goto :eof
    )
)

echo.
echo [WARN] Server did not start within 5 seconds. Check the server window for errors.
