@echo off
REM Claude Code Memory Awareness Hooks - Windows Installation Batch Wrapper
REM This batch file wraps the PowerShell installation script for easy execution

setlocal enabledelayedexpansion

echo.
echo ========================================================
echo  Claude Code Memory Awareness Hooks - Windows Installer
echo ========================================================
echo.

REM Check for help flag
if "%1"=="-h" goto :show_help
if "%1"=="--help" goto :show_help
if "%1"=="/?" goto :show_help

REM Check for Node.js
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo.
    echo Please install Node.js from: https://nodejs.org/
    echo After installation, restart this command prompt and try again.
    echo.
    pause
    exit /b 1
)

REM Get the directory of this script and resolve paths
set SCRIPT_DIR=%~dp0
set PS_SCRIPT=%SCRIPT_DIR%install_claude_hooks_windows.ps1

echo [INFO] Script directory: %SCRIPT_DIR%
echo [INFO] PowerShell script: %PS_SCRIPT%
echo.

REM Check if PowerShell script exists
if not exist "%PS_SCRIPT%" (
    echo [ERROR] PowerShell script not found: %PS_SCRIPT%
    echo.
    echo Please ensure you're running this from the scripts directory
    echo of the mcp-memory-service repository.
    echo.
    echo Expected repository structure:
    echo   mcp-memory-service/
    echo     ├── scripts/
    echo     │   ├── install_claude_hooks_windows.bat  ^<-- This file
    echo     │   └── install_claude_hooks_windows.ps1
    echo     └── claude-hooks/
    echo         ├── core/
    echo         ├── utilities/
    echo         └── config.json
    echo.
    pause
    exit /b 1
)

REM Forward all arguments to PowerShell script for robust argument handling
set "PS_ARGS=%*"

REM Check PowerShell execution policy
powershell -Command "Get-ExecutionPolicy" >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] PowerShell is not available
    echo.
    pause
    exit /b 1
)

REM Run the PowerShell script with appropriate execution policy
echo Running installation script...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %PS_ARGS%

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Installation failed with error code: %errorlevel%
    echo.
    echo Troubleshooting tips:
    echo   1. Run this script as Administrator if permission errors occur
    echo   2. Ensure Node.js is installed and in PATH
    echo   3. Check that the claude-hooks directory exists in the repository
    echo   4. Try running the PowerShell script directly:
    echo      powershell -ExecutionPolicy Bypass -File "%PS_SCRIPT%"
    echo.
    pause
    exit /b %errorlevel%
)

echo.
echo Installation process completed successfully!
echo.
pause
exit /b 0

:show_help
echo Usage: install_claude_hooks_windows.bat [options]
echo.
echo Options:
echo   -h, --help, /?     Show this help message
echo   -uninstall         Remove installed hooks
echo   -test              Run tests only
echo.
echo Examples:
echo   install_claude_hooks_windows.bat              - Install hooks
echo   install_claude_hooks_windows.bat -uninstall   - Remove hooks
echo   install_claude_hooks_windows.bat -test        - Test installation
echo.
echo This script will:
echo   1. Check for Node.js installation
echo   2. Create Claude Code hooks directory at %%USERPROFILE%%\.claude-code\hooks
echo   3. Copy all hook files from the claude-hooks directory
echo   4. Configure the memory service endpoint (default: narrowbox.local:8443)
echo   5. Run integration tests to verify installation
echo.
echo For more information, see the README.md in the claude-hooks directory.
echo.
pause
exit /b 0