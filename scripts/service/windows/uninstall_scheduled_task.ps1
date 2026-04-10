#Requires -Version 5.1
<#
.SYNOPSIS
    Uninstalls the MCP Memory HTTP Server scheduled task.

.DESCRIPTION
    Stops the running server (if any) and removes the scheduled task from Windows Task Scheduler.
    Optionally cleans up log files.

.PARAMETER CleanupLogs
    Also remove log files from %LOCALAPPDATA%\mcp-memory\logs

.PARAMETER Force
    Don't prompt for confirmation.

.EXAMPLE
    .\uninstall_scheduled_task.ps1
    Uninstalls the task (prompts for confirmation).

.EXAMPLE
    .\uninstall_scheduled_task.ps1 -Force -CleanupLogs
    Uninstalls without prompting and removes logs.

.NOTES
    File Name      : uninstall_scheduled_task.ps1
    Location       : scripts/service/windows/
#>

param(
    [switch]$CleanupLogs,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Configuration
$TaskName = "MCPMemoryHTTPServer"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item "$ScriptDir\..\..\..").FullName
$LogDir = Join-Path $env:LOCALAPPDATA "mcp-memory\logs"
$PidFile = Join-Path $env:LOCALAPPDATA "mcp-memory\http-server.pid"

# Load shared server-config helper (reads host/port/https from .env)
. "$ScriptDir\lib\server-config.ps1"
$ServerConfig = Get-McpServerConfig -ProjectRoot $ProjectRoot
$ServerPort = $ServerConfig.Port

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MCP Memory HTTP Server - Uninstaller " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if task exists
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if (-not $ExistingTask) {
    Write-Host "[INFO] Task '$TaskName' is not installed." -ForegroundColor Yellow
    exit 0
}

# Confirm uninstall
if (-not $Force) {
    $Response = Read-Host "Are you sure you want to uninstall the MCP Memory HTTP Server task? (y/N)"
    if ($Response -ne 'y' -and $Response -ne 'Y') {
        Write-Host "[INFO] Uninstallation cancelled." -ForegroundColor Yellow
        exit 0
    }
}

# Stop running task
Write-Host "[INFO] Stopping running task..." -ForegroundColor Yellow
try {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
} catch {
    # Task might not be running
}

# Kill any lingering Python process
if (Test-Path $PidFile) {
    $StoredPid = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($StoredPid) {
        Write-Host "[INFO] Stopping server process (PID: $StoredPid)..." -ForegroundColor Yellow
        Stop-Process -Id $StoredPid -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# Also try to stop via port (from .env)
try {
    $Connection = Get-NetTCPConnection -LocalPort $ServerPort -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }
    if ($Connection) {
        Write-Host "[INFO] Stopping process listening on port $ServerPort..." -ForegroundColor Yellow
        Stop-Process -Id $Connection.OwningProcess -Force -ErrorAction SilentlyContinue
    }
} catch {
    # Port might not be in use
}

# Wait a moment for cleanup
Start-Sleep -Seconds 1

# Unregister the task
Write-Host "[INFO] Removing scheduled task..." -ForegroundColor Yellow
try {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "[SUCCESS] Scheduled task removed." -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to remove scheduled task: $_" -ForegroundColor Red
    exit 1
}

# Cleanup logs if requested
if ($CleanupLogs) {
    Write-Host "[INFO] Cleaning up log files..." -ForegroundColor Yellow
    if (Test-Path $LogDir) {
        Remove-Item "$LogDir\http-server*" -Force -ErrorAction SilentlyContinue
        Write-Host "[SUCCESS] Log files removed." -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "[SUCCESS] MCP Memory HTTP Server has been uninstalled." -ForegroundColor Green
Write-Host ""
Write-Host "Note: Your memory database and configuration are preserved." -ForegroundColor Cyan
Write-Host "To reinstall, run: .\install_scheduled_task.ps1" -ForegroundColor Cyan
