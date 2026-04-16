#Requires -Version 5.1
<#
.SYNOPSIS
    Management script for MCP Memory HTTP Server scheduled task.

.DESCRIPTION
    Provides commands to manage the HTTP server scheduled task:
    - status: Show task and server status
    - start: Start the server
    - stop: Stop the server
    - restart: Restart the server
    - logs: View server logs
    - health: Check server health endpoint

.PARAMETER Action
    The action to perform: status, start, stop, restart, logs, health

.PARAMETER Lines
    Number of log lines to show (default: 50)

.EXAMPLE
    .\manage_service.ps1 status
    Shows the current status of the task and server.

.EXAMPLE
    .\manage_service.ps1 logs -Lines 100
    Shows the last 100 lines of the log file.

.NOTES
    File Name      : manage_service.ps1
    Location       : scripts/service/windows/
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("status", "start", "stop", "restart", "logs", "health", "help")]
    [string]$Action = "help",

    [int]$Lines = 50
)

$ErrorActionPreference = "Stop"

# Load shared server-config helper (reads host/port/https from .env)
. "$PSScriptRoot\lib\server-config.ps1"
$ServerConfig = Get-McpServerConfig
Enable-McpSelfSignedCertBypass
# Per-call extras (adds -SkipCertificateCheck on PS 7+ with HTTPS) because
# Invoke-WebRequest on PS 7+ uses HttpClient which ignores ServicePointManager.
$McpWebExtras = Get-McpWebRequestExtraParams -HttpsEnabled $ServerConfig.HttpsEnabled

# Configuration
$TaskName = "MCPMemoryHTTPServer"
$LogFile = Join-Path $env:LOCALAPPDATA "mcp-memory\logs\http-server.log"
$PidFile = Join-Path $env:LOCALAPPDATA "mcp-memory\http-server.pid"
# Derived from .env — supports both HTTP and HTTPS, any port
$HealthUrl = $ServerConfig.HealthUrl
$DashboardUrl = $ServerConfig.DashboardUrl
$ServerPort = $ServerConfig.Port

function Show-Help {
    Write-Host ""
    Write-Host "MCP Memory HTTP Server Management" -ForegroundColor Cyan
    Write-Host "=================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\manage_service.ps1 <action>" -ForegroundColor White
    Write-Host ""
    Write-Host "Actions:" -ForegroundColor Yellow
    Write-Host "  status   - Show task and server status"
    Write-Host "  start    - Start the server"
    Write-Host "  stop     - Stop the server"
    Write-Host "  restart  - Restart the server"
    Write-Host "  logs     - View server logs (use -Lines N for more/less)"
    Write-Host "  health   - Check server health endpoint"
    Write-Host "  help     - Show this help"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\manage_service.ps1 status"
    Write-Host "  .\manage_service.ps1 logs -Lines 100"
    Write-Host ""
}

function Get-ServerStatus {
    # Check task status
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    $TaskInfo = $null
    if ($Task) {
        $TaskInfo = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue
    }

    # Check process via PID file
    $ProcessRunning = $false
    $ProcessPid = $null
    if (Test-Path $PidFile) {
        $StoredPid = Get-Content $PidFile -ErrorAction SilentlyContinue
        if ($StoredPid) {
            $Process = Get-Process -Id $StoredPid -ErrorAction SilentlyContinue
            if ($Process -and $Process.ProcessName -like "*python*") {
                $ProcessRunning = $true
                $ProcessPid = $StoredPid
            }
        }
    }

    # Check HTTP health (public, no auth — fast liveness check)
    $HttpHealthy = $false
    $HealthResponse = $null
    try {
        $params = @{
            Uri = $HealthUrl
            TimeoutSec = 3
            UseBasicParsing = $true
            ErrorAction = 'SilentlyContinue'
        } + $McpWebExtras
        $Response = Invoke-WebRequest @params
        if ($Response.StatusCode -eq 200) {
            $HttpHealthy = $true
            $HealthResponse = $Response.Content | ConvertFrom-Json
        }
    } catch {
        # Server not responding
    }

    # Fetch version + backend from the authenticated /api/health/detailed
    # endpoint. The public /api/health response only contains status since the
    # v10.21.0 security hardening (GHSA-73hc-m4hx-79pj), so the human-readable
    # status display needs the authenticated endpoint for version/backend.
    # Degrades gracefully if MCP_API_KEY is not set in .env.
    $DetailedHealth = $null
    if ($HttpHealthy) {
        $ApiKey = Get-McpApiKey
        if ($ApiKey) {
            try {
                $DetailedUrl = "$($ServerConfig.BaseUrl)/api/health/detailed"
                $detailedParams = @{
                    Uri = $DetailedUrl
                    Headers = @{ "Authorization" = "Bearer $ApiKey" }
                    TimeoutSec = 3
                    UseBasicParsing = $true
                    ErrorAction = 'SilentlyContinue'
                } + $McpWebExtras
                $DetailedResponse = Invoke-WebRequest @detailedParams
                if ($DetailedResponse.StatusCode -eq 200) {
                    $DetailedHealth = $DetailedResponse.Content | ConvertFrom-Json
                }
            } catch {
                # /health/detailed unreachable — keep fields empty, don't fail status
            }
        }
    }

    return @{
        Task = $Task
        TaskInfo = $TaskInfo
        ProcessRunning = $ProcessRunning
        ProcessPid = $ProcessPid
        HttpHealthy = $HttpHealthy
        HealthResponse = $HealthResponse
        DetailedHealth = $DetailedHealth
    }
}

function Show-Status {
    Write-Host ""
    Write-Host "MCP Memory HTTP Server Status" -ForegroundColor Cyan
    Write-Host "=============================" -ForegroundColor Cyan
    Write-Host ""

    $Status = Get-ServerStatus

    # Task Status
    Write-Host "Scheduled Task:" -ForegroundColor Yellow
    if ($Status.Task) {
        Write-Host "  Name:   $TaskName"
        Write-Host "  State:  $($Status.Task.State)"
        if ($Status.TaskInfo) {
            Write-Host "  Last Run: $($Status.TaskInfo.LastRunTime)"
            Write-Host "  Next Run: $($Status.TaskInfo.NextRunTime)"
            $ResultText = switch ($Status.TaskInfo.LastTaskResult) {
                0 { "Success" }
                267009 { "Running" }
                267014 { "Stopped" }
                default { "Code: $($Status.TaskInfo.LastTaskResult)" }
            }
            Write-Host "  Result: $ResultText"
        }
    } else {
        Write-Host "  [NOT INSTALLED]" -ForegroundColor Red
        Write-Host "  Run .\install_scheduled_task.ps1 to install"
    }

    Write-Host ""

    # Process Status
    Write-Host "Server Process:" -ForegroundColor Yellow
    if ($Status.ProcessRunning) {
        Write-Host "  Status: " -NoNewline
        Write-Host "RUNNING" -ForegroundColor Green
        Write-Host "  PID:    $($Status.ProcessPid)"
    } else {
        Write-Host "  Status: " -NoNewline
        Write-Host "NOT RUNNING" -ForegroundColor Red
    }

    Write-Host ""

    # HTTP Health
    Write-Host "HTTP Endpoint:" -ForegroundColor Yellow
    if ($Status.HttpHealthy) {
        Write-Host "  Status:  " -NoNewline
        Write-Host "HEALTHY" -ForegroundColor Green
        Write-Host "  URL:     $DashboardUrl"
        # Version + backend come from /api/health/detailed (authenticated).
        # Public /api/health has not exposed these since v10.21.0.
        if ($Status.DetailedHealth) {
            $backendValue = $null
            if ($Status.DetailedHealth.storage) {
                # 'backend' is the canonical field per DetailedHealthResponse schema.
                # 'storage_backend' may appear in backend-specific stats as a fallback.
                $backendValue = $Status.DetailedHealth.storage.backend
                if (-not $backendValue) {
                    $backendValue = $Status.DetailedHealth.storage.storage_backend
                }
            }
            Write-Host "  Version: $($Status.DetailedHealth.version)"
            Write-Host "  Backend: $backendValue"
        } else {
            Write-Host "  Version: (unavailable - set MCP_API_KEY in .env for details)" -ForegroundColor DarkGray
            Write-Host "  Backend: (unavailable - set MCP_API_KEY in .env for details)" -ForegroundColor DarkGray
        }
    } else {
        Write-Host "  Status: " -NoNewline
        Write-Host "NOT RESPONDING" -ForegroundColor Red
    }

    Write-Host ""

    # Log file info
    if (Test-Path $LogFile) {
        $LogInfo = Get-Item $LogFile
        Write-Host "Log File:" -ForegroundColor Yellow
        Write-Host "  Path: $LogFile"
        Write-Host "  Size: $([math]::Round($LogInfo.Length / 1KB, 2)) KB"
        Write-Host "  Modified: $($LogInfo.LastWriteTime)"
    }

    Write-Host ""
}

function Start-Server {
    $Status = Get-ServerStatus

    if ($Status.HttpHealthy) {
        Write-Host "[INFO] Server is already running." -ForegroundColor Yellow
        return
    }

    if (-not $Status.Task) {
        Write-Host "[ERROR] Scheduled task is not installed. Run .\install_scheduled_task.ps1 first." -ForegroundColor Red
        return
    }

    Write-Host "[INFO] Starting server..." -ForegroundColor Green
    Start-ScheduledTask -TaskName $TaskName

    # Wait and check
    Write-Host "[INFO] Waiting for server to start..." -ForegroundColor Yellow
    for ($i = 1; $i -le 10; $i++) {
        Start-Sleep -Seconds 1
        try {
            $params = @{
                Uri = $HealthUrl
                TimeoutSec = 2
                UseBasicParsing = $true
                ErrorAction = 'SilentlyContinue'
            } + $McpWebExtras
            $Response = Invoke-WebRequest @params
            if ($Response.StatusCode -eq 200) {
                Write-Host "[SUCCESS] Server started successfully!" -ForegroundColor Green
                Write-Host "Dashboard: $DashboardUrl" -ForegroundColor Cyan
                return
            }
        } catch {
            Write-Host "." -NoNewline
        }
    }

    Write-Host ""
    Write-Host "[WARN] Server may still be starting. Check status in a moment." -ForegroundColor Yellow
}

function Stop-Server {
    $Status = Get-ServerStatus

    if (-not $Status.ProcessRunning -and -not $Status.HttpHealthy) {
        Write-Host "[INFO] Server is not running." -ForegroundColor Yellow
        return
    }

    Write-Host "[INFO] Stopping server..." -ForegroundColor Yellow

    # Stop scheduled task
    if ($Status.Task -and $Status.Task.State -eq "Running") {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    }

    # Kill process if still running
    if ($Status.ProcessPid) {
        Stop-Process -Id $Status.ProcessPid -Force -ErrorAction SilentlyContinue
    }

    # Also try via port (from .env)
    try {
        $Connection = Get-NetTCPConnection -LocalPort $ServerPort -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }
        if ($Connection) {
            Stop-Process -Id $Connection.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    } catch {
        # Ignore
    }

    # Clean up PID file
    if (Test-Path $PidFile) {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }

    Write-Host "[SUCCESS] Server stopped." -ForegroundColor Green
}

function Restart-Server {
    Write-Host "[INFO] Restarting server..." -ForegroundColor Yellow
    Stop-Server
    Start-Sleep -Seconds 2
    Start-Server
}

function Show-Logs {
    if (-not (Test-Path $LogFile)) {
        Write-Host "[INFO] No log file found at: $LogFile" -ForegroundColor Yellow
        return
    }

    Write-Host "Last $Lines lines of $LogFile" -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
    Get-Content $LogFile -Tail $Lines
}

function Check-Health {
    Write-Host "[INFO] Checking health endpoint..." -ForegroundColor Yellow
    try {
        $params = @{
            Uri = $HealthUrl
            TimeoutSec = 5
            UseBasicParsing = $true
        } + $McpWebExtras
        $Response = Invoke-WebRequest @params
        Write-Host "[SUCCESS] Server is healthy!" -ForegroundColor Green
        Write-Host ""
        $Response.Content | ConvertFrom-Json | Format-List
    } catch {
        Write-Host "[ERROR] Server is not responding: $_" -ForegroundColor Red
    }
}

# Main execution
switch ($Action) {
    "status" { Show-Status }
    "start" { Start-Server }
    "stop" { Stop-Server }
    "restart" { Restart-Server }
    "logs" { Show-Logs }
    "health" { Check-Health }
    "help" { Show-Help }
    default { Show-Help }
}
