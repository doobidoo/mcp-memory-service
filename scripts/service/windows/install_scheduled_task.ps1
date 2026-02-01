#Requires -Version 5.1
<#
.SYNOPSIS
    Installs the MCP Memory HTTP Server as a Windows Scheduled Task.

.DESCRIPTION
    Creates a scheduled task that automatically starts the HTTP server when the user logs in.
    The server runs in the background without a visible window and automatically restarts on failure.

.PARAMETER Force
    Overwrite existing task if it exists.

.EXAMPLE
    .\install_scheduled_task.ps1
    Installs the scheduled task (prompts if task already exists).

.EXAMPLE
    .\install_scheduled_task.ps1 -Force
    Overwrites existing task without prompting.

.NOTES
    File Name      : install_scheduled_task.ps1
    Prerequisite   : Windows 10/11, PowerShell 5.1+
    Location       : scripts/service/windows/
#>

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Configuration
$TaskName = "MCPMemoryHTTPServer"
$TaskDescription = "MCP Memory Service HTTP Server - Provides dashboard and REST API for memory management"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item "$ScriptDir\..\..\..").FullName
$WrapperScript = Join-Path $ScriptDir "run_http_server_background.ps1"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MCP Memory HTTP Server - Installer   " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Task Name:     $TaskName"
Write-Host "Project Root:  $ProjectRoot"
Write-Host "Wrapper Script: $WrapperScript"
Write-Host ""

# Check if wrapper script exists
if (-not (Test-Path $WrapperScript)) {
    Write-Host "[ERROR] Wrapper script not found: $WrapperScript" -ForegroundColor Red
    exit 1
}

# Check if task already exists
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($ExistingTask) {
    if ($Force) {
        Write-Host "[INFO] Removing existing task (--Force specified)..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    } else {
        Write-Host "[WARN] Task '$TaskName' already exists." -ForegroundColor Yellow
        $Response = Read-Host "Do you want to overwrite it? (y/N)"
        if ($Response -ne 'y' -and $Response -ne 'Y') {
            Write-Host "[INFO] Installation cancelled." -ForegroundColor Yellow
            exit 0
        }
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
}

Write-Host "[INFO] Creating scheduled task..." -ForegroundColor Green

# Create the action (use full path — Task Scheduler has minimal PATH)
$PowerShellPath = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
if (-not (Test-Path $PowerShellPath)) {
    # Fallback: resolve from current session
    $PowerShellPath = (Get-Command powershell.exe).Source
}
$Action = New-ScheduledTaskAction -Execute $PowerShellPath `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$WrapperScript`"" `
    -WorkingDirectory $ProjectRoot

# Create the trigger (at user logon)
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Create principal (run as current user)
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

# Settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0) `
    -MultipleInstances IgnoreNew `
    -Priority 7

# Register the task
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Description $TaskDescription `
        -Action $Action `
        -Trigger $Trigger `
        -Principal $Principal `
        -Settings $Settings | Out-Null

    Write-Host ""
    Write-Host "[SUCCESS] Scheduled task created successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Task Configuration:" -ForegroundColor Cyan
    Write-Host "  - Name:    $TaskName"
    Write-Host "  - Trigger: At user logon ($env:USERNAME)"
    Write-Host "  - Action:  Run HTTP server in background"
    Write-Host "  - Restart: 3 attempts with 1 minute delay"
    Write-Host ""
    Write-Host "Management Commands:" -ForegroundColor Cyan
    Write-Host "  Status:    .\manage_service.ps1 status"
    Write-Host "  Start:     .\manage_service.ps1 start"
    Write-Host "  Stop:      .\manage_service.ps1 stop"
    Write-Host "  Logs:      .\manage_service.ps1 logs"
    Write-Host ""
    Write-Host "The server will automatically start at your next login." -ForegroundColor Yellow
    Write-Host ""

    # Offer to start now
    $StartNow = Read-Host "Do you want to start the server now? (Y/n)"
    if ($StartNow -ne 'n' -and $StartNow -ne 'N') {
        Write-Host "[INFO] Starting server..." -ForegroundColor Green
        Start-ScheduledTask -TaskName $TaskName
        Start-Sleep -Seconds 3

        # Check if running
        $TaskInfo = Get-ScheduledTaskInfo -TaskName $TaskName
        if ($TaskInfo.LastTaskResult -eq 0 -or $TaskInfo.LastTaskResult -eq 267009) {
            Write-Host "[SUCCESS] Server is starting. Check http://127.0.0.1:8000/ in a few seconds." -ForegroundColor Green
        } else {
            Write-Host "[WARN] Server may have failed to start. Check logs with: .\manage_service.ps1 logs" -ForegroundColor Yellow
        }
    }

} catch {
    Write-Host "[ERROR] Failed to create scheduled task: $_" -ForegroundColor Red
    exit 1
}
