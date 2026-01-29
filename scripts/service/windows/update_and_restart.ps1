#Requires -Version 5.1
<#
.SYNOPSIS
    One-command update and restart for MCP Memory Service on Windows.

.DESCRIPTION
    Streamlined workflow to update MCP Memory Service:
    1. Pull latest changes from git
    2. Install updated dependencies (editable mode)
    3. Restart HTTP dashboard server (via scheduled task or direct)
    4. Verify version and health

    Target time: <2 minutes (typical: 60-90 seconds)

.PARAMETER NoRestart
    Update code but don't restart the server

.PARAMETER Force
    Force update even with uncommitted changes (auto-stash)

.EXAMPLE
    .\update_and_restart.ps1
    Standard update and restart with prompts

.EXAMPLE
    .\update_and_restart.ps1 -Force
    Force update and auto-stash uncommitted changes

.EXAMPLE
    .\update_and_restart.ps1 -NoRestart
    Update code only, don't restart server

.NOTES
    File Name      : update_and_restart.ps1
    Author         : Heinrich Krupp
    Prerequisite   : Git, Python/uv, pip
    Copyright 2026 - Licensed under Apache License 2.0
#>

param(
    [switch]$NoRestart,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$StartTime = Get-Date

# Configuration
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$ManageServiceScript = Join-Path $PSScriptRoot "manage_service.ps1"
# HTTPS is enabled by default - use https and skip certificate validation for self-signed certs
$HealthUrl = "https://127.0.0.1:8000/api/health"

# Skip SSL certificate validation for self-signed certificates
# Compatible with both PowerShell 5.1 and 7+
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12

# Check if we're on PowerShell 7+ (has -SkipCertificateCheck parameter)
$Script:SkipCertCheckSupported = $PSVersionTable.PSVersion.Major -ge 7

# For PowerShell 5.1, use callback approach
if (-not $Script:SkipCertCheckSupported) {
    if (-not ("TrustAllCertsPolicy" -as [type])) {
        Add-Type @"
using System.Net;
using System.Security.Cryptography.X509Certificates;
public class TrustAllCertsPolicy : ICertificatePolicy {
    public bool CheckValidationResult(ServicePoint srvPoint, X509Certificate certificate,
        WebRequest request, int certificateProblem) { return true; }
}
"@
    }
    [System.Net.ServicePointManager]::CertificatePolicy = New-Object TrustAllCertsPolicy
}

# Color helpers
function Write-InfoLog { param($Message) Write-Host "[i] $Message" -ForegroundColor Cyan }
function Write-SuccessLog { param($Message) Write-Host "[+] $Message" -ForegroundColor Green }
function Write-WarningLog { param($Message) Write-Host "[!] $Message" -ForegroundColor Yellow }
function Write-ErrorLog { param($Message) Write-Host "[x] $Message" -ForegroundColor Red }
function Write-StepLog { param($Message) Write-Host "`n>>> $Message" -ForegroundColor Blue }

function Get-ElapsedSeconds {
    return [int]((Get-Date) - $StartTime).TotalSeconds
}

function Get-CurrentVersion {
    $VersionFile = Join-Path $ProjectRoot "src\mcp_memory_service\_version.py"
    if (Test-Path $VersionFile) {
        $Content = Get-Content $VersionFile -Raw
        if ($Content -match '__version__\s*=\s*["\x27]([^"\x27]+)["\x27]') {
            return $Matches[1]
        }
    }
    return "unknown"
}

function Get-ServerVersion {
    try {
        $params = @{
            Uri = $HealthUrl
            TimeoutSec = 3
            ErrorAction = 'SilentlyContinue'
        }
        if ($Script:SkipCertCheckSupported) { $params['SkipCertificateCheck'] = $true }
        $Response = Invoke-RestMethod @params
        return $Response.version
    } catch {
        return "unknown"
    }
}

# Configure Git to use Windows OpenSSH (connects to Windows SSH agent)
function Initialize-GitSsh {
    # Windows OpenSSH path (use forward slashes for git compatibility)
    $WindowsSsh = "C:/Windows/System32/OpenSSH/ssh.exe"
    $WindowsSshAdd = "C:/Windows/System32/OpenSSH/ssh-add.exe"

    if (Test-Path $WindowsSsh) {
        # Set GIT_SSH_COMMAND to use Windows OpenSSH
        # This allows git to use the Windows SSH agent
        $env:GIT_SSH_COMMAND = "`"$WindowsSsh`""
    }

    # Ensure Windows SSH agent service is running
    $Service = Get-Service ssh-agent -ErrorAction SilentlyContinue
    if ($Service -and $Service.Status -ne 'Running') {
        try {
            Start-Service ssh-agent -ErrorAction Stop
            Write-InfoLog "Started Windows SSH agent service"
        } catch {
            Write-WarningLog "Could not start SSH agent service"
        }
    }

    # Check if SSH keys are loaded (with timeout to prevent hanging)
    if (Test-Path $WindowsSshAdd) {
        try {
            $Job = Start-Job -ScriptBlock { & "C:/Windows/System32/OpenSSH/ssh-add.exe" -l 2>&1 }
            $Completed = Wait-Job $Job -Timeout 3
            if ($Completed) {
                $KeyCheck = Receive-Job $Job
                Remove-Job $Job -Force
                if ($KeyCheck -match "no identities" -or $KeyCheck -match "error") {
                    Write-WarningLog "No SSH keys loaded in Windows SSH agent!"
                    Write-WarningLog "Run this command manually to add your key:"
                    Write-Host '  & "C:/Windows/System32/OpenSSH/ssh-add.exe" "$env:USERPROFILE\.ssh\id_ed25519"' -ForegroundColor Yellow
                    Write-Host ""
                }
            } else {
                Stop-Job $Job
                Remove-Job $Job -Force
                Write-WarningLog "SSH agent check timed out - agent may not be responding"
            }
        } catch {
            # Silently continue if SSH check fails
        }
    }
}

# Banner
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  MCP Memory Service - Update & Restart    " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $ProjectRoot

# Configure SSH for git operations
Initialize-GitSsh

# Step 1: Check for uncommitted changes
Write-StepLog "Checking repository status..."

$GitStatus = git status --porcelain 2>&1
if ($GitStatus -and -not $Force) {
    Write-WarningLog "You have uncommitted changes:"
    git status --short | Select-Object -First 10
    Write-Host ""

    $Response = Read-Host "Stash changes and continue? [y/N]"
    if ($Response -match '^[Yy]$') {
        Write-InfoLog "Stashing local changes..."
        $StashMessage = "Auto-stash before update $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        git stash push -m $StashMessage
        Write-SuccessLog "Changes stashed"
    } else {
        Write-ErrorLog "Update cancelled. Use -Force to override."
        exit 1
    }
}

# Step 2: Get current version
Write-StepLog "Recording current version..."

$CurrentVersion = Get-CurrentVersion
Write-InfoLog "Current version: $CurrentVersion"

# Step 3: Pull latest changes
Write-StepLog "Pulling latest changes from git..."

$BeforeCommit = git rev-parse HEAD

# Git writes info messages to stderr even on success, temporarily allow errors
$OldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$PullOutput = git pull --rebase 2>&1
$PullExitCode = $LASTEXITCODE
$ErrorActionPreference = $OldErrorActionPreference

if ($PullExitCode -ne 0) {
    Write-ErrorLog "Git pull failed with exit code $PullExitCode"
    Write-Host $PullOutput
    exit 1
}

$AfterCommit = git rev-parse HEAD

if ($BeforeCommit -eq $AfterCommit) {
    Write-InfoLog "Already up-to-date (no new commits)"
} else {
    $CommitCount = (git rev-list --count "$BeforeCommit..$AfterCommit").Trim()
    Write-SuccessLog "Pulled $CommitCount new commit(s)"

    Write-Host ""
    git log --oneline --graph "$BeforeCommit..$AfterCommit" | Select-Object -First 10
}

# Step 4: Get new version
$NewVersion = Get-CurrentVersion

if ($CurrentVersion -ne $NewVersion) {
    Write-InfoLog "Version change: $CurrentVersion -> $NewVersion"
}

# Step 5: Install dependencies (editable mode)
Write-StepLog "Installing dependencies (editable mode)..."

# Check for uv
$UseUv = Get-Command uv -ErrorAction SilentlyContinue
if ($UseUv) {
    Write-InfoLog "Using uv for faster installation..."
    & uv pip install -e . --quiet 2>&1 | Out-Null
} else {
    Write-InfoLog "Using pip for installation..."
    & pip install -e . --quiet 2>&1 | Out-Null
}

Write-SuccessLog "Dependencies installed"

# Verify installation - use same tool that was used for installation
try {
    if ($UseUv) {
        $PipShow = uv pip show mcp-memory-service 2>&1 | Out-String
    } else {
        $PipShow = pip show mcp-memory-service 2>&1 | Out-String
    }

    if ($PipShow -match 'Version:\s+(.+)') {
        $InstalledVersion = $Matches[1].Trim()
    } else {
        $InstalledVersion = "unknown"
    }

    if ($InstalledVersion -ne $NewVersion) {
        Write-WarningLog "Installation version mismatch! Expected: $NewVersion, Got: $InstalledVersion"
        Write-WarningLog "Retrying installation..."
        if ($UseUv) {
            & uv pip install -e . --reinstall --quiet 2>&1 | Out-Null
        } else {
            & pip install -e . --force-reinstall --quiet 2>&1 | Out-Null
        }

        # Re-check
        if ($UseUv) {
            $PipShow = uv pip show mcp-memory-service 2>&1 | Out-String
        } else {
            $PipShow = pip show mcp-memory-service 2>&1 | Out-String
        }
        if ($PipShow -match 'Version:\s+(.+)') {
            $InstalledVersion = $Matches[1].Trim()
        }
    }

    Write-InfoLog "Installed version: $InstalledVersion"
} catch {
    Write-WarningLog "Could not verify installation version"
}

# Step 6: Restart server (if requested)
if ($NoRestart) {
    Write-WarningLog "Skipping server restart (-NoRestart flag)"
} else {
    Write-StepLog "Restarting HTTP dashboard server..."

    if (Test-Path $ManageServiceScript) {
        # Use manage_service.ps1 for smart restart
        & $ManageServiceScript restart
    } else {
        Write-WarningLog "manage_service.ps1 not found, using manual restart..."

        # Fallback: manual restart
        # Kill any existing server processes
        Get-Process python -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like "*run_http_server*" } |
            Stop-Process -Force -ErrorAction SilentlyContinue

        Start-Sleep -Seconds 2

        # Start server in background
        $ServerScript = Join-Path $ProjectRoot "scripts\server\run_http_server.py"
        $LogFile = Join-Path $env:TEMP "mcp-memory-update.log"

        Start-Process python -ArgumentList $ServerScript -WindowStyle Hidden -RedirectStandardOutput $LogFile -RedirectStandardError $LogFile
        Start-Sleep -Seconds 8
    }

    # Step 7: Health check
    Write-StepLog "Verifying server health..."

    $MaxWait = 15
    $WaitCount = 0
    $Healthy = $false

    while ($WaitCount -lt $MaxWait) {
        try {
            $params = @{
                Uri = $HealthUrl
                TimeoutSec = 2
                ErrorAction = 'SilentlyContinue'
            }
            if ($Script:SkipCertCheckSupported) { $params['SkipCertificateCheck'] = $true }
            $HealthResponse = Invoke-RestMethod @params
            $ServerVersion = $HealthResponse.version

            if ($ServerVersion -eq $NewVersion) {
                Write-SuccessLog "Server healthy and running version $ServerVersion"
                $Healthy = $true
                break
            } else {
                Write-WarningLog "Server running old version: $ServerVersion (expected: $NewVersion)"
                Write-InfoLog "Waiting for server to reload... (${WaitCount}s)"
            }
        } catch {
            # Server not ready yet
        }

        Start-Sleep -Seconds 1
        $WaitCount++
    }

    if (-not $Healthy) {
        Write-ErrorLog "Server health check timeout after ${MaxWait}s"
        Write-InfoLog "Check logs in: $env:LOCALAPPDATA\mcp-memory\logs\"
        exit 1
    }
}

# Step 8: Summary
$TotalTime = Get-ElapsedSeconds

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "          Update Complete!                 " -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-SuccessLog "Version: $CurrentVersion -> $NewVersion"
Write-SuccessLog "Total time: ${TotalTime}s"
Write-Host ""
Write-InfoLog "Dashboard: https://localhost:8000"
Write-InfoLog "API Docs:  https://localhost:8000/api/docs"
Write-Host ""

if ($CurrentVersion -ne $NewVersion) {
    Write-InfoLog "New version deployed. Check CHANGELOG.md for details:"
    Write-Host ""

    $ChangelogPath = Join-Path $ProjectRoot "CHANGELOG.md"
    if (Test-Path $ChangelogPath) {
        $ChangelogContent = Get-Content $ChangelogPath -Raw
        # Build pattern separately to avoid parser confusion
        $EscapedVersion = [regex]::Escape($NewVersion)
        $Pattern = '(?s)## \[' + $EscapedVersion + '\].*?(?=## \[|$)'

        if ($ChangelogContent -match $Pattern) {
            $ChangelogLines = $Matches[0] -split "`n" | Select-Object -First 20
            foreach ($Line in $ChangelogLines) {
                Write-Host $Line
            }
        } else {
            Write-WarningLog "CHANGELOG not updated for $NewVersion"
        }
    }
}

exit 0
