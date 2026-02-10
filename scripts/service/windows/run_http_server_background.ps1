#Requires -Version 5.1
<#
.SYNOPSIS
    Wrapper script to run MCP Memory HTTP Server in background with logging and restart logic.

.DESCRIPTION
    This script is designed to be executed by Windows Task Scheduler.
    It runs the HTTP server with proper environment setup, logging, and automatic restart on failure.

.NOTES
    File Name      : run_http_server_background.ps1
    Prerequisite   : Python, uv package manager
    Location       : scripts/service/windows/
#>

# Configuration
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item "$ScriptDir\..\..\..").FullName
$LogDir = Join-Path $env:LOCALAPPDATA "mcp-memory\logs"
$LogFile = Join-Path $LogDir "http-server.log"
$PidFile = Join-Path $env:LOCALAPPDATA "mcp-memory\http-server.pid"
$MaxRestarts = 3
$RestartDelaySeconds = 60

# Ensure log directory exists
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Logging function
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] [$Level] $Message"
    Add-Content -Path $LogFile -Value $LogMessage

    # Keep log file under 10MB
    if ((Get-Item $LogFile -ErrorAction SilentlyContinue).Length -gt 10MB) {
        $OldLog = "$LogFile.old"
        if (Test-Path $OldLog) { Remove-Item $OldLog -Force }
        Rename-Item $LogFile $OldLog
    }
}

# Load .env file
function Load-EnvFile {
    $EnvFile = Join-Path $ProjectRoot ".env"
    if (Test-Path $EnvFile) {
        Write-Log "Loading environment from $EnvFile"
        Get-Content $EnvFile | ForEach-Object {
            if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
                $Name = $matches[1].Trim()
                $Value = $matches[2].Trim().Trim('"').Trim("'")
                [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
            }
        }
    } else {
        Write-Log "No .env file found at $EnvFile" "WARN"
    }
}

# Find uv executable (Task Scheduler has a minimal PATH)
function Find-Executable {
    # Common install locations for uv on Windows
    $Candidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "uv\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe"),
        (Join-Path $env:ProgramData "uv\uv.exe")
    )

    foreach ($Path in $Candidates) {
        if (Test-Path $Path) {
            Write-Log "Found uv at: $Path"
            return $Path
        }
    }

    # Try PATH resolution (works in interactive shells, may fail in Task Scheduler)
    $PathResolved = Get-Command uv -ErrorAction SilentlyContinue
    if ($PathResolved) {
        Write-Log "Found uv on PATH: $($PathResolved.Source)"
        return $PathResolved.Source
    }

    Write-Log "uv not found, will fall back to python directly" "WARN"
    return $null
}

# Check if server is already running
function Test-ServerRunning {
    if (Test-Path $PidFile) {
        $StoredPid = Get-Content $PidFile -ErrorAction SilentlyContinue
        if ($StoredPid) {
            $Process = Get-Process -Id $StoredPid -ErrorAction SilentlyContinue
            if ($Process -and $Process.ProcessName -like "*python*") {
                return $true
            }
        }
    }

    # Also check via HTTP health endpoint
    try {
        $Response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($Response.StatusCode -eq 200) {
            return $true
        }
    } catch {
        # Server not responding
    }

    return $false
}

# Main execution
Write-Log "========== MCP Memory HTTP Server Starting =========="
Write-Log "Project Root: $ProjectRoot"
Write-Log "Log File: $LogFile"

# Check if already running
if (Test-ServerRunning) {
    Write-Log "Server is already running. Exiting." "WARN"
    exit 0
}

# Change to project directory
Set-Location $ProjectRoot
Write-Log "Working directory: $(Get-Location)"

# Load environment variables
Load-EnvFile

# Restart loop
$RestartCount = 0
while ($RestartCount -lt $MaxRestarts) {
    Write-Log "Starting HTTP server (attempt $($RestartCount + 1)/$MaxRestarts)..."

    try {
        # Resolve executable (uv or python fallback)
        $UvPath = Find-Executable
        $ProcessInfo = New-Object System.Diagnostics.ProcessStartInfo
        if ($UvPath) {
            $ProcessInfo.FileName = $UvPath
            $ProcessInfo.Arguments = "run python scripts/server/run_http_server.py"
        } else {
            # Fallback: run python directly (same PATH issue applies)
            $PythonPath = $null

            # Try py.exe launcher first (installed in SystemRoot, reliably on PATH)
            $PyLauncher = Join-Path $env:SystemRoot "py.exe"
            if (Test-Path $PyLauncher) {
                $PythonPath = $PyLauncher
            }

            # Try common Python install locations
            if (-not $PythonPath) {
                $PythonCandidates = @(
                    (Join-Path $env:LOCALAPPDATA "Programs\Python\Python3*\python.exe"),
                    (Join-Path $env:ProgramFiles "Python3*\python.exe"),
                    "C:\Python3*\python.exe"
                )
                foreach ($Pattern in $PythonCandidates) {
                    $Matches = Get-Item $Pattern -ErrorAction SilentlyContinue
                    if ($Matches) { $PythonPath = ($Matches | Sort-Object DirectoryName -Descending | Select-Object -First 1).FullName; break }
                }
            }

            # Last resort: PATH resolution
            if (-not $PythonPath) {
                $PythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
            }

            if (-not $PythonPath) {
                Write-Log "Neither uv nor python found. Cannot start server." "ERROR"
                throw "No Python executable found"
            }

            $ProcessInfo.FileName = $PythonPath
            $ProcessInfo.Arguments = "scripts/server/run_http_server.py"
        }
        $ProcessInfo.WorkingDirectory = $ProjectRoot
        $ProcessInfo.UseShellExecute = $false
        $ProcessInfo.RedirectStandardOutput = $true
        $ProcessInfo.RedirectStandardError = $true
        $ProcessInfo.CreateNoWindow = $true

        Write-Log "Executable: $($ProcessInfo.FileName) $($ProcessInfo.Arguments)"

        $Process = New-Object System.Diagnostics.Process
        $Process.StartInfo = $ProcessInfo

        # Event handlers for output
        # NOTE: $using: scope does NOT work in .NET event handlers (only in PS jobs/runspaces).
        # Use $script: scope which is captured by the closure.
        $OutputHandler = {
            if (-not [String]::IsNullOrEmpty($EventArgs.Data)) {
                Add-Content -Path $script:LogFile -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [SERVER] $($EventArgs.Data)"
            }
        }

        $Process.add_OutputDataReceived($OutputHandler)
        $Process.add_ErrorDataReceived($OutputHandler)

        $Process.Start() | Out-Null
        $Process.BeginOutputReadLine()
        $Process.BeginErrorReadLine()

        # Save PID
        Set-Content -Path $PidFile -Value $Process.Id
        Write-Log "Server started with PID $($Process.Id)"

        # Wait for process to exit
        $Process.WaitForExit()
        $ExitCode = $Process.ExitCode

        Write-Log "Server exited with code $ExitCode" $(if ($ExitCode -eq 0) { "INFO" } else { "ERROR" })

        # Clean up PID file
        if (Test-Path $PidFile) {
            Remove-Item $PidFile -Force
        }

        # If clean exit (0), don't restart
        if ($ExitCode -eq 0) {
            Write-Log "Server stopped gracefully. Not restarting."
            break
        }

    } catch {
        Write-Log "Error starting server: $_" "ERROR"
    }

    $RestartCount++

    if ($RestartCount -lt $MaxRestarts) {
        Write-Log "Waiting $RestartDelaySeconds seconds before restart..."
        Start-Sleep -Seconds $RestartDelaySeconds
    }
}

if ($RestartCount -ge $MaxRestarts) {
    Write-Log "Max restart attempts ($MaxRestarts) reached. Giving up." "ERROR"
}

Write-Log "========== MCP Memory HTTP Server Stopped =========="
