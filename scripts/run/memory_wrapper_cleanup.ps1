# MCP Memory Service Wrapper with Orphan Cleanup
# 
# Cleans up orphaned MCP memory processes before starting the server.
# Orphaned processes cause SQLite "database locked" errors.
#
# Usage in MCP config:
# {
#   "memory": {
#     "command": "powershell",
#     "args": ["-ExecutionPolicy", "Bypass", "-File", "C:\\path\\to\\scripts\\run\\memory_wrapper_cleanup.ps1"],
#     "env": { ... }
#   }
# }

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent (Split-Path -Parent $ScriptDir)

function Write-Log {
    param([string]$Message)
    Write-Host "[mcp-memory-wrapper] $Message" -ForegroundColor Cyan
}

function Get-OrphanedProcesses {
    <#
    .SYNOPSIS
    Find MCP memory processes whose parent no longer exists (orphaned).
    #>
    $orphans = @()
    
    try {
        # Get all Python processes running mcp-memory-service
        $mcpProcesses = Get-WmiObject Win32_Process -Filter "CommandLine LIKE '%mcp-memory-service%'" -ErrorAction SilentlyContinue
        
        foreach ($proc in $mcpProcesses) {
            # Skip our own process
            if ($proc.ProcessId -eq $PID) {
                continue
            }
            
            # Check if parent process exists
            $parentExists = Get-Process -Id $proc.ParentProcessId -ErrorAction SilentlyContinue
            
            if (-not $parentExists) {
                $orphans += $proc.ProcessId
            }
        }
    }
    catch {
        Write-Log "Warning: Error checking for orphans: $_"
    }
    
    return $orphans
}

function Remove-OrphanedProcesses {
    <#
    .SYNOPSIS
    Kill orphaned MCP memory processes.
    #>
    $orphans = Get-OrphanedProcesses
    
    if ($orphans.Count -gt 0) {
        Write-Log "Found $($orphans.Count) orphaned process(es): $($orphans -join ', ')"
        
        foreach ($pid in $orphans) {
            try {
                Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                Write-Log "Terminated orphaned process: $pid"
            }
            catch {
                Write-Log "Failed to terminate process $pid : $_"
            }
        }
    }
    else {
        Write-Log "No orphaned processes found"
    }
}

function Find-Uv {
    <#
    .SYNOPSIS
    Find the uv executable.
    #>
    
    # Check if uv is in PATH
    $uvPath = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvPath) {
        return $uvPath.Path
    }
    
    # Check common locations
    $commonPaths = @(
        "$env:USERPROFILE\.local\bin\uv.exe",
        "$env:USERPROFILE\.cargo\bin\uv.exe",
        "$env:LOCALAPPDATA\Programs\uv\uv.exe"
    )
    
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            return $path
        }
    }
    
    Write-Log "ERROR: uv not found. Install with: irm https://astral.sh/uv/install.ps1 | iex"
    exit 1
}

function Start-MemoryServer {
    <#
    .SYNOPSIS
    Start the MCP memory server.
    #>
    Set-Location $ProjectDir
    
    $uv = Find-Uv
    Write-Log "Starting server with: $uv run memory"
    
    # Start the server (this replaces our process)
    & $uv run memory $args
    exit $LASTEXITCODE
}

# Main execution
try {
    Write-Log "Starting (Windows $([System.Environment]::OSVersion.Version))"
    
    # Step 1: Cleanup orphans
    Remove-OrphanedProcesses
    
    # Step 2: Start server
    Start-MemoryServer
}
catch {
    Write-Log "Fatal error: $_"
    exit 1
}
