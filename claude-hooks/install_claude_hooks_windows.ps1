# Claude Code Memory Awareness Hooks - Windows Installation Script v2.2.0
# Installs hooks into Claude Code hooks directory for automatic memory awareness
# Enhanced Output Control and Session Management

param(
    [switch]$Uninstall,
    [switch]$Test,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Configuration - Detect proper Claude Code hooks directory
function Get-ClaudeHooksDirectory {
    # Primary location: User profile (updated to match actual Claude Code directory structure)
    $primaryPath = "$env:USERPROFILE\.claude\hooks"

    # Alternative locations to check
    $alternativePaths = @(
        "$env:APPDATA\.claude\hooks",
        "$env:LOCALAPPDATA\.claude\hooks"
    )

    # If primary path already exists, use it
    if (Test-Path $primaryPath) {
        return $primaryPath
    }

    # Check if Claude Code is installed and can tell us the hooks directory
    try {
        $claudeHelp = claude --help 2>$null
        if ($claudeHelp -match "hooks.*directory.*(\S+)") {
            $detectedPath = $matches[1]
            if ($detectedPath -and (Test-Path (Split-Path -Parent $detectedPath) -ErrorAction SilentlyContinue)) {
                return $detectedPath
            }
        }
    } catch {
        # Claude CLI not available or failed
    }

    # Check alternative locations
    foreach ($altPath in $alternativePaths) {
        if (Test-Path $altPath) {
            return $altPath
        }
    }

    # Default to primary path (will be created if needed)
    return $primaryPath
}

$CLAUDE_HOOKS_DIR = Get-ClaudeHooksDirectory

# Script is now in the claude-hooks directory itself
$SCRIPT_DIR = $PSScriptRoot
$SOURCE_DIR = $SCRIPT_DIR

$dateStr = Get-Date -Format "yyyyMMdd-HHmmss"
$BACKUP_DIR = "$env:USERPROFILE\.claude\hooks-backup-$dateStr"

# Debug: Display resolved paths
function Write-Info { Write-Host "[INFO]" -ForegroundColor Green -NoNewline; Write-Host " $args" }
function Write-Warn { Write-Host "[WARN]" -ForegroundColor Yellow -NoNewline; Write-Host " $args" }
function Write-Error { Write-Host "[ERROR]" -ForegroundColor Red -NoNewline; Write-Host " $args" }

Write-Info "Script location: $SCRIPT_DIR"
Write-Info "Repository root: $REPO_ROOT"
Write-Info "Source hooks directory: $SOURCE_DIR"
Write-Info "Target hooks directory: $CLAUDE_HOOKS_DIR"

# Show help
if ($Help) {
    Write-Host @"
Claude Code Memory Awareness Hooks - Windows Installation

Usage: .\install_claude_hooks_windows.ps1 [options]

Options:
  -Help       Show this help message
  -Uninstall  Remove installed hooks
  -Test       Run tests only

Examples:
  .\install_claude_hooks_windows.ps1              # Install hooks
  .\install_claude_hooks_windows.ps1 -Uninstall   # Remove hooks
  .\install_claude_hooks_windows.ps1 -Test        # Test installation
"@
    exit 0
}

# Header
Write-Host ""
Write-Host "Claude Code Memory Awareness Hooks Installation v2.2.0 (Windows)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Claude Code is installed
function Test-ClaudeCode {
    $claudePath = Get-Command claude -ErrorAction SilentlyContinue
    if (-not $claudePath) {
        Write-Warn "Claude Code CLI not found in PATH"
        Write-Warn "Please ensure Claude Code is installed and accessible"
        $response = Read-Host "Continue anyway? (Y/N)"
        if ($response -ne "Y" -and $response -ne "y") {
            exit 1
        }
    } else {
        Write-Info "Claude Code CLI found: $($claudePath.Source)"
    }
}

# Validate source directory exists
function Test-SourceDirectory {
    Write-Info "Validating source directory..."

    if (-not (Test-Path $SOURCE_DIR)) {
        Write-Error "Source hooks directory not found: $SOURCE_DIR"
        Write-Error "Please ensure you are running this script from the mcp-memory-service repository"
        Write-Error "Expected repository structure:"
        Write-Error "  mcp-memory-service/"
        Write-Error "    scripts/"
        Write-Error "      install_claude_hooks_windows.ps1  (This script)"
        Write-Error "    claude-hooks/"
        Write-Error "      core/"
        Write-Error "      utilities/"
        Write-Error "      config.json"
        exit 1
    }

    # Check for required subdirectories
    $requiredDirs = @("core", "utilities", "tests")
    foreach ($dir in $requiredDirs) {
        $dirPath = Join-Path $SOURCE_DIR $dir
        if (-not (Test-Path $dirPath)) {
            Write-Error "Missing required directory: $dirPath"
            Write-Error "The claude-hooks directory appears to be incomplete"
            exit 1
        }
    }

    Write-Info "Source directory validation passed"
}

# Create Claude Code hooks directory if it does not exist
function New-HooksDirectory {
    if (-not (Test-Path $CLAUDE_HOOKS_DIR)) {
        Write-Info "Creating Claude Code hooks directory: $CLAUDE_HOOKS_DIR"
        try {
            New-Item -ItemType Directory -Path $CLAUDE_HOOKS_DIR -Force | Out-Null
            Write-Info "Successfully created hooks directory"
        } catch {
            Write-Error "Failed to create hooks directory: $CLAUDE_HOOKS_DIR"
            Write-Error "Error: $($_.Exception.Message)"
            Write-Error ""
            Write-Error "Possible solutions:"
            Write-Error "  1. Run PowerShell as Administrator"
            Write-Error "  2. Check if the parent directory exists and is writable"
            Write-Error "  3. Manually create the directory: $CLAUDE_HOOKS_DIR"
            exit 1
        }
    } else {
        Write-Info "Claude Code hooks directory exists: $CLAUDE_HOOKS_DIR"
    }

    # Test write access
    $testFile = Join-Path $CLAUDE_HOOKS_DIR "write-test.tmp"
    try {
        "test" | Out-File -FilePath $testFile -Force
        Remove-Item -Path $testFile -Force
        Write-Info "Write access confirmed for hooks directory"
    } catch {
        Write-Error "No write access to hooks directory: $CLAUDE_HOOKS_DIR"
        Write-Error "Please check permissions or run as Administrator"
        exit 1
    }
}

# Backup existing hooks if they exist
function Backup-ExistingHooks {
    $hasExisting = $false

    if ((Test-Path "$CLAUDE_HOOKS_DIR\core") -or
        (Test-Path "$CLAUDE_HOOKS_DIR\utilities") -or
        (Test-Path "$CLAUDE_HOOKS_DIR\config.json")) {
        $hasExisting = $true
    }

    if ($hasExisting) {
        Write-Info "Backing up existing hooks to: $BACKUP_DIR"
        New-Item -ItemType Directory -Path $BACKUP_DIR -Force | Out-Null
        Copy-Item -Path "$CLAUDE_HOOKS_DIR\*" -Destination $BACKUP_DIR -Recurse -Force -ErrorAction SilentlyContinue
        Write-Info "Backup created successfully"
    }
}

# Install hook files
function Install-Hooks {
    Write-Info "Installing memory awareness hooks..."

    # Create necessary directories
    New-Item -ItemType Directory -Path "$CLAUDE_HOOKS_DIR\core" -Force | Out-Null
    New-Item -ItemType Directory -Path "$CLAUDE_HOOKS_DIR\utilities" -Force | Out-Null
    New-Item -ItemType Directory -Path "$CLAUDE_HOOKS_DIR\tests" -Force | Out-Null

    # Copy core hooks
    Copy-Item -Path "$SOURCE_DIR\core\*" -Destination "$CLAUDE_HOOKS_DIR\core\" -Recurse -Force
    Write-Info "Installed core hooks (session-start, session-end, topic-change)"

    # Copy utilities
    Copy-Item -Path "$SOURCE_DIR\utilities\*" -Destination "$CLAUDE_HOOKS_DIR\utilities\" -Recurse -Force
    Write-Info "Installed utility modules"

    # Copy tests
    Copy-Item -Path "$SOURCE_DIR\tests\*" -Destination "$CLAUDE_HOOKS_DIR\tests\" -Recurse -Force
    Write-Info "Installed test suite"

    # Copy documentation and configuration
    Copy-Item -Path "$SOURCE_DIR\README.md" -Destination "$CLAUDE_HOOKS_DIR\" -Force
    Copy-Item -Path "$SOURCE_DIR\config.template.json" -Destination "$CLAUDE_HOOKS_DIR\" -Force
    Write-Info "Installed documentation and templates"
}

# Install or update configuration
function Install-Config {
    $configFile = "$CLAUDE_HOOKS_DIR\config.json"

    if (-not (Test-Path $configFile)) {
        # First installation - use default config
        Copy-Item -Path "$SOURCE_DIR\config.json" -Destination $configFile -Force
        Write-Info "Installed default configuration"
        Write-Warn "Please update config.json with your memory service endpoint and API key"
    } else {
        Write-Info "Configuration file already exists - not overwriting"
        Write-Info "   Compare with config.template.json for new options"
    }
}

# Test installation
function Test-Installation {
    Write-Info "Testing installation..."

    # Check if required files exist
    $requiredFiles = @(
        "core\session-start.js",
        "core\session-end.js",
        "utilities\project-detector.js",
        "utilities\memory-scorer.js",
        "utilities\context-formatter.js",
        "config.json",
        "README.md"
    )

    $missingFiles = @()
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path "$CLAUDE_HOOKS_DIR\$file")) {
            $missingFiles += $file
        }
    }

    if ($missingFiles.Count -gt 0) {
        Write-Error "Installation incomplete - missing files:"
        foreach ($file in $missingFiles) {
            Write-Host "  - $file"
        }
        return $false
    }

    # Test Node.js availability
    $nodeVersion = node --version 2>$null
    if (-not $nodeVersion) {
        Write-Warn "Node.js not found - hooks require Node.js to function"
        Write-Warn "Please install Node.js version 14 or higher"
    } else {
        Write-Info "Node.js available: $nodeVersion"
    }

    # Run integration test
    if (Test-Path "$CLAUDE_HOOKS_DIR\tests\integration-test.js") {
        Write-Info "Running integration tests..."
        Push-Location $CLAUDE_HOOKS_DIR
        try {
            $testResult = node tests\integration-test.js 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Info "Integration tests passed"
            } else {
                Write-Warn "Some integration tests failed - check configuration"
                Write-Host $testResult
            }
        } finally {
            Pop-Location
        }
    }

    return $true
}

# Display post-installation instructions
function Show-PostInstallInstructions {
    Write-Host ""
    Write-Host "Installation Complete!" -ForegroundColor Green
    Write-Host "=====================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Configure your memory service endpoint:"
    Write-Host "   Edit: $CLAUDE_HOOKS_DIR\config.json"
    Write-Host "   Update endpoint and apiKey values"
    Write-Host ""
    Write-Host "2. Test the hooks:"
    Write-Host "   cd $CLAUDE_HOOKS_DIR"
    Write-Host "   node tests\integration-test.js"
    Write-Host ""
    Write-Host "3. Start using Claude Code:"
    Write-Host "   The hooks will automatically activate on session start/end"
    Write-Host ""
    Write-Host "Installation Details:" -ForegroundColor Cyan
    Write-Host "   Hooks Directory: $CLAUDE_HOOKS_DIR"
    if (Test-Path $BACKUP_DIR) {
        Write-Host "   Backup Directory: $BACKUP_DIR"
    }
    Write-Host ""

    # Try to read and display current configuration
    $configPath = Join-Path $CLAUDE_HOOKS_DIR "config.json"
    if (Test-Path $configPath) {
        try {
            $config = Get-Content $configPath | ConvertFrom-Json
            Write-Host "Configuration:" -ForegroundColor Cyan
            Write-Host "   Memory Service: $($config.memoryService.endpoint)"
            Write-Host "   Max Memories: $($config.memoryService.maxMemoriesPerSession)"
        } catch {
            Write-Warn "Could not read configuration file"
        }
    }
    Write-Host ""
    $readmePath = Join-Path $CLAUDE_HOOKS_DIR "README.md"
    Write-Host "For troubleshooting, see: $readmePath"
}

# Uninstall function
function Uninstall-Hooks {
    if (Test-Path $CLAUDE_HOOKS_DIR) {
        $response = Read-Host "Remove all Claude Code memory awareness hooks? (Y/N)"
        if ($response -eq "Y" -or $response -eq "y") {
            Remove-Item -Path "$CLAUDE_HOOKS_DIR\core" -Recurse -Force -ErrorAction SilentlyContinue
            Remove-Item -Path "$CLAUDE_HOOKS_DIR\utilities" -Recurse -Force -ErrorAction SilentlyContinue
            Remove-Item -Path "$CLAUDE_HOOKS_DIR\tests" -Recurse -Force -ErrorAction SilentlyContinue
            Remove-Item -Path "$CLAUDE_HOOKS_DIR\config.json" -Force -ErrorAction SilentlyContinue
            Remove-Item -Path "$CLAUDE_HOOKS_DIR\config.template.json" -Force -ErrorAction SilentlyContinue
            Remove-Item -Path "$CLAUDE_HOOKS_DIR\README.md" -Force -ErrorAction SilentlyContinue
            Write-Info "Hooks uninstalled successfully"
        }
    } else {
        Write-Info "No hooks found to uninstall"
    }
}

# Test only function
function Test-Only {
    if (Test-Path "$CLAUDE_HOOKS_DIR\tests\integration-test.js") {
        Push-Location $CLAUDE_HOOKS_DIR
        try {
            node tests\integration-test.js
        } finally {
            Pop-Location
        }
    } else {
        Write-Error "Tests not found - please install first"
        exit 1
    }
}

# Main execution
try {
    if ($Uninstall) {
        Uninstall-Hooks
    } elseif ($Test) {
        Test-Only
    } else {
        # Main installation process
        Test-SourceDirectory
        Test-ClaudeCode
        New-HooksDirectory
        Backup-ExistingHooks
        Install-Hooks
        Install-Config
        if (Test-Installation) {
            Show-PostInstallInstructions
        }
    }
} catch {
    Write-Host "ERROR: Installation failed" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}
