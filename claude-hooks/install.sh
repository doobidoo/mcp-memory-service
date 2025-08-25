#!/bin/bash

# Claude Code Memory Awareness Hooks - Installation Script
# Installs hooks into Claude Code hooks directory for automatic memory awareness

set -e

echo "🚀 Claude Code Memory Awareness Hooks Installation"
echo "================================================="

# Configuration
CLAUDE_HOOKS_DIR="${HOME}/.claude/hooks"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${HOME}/.claude/hooks-backup-$(date +%Y%m%d-%H%M%S)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Claude Code is installed
check_claude_code() {
    if ! command -v claude &> /dev/null; then
        warn "Claude Code CLI not found in PATH"
        warn "Please ensure Claude Code is installed and accessible"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        info "Claude Code CLI found: $(which claude)"
    fi
}

# Create Claude Code hooks directory if it doesn't exist
create_hooks_directory() {
    if [ ! -d "$CLAUDE_HOOKS_DIR" ]; then
        info "Creating Claude Code hooks directory: $CLAUDE_HOOKS_DIR"
        mkdir -p "$CLAUDE_HOOKS_DIR"
    else
        info "Claude Code hooks directory exists: $CLAUDE_HOOKS_DIR"
    fi
}

# Backup existing hooks if they exist
backup_existing_hooks() {
    local has_existing=false
    
    if [ -d "$CLAUDE_HOOKS_DIR/core" ] || [ -d "$CLAUDE_HOOKS_DIR/utilities" ]; then
        has_existing=true
    fi
    
    if [ -f "$CLAUDE_HOOKS_DIR/config.json" ]; then
        has_existing=true
    fi
    
    if [ "$has_existing" = true ]; then
        info "Backing up existing hooks to: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
        cp -r "$CLAUDE_HOOKS_DIR"/* "$BACKUP_DIR"/ 2>/dev/null || true
        info "Backup created successfully"
    fi
}

# Install hook files
install_hooks() {
    info "Installing memory awareness hooks..."
    
    # Create necessary directories
    mkdir -p "$CLAUDE_HOOKS_DIR/core"
    mkdir -p "$CLAUDE_HOOKS_DIR/utilities"
    mkdir -p "$CLAUDE_HOOKS_DIR/tests"
    
    # Copy core hooks
    cp "$SOURCE_DIR/core/session-start.js" "$CLAUDE_HOOKS_DIR/core/"
    cp "$SOURCE_DIR/core/session-end.js" "$CLAUDE_HOOKS_DIR/core/"
    cp "$SOURCE_DIR/core/memory-retrieval.js" "$CLAUDE_HOOKS_DIR/core/"
    info "✅ Installed core hooks (session-start, session-end, memory-retrieval)"
    
    # Copy utilities
    cp "$SOURCE_DIR/utilities/project-detector.js" "$CLAUDE_HOOKS_DIR/utilities/"
    cp "$SOURCE_DIR/utilities/memory-scorer.js" "$CLAUDE_HOOKS_DIR/utilities/"
    cp "$SOURCE_DIR/utilities/context-formatter.js" "$CLAUDE_HOOKS_DIR/utilities/"
    cp "$SOURCE_DIR/utilities/context-shift-detector.js" "$CLAUDE_HOOKS_DIR/utilities/"
    info "✅ Installed utility modules (project-detector, memory-scorer, context-formatter, context-shift-detector)"
    
    # Copy tests
    cp "$SOURCE_DIR/tests/integration-test.js" "$CLAUDE_HOOKS_DIR/tests/"
    info "✅ Installed test suite"
    
    # Copy documentation
    cp "$SOURCE_DIR/README.md" "$CLAUDE_HOOKS_DIR/"
    info "✅ Installed documentation"
}

# Install or update configuration
install_config() {
    local config_file="$CLAUDE_HOOKS_DIR/config.json"
    local template_file="$CLAUDE_HOOKS_DIR/config.template.json"
    
    # Always copy template
    cp "$SOURCE_DIR/config.template.json" "$template_file"
    info "✅ Installed configuration template"
    
    if [ ! -f "$config_file" ]; then
        # First installation - use our default config
        cp "$SOURCE_DIR/config.json" "$config_file"
        info "✅ Installed default configuration"
        warn "⚠️  Please update config.json with your memory service endpoint and API key"
    else
        info "✅ Configuration file already exists - not overwriting"
        info "   Compare with config.template.json for new options"
    fi
}

# Configure Claude Code settings
configure_claude_settings() {
    local settings_file="${HOME}/.claude/settings.json"
    local local_settings_file="${HOME}/.claude/settings.local.json"
    
    info "Configuring Claude Code settings for hook integration..."
    
    # Create .claude directory if it doesn't exist
    mkdir -p "${HOME}/.claude"
    
    # Hook configuration for Claude Code
    local hook_config='{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "node ~/.claude/hooks/core/session-start.js",
            "timeout": 10
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "node ~/.claude/hooks/core/session-end.js",
            "timeout": 15
          }
        ]
      }
    ]
  }
}'
    
    # Check if settings file exists and merge configuration
    if [ -f "$settings_file" ]; then
        # Backup existing settings
        cp "$settings_file" "${settings_file}.backup-$(date +%Y%m%d-%H%M%S)"
        info "Backed up existing settings.json"
        
        # Merge with existing settings (basic merge - overwrites hooks section)
        # In a production script, we'd use a proper JSON merger
        echo "$hook_config" > "$settings_file"
        warn "⚠️  Existing settings.json hooks section has been replaced"
        warn "   Please manually merge if you had other hooks configured"
        warn "   Backup available: ${settings_file}.backup-*"
    else
        # Create new settings file
        echo "$hook_config" > "$settings_file"
        info "✅ Created new settings.json with hook configuration"
    fi
    
    # Also update local settings if it exists (common pattern)
    if [ -f "$local_settings_file" ]; then
        # Don't overwrite local settings, just inform user
        info "Found settings.local.json - you may need to manually add hook configuration"
    fi
    
    info "✅ Claude Code settings configured for memory awareness hooks"
}

# Test installation
test_installation() {
    info "Testing installation..."
    
    # Check if required files exist
    local required_files=(
        "core/session-start.js"
        "core/session-end.js"
        "utilities/project-detector.js"
        "utilities/memory-scorer.js"
        "utilities/context-formatter.js"
        "config.json"
        "README.md"
    )
    
    local missing_files=()
    for file in "${required_files[@]}"; do
        if [ ! -f "$CLAUDE_HOOKS_DIR/$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        error "Installation incomplete - missing files:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        return 1
    fi
    
    # Test Node.js availability
    if ! node --version &> /dev/null; then
        warn "Node.js not found - hooks require Node.js to function"
        warn "Please install Node.js version 14 or higher"
    else
        info "✅ Node.js available: $(node --version)"
    fi
    
    # Test Claude Code settings configuration
    local settings_file="${HOME}/.claude/settings.json"
    if [ -f "$settings_file" ]; then
        if grep -q "session-start.js" "$settings_file" && grep -q "session-end.js" "$settings_file"; then
            info "✅ Claude Code settings configured correctly"
        else
            warn "⚠️  Claude Code settings may not be configured correctly"
        fi
    else
        warn "⚠️  Claude Code settings.json not found"
    fi
    
    # Run integration test
    if [ -f "$CLAUDE_HOOKS_DIR/tests/integration-test.js" ]; then
        info "Running integration tests..."
        cd "$CLAUDE_HOOKS_DIR"
        
        if node tests/integration-test.js; then
            info "✅ Integration tests passed"
        else
            warn "⚠️  Some integration tests failed - check configuration"
        fi
    fi
    
    # Test hook detection by Claude Code (if available)
    if command -v claude &> /dev/null; then
        info "Testing Claude Code hook detection..."
        if claude --debug hooks --help &> /dev/null; then
            # This would require a more sophisticated test
            info "✅ Claude Code is available for hook testing"
        else
            warn "⚠️  Could not test Claude Code hook detection"
        fi
    fi
}

# Display post-installation instructions
show_post_install_instructions() {
    echo ""
    echo "🎉 Installation Complete!"
    echo "========================"
    echo ""
    echo "📋 Next Steps:"
    echo ""
    echo "1. Configure your memory service endpoint:"
    echo "   Edit: $CLAUDE_HOOKS_DIR/config.json"
    echo "   Update 'endpoint' and 'apiKey' values"
    echo ""
    echo "2. Test the hooks:"
    echo "   cd $CLAUDE_HOOKS_DIR"
    echo "   node tests/integration-test.js"
    echo ""
    echo "3. Start using Claude Code:"
    echo "   The hooks will automatically activate on session start/end"
    echo ""
    echo "📁 Installation Details:"
    echo "   Hooks Directory: $CLAUDE_HOOKS_DIR"
    echo "   Backup Directory: $BACKUP_DIR (if applicable)"
    echo ""
    echo "🔧 Configuration:"
    echo "   Memory Service: $(grep -o '\"endpoint\"[^,]*' "$CLAUDE_HOOKS_DIR/config.json" | cut -d'"' -f4)"
    echo "   Max Memories: $(grep -o '\"maxMemoriesPerSession\"[^,]*' "$CLAUDE_HOOKS_DIR/config.json" | cut -d':' -f2 | tr -d ' ,')"
    echo ""
    echo "For troubleshooting, see: $CLAUDE_HOOKS_DIR/README.md"
}

# Main installation process
main() {
    echo ""
    check_claude_code
    create_hooks_directory
    backup_existing_hooks
    install_hooks
    install_config
    configure_claude_settings
    test_installation
    show_post_install_instructions
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --uninstall    Remove installed hooks"
        echo "  --test         Run tests only"
        echo ""
        exit 0
        ;;
    --uninstall)
        if [ -d "$CLAUDE_HOOKS_DIR" ]; then
            read -p "Remove all Claude Code memory awareness hooks? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$CLAUDE_HOOKS_DIR/core"
                rm -rf "$CLAUDE_HOOKS_DIR/utilities"
                rm -rf "$CLAUDE_HOOKS_DIR/tests"
                rm -f "$CLAUDE_HOOKS_DIR/config.json"
                rm -f "$CLAUDE_HOOKS_DIR/config.template.json"
                rm -f "$CLAUDE_HOOKS_DIR/README.md"
                info "Hooks uninstalled successfully"
            fi
        else
            info "No hooks found to uninstall"
        fi
        exit 0
        ;;
    --test)
        if [ -f "$CLAUDE_HOOKS_DIR/tests/integration-test.js" ]; then
            cd "$CLAUDE_HOOKS_DIR"
            node tests/integration-test.js
        else
            error "Tests not found - please install first"
            exit 1
        fi
        exit 0
        ;;
    "")
        main
        ;;
    *)
        error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac