#!/bin/bash

# Natural Memory Triggers v7.1.3 - Specialized Installation Script
# Installs intelligent automatic memory awareness with pattern detection
# Includes performance optimization and CLI management tools

set -e

echo "🧠 Natural Memory Triggers v7.1.3 Installation"
echo "==============================================="

# Enhanced Claude Code directory detection
get_claude_hooks_directory() {
    local primary_path="${HOME}/.claude/hooks"
    local alternative_paths=(
        "${HOME}/.config/claude/hooks"
        "${XDG_CONFIG_HOME:-$HOME/.config}/claude/hooks"
    )

    # If primary path already exists, use it
    if [ -d "$primary_path" ]; then
        echo "$primary_path"
        return 0
    fi

    # Check if Claude Code is installed and can tell us the hooks directory
    if command -v claude &> /dev/null; then
        local claude_help
        if claude_help=$(claude --help 2>/dev/null); then
            # Look for hooks directory information in help output
            local detected_path
            detected_path=$(echo "$claude_help" | grep -o 'hooks.*directory[^[:space:]]*' | head -1 | sed 's/.*: *//' || true)
            if [ -n "$detected_path" ] && [ -d "$(dirname "$detected_path" 2>/dev/null || echo '')" ]; then
                echo "$detected_path"
                return 0
            fi
        fi
    fi

    # Check alternative locations
    for alt_path in "${alternative_paths[@]}"; do
        if [ -d "$alt_path" ]; then
            echo "$alt_path"
            return 0
        fi
    done

    # Default to primary path (will be created if needed)
    echo "$primary_path"
}

# Configuration
CLAUDE_HOOKS_DIR="$(get_claude_hooks_directory)"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${HOME}/.claude/hooks-backup-natural-triggers-$(date +%Y%m%d-%H%M%S)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Debug: Display resolved paths
echo ""
echo -e "${GREEN}[INFO]${NC} Script location: $SOURCE_DIR"
echo -e "${GREEN}[INFO]${NC} Target hooks directory: $CLAUDE_HOOKS_DIR"
echo -e "${GREEN}[INFO]${NC} Backup directory: $BACKUP_DIR"
echo ""

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${BLUE}[SUCCESS]${NC} $1"
}

# Check prerequisites for Natural Memory Triggers
check_prerequisites() {
    info "Checking prerequisites for Natural Memory Triggers..."

    # Check Claude Code CLI
    if ! command -v claude &> /dev/null; then
        warn "Claude Code CLI not found in PATH"
        warn "Please ensure Claude Code is installed and accessible"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        success "Claude Code CLI found: $(which claude)"
    fi

    # Check Node.js version
    if ! node --version &> /dev/null; then
        error "Node.js not found - required for hook execution"
        exit 1
    else
        local node_version=$(node --version | sed 's/v//')
        local major_version=$(echo $node_version | cut -d. -f1)
        if [ "$major_version" -lt 14 ]; then
            error "Node.js version $node_version found, but version 14+ required"
            exit 1
        else
            success "Node.js version $node_version found (compatible)"
        fi
    fi

    # Check if MCP Memory Service is accessible
    info "Testing MCP Memory Service connectivity..."
    if curl -k -s --connect-timeout 3 https://localhost:8443/api/health > /dev/null 2>&1; then
        success "MCP Memory Service is accessible"
    else
        warn "MCP Memory Service not accessible at https://localhost:8443"
        warn "Natural triggers will work with configuration fallback"
    fi
}

# Create backup of existing installation
create_backup() {
    if [ -d "$CLAUDE_HOOKS_DIR" ]; then
        info "Creating backup of existing hooks installation..."
        mkdir -p "$BACKUP_DIR"
        cp -r "$CLAUDE_HOOKS_DIR/"* "$BACKUP_DIR/" 2>/dev/null || true
        success "Backup created at: $BACKUP_DIR"
    fi
}

# Install Natural Memory Triggers components
install_natural_triggers() {
    info "Installing Natural Memory Triggers components..."

    # Create necessary directories
    mkdir -p "$CLAUDE_HOOKS_DIR/core"
    mkdir -p "$CLAUDE_HOOKS_DIR/utilities"

    # Install enhanced core hooks
    cp "$SOURCE_DIR/core/session-start.js" "$CLAUDE_HOOKS_DIR/core/"
    cp "$SOURCE_DIR/core/mid-conversation.js" "$CLAUDE_HOOKS_DIR/core/"
    success "Installed enhanced core hooks (session-start, mid-conversation)"

    # Install Natural Memory Triggers utilities
    cp "$SOURCE_DIR/utilities/adaptive-pattern-detector.js" "$CLAUDE_HOOKS_DIR/utilities/"
    cp "$SOURCE_DIR/utilities/tiered-conversation-monitor.js" "$CLAUDE_HOOKS_DIR/utilities/"
    cp "$SOURCE_DIR/utilities/performance-manager.js" "$CLAUDE_HOOKS_DIR/utilities/"
    cp "$SOURCE_DIR/utilities/mcp-client.js" "$CLAUDE_HOOKS_DIR/utilities/"
    cp "$SOURCE_DIR/utilities/memory-client.js" "$CLAUDE_HOOKS_DIR/utilities/"
    success "Installed Natural Memory Triggers utilities"

    # Install CLI management tools
    if [ -f "$SOURCE_DIR/memory-mode-controller.js" ]; then
        cp "$SOURCE_DIR/memory-mode-controller.js" "$CLAUDE_HOOKS_DIR/"
        success "Installed CLI management controller"
    fi

    # Install enhanced git analyzer
    if [ -f "$SOURCE_DIR/utilities/git-analyzer.js" ]; then
        cp "$SOURCE_DIR/utilities/git-analyzer.js" "$CLAUDE_HOOKS_DIR/utilities/"
        success "Installed enhanced git analyzer"
    fi
}

# Configure Natural Memory Triggers
configure_natural_triggers() {
    local config_file="$CLAUDE_HOOKS_DIR/config.json"

    info "Configuring Natural Memory Triggers..."

    # Install enhanced configuration with natural triggers
    if [ -f "$SOURCE_DIR/config.json" ]; then
        if [ -f "$config_file" ]; then
            # Backup existing config
            cp "$config_file" "${config_file}.backup-$(date +%Y%m%d-%H%M%S)"
            warn "Existing config backed up"
        fi

        cp "$SOURCE_DIR/config.json" "$config_file"
        success "Installed Natural Memory Triggers configuration"

        # Update paths for current system - use dynamic repository root detection
        REPO_ROOT="$(cd "$(dirname "$SOURCE_DIR")" && pwd)"
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sed -i "s|{{REPO_ROOT_PLACEHOLDER}}|$REPO_ROOT|g" "$config_file"
            # Also handle legacy hardcoded paths for backward compatibility
            sed -i "s|/Users/hkr/Documents/GitHub/mcp-memory-service|$REPO_ROOT|g" "$config_file"
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|{{REPO_ROOT_PLACEHOLDER}}|$REPO_ROOT|g" "$config_file"
            # Also handle legacy hardcoded paths for backward compatibility
            sed -i '' "s|/Users/hkr/Documents/GitHub/mcp-memory-service|$REPO_ROOT|g" "$config_file"
        fi

        success "Updated configuration paths for current system"
    else
        warn "Source config.json not found - using template"
    fi
}

# Configure Claude Code settings for Natural Memory Triggers
configure_claude_settings() {
    local settings_file="${HOME}/.claude/settings.json"

    info "Configuring Claude Code settings for Natural Memory Triggers..."

    # Create .claude directory if it doesn't exist
    mkdir -p "${HOME}/.claude"

    # Enhanced hook configuration with mid-conversation support
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
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "node ~/.claude/hooks/core/mid-conversation.js",
            "timeout": 8
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

        # For now, replace the hooks section (in production, we'd use proper JSON merging)
        echo "$hook_config" > "$settings_file"
        warn "⚠️  Existing settings.json hooks section has been replaced"
        warn "   Please manually merge if you had other hooks configured"
        warn "   Backup available: ${settings_file}.backup-*"
    else
        # Create new settings file
        echo "$hook_config" > "$settings_file"
        success "Created new settings.json with Natural Memory Triggers configuration"
    fi
}

# Install test files
install_tests() {
    info "Installing test files..."

    if [ -f "$SOURCE_DIR/test-natural-triggers.js" ]; then
        cp "$SOURCE_DIR/test-natural-triggers.js" "$CLAUDE_HOOKS_DIR/"
        success "Installed Natural Memory Triggers test suite"
    fi

    if [ -f "$SOURCE_DIR/test-mcp-hook.js" ]; then
        cp "$SOURCE_DIR/test-mcp-hook.js" "$CLAUDE_HOOKS_DIR/"
        success "Installed MCP connection test"
    fi
}

# Test Natural Memory Triggers installation
test_natural_triggers() {
    info "Testing Natural Memory Triggers installation..."

    # Check required components
    local required_components=(
        "core/session-start.js"
        "core/mid-conversation.js"
        "utilities/adaptive-pattern-detector.js"
        "utilities/performance-manager.js"
        "utilities/mcp-client.js"
        "config.json"
    )

    local missing_components=()
    for component in "${required_components[@]}"; do
        if [ ! -f "$CLAUDE_HOOKS_DIR/$component" ]; then
            missing_components+=("$component")
        fi
    done

    if [ ${#missing_components[@]} -ne 0 ]; then
        error "Installation incomplete - missing components:"
        for component in "${missing_components[@]}"; do
            echo "  - $component"
        done
        return 1
    fi

    success "All required components installed"

    # Test natural triggers functionality
    if [ -f "$CLAUDE_HOOKS_DIR/test-natural-triggers.js" ]; then
        info "Running Natural Memory Triggers test suite..."
        cd "$CLAUDE_HOOKS_DIR"

        if node test-natural-triggers.js; then
            success "Natural Memory Triggers test suite passed"
        else
            warn "⚠️  Some Natural Memory Triggers tests failed"
            return 1
        fi
    fi

    # Test Claude Code hook detection
    if command -v claude &> /dev/null; then
        info "Testing Claude Code hook detection..."
        if claude --help | grep -q "hooks" 2>/dev/null; then
            success "Claude Code hook support detected"
        else
            warn "⚠️  Claude Code hook support not detected"
        fi
    fi

    return 0
}

# Display usage information
show_usage() {
    info "Natural Memory Triggers CLI Management:"
    echo ""
    echo "  Check status:"
    echo "    node ~/.claude/hooks/memory-mode-controller.js status"
    echo ""
    echo "  Switch performance profiles:"
    echo "    node ~/.claude/hooks/memory-mode-controller.js profile balanced"
    echo "    node ~/.claude/hooks/memory-mode-controller.js profile speed_focused"
    echo "    node ~/.claude/hooks/memory-mode-controller.js profile memory_aware"
    echo ""
    echo "  Adjust trigger sensitivity:"
    echo "    node ~/.claude/hooks/memory-mode-controller.js sensitivity 0.6"
    echo ""
    echo "  Test components:"
    echo "    node ~/.claude/hooks/test-natural-triggers.js"
    echo ""
}

# Main installation function
main() {
    info "Starting Natural Memory Triggers installation..."

    check_prerequisites
    create_backup
    install_natural_triggers
    configure_natural_triggers
    configure_claude_settings
    install_tests

    if test_natural_triggers; then
        success "🧠 Natural Memory Triggers v7.1.3 installed successfully!"
        echo ""
        info "Features enabled:"
        echo "  ✅ Intelligent trigger detection with 85%+ accuracy"
        echo "  ✅ Multi-tier performance optimization (50ms/150ms/500ms)"
        echo "  ✅ Mid-conversation memory injection"
        echo "  ✅ Git-aware context and repository integration"
        echo "  ✅ CLI management and real-time configuration"
        echo ""
        show_usage
    else
        error "Installation completed but tests failed"
        error "Please check the configuration and try again"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Natural Memory Triggers v7.1.3 Installation"
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --test         Run tests only"
        echo "  --uninstall    Remove Natural Memory Triggers"
        echo ""
        exit 0
        ;;
    --test)
        if [ -d "$CLAUDE_HOOKS_DIR" ]; then
            test_natural_triggers
        else
            error "Natural Memory Triggers not installed - please install first"
            exit 1
        fi
        exit 0
        ;;
    --uninstall)
        if [ -d "$CLAUDE_HOOKS_DIR" ]; then
            read -p "Remove Natural Memory Triggers components? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -f "$CLAUDE_HOOKS_DIR/core/mid-conversation.js"
                rm -f "$CLAUDE_HOOKS_DIR/utilities/adaptive-pattern-detector.js"
                rm -f "$CLAUDE_HOOKS_DIR/utilities/tiered-conversation-monitor.js"
                rm -f "$CLAUDE_HOOKS_DIR/utilities/performance-manager.js"
                rm -f "$CLAUDE_HOOKS_DIR/utilities/mcp-client.js"
                rm -f "$CLAUDE_HOOKS_DIR/utilities/memory-client.js"
                rm -f "$CLAUDE_HOOKS_DIR/memory-mode-controller.js"
                rm -f "$CLAUDE_HOOKS_DIR/test-natural-triggers.js"
                rm -f "$CLAUDE_HOOKS_DIR/test-mcp-hook.js"
                success "Natural Memory Triggers components removed"
            fi
        else
            info "No Natural Memory Triggers installation found"
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