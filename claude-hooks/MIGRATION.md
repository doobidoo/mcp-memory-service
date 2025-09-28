# Migration Guide: Unified Python Hook Installer

## 🎯 Overview

The Claude Code Memory Awareness Hooks have been **consolidated into a single, unified Python installer** that replaces all previous platform-specific installers.

## 📋 **What Changed**

### ❌ **Deprecated (Removed)**
- `install.sh` - Legacy shell installer
- `install-natural-triggers.sh` - Natural triggers shell installer
- `install_claude_hooks_windows.bat` - Windows batch installer

### ✅ **New Unified Solution**
- `install_hooks.py` - **Single cross-platform Python installer**

## 🚀 **Migration Steps**

### For New Installations
```bash
# Navigate to hooks directory
cd claude-hooks

# Install basic memory awareness hooks
python install_hooks.py --basic

# OR install Natural Memory Triggers v7.1.3 (recommended)
python install_hooks.py --natural-triggers

# OR install everything
python install_hooks.py --all
```

### For Existing Users
```bash
# Uninstall old hooks (optional, installer handles upgrades)
python install_hooks.py --uninstall

# Install fresh with new installer
python install_hooks.py --natural-triggers
```

## ✨ **Benefits of Unified Installer**

### 🔧 **Technical Improvements**
- **Intelligent JSON merging** - Preserves existing Claude Code hook configurations
- **Cross-platform compatibility** - Works on Windows, macOS, and Linux
- **Dynamic path resolution** - No hardcoded paths, works in any location
- **Atomic installations** - Automatic rollback on failure
- **Comprehensive backups** - Timestamped backups before changes
- **Empty directory cleanup** - Proper uninstall process

### 🎯 **User Experience**
- **Single installation method** across all platforms
- **Consistent CLI interface** with clear options
- **Dry-run support** for testing without changes
- **Enhanced error handling** with detailed feedback
- **CLI management tools** for real-time configuration

## 📖 **Advanced Usage**

### Available Options
```bash
# Test installation without making changes
python install_hooks.py --dry-run --natural-triggers

# Install only basic hooks
python install_hooks.py --basic

# Install Natural Memory Triggers (recommended)
python install_hooks.py --natural-triggers

# Install everything (basic + natural triggers)
python install_hooks.py --all

# Uninstall all hooks
python install_hooks.py --uninstall

# Get help
python install_hooks.py --help
```

### CLI Management (Natural Memory Triggers)
```bash
# Check status
node ~/.claude/hooks/memory-mode-controller.js status

# Switch performance profiles
node ~/.claude/hooks/memory-mode-controller.js profile balanced
node ~/.claude/hooks/memory-mode-controller.js profile speed_focused
node ~/.claude/hooks/memory-mode-controller.js profile memory_aware

# Adjust sensitivity
node ~/.claude/hooks/memory-mode-controller.js sensitivity 0.7
```

## 🔧 **Integration with Main Installer**

The main MCP Memory Service installer now uses the unified hook installer:

```bash
# Install service + basic hooks
python scripts/installation/install.py --install-hooks

# Install service + Natural Memory Triggers
python scripts/installation/install.py --install-natural-triggers
```

## 🛠 **Troubleshooting**

### Common Issues

**Q: Can I still use the old shell scripts?**
A: No, they have been removed. The unified Python installer provides all functionality with improved reliability.

**Q: Will my existing hook configuration be preserved?**
A: Yes, the unified installer intelligently merges configurations and preserves existing hooks.

**Q: What if I have custom modifications to the old installers?**
A: The unified installer is designed to be extensible. Please file an issue if you need specific functionality.

**Q: Does this work on Windows?**
A: Yes, the unified Python installer provides full Windows support with proper path handling.

## 📞 **Support**

If you encounter issues:

1. **Check prerequisites**: Ensure Claude Code CLI, Node.js, and Python are installed
2. **Test with dry-run**: Use `--dry-run` flag to identify issues
3. **Check logs**: The installer provides detailed error messages
4. **File an issue**: [GitHub Issues](https://github.com/doobidoo/mcp-memory-service/issues)

## 🎉 **Benefits Summary**

The unified installer provides:
- ✅ **Better reliability** across all platforms
- ✅ **Safer installations** with intelligent configuration merging
- ✅ **Consistent experience** regardless of operating system
- ✅ **Advanced features** like Natural Memory Triggers v7.1.3
- ✅ **Professional tooling** with comprehensive testing and validation

This migration represents a significant improvement in the installation experience while maintaining full backward compatibility for existing users.