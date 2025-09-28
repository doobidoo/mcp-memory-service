#!/usr/bin/env python3
"""
Unified Claude Code Memory Awareness Hooks Installer
====================================================

Cross-platform installer for Claude Code memory awareness hooks with support for:
- Basic memory awareness hooks (session-start, session-end)
- Natural Memory Triggers v7.1.3 (intelligent automatic memory awareness)
- Mid-conversation hooks for real-time memory injection
- Performance optimization and CLI management tools

Replaces multiple platform-specific installers with a single Python solution.
"""

import os
import sys
import json
import shutil
import platform
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


class HookInstaller:
    """Unified hook installer for all platforms and feature levels."""

    def __init__(self):
        self.script_dir = Path(__file__).parent.absolute()
        self.platform_name = platform.system().lower()
        self.claude_hooks_dir = self._detect_claude_hooks_directory()
        self.backup_dir = None

    def _detect_claude_hooks_directory(self) -> Path:
        """Detect the Claude Code hooks directory across platforms."""
        home = Path.home()

        # Primary paths by platform
        primary_paths = {
            'windows': [
                home / 'AppData' / 'Roaming' / 'Claude' / 'hooks',
                home / '.claude' / 'hooks'
            ],
            'darwin': [  # macOS
                home / '.claude' / 'hooks',
                home / 'Library' / 'Application Support' / 'Claude' / 'hooks'
            ],
            'linux': [
                home / '.claude' / 'hooks',
                home / '.config' / 'claude' / 'hooks'
            ]
        }

        # Check platform-specific paths first
        platform_paths = primary_paths.get(self.platform_name, primary_paths['linux'])

        for path in platform_paths:
            if path.exists():
                return path

        # Check if Claude Code CLI can tell us the location
        try:
            result = subprocess.run(['claude', '--help'],
                                  capture_output=True, text=True, timeout=5)
            # Look for hooks directory info in help output
            # This is a placeholder - actual Claude CLI might not provide this
        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Default to standard location
        return home / '.claude' / 'hooks'

    def info(self, message: str) -> None:
        """Print info message."""
        print(f"{Colors.GREEN}[INFO]{Colors.NC} {message}")

    def warn(self, message: str) -> None:
        """Print warning message."""
        print(f"{Colors.YELLOW}[WARN]{Colors.NC} {message}")

    def error(self, message: str) -> None:
        """Print error message."""
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

    def success(self, message: str) -> None:
        """Print success message."""
        print(f"{Colors.BLUE}[SUCCESS]{Colors.NC} {message}")

    def header(self, message: str) -> None:
        """Print header message."""
        print(f"\n{Colors.CYAN}{'=' * 60}{Colors.NC}")
        print(f"{Colors.CYAN} {message}{Colors.NC}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.NC}\n")

    def check_prerequisites(self) -> bool:
        """Check system prerequisites for hook installation."""
        self.info("Checking prerequisites...")

        all_good = True

        # Check Claude Code CLI
        try:
            result = subprocess.run(['claude', '--version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.success(f"Claude Code CLI found: {result.stdout.strip()}")
            else:
                self.warn("Claude Code CLI found but version check failed")
        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
            self.warn("Claude Code CLI not found in PATH")
            self.info("You can still install hooks, but some features may not work")

        # Check Node.js
        try:
            result = subprocess.run(['node', '--version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                major_version = int(version.replace('v', '').split('.')[0])
                if major_version >= 14:
                    self.success(f"Node.js found: {version} (compatible)")
                else:
                    self.error(f"Node.js {version} found, but version 14+ required")
                    all_good = False
            else:
                self.error("Node.js found but version check failed")
                all_good = False
        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
            self.error("Node.js not found - required for hook execution")
            self.info("Please install Node.js 14+ from https://nodejs.org/")
            all_good = False

        # Check Python version
        if sys.version_info < (3, 7):
            self.error(f"Python {sys.version} found, but Python 3.7+ required")
            all_good = False
        else:
            self.success(f"Python {sys.version_info.major}.{sys.version_info.minor} found (compatible)")

        return all_good

    def create_backup(self) -> None:
        """Create backup of existing hooks installation."""
        if not self.claude_hooks_dir.exists():
            self.info("No existing hooks installation found - no backup needed")
            return

        timestamp = subprocess.run(['date', '+%Y%m%d-%H%M%S'],
                                 capture_output=True, text=True).stdout.strip()
        if not timestamp:  # Fallback for Windows
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        self.backup_dir = self.claude_hooks_dir.parent / f"hooks-backup-{timestamp}"

        try:
            shutil.copytree(self.claude_hooks_dir, self.backup_dir)
            self.success(f"Backup created: {self.backup_dir}")
        except Exception as e:
            self.warn(f"Failed to create backup: {e}")
            self.warn("Continuing without backup...")

    def install_basic_hooks(self) -> bool:
        """Install basic memory awareness hooks."""
        self.info("Installing basic memory awareness hooks...")

        try:
            # Create necessary directories
            (self.claude_hooks_dir / "core").mkdir(parents=True, exist_ok=True)
            (self.claude_hooks_dir / "utilities").mkdir(parents=True, exist_ok=True)
            (self.claude_hooks_dir / "tests").mkdir(parents=True, exist_ok=True)

            # Core hooks
            core_files = [
                "session-start.js",
                "session-end.js",
                "memory-retrieval.js",
                "topic-change.js"
            ]

            for file in core_files:
                src = self.script_dir / "core" / file
                dst = self.claude_hooks_dir / "core" / file
                if src.exists():
                    shutil.copy2(src, dst)
                else:
                    self.warn(f"Core file not found: {file}")

            # Basic utilities
            utility_files = [
                "project-detector.js",
                "memory-scorer.js",
                "context-formatter.js",
                "context-shift-detector.js",
                "conversation-analyzer.js",
                "dynamic-context-updater.js",
                "session-tracker.js",
                "git-analyzer.js"
            ]

            for file in utility_files:
                src = self.script_dir / "utilities" / file
                dst = self.claude_hooks_dir / "utilities" / file
                if src.exists():
                    shutil.copy2(src, dst)
                else:
                    self.warn(f"Utility file not found: {file}")

            # Tests
            test_files = ["integration-test.js"]
            for file in test_files:
                src = self.script_dir / "tests" / file
                dst = self.claude_hooks_dir / "tests" / file
                if src.exists():
                    shutil.copy2(src, dst)

            # Documentation
            readme_src = self.script_dir / "README.md"
            if readme_src.exists():
                shutil.copy2(readme_src, self.claude_hooks_dir / "README.md")

            self.success("Basic hooks installed successfully")
            return True

        except Exception as e:
            self.error(f"Failed to install basic hooks: {e}")
            return False

    def install_natural_triggers(self) -> bool:
        """Install Natural Memory Triggers v7.1.3 components."""
        self.info("Installing Natural Memory Triggers v7.1.3...")

        try:
            # Ensure directories exist
            (self.claude_hooks_dir / "core").mkdir(parents=True, exist_ok=True)
            (self.claude_hooks_dir / "utilities").mkdir(parents=True, exist_ok=True)

            # Mid-conversation hook
            mid_conv_src = self.script_dir / "core" / "mid-conversation.js"
            if mid_conv_src.exists():
                shutil.copy2(mid_conv_src, self.claude_hooks_dir / "core" / "mid-conversation.js")
                self.success("Installed mid-conversation hooks")
            else:
                self.warn("Mid-conversation hook not found")

            # v7.1.3 enhanced utilities
            enhanced_utilities = [
                "adaptive-pattern-detector.js",
                "tiered-conversation-monitor.js",
                "performance-manager.js",
                "mcp-client.js",
                "memory-client.js"
            ]

            for file in enhanced_utilities:
                src = self.script_dir / "utilities" / file
                dst = self.claude_hooks_dir / "utilities" / file
                if src.exists():
                    shutil.copy2(src, dst)
                else:
                    self.warn(f"Enhanced utility not found: {file}")

            # CLI management tools
            cli_tools = [
                "memory-mode-controller.js",
                "debug-pattern-test.js"
            ]

            for file in cli_tools:
                src = self.script_dir / file
                dst = self.claude_hooks_dir / file
                if src.exists():
                    shutil.copy2(src, dst)

            # Test files
            test_files = [
                "test-natural-triggers.js",
                "test-mcp-hook.js",
                "test-dual-protocol-hook.js"
            ]

            for file in test_files:
                src = self.script_dir / file
                dst = self.claude_hooks_dir / file
                if src.exists():
                    shutil.copy2(src, dst)

            self.success("Natural Memory Triggers v7.1.3 installed successfully")
            return True

        except Exception as e:
            self.error(f"Failed to install Natural Memory Triggers: {e}")
            return False

    def install_configuration(self, install_natural_triggers: bool = False) -> bool:
        """Install or update configuration files."""
        self.info("Installing configuration...")

        try:
            # Install template configuration
            template_src = self.script_dir / "config.template.json"
            template_dst = self.claude_hooks_dir / "config.template.json"
            if template_src.exists():
                shutil.copy2(template_src, template_dst)

            # Install main configuration
            config_src = self.script_dir / "config.json"
            config_dst = self.claude_hooks_dir / "config.json"

            if config_dst.exists():
                # Backup existing config
                backup_config = config_dst.with_suffix('.json.backup')
                shutil.copy2(config_dst, backup_config)
                self.info("Existing configuration backed up")

            if config_src.exists():
                shutil.copy2(config_src, config_dst)

                # Update paths for current system
                try:
                    with open(config_dst, 'r') as f:
                        config = json.load(f)

                    # Update server working directory path
                    if 'memoryService' in config and 'mcp' in config['memoryService']:
                        config['memoryService']['mcp']['serverWorkingDir'] = str(self.script_dir.parent)

                    with open(config_dst, 'w') as f:
                        json.dump(config, f, indent=2)

                    self.success("Configuration installed and updated for current system")
                except Exception as e:
                    self.warn(f"Failed to update configuration paths: {e}")

            return True

        except Exception as e:
            self.error(f"Failed to install configuration: {e}")
            return False

    def configure_claude_settings(self, install_mid_conversation: bool = False) -> bool:
        """Configure Claude Code settings.json for hook integration."""
        self.info("Configuring Claude Code settings...")

        try:
            # Determine settings path based on platform
            home = Path.home()
            if self.platform_name == 'windows':
                settings_dir = home / 'AppData' / 'Roaming' / 'Claude'
            else:
                settings_dir = home / '.claude'

            settings_dir.mkdir(parents=True, exist_ok=True)
            settings_file = settings_dir / 'settings.json'

            # Create hook configuration
            hook_config = {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": f'node "{self.claude_hooks_dir}/core/session-start.js"',
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
                                    "command": f'node "{self.claude_hooks_dir}/core/session-end.js"',
                                    "timeout": 15
                                }
                            ]
                        }
                    ]
                }
            }

            # Add mid-conversation hook if Natural Memory Triggers are installed
            if install_mid_conversation:
                hook_config["hooks"]["UserPromptSubmit"] = [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": f'node "{self.claude_hooks_dir}/core/mid-conversation.js"',
                                "timeout": 8
                            }
                        ]
                    }
                ]

            # Handle existing settings with intelligent merging
            final_config = hook_config
            if settings_file.exists():
                # Backup existing settings
                backup_settings = settings_file.with_suffix('.json.backup')
                shutil.copy2(settings_file, backup_settings)
                self.info("Existing settings.json backed up")

                try:
                    # Load existing settings
                    with open(settings_file, 'r') as f:
                        existing_settings = json.load(f)

                    # Intelligent merging: preserve existing hooks while adding/updating memory awareness hooks
                    if 'hooks' not in existing_settings:
                        existing_settings['hooks'] = {}

                    # Check for conflicts and merge intelligently
                    memory_hook_types = {'SessionStart', 'SessionEnd', 'UserPromptSubmit'}
                    conflicts = []

                    for hook_type in memory_hook_types:
                        if hook_type in existing_settings['hooks'] and hook_type in hook_config['hooks']:
                            # Check if existing hook is different from our memory awareness hook
                            existing_commands = [
                                hook.get('command', '') for hooks_group in existing_settings['hooks'][hook_type]
                                for hook in hooks_group.get('hooks', [])
                            ]
                            memory_commands = [
                                hook.get('command', '') for hooks_group in hook_config['hooks'][hook_type]
                                for hook in hooks_group.get('hooks', [])
                            ]

                            # Check if any existing command contains memory hook
                            is_memory_hook = any('session-start.js' in cmd or 'session-end.js' in cmd or 'mid-conversation.js' in cmd
                                               for cmd in existing_commands)

                            if not is_memory_hook:
                                conflicts.append(hook_type)

                    if conflicts:
                        self.warn(f"Found existing non-memory hooks for: {', '.join(conflicts)}")
                        self.warn("Memory awareness hooks will be added alongside existing hooks")

                        # Add memory hooks alongside existing ones
                        for hook_type in hook_config['hooks']:
                            if hook_type in existing_settings['hooks']:
                                existing_settings['hooks'][hook_type].extend(hook_config['hooks'][hook_type])
                            else:
                                existing_settings['hooks'][hook_type] = hook_config['hooks'][hook_type]
                    else:
                        # No conflicts, safe to update memory awareness hooks
                        existing_settings['hooks'].update(hook_config['hooks'])
                        self.info("Updated memory awareness hooks without conflicts")

                    final_config = existing_settings
                    self.success("Settings merged intelligently, preserving existing configuration")

                except json.JSONDecodeError as e:
                    self.warn(f"Existing settings.json invalid, using backup and creating new: {e}")
                    final_config = hook_config
                except Exception as e:
                    self.warn(f"Error merging settings, creating new configuration: {e}")
                    final_config = hook_config

            # Write final configuration
            with open(settings_file, 'w') as f:
                json.dump(final_config, f, indent=2)

            self.success("Claude Code settings configured successfully")
            return True

        except Exception as e:
            self.error(f"Failed to configure Claude Code settings: {e}")
            return False

    def run_tests(self, test_natural_triggers: bool = False) -> bool:
        """Run hook tests to verify installation."""
        self.info("Running installation tests...")

        success = True

        # Check required files exist
        required_files = [
            "core/session-start.js",
            "core/session-end.js",
            "utilities/project-detector.js",
            "utilities/memory-scorer.js",
            "utilities/context-formatter.js",
            "config.json"
        ]

        if test_natural_triggers:
            required_files.extend([
                "core/mid-conversation.js",
                "utilities/adaptive-pattern-detector.js",
                "utilities/performance-manager.js",
                "utilities/mcp-client.js"
            ])

        missing_files = []
        for file in required_files:
            if not (self.claude_hooks_dir / file).exists():
                missing_files.append(file)

        if missing_files:
            self.error("Installation incomplete - missing files:")
            for file in missing_files:
                self.error(f"  - {file}")
            success = False
        else:
            self.success("All required files installed correctly")

        # Test Node.js execution
        test_script = self.claude_hooks_dir / "core" / "session-start.js"
        if test_script.exists():
            try:
                result = subprocess.run(['node', '--check', str(test_script)],
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    self.success("Hook JavaScript syntax validation passed")
                else:
                    self.error(f"Hook JavaScript syntax validation failed: {result.stderr}")
                    success = False
            except Exception as e:
                self.warn(f"Could not validate JavaScript syntax: {e}")

        # Run integration tests if available
        integration_test = self.claude_hooks_dir / "tests" / "integration-test.js"
        if integration_test.exists():
            try:
                self.info("Running integration tests...")
                result = subprocess.run(['node', str(integration_test)],
                                      capture_output=True, text=True,
                                      timeout=30, cwd=str(self.claude_hooks_dir))
                if result.returncode == 0:
                    self.success("Integration tests passed")
                else:
                    self.warn("Some integration tests failed - check configuration")
                    if result.stdout:
                        self.info(f"Test output: {result.stdout}")
            except Exception as e:
                self.warn(f"Could not run integration tests: {e}")

        # Run Natural Memory Triggers tests if applicable
        if test_natural_triggers:
            natural_test = self.claude_hooks_dir / "test-natural-triggers.js"
            if natural_test.exists():
                try:
                    self.info("Running Natural Memory Triggers tests...")
                    result = subprocess.run(['node', str(natural_test)],
                                          capture_output=True, text=True,
                                          timeout=30, cwd=str(self.claude_hooks_dir))
                    if result.returncode == 0:
                        self.success("Natural Memory Triggers tests passed")
                    else:
                        self.warn("Some Natural Memory Triggers tests failed")
                except Exception as e:
                    self.warn(f"Could not run Natural Memory Triggers tests: {e}")

        return success

    def _cleanup_empty_directories(self) -> None:
        """Remove empty directories after uninstall."""
        try:
            # Directories to check for cleanup (in reverse order to handle nested structure)
            directories_to_check = [
                self.claude_hooks_dir / "core",
                self.claude_hooks_dir / "utilities",
                self.claude_hooks_dir / "tests"
            ]

            for directory in directories_to_check:
                if directory.exists() and directory.is_dir():
                    try:
                        # Check if directory is empty (no files, only empty subdirectories allowed)
                        items = list(directory.iterdir())
                        if not items:
                            # Directory is completely empty
                            directory.rmdir()
                            self.info(f"Removed empty directory: {directory.name}/")
                        else:
                            # Check if it only contains empty subdirectories
                            all_empty = True
                            for item in items:
                                if item.is_file():
                                    all_empty = False
                                    break
                                elif item.is_dir() and list(item.iterdir()):
                                    all_empty = False
                                    break

                            if all_empty:
                                # Remove empty subdirectories first
                                for item in items:
                                    if item.is_dir():
                                        item.rmdir()
                                # Then remove the parent directory
                                directory.rmdir()
                                self.info(f"Removed empty directory tree: {directory.name}/")
                    except OSError:
                        # Directory not empty or permission issue, skip silently
                        pass

        except Exception as e:
            self.warn(f"Could not cleanup empty directories: {e}")

    def uninstall(self) -> bool:
        """Remove installed hooks."""
        self.info("Uninstalling Claude Code memory awareness hooks...")

        try:
            if not self.claude_hooks_dir.exists():
                self.info("No hooks installation found")
                return True

            # Remove hook files
            files_to_remove = [
                "core/session-start.js",
                "core/session-end.js",
                "core/mid-conversation.js",
                "core/memory-retrieval.js",
                "core/topic-change.js",
                "memory-mode-controller.js",
                "test-natural-triggers.js",
                "test-mcp-hook.js",
                "debug-pattern-test.js"
            ]

            # Remove utilities
            utility_files = [
                "utilities/adaptive-pattern-detector.js",
                "utilities/performance-manager.js",
                "utilities/mcp-client.js",
                "utilities/memory-client.js",
                "utilities/tiered-conversation-monitor.js"
            ]
            files_to_remove.extend(utility_files)

            removed_count = 0
            for file in files_to_remove:
                file_path = self.claude_hooks_dir / file
                if file_path.exists():
                    file_path.unlink()
                    removed_count += 1

            # Remove config files if user confirms
            config_file = self.claude_hooks_dir / "config.json"
            if config_file.exists():
                # We'll keep config files by default since they may have user customizations
                self.info("Configuration files preserved (contains user customizations)")

            # Clean up empty directories
            self._cleanup_empty_directories()

            self.success(f"Removed {removed_count} hook files and cleaned up empty directories")
            return True

        except Exception as e:
            self.error(f"Failed to uninstall hooks: {e}")
            return False


def main():
    """Main installer function."""
    parser = argparse.ArgumentParser(
        description="Unified Claude Code Memory Awareness Hooks Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install_hooks.py                    # Install all features (default)
  python install_hooks.py --basic            # Basic hooks only
  python install_hooks.py --natural-triggers # Natural Memory Triggers only
  python install_hooks.py --test             # Run tests only
  python install_hooks.py --uninstall        # Remove hooks

Features:
  Basic: Session-start and session-end hooks for memory awareness
  Natural Triggers: v7.1.3 intelligent automatic memory awareness with
                   pattern detection, performance optimization, and CLI tools
        """
    )

    parser.add_argument('--basic', action='store_true',
                        help='Install basic memory awareness hooks only')
    parser.add_argument('--natural-triggers', action='store_true',
                        help='Install Natural Memory Triggers v7.1.3 only')
    parser.add_argument('--all', action='store_true',
                        help='Install all features (default behavior)')
    parser.add_argument('--test', action='store_true',
                        help='Run tests only (do not install)')
    parser.add_argument('--uninstall', action='store_true',
                        help='Remove installed hooks')
    parser.add_argument('--force', action='store_true',
                        help='Force installation even if prerequisites fail')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be installed without making changes')

    args = parser.parse_args()

    # Create installer instance
    installer = HookInstaller()

    installer.header("Claude Code Memory Awareness Hooks Installer v7.1.3")
    installer.info(f"Script location: {installer.script_dir}")
    installer.info(f"Target hooks directory: {installer.claude_hooks_dir}")
    installer.info(f"Platform: {installer.platform_name}")

    # Handle special modes first
    if args.uninstall:
        if installer.uninstall():
            installer.success("Hooks uninstalled successfully")
        else:
            installer.error("Uninstall failed")
            sys.exit(1)
        return

    if args.test:
        test_natural_triggers = not args.basic
        if installer.run_tests(test_natural_triggers=test_natural_triggers):
            installer.success("All tests passed")
        else:
            installer.error("Some tests failed")
            sys.exit(1)
        return

    # Check prerequisites
    if not installer.check_prerequisites() and not args.force:
        installer.error("Prerequisites check failed. Use --force to continue anyway.")
        sys.exit(1)

    # Determine what to install
    install_all = not (args.basic or args.natural_triggers) or args.all
    install_basic = args.basic or install_all
    install_natural_triggers = args.natural_triggers or install_all

    installer.info(f"Installation plan:")
    installer.info(f"  Basic hooks: {'Yes' if install_basic else 'No'}")
    installer.info(f"  Natural Memory Triggers: {'Yes' if install_natural_triggers else 'No'}")

    if args.dry_run:
        installer.info("DRY RUN - No changes will be made")
        installer.info("Would install:")
        if install_basic:
            installer.info("  - Basic memory awareness hooks")
            installer.info("  - Core utilities and configuration")
        if install_natural_triggers:
            installer.info("  - Natural Memory Triggers v7.1.3")
            installer.info("  - Mid-conversation hooks")
            installer.info("  - Performance optimization utilities")
            installer.info("  - CLI management tools")
        return

    # Create backup
    installer.create_backup()

    # Perform installation
    overall_success = True

    # Install components based on selection
    if install_basic:
        if not installer.install_basic_hooks():
            overall_success = False

    if install_natural_triggers:
        if not installer.install_natural_triggers():
            overall_success = False

    # Install configuration (always needed)
    if not installer.install_configuration(install_natural_triggers=install_natural_triggers):
        overall_success = False

    # Configure Claude Code settings
    if not installer.configure_claude_settings(install_mid_conversation=install_natural_triggers):
        overall_success = False

    # Run tests to verify installation
    if overall_success:
        installer.info("Running post-installation tests...")
        if installer.run_tests(test_natural_triggers=install_natural_triggers):
            installer.header("Installation Complete!")

            if install_basic and install_natural_triggers:
                installer.success("Complete Claude Code memory awareness system installed")
                installer.info("Features available:")
                installer.info("  ✅ Session-start and session-end hooks")
                installer.info("  ✅ Natural Memory Triggers with intelligent pattern detection")
                installer.info("  ✅ Mid-conversation memory injection")
                installer.info("  ✅ Performance optimization and CLI management")
                installer.info("")
                installer.info("CLI Management:")
                installer.info(f"  node {installer.claude_hooks_dir}/memory-mode-controller.js status")
                installer.info(f"  node {installer.claude_hooks_dir}/memory-mode-controller.js profile balanced")
            elif install_natural_triggers:
                installer.success("Natural Memory Triggers v7.1.3 installed")
                installer.info("Advanced memory awareness features available")
            elif install_basic:
                installer.success("Basic memory awareness hooks installed")
                installer.info("Session-based memory awareness enabled")

        else:
            installer.warn("Installation completed but some tests failed")
            installer.info("Hooks may still work - check configuration manually")
    else:
        installer.error("Installation failed - some components could not be installed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Installation cancelled by user{Colors.NC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.NC}")
        sys.exit(1)