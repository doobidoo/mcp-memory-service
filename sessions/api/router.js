#!/usr/bin/env node

// ==== IMPORTS ===== //

// ===== STDLIB ===== //
//--//

// ===== LOCAL ===== //
const {
    handleStateCommand,
    handleModeCommand,
    handleFlagsCommand,
    handleStatusCommand,
    handleVersionCommand,
    handleTodosCommand
} = require('./state_commands.js');
const { handleConfigCommand } = require('./config_commands.js');
const { handleProtocolCommand } = require('./protocol_commands.js');
const { handleTaskCommand } = require('./task_commands.js');
let handleKickstartCommand;
let _HAS_KICKSTART = false;
try {
    // Optional: kickstart is only available before completion/cleanup
    handleKickstartCommand = require('./kickstart_commands.js').handleKickstartCommand;
    _HAS_KICKSTART = typeof handleKickstartCommand === 'function';
} catch (e) {
    _HAS_KICKSTART = false;
}
const { handleUninstallCommand } = require('./uninstall_commands.js');
//--//

//-#

// ==== GLOBALS ===== //

const COMMAND_HANDLERS = {
    'protocol': handleProtocolCommand,
    'state': handleStateCommand,
    'mode': handleModeCommand,
    'flags': handleFlagsCommand,
    'status': handleStatusCommand,
    'version': handleVersionCommand,
    'config': handleConfigCommand,
    'todos': handleTodosCommand,
    'tasks': handleTaskCommand,
    'uninstall': handleUninstallCommand,
};

// Register kickstart only if available
if (_HAS_KICKSTART) {
    COMMAND_HANDLERS['kickstart'] = handleKickstartCommand;
}

// Help dictionary for progressive disclosure
const HELP_MESSAGES = {
    "root": `Available subsystems:
  state    - show, mode, task, todos, flags, update
  config   - show, phrases, git, env, features, read, write, tools
  tasks    - idx, start
  protocol - startup-load
  uninstall - Remove cc-sessions framework${_HAS_KICKSTART ? `
  kickstart - full, subagents, next, complete` : ''}`,

    "state": `Available state commands:
  show [section]   - Display state (task, todos, flags, mode)
  mode <mode>      - Switch mode (discussion/no, bypass/off, implementation/go)
  task <action>    - Manage task (clear, show, restore <file>)
  todos <action>   - Manage todos (clear)
  flags <action>   - Manage flags (clear, clear-context)
  update <action>  - Manage updates (status, suppress, check)`,

    "config": `Available config commands:
  show             - Display current configuration
  phrases <action> - Manage trigger phrases (list, add, remove)
  git <action>     - Manage git preferences (show, add, branch, commit, merge, push, repo)
  env <action>     - Manage environment (show, os, shell, name)
  features <action> - Manage features (show, set, toggle)
  read <action>    - Manage bash read patterns (list, add, remove)
  write <action>   - Manage bash write patterns (list, add, remove)
  tools <action>   - Manage blocked tools (list, block, unblock)`,

    "config.phrases": `Available phrases commands:
  list [category]             - List trigger phrases
  add <category> "<phrase>"   - Add trigger phrase
  remove <category> "<phrase>" - Remove trigger phrase

Valid categories: go, no, create, start, complete, compact`,

    "config.git": `Available git commands:
  show                - Display git preferences
  add <ask|all>       - Set staging behavior
  branch <name>       - Set default branch
  commit <style>      - Set commit style (conventional, simple, detailed)
  merge <auto|ask>    - Set merge behavior
  push <auto|ask>     - Set push behavior
  repo <super|mono>   - Set repository type`,

    "config.env": `Available env commands:
  show            - Display environment settings
  os <os>         - Set operating system (linux, macos, windows)
  shell <shell>   - Set shell (bash, zsh, fish, powershell, cmd)
  name <name>     - Set developer name`,

    "config.features": `Available features commands:
  show              - Display all feature flags
  set <key> <value> - Set feature value
  toggle <key>      - Toggle feature boolean or cycle enum

Features: branch_enforcement, task_detection, auto_ultrathink, icon_style, warn_85, warn_90`,

    "config.read": `Available read commands:
  list              - List all bash read patterns
  add <pattern>     - Add pattern to read list
  remove <pattern>  - Remove pattern from read list`,

    "config.write": `Available write commands:
  list              - List all bash write patterns
  add <pattern>     - Add pattern to write list
  remove <pattern>  - Remove pattern from write list`,

    "config.tools": `Available tools commands:
  list                - List all blocked tools
  block <ToolName>    - Block tool in discussion mode
  unblock <ToolName>  - Unblock tool`,

    "tasks": `Available tasks commands:
  idx list        - List all task indexes
  idx <name>      - Show tasks in specific index
  start @<task>   - Start working on a task`,
};

//-#

// ==== DECLARATIONS ===== //
//-#

// ==== CLASSES ===== //
//-#

// ==== FUNCTIONS ===== //

function resolveHelp(commandPath) {
    /**
     * Resolve help text for failed command parsing.
     *
     * Args:
     *     commandPath: Array of successfully parsed command tokens before failure
     *                 Example: [] for root, ['config'] for config subsystem,
     *                          ['config', 'phrases'] for phrases commands
     *
     * Returns:
     *     Appropriate help text for the command level
     */
    // Build key from command path
    const key = commandPath.length > 0 ? commandPath.join('.') : 'root';

    // Return help for this level, or root help if not found
    return HELP_MESSAGES[key] || HELP_MESSAGES['root'];
}

function routeCommand(command, args, jsonOutput = false, fromSlash = false) {
    /**
     * Route a command to the appropriate handler.
     *
     * Args:
     *     command: Main command to execute
     *     args: Additional arguments for the command
     *     jsonOutput: Whether to format output as JSON
     *     fromSlash: Whether the command came from a slash command
     *
     * Returns:
     *     Command result (dict for JSON, string for human-readable)
     *
     * Throws:
     *     Error: If command is unknown or invalid
     */

    // Special handling for slash command router
    if (command === 'slash') {
        if (!args || args.length === 0) {
            return formatSlashHelp();
        }

        const subsystem = args[0].toLowerCase();
        const subsystemArgs = args.length > 1 ? args.slice(1) : [];

        // Route to appropriate subsystem
        const subsystems = ['tasks', 'state', 'config', 'uninstall'];
        if (_HAS_KICKSTART) subsystems.push('kickstart');
        if (subsystems.includes(subsystem)) {
            return routeCommand(subsystem, subsystemArgs, jsonOutput, true);
        } else if (subsystem === 'bypass') {
            return routeCommand('mode', ['bypass'], jsonOutput, true);
        } else if (subsystem === 'help') {
            return formatSlashHelp();
        } else {
            return `Unknown subsystem: ${subsystem}\n\nValid subsystems: tasks, state, config, uninstall, bypass\n\nUse '/sessions help' for full usage information.`;
        }
    }

    if (!(command in COMMAND_HANDLERS)) {
        if (fromSlash) {
            return resolveHelp([]);
        }
        throw new Error(`Unknown command: ${command}. Available commands: ${Object.keys(COMMAND_HANDLERS).join(', ')}`);
    }

    const handler = COMMAND_HANDLERS[command];

    // Wrap handler calls with error recovery when called from slash
    if (fromSlash) {
        try {
            // Pass fromSlash to commands that support it
            if (['config', 'state', 'tasks', 'uninstall'].includes(command)) {
                return handler(args, jsonOutput, fromSlash);
            } else {
                // For commands that don't support fromSlash, add it to args for backward compatibility
                if (!args.includes('--from-slash')) {
                    args = [...args, '--from-slash'];
                }
                return handler(args, jsonOutput);
            }
        } catch (e) {
            // Return contextual help instead of throwing
            // Try to determine where in the command tree we are
            return resolveHelp([command]);
        }
    } else {
        // Normal API calls - let exceptions propagate
        if (['config', 'state', 'tasks', 'uninstall'].includes(command)) {
            return handler(args, jsonOutput, fromSlash);
        } else {
            // For commands that don't support fromSlash, add it to args for backward compatibility
            if (!args.includes('--from-slash')) {
                args = [...args, '--from-slash'];
            }
            return handler(args, jsonOutput);
        }
    }
}

function formatSlashHelp() {
    /**Format help output for unified /sessions slash command.*/
    const lines = [
        "# /sessions - Unified Sessions Management",
        "",
        "Manage all aspects of your Claude Code session from one command.",
        "",
        "## Available Subsystems",
        "",
        "### Tasks",
        "  /sessions tasks idx list        - List all task indexes",
        "  /sessions tasks idx <name>      - Show pending tasks in index",
        "  /sessions tasks start @<name>   - Start working on a task",
        "",
        "### State",
        "  /sessions state                 - Display current state",
        "  /sessions state show [section]  - Show specific section (task, todos, flags, mode)",
        "  /sessions state mode <mode>     - Switch mode (discussion/no, bypass/off)",
        "  /sessions state task <action>   - Manage task (clear, show, restore <file>)",
        "  /sessions state todos <action>  - Manage todos (clear)",
        "  /sessions state flags <action>  - Manage flags (clear, clear-context)",
        "  /sessions state update ...      - Manage update notifications (status, suppress, check)",
        "",
        "### Config",
        "  /sessions config show           - Display current configuration",
        "  /sessions config trigger ...    - Manage trigger phrases",
        "  /sessions config git ...        - Manage git preferences",
        "  /sessions config env ...        - Manage environment settings",
        "  /sessions config features ...   - Manage feature toggles",
        "  /sessions config read ...       - Manage bash read patterns",
        "  /sessions config write ...      - Manage bash write patterns",
        "  /sessions config tools ...      - Manage blocked tools",
        "",
    ];
    if (_HAS_KICKSTART) {
        lines.push(
            "### Kickstart",
            "  /sessions kickstart full          - Initialize full kickstart onboarding",
            "  /sessions kickstart subagents     - Initialize subagents-only onboarding",
            "  /sessions kickstart next          - Load the next module",
            "  /sessions kickstart complete      - Finish kickstart and cleanup",
            ""
        );
    }
    lines.push(
        "### Uninstall",
        "  /sessions uninstall             - Safely remove cc-sessions framework",
        "  /sessions uninstall --dry-run   - Preview what would be removed",
        "",
        "### Quick Shortcuts",
        "  /sessions bypass                - Disable bypass mode (return to normal)",
        "",
        "## Quick Reference",
        "",
        "**Common Operations:**",
        "  /sessions tasks idx list                    # Browse available tasks",
        "  /sessions tasks start @my-task              # Start a task",
        "  /sessions state show task                   # Check current task",
        "  /sessions state mode no                     # Switch to discussion mode",
        "  /sessions config show                       # View all settings",
        "",
        "**Use '/sessions <subsystem> help' for detailed help on each subsystem**",
    );
    return lines.join('\n');
}

//-#

// ==== EXPORTS ===== //
module.exports = {
    routeCommand,
    formatSlashHelp
};
//-#
