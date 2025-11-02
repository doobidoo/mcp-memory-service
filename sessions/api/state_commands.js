#!/usr/bin/env node

// ==== IMPORTS ===== //

// ===== STDLIB ===== //
const fs = require('fs');
const path = require('path');
//--//

// ===== 3RD-PARTY ===== //
//--//

// ===== LOCAL ===== //
const {
    loadState,
    editState,
    loadConfig,
    Mode,
    TaskState,
    PROJECT_ROOT,
    STATE_FILE
} = require('../hooks/shared_state.js');
//--//

//-#

// ==== GLOBALS ===== //
const STATE = loadState();
//-#

// ==== FUNCTIONS ===== //

//!> State inspection handlers
function handleStateCommand(args, jsonOutput = false, fromSlash = false) {
    /**
     * Handle state inspection and management commands.
     *
     * Usage:
     *     state                       - Show full state
     *     state help                  - Show help information
     *     state show [section]        - Show state (all, task, todos, flags, mode)
     *     state mode <mode>           - Switch mode (discussion/no, bypass/off)
     *     state task <action>         - Manage current task
     *     state todos <action>        - Manage todos
     *     state flags <action>        - Manage flags
     */
    // Handle help command
    if (!args || args.length === 0 || (args.length > 0 && ['help', ''].includes(args[0].toLowerCase()))) {
        if (fromSlash && (!args || args.length === 0 || args[0].toLowerCase() === 'help')) {
            return formatStateHelp();
        } else if (!args || args.length === 0) {
            // Show full state when no args
            if (jsonOutput) {
                return STATE.toDict();
            }
            return formatStateHuman(STATE);
        }
    }

    const section = args[0].toLowerCase();
    const sectionArgs = args.length > 1 ? args.slice(1) : [];

    // Remove --from-slash from sectionArgs if present
    const cleanedSectionArgs = sectionArgs.filter(arg => arg !== '--from-slash');

    // Route to appropriate handler
    if (section === 'show') {
        return handleShowCommand(cleanedSectionArgs, jsonOutput);
    } else if (section === 'mode') {
        return handleModeCommand(cleanedSectionArgs, jsonOutput, fromSlash);
    } else if (section === 'task') {
        return handleTaskCommand(cleanedSectionArgs, jsonOutput, fromSlash);
    } else if (section === 'todos') {
        return handleTodosCommand(cleanedSectionArgs, jsonOutput);
    } else if (section === 'flags') {
        return handleFlagsCommand(cleanedSectionArgs, jsonOutput);
    } else if (section === 'update') {
        return handleUpdateCommand(cleanedSectionArgs, jsonOutput, fromSlash);
    } else {
        // For backward compatibility, support direct component access
        const component = section;

        // Handle nested access (e.g., task.name, flags.noob)
        if (component.includes('.')) {
            const parts = component.split('.');
            let result = STATE.toDict();
            try {
                for (const part of parts) {
                    if (typeof result === 'object' && result !== null) {
                        if (part in result) {
                            result = result[part];
                        } else if (typeof result[part] !== 'undefined') {
                            result = result[part];
                        } else {
                            throw new Error(`Invalid state path: ${component}`);
                        }
                    } else {
                        throw new Error(`Invalid state path: ${component}`);
                    }
                }

                if (jsonOutput) {
                    return { [component]: result };
                } else {
                    return `${component}: ${result}`;
                }
            } catch (error) {
                throw new Error(`Invalid state path: ${component}`);
            }
        }

        // Handle top-level components for backward compatibility
        if (component === 'mode') {
            if (jsonOutput) {
                return { mode: STATE.mode.value || STATE.mode };
            }
            return `Mode: ${STATE.mode.value || STATE.mode}`;
        } else if (component === 'task') {
            if (jsonOutput) {
                return { task: STATE.current_task ? STATE.current_task.taskState : null };
            }
            if (STATE.current_task) {
                return formatTaskHuman(STATE.current_task);
            }
            return "No active task";
        } else if (component === 'todos') {
            if (jsonOutput) {
                return { todos: STATE.todos.toDict() };
            }
            return formatTodosHuman(STATE.todos);
        } else if (component === 'flags') {
            if (jsonOutput) {
                return { flags: { ...STATE.flags } };
            }
            return formatFlagsHuman(STATE.flags);
        } else if (component === 'metadata') {
            if (jsonOutput) {
                return { metadata: STATE.metadata };
            }
            return `Metadata: ${JSON.stringify(STATE.metadata, null, 2)}`;
        } else {
            if (fromSlash) {
                return `Unknown command: ${section}\n\n${formatStateHelp()}`;
            }
            throw new Error(`Unknown state component: ${component}`);
        }
    }
}

function formatStateHelp() {
    /**Format help output for slash command.*/
    const lines = [
        "Sessions State Commands:",
        "",
        "  /sessions state                 - Display current state",
        "  /sessions state show [section]  - Show specific section (task, todos, flags, mode)",
        "  /sessions state mode <mode>     - Switch mode (discussion/no, bypass/off)",
        "  /sessions state task <action>   - Manage task (clear, show, restore <file>)",
        "  /sessions state todos <action>  - Manage todos (clear)",
        "  /sessions state flags <action>  - Manage flags (clear, clear-context)",
        "  /sessions state update ...      - Manage update notifications (see update help)",
        "",
        "Mode Aliases:",
        "  no   → discussion mode",
        "  off  → bypass mode toggle",
        "  go   → implementation mode (use trigger phrases, not slash commands)",
        "",
        "Security Boundaries:",
        "  Mode switching:",
        "    • User can switch between modes freely via slash commands",
        "    • API can only switch implementation → discussion (one-way safety)",
        "    • Use trigger phrases for discussion → implementation transitions",
        "  Bypass mode:",
        "    • Deactivation: Available anytime (return to normal DAIC enforcement)",
        "    • Activation: Requires user-initiated slash command (safety mechanism)",
        "  Permission-based operations:",
        "    • 'todos clear' requires special permission flag (api.todos_clear)",
        "    • Only available immediately after session restoration",
        "",
        "Examples:",
        "  /sessions state show task       - Show current task",
        "  /sessions state mode no         - Switch to discussion mode",
        "  /sessions state task restore m-refactor-commands.md",
    ];
    return lines.join('\n');
}

function formatStateHuman(state) {
    /**Format full state for human reading.*/
    const lines = [
        "=== Session State ===",
        `Mode: ${state.mode.value || state.mode}`,
        "",
    ];

    if (state.current_task) {
        lines.push("Current Task:");
        lines.push(`  Name: ${state.current_task.name}`);
        lines.push(`  File: ${state.current_task.file}`);
        lines.push(`  Branch: ${state.current_task.branch}`);
        lines.push(`  Status: ${state.current_task.status}`);
    } else {
        lines.push("Current Task: None");
    }

    lines.push("");
    lines.push(`Active Todos: ${state.todos.active ? state.todos.active.length : 0}`);
    if (state.todos.active) {
        for (const todo of state.todos.active) {
            const statusIcon = todo.status === 'completed' ? "✓" : "○";
            lines.push(`  ${statusIcon} ${todo.content || 'Unknown'}`);
        }
    }

    if (state.todos.stashed && state.todos.stashed.length > 0) {
        lines.push(`Stashed Todos: ${state.todos.stashed.length}`);
    }

    lines.push("");
    lines.push("Flags:");
    lines.push(`  Context 85% warning: ${state.flags.context_85}`);
    lines.push(`  Context 90% warning: ${state.flags.context_90}`);
    lines.push(`  Subagent mode: ${state.flags.subagent}`);
    lines.push(`  Noob mode: ${state.flags.noob}`);
    lines.push(`  Bypass mode: ${state.flags.bypass_mode}`);

    return lines.join('\n');
}

function formatTaskHuman(task) {
    /**Format task for human reading.*/
    const lines = ["Current Task:"];
    if (task.name) {
        lines.push(`  Name: ${task.name}`);
    }
    if (task.file) {
        lines.push(`  File: ${task.file}`);
    }
    if (task.branch) {
        lines.push(`  Branch: ${task.branch}`);
    }
    if (task.status) {
        lines.push(`  Status: ${task.status}`);
    }
    if (task.created) {
        lines.push(`  Created: ${task.created}`);
    }
    if (task.started) {
        lines.push(`  Started: ${task.started}`);
    }
    if (task.submodules && task.submodules.length > 0) {
        lines.push(`  Submodules: ${task.submodules.join(', ')}`);
    }
    return lines.join('\n');
}

function formatTodosHuman(todos) {
    /**Format todos for human reading.*/
    const lines = [];

    if (todos.active && todos.active.length > 0) {
        lines.push(`Active Todos (${todos.active.length}):`);
        todos.active.forEach((todo, i) => {
            const status = todo.status || 'pending';
            const icon = { "completed": "✓", "in_progress": "→", "pending": "○" }[status] || "?";
            lines.push(`  ${i + 1}. [${icon}] ${todo.content || 'Unknown'}`);
        });
    } else {
        lines.push("Active Todos: None");
    }

    if (todos.stashed && todos.stashed.length > 0) {
        lines.push(`\nStashed Todos (${todos.stashed.length}):`);
        todos.stashed.forEach((todo, i) => {
            lines.push(`  ${i + 1}. ${todo.content || 'Unknown'}`);
        });
    }

    return lines.join('\n');
}

function formatFlagsHuman(flags) {
    /**Format flags for human reading.*/
    const lines = ["Flags:"];
    lines.push(`  context_85: ${flags.context_85}`);
    lines.push(`  context_90: ${flags.context_90}`);
    lines.push(`  subagent: ${flags.subagent}`);
    lines.push(`  noob: ${flags.noob}`);
    lines.push(`  bypass_mode: ${flags.bypass_mode}`);
    return lines.join('\n');
}
//!<

//!> Mode command handler
function handleModeCommand(args, jsonOutput = false, fromSlash = false) {
    /**
     * Handle mode switching commands.
     *
     * Usage:
     *     mode discussion / mode no   - Switch to discussion mode (one-way only)
     *     mode bypass / mode off      - Toggle bypass mode (disables behavioral constraints)
     *     mode go                     - Switch to implementation mode (not allowed via API)
     */
    if (!args || args.length === 0) {
        // Just show current mode and bypass status
        if (jsonOutput) {
            return { mode: STATE.mode.value || STATE.mode, bypass_mode: STATE.flags.bypass_mode };
        }
        let result = `Current mode: ${STATE.mode.value || STATE.mode}`;
        if (STATE.flags.bypass_mode) {
            result += "\nBypass mode: ACTIVE (behavioral constraints disabled)";
        }
        return result;
    }

    let targetMode = args[0].toLowerCase();

    // Friendly name mapping
    const modeAliases = {
        'no': 'discussion',
        'go': 'implementation',
        'off': 'bypass'
    };

    targetMode = modeAliases[targetMode] || targetMode;

    if (targetMode === 'discussion') {
        // One-way switch to discussion allowed
        let result;
        editState(state => {
            if (state.mode === Mode.GO || state.mode === 'implementation') {
                state.mode = Mode.NO;
                result = "Switched to discussion mode";
            } else {
                result = "Already in discussion mode";
            }
        });

        if (jsonOutput) {
            return { mode: "discussion", message: result };
        }
        return result;

    } else if (targetMode === 'bypass') {
        // Check if this is a slash command (user-initiated) or API call (Claude-initiated)
        const isSlashCommand = fromSlash || args.includes('--from-slash');

        let result = null;
        let bypassActive = null;

        editState(state => {
            if (state.flags.bypass_mode) {
                // Always allow deactivating bypass mode (returning to safety)
                state.flags.bypass_mode = false;
                bypassActive = false;
                result = "Bypass mode INACTIVE - behavioral constraints enabled";
            } else {
                // Only allow activating bypass if it's from a slash command (user-initiated)
                if (!isSlashCommand) {
                    throw new Error("Cannot activate bypass mode via API. Only the user can enable bypass mode.");
                }
                state.flags.bypass_mode = true;
                bypassActive = true;
                result = "Bypass mode ACTIVE - behavioral constraints disabled";
            }
        });

        if (jsonOutput) {
            return { bypass_mode: bypassActive, message: result };
        }
        return result;

    } else if (targetMode === 'implementation') {
        // Allow via slash command (user-initiated), block via direct API call (Claude-initiated)
        if (!fromSlash) {
            throw new Error("Cannot switch to implementation mode via API. Use trigger phrases or slash command instead.");
        }

        // User-initiated via slash command - allow the switch
        let result;
        editState(state => {
            if (state.mode === Mode.GO || state.mode === 'implementation') {
                result = "Already in implementation mode";
            } else {
                state.mode = Mode.GO;
                result = "Mode switched: discussion → implementation\n\nYou are now in Implementation Mode and may use tools to execute agreed upon actions.\n\nRemember to return to Discussion Mode when done:\n  /sessions state mode no\n  OR use your discussion mode trigger phrases";
            }
        });

        if (jsonOutput) {
            return { mode: "implementation", message: result };
        }
        return result;

    } else {
        const validModes = "discussion (no), implementation (go), bypass (off)";
        if (fromSlash) {
            return `Unknown mode: ${args[0]}\n\nValid modes: ${validModes}\n\nUsage:\n  mode discussion / mode no        - Switch to discussion mode\n  mode implementation / mode go    - Switch to implementation mode\n  mode bypass / mode off           - Toggle bypass mode`;
        }
        throw new Error(`Unknown mode: ${args[0]}. Valid modes: ${validModes}`);
    }
}
//!<

//!> Flags command handler
function handleFlagsCommand(args, jsonOutput = false) {
    /**
     * Handle flag management commands.
     *
     * Usage:
     *     flags clear         - Clear all flags
     *     flags clear-context - Clear context warnings only
     */
    if (!args || args.length === 0) {
        // Show current flags
        if (jsonOutput) {
            return { flags: { ...STATE.flags } };
        }
        return formatFlagsHuman(STATE.flags);
    }

    const action = args[0].toLowerCase();

    if (action === 'clear') {
        editState(state => {
            state.flags.context_85 = false;
            state.flags.context_90 = false;
            state.flags.subagent = false;
            state.flags.noob = false;
        });

        if (jsonOutput) {
            return { message: "All flags cleared" };
        }
        return "All flags cleared";

    } else if (action === 'clear-context') {
        editState(state => {
            state.flags.context_85 = false;
            state.flags.context_90 = false;
        });

        if (jsonOutput) {
            return { message: "Context warnings cleared" };
        }
        return "Context warnings cleared";

    } else {
        throw new Error(`Unknown flags action: ${action}. Valid actions: clear, clear-context`);
    }
}
//!<

//!> Todos handler
function handleTodosCommand(args, jsonOutput = false) {
    /**
     * Handle todos management commands.
     *
     * Usage:
     *     todos clear - Clear all active todos (requires api.todos_clear permission for API, always available via slash command)
     */
    if (!args || args.length === 0) {
        // Show current todos
        if (jsonOutput) {
            return { todos: STATE.todos.toList('active') };
        }
        const lines = ["Active Todos:"];
        if (STATE.todos.active && STATE.todos.active.length > 0) {
            for (const todo of STATE.todos.active) {
                const status = todo.status || 'pending';
                const content = todo.content || 'Unknown';
                lines.push(`  [${status}] ${content}`);
            }
        } else {
            lines.push("  (none)");
        }
        return lines.join('\n');
    }

    // Check if --from-slash flag is present
    const isSlashCommand = args.includes('--from-slash');
    const cleanedArgs = args.filter(arg => arg !== '--from-slash');

    if (cleanedArgs.length === 0) {
        throw new Error("todos command requires an action. Valid actions: clear");
    }

    const action = cleanedArgs[0].toLowerCase();

    if (action === 'clear') {
        // Check if we have permission (only for API calls, not slash commands)
        if (!isSlashCommand && !STATE.api.todos_clear) {
            if (jsonOutput) {
                return { error: "Permission denied: todos clear command is not available in this context" };
            }
            return "Permission denied: The todos clear command is only available immediately after todos are restored";
        }

        // Clear the todos
        editState(state => {
            state.todos.clearActive();
            // Only disable permission if this was an API call
            if (!isSlashCommand && state.api.todos_clear) {
                state.api.todos_clear = false;
            }
        });

        if (jsonOutput) {
            return { message: "Active todos cleared" };
        }
        return "Active todos cleared";

    } else {
        throw new Error(`Unknown todos action: ${action}. Valid actions: clear`);
    }
}
//!<

//!> Task management handler
function handleTaskCommand(args, jsonOutput = false, fromSlash = false) {
    /**
     * Handle task management commands.
     *
     * Usage:
     *     task clear          - Clear current task
     *     task show           - Show current task details
     *     task restore <file> - Restore task from file frontmatter
     */
    if (!args || args.length === 0) {
        // Show current task by default
        if (STATE.current_task) {
            if (jsonOutput) {
                return { task: STATE.current_task.task_state };
            }
            return formatTaskHuman(STATE.current_task);
        } else {
            if (jsonOutput) {
                return { task: null };
            }
            return "No active task";
        }
    }

    const action = args[0].toLowerCase();

    if (action === 'clear') {
        editState(s => {
            s.current_task.clearTask();
        });

        if (jsonOutput) {
            return { message: "Task cleared" };
        }
        return "Task cleared";

    } else if (action === 'show') {
        if (STATE.current_task) {
            if (jsonOutput) {
                return { task: STATE.current_task.task_state };
            }
            return formatTaskHuman(STATE.current_task);
        } else {
            if (jsonOutput) {
                return { task: null };
            }
            return "No active task";
        }

    } else if (action === 'restore') {
        if (args.length < 2) {
            throw new Error("task restore requires a task file path");
        }

        const taskFile = args[1];

        // Use TaskState.loadTask() to properly load task from file
        let taskState;
        try {
            taskState = TaskState.loadTask({ file: taskFile });
        } catch (error) {
            if (fromSlash) {
                return `Failed to restore task: ${error.message || String(error)}`;
            }
            throw new Error(`Failed to restore task: ${error.message || String(error)}`);
        }

        // Update state with loaded task
        editState(s => {
            s.current_task = taskState;
        });

        // Guidance message for Claude
        const guidance = `\n\nTask restored. If you don't have sessions/tasks/${taskFile} in your context, read it to understand the task requirements.`;

        if (jsonOutput) {
            return { message: `Task '${taskState.name}' restored from ${taskFile}`, guidance: guidance };
        }

        return `Task '${taskState.name}' restored from ${taskFile}${guidance}`;

    } else {
        const validActions = "clear, show, restore";
        if (fromSlash) {
            return `Unknown task action: ${action}\n\nValid actions: ${validActions}\n\nUsage:\n  task clear          - Clear current task\n  task show           - Show current task details\n  task restore <file> - Restore task from file frontmatter`;
        }
        throw new Error(`Unknown task action: ${action}. Valid actions: ${validActions}`);
    }
}
//!<

//!> Show subsection handler
function handleShowCommand(args, jsonOutput = false) {
    /**
     * Handle show subsection commands for convenient access.
     *
     * Usage:
     *     show task   - Show current task
     *     show todos  - Show active todos
     *     show flags  - Show session flags
     *     show mode   - Show current mode
     */
    if (!args || args.length === 0) {
        // Default to full state
        return handleStateCommand([], jsonOutput);
    }

    const subsection = args[0].toLowerCase();

    // Route to appropriate handler
    if (subsection === 'task') {
        return handleTaskCommand(['show'], jsonOutput);
    } else if (subsection === 'todos') {
        return handleTodosCommand([], jsonOutput);
    } else if (subsection === 'flags') {
        return handleFlagsCommand([], jsonOutput);
    } else if (subsection === 'mode') {
        if (jsonOutput) {
            return { mode: STATE.mode.value || STATE.mode };
        }
        return `Mode: ${STATE.mode.value || STATE.mode}`;
    } else {
        throw new Error(`Unknown show subsection: ${subsection}. Valid: task, todos, flags, mode`);
    }
}
//!<

//!> Status and version handlers
function handleStatusCommand(args, jsonOutput = false) {
    /**
     * Handle status command - human-readable summary of current state.
     */
    if (jsonOutput) {
        // For JSON, just return the full state
        return STATE.toDict();
    }

    // Human-readable status summary
    const lines = [
        "╔══════════════════════════════════════╗",
        "║      cc-sessions Status Summary      ║",
        "╚══════════════════════════════════════╝",
        "",
        `Mode: ${(STATE.mode.value || STATE.mode || '').toUpperCase()}`,
    ];

    if (STATE.current_task && STATE.current_task.name) {
        lines.push(`Task: ${STATE.current_task.name} (${STATE.current_task.status || 'unknown'})`);
    } else {
        lines.push("Task: None");
    }

    const activeCount = STATE.todos.active ? STATE.todos.active.length : 0;
    const completed = STATE.todos.active ? STATE.todos.active.filter(t => t.status === 'completed').length : 0;
    lines.push(`Todos: ${completed}/${activeCount} completed`);

    // Warnings
    const warnings = [];
    if (STATE.flags.context_85) {
        warnings.push("85% context usage");
    }
    if (STATE.flags.context_90) {
        warnings.push("90% context usage");
    }
    if (warnings.length > 0) {
        lines.push(`⚠ Warnings: ${warnings.join(', ')}`);
    }

    return lines.join('\n');
}

function handleVersionCommand(args, jsonOutput = false) {
    /**
     * Handle version command - show package version.
     */
    let pkgVersion = "development";

    // Try to get version from package.json
    try {
        const packagePath = path.join(PROJECT_ROOT, 'package.json');
        const packageData = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
        pkgVersion = packageData.version || "development";
    } catch (error) {
        // Fallback to development if we can't read package.json
        pkgVersion = "development";
    }

    if (jsonOutput) {
        return { version: pkgVersion };
    }
    return `cc-sessions version: ${pkgVersion}`;
}

function handleUpdateCommand(args, jsonOutput = false, fromSlash = false) {
    /**
     * Handle update management commands.
     *
     * Usage:
     *     update suppress  - Suppress update notifications
     *     update check     - Force re-check for updates
     *     update status    - Show current update status
     */
    if (!args || args.length === 0) {
        throw new Error("update command requires a subcommand. Valid: suppress, check, status");
    }

    if (args[0].toLowerCase() === 'help') {
        return formatUpdateHelp();
    }

    const subcommand = args[0].toLowerCase();

    if (subcommand === 'suppress') {
        editState(s => {
            s.metadata.update_available = false;
        });

        if (jsonOutput) {
            return { message: "Update notifications suppressed" };
        }
        if (fromSlash) {
            return "✓ Update notifications suppressed\n\nUpdate checks will resume after next package update.";
        }
        return "Update notifications suppressed";

    } else if (subcommand === 'check') {
        editState(s => {
            delete s.metadata.update_available;
            delete s.metadata.latest_version;
            delete s.metadata.current_version;
        });

        if (jsonOutput) {
            return { message: "Update check flag cleared" };
        }
        if (fromSlash) {
            return "✓ Update check flag cleared\n\nVersion check will run on next session start.";
        }
        return "Update check flag cleared - will re-check on next session start";

    } else if (subcommand === 'status') {
        const updateFlag = STATE.metadata.update_available;
        const latestVersion = STATE.metadata.latest_version;
        const currentVersion = STATE.metadata.current_version;

        if (jsonOutput) {
            return {
                current_version: currentVersion || "unknown",
                latest_version: latestVersion || "not checked",
                update_available: updateFlag
            };
        }

        if (fromSlash) {
            const lines = [
                "Update Status:",
                `  Current version: ${currentVersion || 'unknown'}`,
                `  Latest version: ${latestVersion || 'not checked'}`,
                `  Update available: ${updateFlag ? 'Yes' : (updateFlag === false ? 'No' : 'Unknown (not checked)')}`
            ];
            return lines.join('\n');
        }

        if (updateFlag === undefined || updateFlag === null) {
            return "Update check: Not performed yet";
        } else if (updateFlag) {
            return `Update available: ${currentVersion} → ${latestVersion}`;
        } else {
            return `Up to date: ${currentVersion}`;
        }

    } else {
        if (fromSlash) {
            return `Unknown subcommand: ${subcommand}\n\n${formatUpdateHelp()}`;
        }
        throw new Error(`Unknown update subcommand: ${subcommand}. Valid: suppress, check, status`);
    }
}

function formatUpdateHelp() {
    /**Format update help for slash command.*/
    const lines = [
        "Update Management Commands:",
        "",
        "  /sessions state update status    - Show current update status",
        "  /sessions state update suppress  - Suppress update notifications",
        "  /sessions state update check     - Force re-check for updates",
        "",
        "Details:",
        "  status    - Display current and latest versions with update availability",
        "  suppress  - Silence update notifications until next actual update",
        "  check     - Clear cached update flags and re-check on next session start",
        "",
        "Examples:",
        "  /sessions state update status",
        "  /sessions state update suppress"
    ];
    return lines.join('\n');
}
//!<

//-#

// ==== EXPORTS ===== //
module.exports = {
    handleStateCommand,
    handleModeCommand,
    handleFlagsCommand,
    handleStatusCommand,
    handleVersionCommand,
    handleTodosCommand
};
//-#
