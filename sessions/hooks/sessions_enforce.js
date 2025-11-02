#!/usr/bin/env node

// ===== IMPORTS ===== //

/// ===== STDLIB ===== ///
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
///-///

/// ===== 3RD-PARTY ===== ///
///-///

/// ===== LOCAL ===== ///
const {
    editState,
    loadState,
    Mode,
    PROJECT_ROOT,
    loadConfig,
    findGitRepo,
    CCTools
} = require('./shared_state.js');
///-///

//-//

// ===== GLOBALS ===== //
// Load input
let inputData = {};
try {
    const stdin = fs.readFileSync(0, 'utf-8');
    inputData = JSON.parse(stdin);
} catch (e) {
    inputData = {};
}

const toolName = inputData.tool_name || "";
const toolInput = inputData.tool_input || {};

let filePath = null;
const filePathString = toolInput.file_path || "";
if (filePathString) {
    filePath = path.resolve(filePathString);
}

const STATE = loadState();
const CONFIG = loadConfig();

let command = "";
let incomingTodos = [];
if (toolName === "Bash") {
    command = (toolInput.command || "").trim();
}
if (toolName === "TodoWrite") {
    incomingTodos = toolInput.todos || [];
}

/// ===== PATTERNS ===== ///
const READONLY_FIRST = new Set([
    // Basic file reading
    'cat', 'less', 'more', 'head', 'tail', 'wc', 'nl', 'tac', 'rev',
    // Text search and filtering
    'grep', 'egrep', 'fgrep', 'rg', 'ripgrep', 'ag', 'ack',
    // Text processing (all safe for reading)
    'sort', 'uniq', 'cut', 'paste', 'join', 'comm', 'column',
    'tr', 'expand', 'unexpand', 'fold', 'fmt', 'pr', 'shuf', 'tsort',
    // Comparison
    'diff', 'cmp', 'sdiff', 'vimdiff',
    // Checksums
    'md5sum', 'sha1sum', 'sha256sum', 'sha512sum', 'cksum', 'sum',
    // Binary inspection
    'od', 'hexdump', 'xxd', 'strings', 'file', 'readelf', 'objdump', 'nm',
    // File system inspection
    'ls', 'dir', 'vdir', 'pwd', 'which', 'type', 'whereis', 'locate', 'find',
    'basename', 'dirname', 'readlink', 'realpath', 'stat',
    // User/system info
    'whoami', 'id', 'groups', 'users', 'who', 'w', 'last', 'lastlog',
    'hostname', 'uname', 'arch', 'lsb_release', 'hostnamectl',
    'date', 'cal', 'uptime', 'df', 'du', 'free', 'vmstat', 'iostat',
    // Process monitoring
    'ps', 'pgrep', 'pidof', 'top', 'htop', 'iotop', 'atop',
    'lsof', 'jobs', 'pstree', 'fuser',
    // Network monitoring
    'netstat', 'ss', 'ip', 'ifconfig', 'route', 'arp',
    'ping', 'traceroute', 'tracepath', 'mtr', 'nslookup', 'dig', 'host', 'whois',
    // Environment
    'printenv', 'env', 'set', 'export', 'alias', 'history', 'fc',
    // Output
    'echo', 'printf', 'yes', 'seq', 'jot',
    // Testing
    'test', '[', '[[', 'true', 'false',
    // Calculation
    'bc', 'dc', 'expr', 'factor', 'units',
    // Modern tools
    'jq', 'yq', 'xmlstarlet', 'xmllint', 'xsltproc',
    'bat', 'fd', 'fzf', 'tree', 'ncdu', 'exa', 'lsd',
    'tldr', 'cheat',
    // Note: awk/sed are here but need special argument checking
    'awk', 'sed', 'gawk', 'mawk', 'gsed'
]);

// Add user-configured readonly patterns
CONFIG.blocked_actions.bash_read_patterns.forEach(pattern => {
    READONLY_FIRST.add(pattern);
});

const WRITE_FIRST = new Set([
    // File operations
    'rm', 'rmdir', 'unlink', 'shred',
    'mv', 'rename', 'cp', 'install', 'dd',
    'mkdir', 'mkfifo', 'mknod', 'mktemp', 'touch', 'truncate',
    // Permissions
    'chmod', 'chown', 'chgrp', 'umask',
    'ln', 'link', 'symlink',
    'setfacl', 'setfattr', 'chattr',
    // System management
    'useradd', 'userdel', 'usermod', 'groupadd', 'groupdel',
    'passwd', 'chpasswd', 'systemctl', 'service',
    // Package managers
    'apt', 'apt-get', 'dpkg', 'snap', 'yum', 'dnf', 'rpm',
    'pip', 'pip3', 'npm', 'yarn', 'gem', 'cargo',
    // Build tools
    'make', 'cmake', 'ninja', 'meson',
    // Other dangerous
    'sudo', 'doas', 'su', 'crontab', 'at', 'batch',
    'kill', 'pkill', 'killall', 'tee'
]);

// Add user-configured write patterns
CONFIG.blocked_actions.bash_write_patterns.forEach(pattern => {
    WRITE_FIRST.add(pattern);
});

// Enhanced redirection detection (includes stderr redirections)
const REDIR_PATTERNS = [
    /(?:^|\s)(?:>>?|<<?|<<<)\s/,           // Basic redirections
    /(?:^|\s)\d*>&?\d*(?:\s|$)/,            // File descriptor redirections (2>&1, etc)
    /(?:^|\s)&>/                            // Combined stdout/stderr redirect
];
const REDIR = new RegExp(REDIR_PATTERNS.map(p => p.source).join('|'));
///-///

/// ===== CI DETECTION ===== ///
function isCIEnvironment() {
    // Check if running in a CI environment (GitHub Actions)
    const ciIndicators = [
        'GITHUB_ACTIONS',         // GitHub Actions
        'GITHUB_WORKFLOW',        // GitHub Actions workflow
        'CI',                     // Generic CI indicator (set by GitHub Actions)
        'CONTINUOUS_INTEGRATION', // Generic CI (alternative)
    ];
    return ciIndicators.some(indicator => process.env[indicator]);
}
///-///

//-//

/*
╔══════════════════════════════════════════════════════════════════════════════╗
║ ██████╗ █████╗ ██████╗ ██████╗ █████╗  █████╗ ██╗      ██╗ ██╗██████╗██████╗ ║
║ ██╔══██╗██╔═██╗██╔═══╝ ╚═██╔═╝██╔══██╗██╔══██╗██║      ██║ ██║██╔═══╝██╔═══╝ ║
║ ██████╔╝█████╔╝█████╗    ██║  ██║  ██║██║  ██║██║      ██║ ██║██████╗█████╗  ║
║ ██╔═══╝ ██╔═██╗██╔══╝    ██║  ██║  ██║██║  ██║██║      ██║ ██║╚═══██║██╔══╝  ║
║ ██║     ██║ ██║██████╗   ██║  ╚█████╔╝╚█████╔╝███████╗ ╚████╔╝██████║██████╗ ║
║ ╚═╝     ╚═╝ ╚═╝╚═════╝   ╚═╝   ╚════╝  ╚════╝ ╚══════╝  ╚═══╝ ╚═════╝╚═════╝ ║
╚══════════════════════════════════════════════════════════════════════════════╝
PreToolUse Hook

Trigger conditions:
- Write/subagent tool invocation (Bash, Write, Edit, MultiEdit, Task, TodoWrite)

Enforces DAIC (Discussion, Alignment, Implementation, Check) workflow:
- Blocks write tools in discussion mode
- Validates TodoWrite operations for proper scope management
- Enforces git branch consistency with task requirements
- Protects system state files from unauthorized modification
*/

// ===== FUNCTIONS ===== //

/// ===== HELPERS ===== ///
function checkCommandArguments(parts) {
    // Check if command arguments indicate write operations
    if (!parts || parts.length === 0) return true;

    const cmd = parts[0].toLowerCase();
    const args = parts.slice(1);

    // Check sed for in-place editing
    if (cmd === 'sed' || cmd === 'gsed') {
        for (const arg of args) {
            if (arg.startsWith('-i') || arg === '--in-place') {
                return false;  // sed -i is a write operation
            }
        }
    }

    // Check awk for file output operations
    if (['awk', 'gawk', 'mawk'].includes(cmd)) {
        const script = args.join(' ');
        // Check for output redirection within awk script
        if (/>s*["'].*["']/.test(script) || />>s*["'].*["']/.test(script)) {
            return false;
        }
        if (script.includes('print >') || script.includes('print >>') ||
            script.includes('printf >') || script.includes('printf >>')) {
            return false;
        }
    }

    // Check find for dangerous operations
    if (cmd === 'find') {
        if (args.includes('-delete')) {
            return false;
        }
        for (let i = 0; i < args.length; i++) {
            if (args[i] === '-exec' || args[i] === '-execdir') {
                if (i + 1 < args.length) {
                    const execCmd = args[i + 1].toLowerCase();
                    if (WRITE_FIRST.has(execCmd) || ['rm', 'mv', 'cp', 'shred'].includes(execCmd)) {
                        return false;
                    }
                }
            }
        }
    }

    // Check xargs for dangerous commands
    if (cmd === 'xargs') {
        for (const writeCmd of WRITE_FIRST) {
            if (args.some(arg => arg === writeCmd)) {
                return false;
            }
        }
        // Check for sed -i through xargs
        const sedIndex = args.indexOf('sed');
        if (sedIndex > -1 && sedIndex + 1 < args.length && args[sedIndex + 1].startsWith('-i')) {
            return false;
        }
    }

    return true;
}

// Check if a bash command is read-only (no writes, no redirections)
function isBashReadOnly(command, extrasafe = CONFIG.blocked_actions.extrasafe || true) {
    /*Determine if a bash command is read-only.

    Enhanced to check command arguments for operations like:
    - sed -i (in-place editing)
    - awk with file output
    - find -delete or -exec rm
    - xargs with write commands

    Args:
        command (str): The bash command to evaluate.
        extrasafe (bool): If True, unrecognized commands are treated as write-like.*/

    const s = (command || '').trim();
    if (!s) return true;

    if (REDIR.test(s)) {
        return false;
    }

    // Split on |, && and || while avoiding splitting on escaped pipes
    const segments = s.split(/(?<!\|)\|(?!\|)|&&|\|\|/).map(seg => seg.trim());

    for (const segment of segments) {
        if (!segment) continue;

        // Parse command parts (handling quotes)
        let parts = [];
        try {
            // Simple shlex-like splitting for JavaScript
            const regex = /[^\s"']+|"([^"]*)"|'([^']*)'/g;
            let match;
            while ((match = regex.exec(segment)) !== null) {
                parts.push(match[1] || match[2] || match[0]);
            }
        } catch (error) {
            return !CONFIG.blocked_actions.extrasafe;
        }

        if (parts.length === 0) continue;

        const first = parts[0].toLowerCase();
        if (first === 'cd') continue;

        // Special case: Commands with read-only subcommands
        if (['pip', 'pip3'].includes(first)) {
            const subcommand = parts[1]?.toLowerCase() || '';
            if (['show', 'list', 'search', 'check', 'freeze', 'help'].includes(subcommand)) {
                continue;  // Allow read-only pip operations
            }
            return false;  // Block write operations
        }

        if (['npm', 'yarn'].includes(first)) {
            const subcommand = parts[1]?.toLowerCase() || '';
            if (['list', 'ls', 'view', 'show', 'search', 'help'].includes(subcommand)) {
                continue;  // Allow read-only npm/yarn operations
            }
            return false;  // Block write operations
        }

        if (['python', 'python3'].includes(first)) {
            // Allow python -c for simple expressions and python -m for module execution
            if (parts.length > 1 && ['-c', '-m'].includes(parts[1])) {
                continue;  // These are typically read-only operations in our context
            }
            // Block other python invocations as potentially write-like
            return false;
        }

        if (WRITE_FIRST.has(first)) return false;

        // Check command arguments for write operations
        if (!checkCommandArguments(parts)) return false;

        // Check if command is in user's custom readonly list
        if (CONFIG.blocked_actions.bash_read_patterns.includes(first)) continue;  // Allow custom readonly commands

        // If extrasafe is on and command not in readonly list, block it
        if (!READONLY_FIRST.has(first) && CONFIG.blocked_actions.extrasafe) return false;
    }

    return true;
}
///-///

//-//

// ===== EXECUTION ===== //

// Skip DAIC enforcement in CI environments
if (isCIEnvironment()) {
    process.exit(0);
}

//!> Bash command handling
// For Bash commands, check if it's a read-only operation
if (toolName === "Bash" && STATE.mode === Mode.NO && !STATE.flags.bypass_mode) {
    // Special case: Allow sessions.api commands in discussion mode
    if (command && (command.includes('sessions ') || command.includes('python -m cc_sessions.scripts.api'))) {
        // API commands are allowed in discussion mode for state inspection and safe config operations
        process.exit(0);
    }

    if (!isBashReadOnly(command)) {
        // Detect OS for correct sessions command
        const isWindows = process.platform === "win32";
        const sessionsCmd = isWindows ? "sessions/bin/sessions.bat" : "sessions/bin/sessions";

        console.error(`[DAIC] Blocked write-like Bash command in Discussion mode. Only the user can activate implementation mode. Explain what you want to do and seek alignment and approval first.\n` +
                      `Note: Both Claude and the user can configure allowed commands:\n` +
                      `  - View allowed: ${sessionsCmd} config read list\n` +
                      `  - Add command: ${sessionsCmd} config read add <command>\n` +
                      `  - Remove command: ${sessionsCmd} config read remove <command>`);
        process.exit(2);  // Block with feedback
    } else {
        process.exit(0);
    }
}
//!<

//!> Block any attempt to modify sessions-state.json directly
if (filePath && toolName === "Bash" &&
    path.basename(filePath) === 'sessions-state.json' &&
    path.basename(path.dirname(filePath)) === 'sessions') {
    // Check if it's a modifying operation
    if (!isBashReadOnly(command)) {
        console.error("[Security] Direct modification of sessions-state.json is not allowed. " +
                      "This file should only be modified through the TodoWrite tool and approved commands.");
        process.exit(2);
    }
}
//!<

// --- All commands beyond here contain write patterns (read patterns exit early) ---

//!> Discussion mode guard (block write tools)
if (STATE.mode === Mode.NO && !STATE.flags.bypass_mode) {
    if (CONFIG.blocked_actions.isToolBlocked(toolName)) {
        console.error(`[DAIC: Tool Blocked] You're in discussion mode. The ${toolName} tool is not allowed. You need to seek alignment first.`);
        process.exit(2);  // Block with feedback
    } else {
        process.exit(0);  // Allow read-only tools
    }
}
//!<

//!> TodoWrite tool handling
if (toolName === "TodoWrite" && !STATE.flags.bypass_mode) {
    // Check for name mismatch first (regardless of completion state)
    if (STATE.todos.active && STATE.todos.active.length > 0) {
        const activeNames = STATE.todos.active.map(t => t.content);
        const incomingNames = incomingTodos.map(t => t.content || '');

        if (JSON.stringify(activeNames) !== JSON.stringify(incomingNames)) {
            // Todo names changed - safety violation
            // Prepare detailed diff for Claude before clearing state
            const originalCount = activeNames.length;
            const proposedCount = incomingNames.length;

            // Format original todos
            const originalDisplay = activeNames.map((name, i) => `  ${i+1}. ${name}`).join('\n');

            // Format proposed todos
            const proposedDisplay = incomingNames.map((name, i) => `  ${i+1}. ${name}`).join('\n');

            // Get user's implementation trigger phrases
            const triggerPhrases = CONFIG.trigger_phrases.implementation_mode;
            const triggerList = triggerPhrases.map(p => `"${p}"`).join(', ');

            // Clear todos and revert to discussion mode (preparing for re-approval)
            editState(s => {
                s.todos.clearActive();
                s.mode = Mode.NO;
            });

            // Construct message directed at Claude with prescribed format
            const message = `[DAIC: Todo Change Blocked]

You attempted to modify the agreed-upon todo list without user approval.

ORIGINAL TODOS (${originalCount} items):
${originalDisplay}

PROPOSED TODOS (${proposedCount} items):
${proposedDisplay}

The original todos have been cleared and you have been returned to discussion mode.

YOUR NEXT MESSAGE MUST use this exact format:

---
[SHAME RITUAL]
I made a boo boo. I just tried to change the plan.

The todos you approved were:
${originalDisplay}

I tried to change them by [adding/removing/modifying] them:
[Show the changes - use + for added items, - for removed items, -> for modifications]

This [seems fine/is unimportant | was a violation of the execution boundary].

If you approve of the change, you can let me cook by saying: ${triggerList}

Or, feel free to yell at me or redirect me like I'm a 5 year old child.
---

After the user approves with a trigger phrase, you may re-submit the updated todo list using TodoWrite.`;

            console.error(message);
            process.exit(2);
        }
    }

    editState(s => {
        if (!s.todos.storeTodos(incomingTodos)) {
            console.error("[TodoWrite Error] Failed to store todos - check format");
            process.exit(2);
        }
    });
}
//!<

//!> TodoList modification guard
// Get the file path being edited
if (!filePath) {
    process.exit(0); // No file path, allow to proceed
}

// Block direct modification of state file via Write/Edit/MultiEdit
if (["Write", "Edit", "MultiEdit", "NotebookEdit"].includes(toolName) &&
    path.basename(filePath) === 'sessions-state.json' &&
    path.basename(path.dirname(filePath)) === 'sessions' &&
    !STATE.flags.bypass_mode) {
    console.error("[Security] Direct modification of sessions-state.json is not allowed. " +
                  "This file should only be modified through the TodoWrite tool and approved commands.");
    process.exit(2);
}
//!<

//!> Git branch/task submodules enforcement
const expectedBranch = STATE.current_task?.branch;
if (!expectedBranch) {
    process.exit(0); // No branch/task info, allow to proceed
}

// Check if branch enforcement is enabled
if (!CONFIG.features.branch_enforcement) {
    process.exit(0); // Branch enforcement disabled, allow to proceed
}

const repoPath = findGitRepo(path.dirname(filePath));

if (repoPath) {
    try {
        const result = execSync("git branch --show-current", {
            cwd: repoPath,
            encoding: 'utf8',
            timeout: 2000
        });
        const currentBranch = result.trim();

        // Extract the submodule name from the repo path
        const submoduleName = path.basename(repoPath);

        // Check both conditions: branch status and task inclusion
        const branchCorrect = (currentBranch === expectedBranch);
        const inTask = (STATE.current_task.submodules && STATE.current_task.submodules.includes(submoduleName)) ||
                       (repoPath === PROJECT_ROOT); // Root repo - always considered in task

        // Scenario 1: Everything is correct - allow to proceed
        if (inTask && branchCorrect) {
            // Allow
        }
        // Scenario 2: Submodule is in task but on wrong branch
        else if (inTask && !branchCorrect) {
            console.error(`[Branch Mismatch] Submodule '${submoduleName}' is part of this task but is on branch '${currentBranch}' instead of '${expectedBranch}'.`);
            console.error(`Please run: cd ${path.relative(PROJECT_ROOT, repoPath)} && git checkout ${expectedBranch}`);
            process.exit(2);
        }
        // Scenario 3: Submodule not in task but already on correct branch
        else if (!inTask && branchCorrect) {
            console.error(`[Submodule Not in Task] Submodule '${submoduleName}' is on the correct branch '${expectedBranch}' but is not listed in the task file.`);
            console.error(`Please update the task file to include '${submoduleName}' in the submodules list.`);
            process.exit(2);
        }
        // Scenario 4: Submodule not in task AND on wrong branch
        else {
            console.error(`[Submodule Not in Task + Wrong Branch] Submodule '${submoduleName}' has two issues:`);
            console.error(`  1. Not listed in the task file's submodules`);
            console.error(`  2. On branch '${currentBranch}' instead of '${expectedBranch}'`);
            console.error(`To fix: cd ${path.relative(PROJECT_ROOT, repoPath)} && git checkout -b ${expectedBranch}`);
            console.error(`Then update the task file to include '${submoduleName}' in the submodules list.`);
            process.exit(2);
        }
    } catch (error) {
        // Can't check branch, allow to proceed but warn
        console.error(`Warning: Could not verify branch for ${path.basename(repoPath)}: ${error.message}`);
    }
}
//!<

//-//

// Allow tool to proceed
process.exit(0);
