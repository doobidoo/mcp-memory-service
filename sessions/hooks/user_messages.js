#!/usr/bin/env node

// ===== IMPORTS ===== //

/// ===== STDLIB ===== ///
const fs = require('fs');
const path = require('path');
///-///

/// ===== 3RD-PARTY ===== ///
///-///

/// ===== LOCAL ===== ///
const {
    loadState,
    editState,
    Mode,
    PROJECT_ROOT,
    CCTodo,
    loadConfig,
    SessionsProtocol,
    isDirectoryTask,
    isSubtask,
    isParentTask
} = require('./shared_state.js');
///-///

//-//

// ===== GLOBALS ===== //

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

// Skip user messages hook in CI environments
if (isCIEnvironment()) {
    process.exit(0);
}
///-///

// Read stdin synchronously
let inputData = {};
try {
    const stdin = fs.readFileSync(0, 'utf-8');
    inputData = JSON.parse(stdin);
} catch (e) {
    // If parsing fails, treat as empty
    inputData = {};
}

const prompt = inputData.prompt || "";
const transcriptPath = inputData.transcript_path || "";

const STATE = loadState();
const CONFIG = loadConfig();

// Check if this is a slash command we handle via API
const promptStripped = prompt.trim();
const apiCommands = ['/mode', '/state', '/config', '/add-trigger', '/remove-trigger'];
const isApiCommand = promptStripped ? apiCommands.some(cmd => promptStripped.startsWith(cmd)) : false;

// Only add ultrathink if not an API command
let context = "";
if (CONFIG.features.auto_ultrathink && !isApiCommand) {
    context = "[[ ultrathink ]]\n\n";
}

//!> Trigger phrase detection
function phraseMatches(phrase, text) {
    // Case-sensitive only if phrase contains letters and all letters are uppercase.
    const hasLetters = /[A-Za-z]/.test(phrase);
    const isAllCaps = hasLetters && phrase === phrase.toUpperCase();
    if (isAllCaps) return text.includes(phrase);
    return text.toLowerCase().includes(phrase.toLowerCase());
}

const implementationPhraseDetected = CONFIG.trigger_phrases.implementation_mode.some(
    phrase => phraseMatches(phrase, prompt)
);
const discussionPhraseDetected = CONFIG.trigger_phrases.discussion_mode.some(
    phrase => phraseMatches(phrase, prompt)
);
const taskCreationDetected = CONFIG.trigger_phrases.task_creation.some(
    phrase => phraseMatches(phrase, prompt)
);
const taskCompletionDetected = CONFIG.trigger_phrases.task_completion.some(
    phrase => phraseMatches(phrase, prompt)
);
const taskStartDetected = CONFIG.trigger_phrases.task_startup.some(
    phrase => phraseMatches(phrase, prompt)
);
const compactionDetected = CONFIG.trigger_phrases.context_compaction.some(
    phrase => phraseMatches(phrase, prompt)
);
//!<

//!> Flags
let hadActiveTodos = false;
//!<

//-//

/*
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║ ██████╗ █████╗  █████╗ ███╗  ███╗██████╗ ██████╗  ██╗  ██╗ █████╗  █████╗ ██╗  ██╗██████╗ ║
║ ██╔══██╗██╔═██╗██╔══██╗████╗████║██╔══██╗╚═██╔═╝  ██║  ██║██╔══██╗██╔══██╗██║ ██╔╝██╔═══╝ ║
║ ██████╔╝█████╔╝██║  ██║██╔███║██║██████╔╝  ██║    ███████║██║  ██║██║  ██║█████╔╝ ██████╗ ║
║ ██╔═══╝ ██╔═██╗██║  ██║██║╚══╝██║██╔═══╝   ██║    ██╔══██║██║  ██║██║  ██║██╔═██╗ ╚═══██║ ║
║ ██║     ██║ ██║╚█████╔╝██║    ██║██║       ██║    ██║  ██║╚█████╔╝╚█████╔╝██║  ██╗██████║ ║
║ ╚═╝     ╚═╝ ╚═╝ ╚════╝ ╚═╝    ╚═╝╚═╝       ╚═╝    ╚═╝  ╚═╝ ╚════╝  ╚════╝ ╚═╝  ╚═╝╚═════╝ ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
UserPromptSubmit Hook

Manages DAIC mode transitions and protocol triggers:
- Detects trigger phrases for mode switching and protocol activation
- Monitors context window usage and provides warnings
- Auto-loads protocol todos when protocols are triggered
- Clears active todos when switching contexts
*/

// ===== FUNCTIONS ===== //
function loadProtocolFile(relativePath) {
    // Load a protocol file or chunk from sessions/protocols/
    const filePath = path.join(PROJECT_ROOT, 'sessions', 'protocols', relativePath);
    if (!fs.existsSync(filePath)) {
        return "";
    }
    return fs.readFileSync(filePath, 'utf8');
}

function formatTodosForProtocol(todos) {
    // Format a list of CCTodo objects for display in protocols
    const lines = ["## Protocol Todos", "<!-- Use TodoWrite to add these todos exactly as written -->"];
    for (const todo of todos) {
        lines.push(`□ ${todo.content}`);
    }
    return lines.join("\n");
}

function getContextLengthFromTranscript(transcriptPath) {
    // Get current context length from the most recent main-chain message in transcript
    try {
        const lines = fs.readFileSync(transcriptPath, 'utf8').split('\n');

        let mostRecentUsage = null;
        let mostRecentTimestamp = null;

        // Parse each JSONL entry
        for (const line of lines) {
            if (!line.trim()) continue;
            try {
                const data = JSON.parse(line);
                // Skip sidechain entries (subagent calls)
                if (data.isSidechain) continue;

                // Check if this entry has usage data
                if (data.message?.usage) {
                    const entryTime = data.timestamp;
                    // Track the most recent main-chain entry with usage
                    if (entryTime && (!mostRecentTimestamp || entryTime > mostRecentTimestamp)) {
                        mostRecentTimestamp = entryTime;
                        mostRecentUsage = data.message.usage;
                    }
                }
            } catch (e) {
                continue;
            }
        }

        // Calculate context length from most recent usage
        if (mostRecentUsage) {
            const contextLength = (
                (mostRecentUsage.input_tokens || 0) +
                (mostRecentUsage.cache_read_input_tokens || 0) +
                (mostRecentUsage.cache_creation_input_tokens || 0)
            );
            return contextLength;
        }
    } catch (e) {
        // Ignore errors
    }
    return 0;
}
//-//

// ===== EXECUTION ===== //

/// ===== TOKEN MONITORING ===== ///
// Check context usage and warn if needed
if (transcriptPath && fs.existsSync(transcriptPath)) {
    const contextLength = getContextLengthFromTranscript(transcriptPath);

    if (contextLength > 0) {
        // Calculate percentage of usable context (opus 160k/sonnet 800k practical limit before auto-compact)
        let usableTokens = 160000;
        if (STATE.model === "sonnet") {
            usableTokens = 800000;
        }
        const usablePercentage = (contextLength / usableTokens) * 100;

        // Token warnings (only show once per session)
        if (usablePercentage >= 90 && !STATE.flags.context_90 && CONFIG.features.context_warnings.warn_90) {
            context += `\n[90% WARNING] ${contextLength.toLocaleString()}/${usableTokens.toLocaleString()} tokens used (${usablePercentage.toFixed(1)}%). CRITICAL: Run sessions/protocols/task-completion.md to wrap up this task cleanly!\n`;
            editState(s => {
                s.flags.context_90 = true;
            });
        } else if (usablePercentage >= 85 && !STATE.flags.context_85 && CONFIG.features.context_warnings.warn_85) {
            context += `\n[Warning] Context window is ${usablePercentage.toFixed(1)}% full (${contextLength.toLocaleString()}/${usableTokens.toLocaleString()} tokens). The danger zone is >90%. You will receive another warning when you reach 90% - don't panic but gently guide towards context compaction or task completion (if task is nearly complete). Task completion often satisfies compaction requirements and should allow the user to clear context safely, so you do not need to worry about fitting in both processes.\n`;
            editState(s => {
                s.flags.context_85 = true;
            });
        }
    }
}
///-///

/// ===== TRIGGER DETECTION ===== ///

//!> Discussion/Implementation mode toggling
// Implementation triggers (only work in discussion mode, skip for /add-trigger)
if (!isApiCommand && STATE.mode === Mode.NO && implementationPhraseDetected) {
    editState(s => {
        s.mode = Mode.GO;
    });
    context += `[DAIC: Implementation Mode Activated]
CRITICAL RULES:
- Convert your proposed todos to TodoWrite EXACTLY as written
- Do NOT add new todos - only implement approved items
- Do NOT remove todos - complete them or return to discussion
- Check off each todo as you complete it
- If you discover you need to change your approach, return to discussion mode using the API command
- Todo list defines your execution boundary
- When all todos are complete, you'll auto-return to discussion
`;
}

// Emergency stop (works in any mode)
if (STATE.mode === Mode.GO && discussionPhraseDetected) {
    // DEBUG: Log what triggered this
    const debugLogPath = path.join(PROJECT_ROOT, "sessions", "mode-revert-debug.log");
    const debugLog = `
[${new Date().toISOString()}] EMERGENCY STOP TRIGGERED
  Prompt: ${prompt.substring(0, 200)}...
  discussion_phrase_detected: ${discussionPhraseDetected}
  Trigger phrases: ${JSON.stringify(CONFIG.trigger_phrases.discussion_mode)}
`;
    fs.appendFileSync(debugLogPath, debugLog);

    editState(s => {
        s.mode = Mode.NO;
        s.todos.clearActive();
    });
    context += "[DAIC: EMERGENCY STOP] All tools locked. You are now in discussion mode. Re-align with your pair programmer.\n";
}
//!<

//!> Task creation
if (!isApiCommand && taskCreationDetected) {
    // Define todos for this protocol
    const todos = [
        new CCTodo({
            content: 'Create task file from template with appropriate priority, type, and structure',
            activeForm: 'Creating task file from template'
        }),
        new CCTodo({
            content: 'Ask user about task success and propose success criteria',
            activeForm: 'Asking user about task success and proposing success criteria'
        }),
        new CCTodo({
            content: 'Run context-gathering agent to create context manifest',
            activeForm: 'Running context-gathering agent to create context manifest'
        }),
        new CCTodo({
            content: 'Update appropriate service index files',
            activeForm: 'Updating appropriate service index files'
        }),
        new CCTodo({
            content: 'Commit the new task file',
            activeForm: 'Committing the new task file'
        })
    ];

    // Load and compose protocol based on config
    let protocolContent = loadProtocolFile('task-creation/task-creation.md');

    // Build template variables
    const submodulesField = CONFIG.git_preferences.has_submodules
        ? "\n  - submodules: List all submodules requiring git branches for the task (all that will be affected)"
        : "";

    const templateVars = {
        submodules_field: submodulesField,
        todos: formatTodosForProtocol(todos)
    };

    // Format protocol with template variables
    if (protocolContent) {
        protocolContent = protocolContent.replace(/\{(\w+)\}/g, (match, key) => templateVars[key] || match);
    }

    editState(s => {
        s.mode = Mode.GO;
        s.active_protocol = SessionsProtocol.CREATE;
        if (s.todos.active.length > 0) {
            hadActiveTodos = true;
            s.todos.stashActive();
        }
        s.todos.active = todos;
    });

    context += "[Task Creation Notice]\n";

    if (protocolContent) {
        context += `User triggered task creation. Protocol:\n${protocolContent}\n`;
    } else {
        // Fallback to old behavior if protocol not found
        context += "User triggered task creation. Read sessions/protocols/task-creation.md\n";
    }

    if (hadActiveTodos) {
        context += "\nYour previous todos have been stashed and will be restored after task creation is complete.\n";
    }
}
//!<

//!> Task completion
if (!isApiCommand && taskCompletionDetected) {
    // Define todos for this protocol
    const todos = [
        new CCTodo({
            content: 'Verify all success criteria are checked off',
            activeForm: 'Verifying status of success criteria'
        }),
        new CCTodo({
            content: 'Run code-review agent and address any critical issues',
            activeForm: 'Running code-review agent'
        }),
        new CCTodo({
            content: 'Run logging agent to consolidate work logs',
            activeForm: 'Running logging agent to consolidate work logs'
        }),
        new CCTodo({
            content: 'Run service-documentation agent to update CLAUDE.md files and other documentation',
            activeForm: 'Running service-documentation agent to update documentation'
        }),
        new CCTodo({
            content: 'Mark task file complete and move to tasks/done/',
            activeForm: 'Archiving task file'
        })
    ];

    // Build commit todo based on auto_merge preference and directory task status
    let commitContent = 'Commit changes';
    // Check if this is a directory task - if so, don't merge until all subtasks complete
    if (STATE.current_task.file && isDirectoryTask(STATE.current_task.file)) {
        commitContent += ' (directory task - no merge until all subtasks complete)';
    } else if (CONFIG.git_preferences.auto_merge) {
        commitContent += ` and merge to ${CONFIG.git_preferences.default_branch}`;
    } else {
        commitContent += ` and ask if user wants to merge to ${CONFIG.git_preferences.default_branch}`;
    }

    todos.push(new CCTodo({
        content: commitContent,
        activeForm: 'Committing and handling merge'
    }));

    // Add push todo based on auto_push preference
    if (CONFIG.git_preferences.auto_push) {
        todos.push(new CCTodo({
            content: 'Push changes to remote',
            activeForm: 'Pushing changes to remote'
        }));
    } else {
        todos.push(new CCTodo({
            content: 'Ask if user wants to push changes to remote',
            activeForm: 'Asking about pushing to remote'
        }));
    }

    // Load and compose protocol based on config
    let protocolContent = loadProtocolFile('task-completion/task-completion.md');

    // Build template variables based on configuration
    const templateVars = {
        default_branch: CONFIG.git_preferences.default_branch,
        todos: formatTodosForProtocol(todos)
    };

    // Git add warning (only for add_pattern == "all")
    templateVars.git_add_warning = CONFIG.git_preferences.add_pattern === 'all'
        ? loadProtocolFile('task-completion/git-add-warning.md')
        : '';

    // Staging instructions based on add_pattern
    templateVars.staging_instructions = CONFIG.git_preferences.add_pattern === 'all'
        ? loadProtocolFile('task-completion/staging-all.md')
        : loadProtocolFile('task-completion/staging-ask.md');  // Default to 'ask' for safety

    // Commit instructions based on has_submodules
    const commitInstructionsContent = CONFIG.git_preferences.has_submodules
        ? loadProtocolFile('task-completion/commit-superrepo.md')
        : loadProtocolFile('task-completion/commit-standard.md');

    // Directory task completion check - simplified to just control merge behavior
    let directoryCompletionCheck = '';
    if (STATE.current_task.file && isDirectoryTask(STATE.current_task.file)) {
        if (isParentTask(STATE.current_task.file)) {
            // Completing parent README.md - normal merge behavior
            directoryCompletionCheck = loadProtocolFile('task-completion/directory-task-completion.md');
            directoryCompletionCheck = directoryCompletionCheck.replace('{default_branch}', CONFIG.git_preferences.default_branch);
        } else if (isSubtask(STATE.current_task.file)) {
            // Completing a subtask - commit but don't merge
            directoryCompletionCheck = loadProtocolFile('task-completion/subtask-completion.md');
            directoryCompletionCheck = directoryCompletionCheck.replace('{default_branch}', CONFIG.git_preferences.default_branch);
        }
    }

    // Build merge and push instructions based on auto preferences (but override for subtasks)
    let mergeInstruction;
    if (STATE.current_task.file && isSubtask(STATE.current_task.file)) {
        mergeInstruction = 'Do not merge yet - subtask in directory task';
    } else if (CONFIG.git_preferences.auto_merge) {
        mergeInstruction = `Merge into ${CONFIG.git_preferences.default_branch}`;
    } else {
        mergeInstruction = `Ask user if they want to merge into ${CONFIG.git_preferences.default_branch}`;
    }

    const pushInstruction = CONFIG.git_preferences.auto_push
        ? 'Push the merged branch to remote'
        : 'Ask user if they want to push to remote';

    // Load commit style guidance based on preference
    let commitStyleGuidance = '';
    if (CONFIG.git_preferences.commit_style === 'conventional') {
        commitStyleGuidance = loadProtocolFile('task-completion/commit-style-conventional.md');
    } else if (CONFIG.git_preferences.commit_style === 'simple') {
        commitStyleGuidance = loadProtocolFile('task-completion/commit-style-simple.md');
    } else if (CONFIG.git_preferences.commit_style === 'detailed') {
        commitStyleGuidance = loadProtocolFile('task-completion/commit-style-detailed.md');
    } else {
        // Default to conventional if not specified
        commitStyleGuidance = loadProtocolFile('task-completion/commit-style-conventional.md');
    }
    templateVars.commit_style_guidance = commitStyleGuidance;

    // Format commit instructions with merge/push
    templateVars.commit_instructions = commitInstructionsContent
        .replace('{merge_instruction}', mergeInstruction)
        .replace('{push_instruction}', pushInstruction)
        .replace('{commit_style_guidance}', commitStyleGuidance)
        .replace('{default_branch}', CONFIG.git_preferences.default_branch);

    // Add directory task completion check
    templateVars.directory_completion_check = directoryCompletionCheck;

    // Format protocol with all template variables
    if (protocolContent) {
        protocolContent = protocolContent.replace(/\{(\w+)\}/g, (match, key) => templateVars[key] || match);
    }

    editState(s => {
        s.mode = Mode.GO;
        s.active_protocol = SessionsProtocol.COMPLETE;
        s.todos.active = todos;
    });

    context += "[Task Completion Notice]\n";

    if (protocolContent) {
        context += `User triggered task completion. Protocol:\n${protocolContent}\n`;
    } else {
        context += "User triggered task completion. Read sessions/protocols/task-completion.md\n";
    }
}
//!<

//!> Task startup
if (!isApiCommand && taskStartDetected) {
    let taskReference = null;
    const words = prompt.split(' ');
    for (const word of words) {
        if (word.startsWith("@") && word.includes("sessions/tasks/") && word.endsWith(".md")) {
            taskReference = word.split('sessions/tasks/').pop();
            break;
        }
    }

    // Load and compose protocol based on config
    let protocolContent = loadProtocolFile('task-startup/task-startup.md');

    // Load conditional chunks
    let submoduleManagement = '';
    let resumeNotes = '';

    if (CONFIG.git_preferences.has_submodules) {
        const submoduleManagementRaw = loadProtocolFile('task-startup/submodule-management.md');
        // Format the submodule management content with default_branch
        submoduleManagement = submoduleManagementRaw
            ? submoduleManagementRaw.replace('{default_branch}', CONFIG.git_preferences.default_branch)
            : "";
        resumeNotes = loadProtocolFile('task-startup/resume-notes-superrepo.md');
    } else {
        submoduleManagement = "";
        resumeNotes = loadProtocolFile('task-startup/resume-notes-standard.md');
    }

    // Check if this is a directory task and load appropriate guidance
    let directoryGuidance = '';
    if (STATE.current_task.file && isDirectoryTask(STATE.current_task.file)) {
        if (isParentTask(STATE.current_task.file)) {
            // Starting parent README.md - create task branch
            directoryGuidance = loadProtocolFile('task-startup/directory-task-startup.md');
        } else if (isSubtask(STATE.current_task.file)) {
            // Starting a subtask - ensure on parent task branch
            directoryGuidance = loadProtocolFile('task-startup/subtask-startup.md');
        }
    }

    // Set todos based on config
    const todoBranchContent = CONFIG.git_preferences.has_submodules
        ? 'Create/checkout task branch and matching submodule branches'
        : 'Create/checkout task branch';
    const todoBranchActive = CONFIG.git_preferences.has_submodules
        ? 'Creating/checking out task branches'
        : 'Creating/checking out task branch';

    // Build todos list - will add read task todo conditionally
    const todos = [
        new CCTodo({
            content: 'Check git status and handle any uncommitted changes',
            activeForm: 'Checking git status and handling uncommitted changes'
        }),
        new CCTodo({
            content: todoBranchContent,
            activeForm: todoBranchActive
        }),
        new CCTodo({
            content: 'Verify context manifest for the task',
            activeForm: 'Verifying context manifest'
        }),
        new CCTodo({
            content: 'Gather context for the task',
            activeForm: 'Catching up to speed...'
        })
    ];

    // Check if task will be auto-loaded
    // Detect OS for correct sessions command
    const isWindows = process.platform === "win32";
    const sessionsCmd = isWindows ? "sessions/bin/sessions.bat" : "sessions/bin/sessions";

    context += "[Task Startup Notice]\n**If the user mentioned which task to start, *YOU MUST***:\n";
    context += "1. Return to project root directory\n";
    context += `2. Run: \`${sessionsCmd} protocol startup-load <task-file>\`\n`;
    context += "You must do this *BEFORE* the task startup protocol.\n";
    context += "Otherwise, ask which task they want to start, then use the command from project root.\n\n";

    // Build template variables for protocol
    const gitStatusScope = CONFIG.git_preferences.has_submodules
        ? 'in both super-repo and all submodules'
        : '';

    // Build git handling instructions based on add_pattern
    const gitHandling = CONFIG.git_preferences.add_pattern === 'all'
        ? '- Commit ALL changes'
        : '- Either commit changes or explicitly discuss with user';

    const templateVars = {
        default_branch: CONFIG.git_preferences.default_branch,
        submodule_branch_todo: CONFIG.git_preferences.has_submodules ? ' and matching submodule branches' : '',
        submodule_context: CONFIG.git_preferences.has_submodules ? ' (and submodules list)' : '',
        submodule_management_section: submoduleManagement,
        resume_notes: resumeNotes,
        directory_guidance: directoryGuidance,
        git_status_scope: gitStatusScope,
        git_handling: gitHandling,
        todos: formatTodosForProtocol(todos),
        implementation_mode_triggers: CONFIG.trigger_phrases.implementation_mode.length > 0
            ? `[${CONFIG.trigger_phrases.implementation_mode.join(', ')}]`
            : "[]"
    };

    // Format protocol with template variables
    if (protocolContent) {
        protocolContent = protocolContent.replace(/\{(\w+)\}/g, (match, key) => templateVars[key] || match);
    }

    // Set state with todos
    editState(s => {
        s.mode = Mode.GO;
        s.active_protocol = SessionsProtocol.START;
        s.api.startup_load = true;
        s.todos.clearActive();
        s.todos.active = todos;
    });

    // Auto-load protocol content
    if (protocolContent) {
        context += `User triggered task startup. Protocol:\n${protocolContent}\n`;
    } else {
        context += "User triggered task startup. Read sessions/protocols/task-startup.md\n";
    }
}
//!<

//!> Context compaction
if (!isApiCommand && compactionDetected) {
    // Define todos for this protocol
    const todos = [
        new CCTodo({
            content: 'Run logging agent to update work logs',
            activeForm: 'Running logging agent to update work logs'
        }),
        new CCTodo({
            content: 'Run context-refinement agent to check for discoveries',
            activeForm: 'Running context-refinement agent to check for discoveries'
        }),
        new CCTodo({
            content: 'Run service-documentation agent if service interfaces changed',
            activeForm: 'Running service-documentation agent if service interfaces changed'
        })
    ];

    // Load protocol content
    let protocolContent = loadProtocolFile('context-compaction/context-compaction.md');

    // Build template variables
    const templateVars = {
        todos: formatTodosForProtocol(todos)
    };

    // Format protocol with template variables
    if (protocolContent) {
        protocolContent = protocolContent.replace(/\{(\w+)\}/g, (match, key) => templateVars[key] || match);
    }

    if (STATE.todos.active.length > 0) {
        hadActiveTodos = true;
        editState(s => {
            s.todos.stashActive();
        });
    }

    editState(s => {
        s.mode = Mode.GO;
        s.active_protocol = SessionsProtocol.COMPACT;
        s.todos.active = todos;
    });

    context += "[Context Compaction Notice]\n";

    if (protocolContent) {
        context += `User triggered context compaction. Protocol:\n${protocolContent}\n`;
    } else {
        // Fallback to old behavior if protocol not found
        context += "User triggered context compaction. Read sessions/protocols/context-compaction.md\n";
    }

    if (hadActiveTodos) {
        context += "Your todos have been stashed and will be restored in the next session after the user clears context. Do not attempt to update or complete your previous todo list (context compaction todos are now active).\n";
    }
}
//!<

//!> Iterloop detection
if (prompt.toLowerCase().includes("iterloop")) {
    context += "ITERLOOP DETECTED:\nYou have been instructed to iteratively loop over a list. Identify what list the user is referring to, then follow this loop: present one item, wait for the user to respond with questions and discussion points, only continue to the next item when the user explicitly says 'continue' or something similar\n";
}
//!<

///-///

//-//

// Output the context additions
const output = {
    hookSpecificOutput: {
        hookEventName: "UserPromptSubmit",
        additionalContext: context
    }
};
console.log(JSON.stringify(output));

process.exit(0);
