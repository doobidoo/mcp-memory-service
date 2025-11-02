#!/usr/bin/env node

// ==== IMPORTS ===== //

// ===== STDLIB ===== //
const fs = require('fs');
const path = require('path');
//--//

// ===== 3RD-PARTY ===== //
// No third-party dependencies needed
//--//

// ===== LOCAL ===== //
const {
    loadState,
    editState,
    loadConfig,
    TaskState,
    SessionsProtocol,
    PROJECT_ROOT,
    getTaskFilePath,
    isDirectoryTask
} = require('../hooks/shared_state.js');
//--//

//-#

// ==== GLOBALS ===== //
//-#

// ==== DECLARATIONS ===== //
//-#

// ==== CLASSES ===== //
//-#

// ==== FUNCTIONS ===== //

//!> Protocol loading helper
function loadProtocolFile(relativePath) {
    /**Load a protocol file from sessions/protocols/ directory.*/
    const protocolPath = path.join(PROJECT_ROOT, 'sessions', 'protocols', relativePath);
    if (fs.existsSync(protocolPath)) {
        try {
            return fs.readFileSync(protocolPath, 'utf8');
        } catch (error) {
            return null;
        }
    }
    return null;
}
//!<

//!> Index file parsing
function parseIndexFile(indexPath) {
    /**Parse an index file and extract metadata and task lines using simple string parsing.*/
    if (!fs.existsSync(indexPath)) {
        return null;
    }

    let content;
    try {
        content = fs.readFileSync(indexPath, 'utf8');
    } catch (error) {
        return null;
    }

    const lines = content.split('\n');

    // Extract frontmatter using simple string parsing
    const metadata = {};
    if (lines.length > 0 && lines[0] === '---') {
        for (let i = 1; i < lines.length; i++) {
            const line = lines[i];
            if (line === '---') {
                break;
            }
            if (line.includes(':')) {
                const colonIndex = line.indexOf(':');
                const key = line.substring(0, colonIndex).trim();
                const value = line.substring(colonIndex + 1).trim();
                metadata[key] = value;
            }
        }
    }

    // Extract task lines (those starting with - `)
    const taskLines = [];
    for (const line of lines) {
        if (line.trim().startsWith('- `')) {
            taskLines.push(line.trim());
        }
    }

    return { metadata, taskLines };
}

function parseFrontmatter(content) {
    /**Parse frontmatter from file content using simple string parsing.*/
    if (!content.startsWith('---')) {
        return null;
    }

    const parts = content.split('---');
    if (parts.length < 3) {
        return null;
    }

    const frontmatterLines = parts[1].split('\n');
    const frontmatter = {};

    for (const line of frontmatterLines) {
        if (line.includes(':')) {
            const colonIndex = line.indexOf(':');
            const key = line.substring(0, colonIndex).trim();
            let value = line.substring(colonIndex + 1).trim();

            // Handle special cases
            if (key === 'submodules' || key === 'dependencies') {
                // Parse arrays formatted as: [item1, item2]
                if (value.startsWith('[') && value.endsWith(']')) {
                    value = value.slice(1, -1)
                        .split(',')
                        .map(s => s.trim())
                        .filter(s => s.length > 0);
                } else if (value === 'null' || value === '') {
                    value = null;
                }
            } else if (value === 'null' || value === '') {
                value = null;
            }

            frontmatter[key] = value;
        }
    }

    return frontmatter;
}
//!<

//!> Task status extraction
function getTaskStatusMap() {
    /**Build a map of task names to their status.*/
    const tasksDir = path.join(PROJECT_ROOT, 'sessions', 'tasks');
    const taskStatus = {};

    let taskFiles = [];

    // Get regular .md files
    if (fs.existsSync(tasksDir)) {
        const files = fs.readdirSync(tasksDir)
            .filter(f => f.endsWith('.md') && f !== 'TEMPLATE.md')
            .map(f => path.join(tasksDir, f))
            .sort();
        taskFiles.push(...files);
    }

    // Get task directories with README.md files
    if (fs.existsSync(tasksDir)) {
        const dirs = fs.readdirSync(tasksDir)
            .filter(f => {
                const fullPath = path.join(tasksDir, f);
                return fs.statSync(fullPath).isDirectory() &&
                       !['done', 'indexes'].includes(f);
            })
            .sort();

        for (const dir of dirs) {
            const readmePath = path.join(tasksDir, dir, 'README.md');
            if (fs.existsSync(readmePath)) {
                taskFiles.push(path.join(tasksDir, dir));
            }

            // Get subtask files
            const subtasks = fs.readdirSync(path.join(tasksDir, dir))
                .filter(f => f.endsWith('.md') && !['TEMPLATE.md', 'README.md'].includes(f))
                .map(f => path.join(tasksDir, dir, f))
                .sort();
            taskFiles.push(...subtasks);
        }
    }

    // Extract status from each task file
    for (const taskFile of taskFiles) {
        const filePath = getTaskFilePath(taskFile);

        if (!fs.existsSync(filePath)) {
            continue;
        }

        try {
            const content = fs.readFileSync(filePath, 'utf8');
            const lines = content.split('\n').slice(0, 10);

            const taskName = isDirectoryTask(taskFile)
                ? path.basename(taskFile) + '/'
                : path.basename(taskFile);

            let status = null;
            for (const line of lines) {
                if (line.startsWith('status:')) {
                    status = line.split(':')[1].trim();
                    break;
                }
            }

            if (status) {
                taskStatus[taskName] = status;
            }
        } catch (error) {
            continue;
        }
    }

    return taskStatus;
}
//!<

//!> Index operations
function handleIdxList(jsonOutput = false) {
    /**List all available index files.*/
    const indexesDir = path.join(PROJECT_ROOT, 'sessions', 'tasks', 'indexes');

    if (!fs.existsSync(indexesDir)) {
        if (jsonOutput) {
            return { indexes: [], message: "No indexes directory found" };
        }
        return "No indexes directory found at sessions/tasks/indexes";
    }

    const indexFiles = fs.readdirSync(indexesDir)
        .filter(f => f.endsWith('.md'))
        .map(f => path.join(indexesDir, f))
        .sort();

    if (indexFiles.length === 0) {
        if (jsonOutput) {
            return { indexes: [], message: "No index files found" };
        }
        return "No index files found in sessions/tasks/indexes";
    }

    const indexesInfo = [];
    for (const indexFile of indexFiles) {
        const result = parseIndexFile(indexFile);
        if (result) {
            const { metadata, taskLines } = result;
            indexesInfo.push({
                file: path.basename(indexFile),
                id: metadata.index || path.basename(indexFile, '.md'),
                name: metadata.name || path.basename(indexFile, '.md'),
                description: metadata.description || '',
                task_count: taskLines.length
            });
        }
    }

    if (jsonOutput) {
        return { indexes: indexesInfo };
    }

    // Format human-readable output
    const lines = ["Available Task Indexes:", ""];
    for (const info of indexesInfo) {
        lines.push(`  • ${info.name} (${info.file})`);
        if (info.description) {
            lines.push(`    ${info.description}`);
        }
        lines.push(`    Tasks: ${info.task_count}`);
        lines.push("");
    }

    lines.push("Use '/sessions tasks idx <name>' to view tasks in a specific index");
    return lines.join('\n');
}

function handleIdxShow(indexName, jsonOutput = false) {
    /**Show pending tasks in a specific index file.*/
    const indexesDir = path.join(PROJECT_ROOT, 'sessions', 'tasks', 'indexes');

    // Try to find the index file (with or without .md extension)
    let indexPath = path.join(indexesDir, indexName.endsWith('.md') ? indexName : `${indexName}.md`);

    if (!fs.existsSync(indexPath)) {
        // Try alternate name
        const altName = indexName.endsWith('.md') ? indexName.slice(0, -3) : `${indexName}.md`;
        indexPath = path.join(indexesDir, altName);
    }

    if (!fs.existsSync(indexPath)) {
        if (jsonOutput) {
            return { error: `Index file not found: ${indexName}` };
        }
        return `Index file not found: ${indexName}\n\nUse '/sessions tasks idx list' to see available indexes`;
    }

    const result = parseIndexFile(indexPath);
    if (!result) {
        if (jsonOutput) {
            return { error: `Failed to parse index file: ${indexName}` };
        }
        return `Failed to parse index file: ${indexName}`;
    }

    const { metadata, taskLines } = result;
    const taskStatus = getTaskStatusMap();

    // Extract task names and filter by status
    const pendingTasks = [];
    for (const line of taskLines) {
        if (line.includes('`')) {
            try {
                const startIdx = line.indexOf('`') + 1;
                const endIdx = line.indexOf('`', startIdx);
                if (endIdx !== -1) {
                    const taskName = line.substring(startIdx, endIdx);
                    const status = taskStatus[taskName] || 'unknown';
                    if (status === 'pending' || status === 'in-progress') {
                        // Extract description if present
                        let desc = line.substring(endIdx + 1).trim();
                        if (desc.startsWith(' - ')) {
                            desc = desc.substring(3).trim();
                        }
                        pendingTasks.push({
                            name: taskName,
                            status: status,
                            description: desc
                        });
                    }
                }
            } catch (error) {
                continue;
            }
        }
    }

    if (jsonOutput) {
        return {
            index: metadata.index || indexName,
            name: metadata.name || indexName,
            description: metadata.description || '',
            pending_tasks: pendingTasks
        };
    }

    // Format human-readable output
    const lines = [
        `# ${metadata.name || indexName}`,
        ""
    ];

    if (metadata.description) {
        lines.push(metadata.description);
        lines.push("");
    }

    if (pendingTasks.length === 0) {
        lines.push("No pending tasks in this index");
    } else {
        lines.push("Pending Tasks:");
        lines.push("");
        for (const task of pendingTasks) {
            lines.push(`  @${task.name} (${task.status})`);
            if (task.description) {
                lines.push(`    ${task.description}`);
            }
        }

        lines.push("");
        lines.push("To start a task, use: /sessions tasks start @<task-name>");
    }

    return lines.join('\n');
}
//!<

//!> Task startup
function handleTaskStart(taskName, jsonOutput = false, fromSlash = false) {
    /**Start a task with validation and protocol loading.*/
    const config = loadConfig();
    const state = loadState();

    // Check if there's already an active task
    if (state.current_task && state.current_task.name) {
        const errorMsg = `Cannot start task - there is already an active task: ${state.current_task.name}

To clear the current task and start a new one:
  1. Clear the current task: /sessions state task clear
  2. Start the new task: /sessions tasks start ${taskName}

To restore the previous task later:
  /sessions state task restore ${state.current_task.file}
`;
        if (jsonOutput) {
            return {
                error: "Task already active",
                current_task: state.current_task.name,
                message: errorMsg
            };
        }
        return errorMsg;
    }

    // Strip @ symbol if present
    if (taskName.startsWith('@')) {
        taskName = taskName.substring(1);
    }

    // Resolve task file path
    const tasksDir = path.join(PROJECT_ROOT, 'sessions', 'tasks');
    const taskPath = path.join(tasksDir, taskName);

    if (!fs.existsSync(taskPath)) {
        const errorMsg = `Task file not found: ${taskName}\n\nMake sure the task file exists in sessions/tasks/`;
        if (jsonOutput) {
            return { error: "Task not found", message: errorMsg };
        }
        return errorMsg;
    }

    // Read and parse task frontmatter
    let content;
    try {
        content = fs.readFileSync(taskPath, 'utf8');
    } catch (error) {
        const errorMsg = `Failed to read task file: ${taskName}`;
        if (jsonOutput) {
            return { error: "Read failed", message: errorMsg };
        }
        return errorMsg;
    }

    if (!content.startsWith('---')) {
        const errorMsg = `Task file missing frontmatter: ${taskName}`;
        if (jsonOutput) {
            return { error: "Invalid format", message: errorMsg };
        }
        return errorMsg;
    }

    const frontmatter = parseFrontmatter(content);
    if (!frontmatter) {
        const errorMsg = `Invalid frontmatter format in: ${taskName}`;
        if (jsonOutput) {
            return { error: "Invalid format", message: errorMsg };
        }
        return errorMsg;
    }

    // Load and compose protocol based on config
    let protocolContent = loadProtocolFile('task-startup/task-startup.md');

    // Load conditional chunks
    let submoduleManagement = "";
    let resumeNotes;

    if (config.git_preferences.has_submodules) {
        const submoduleManagementRaw = loadProtocolFile('task-startup/submodule-management.md');
        if (submoduleManagementRaw) {
            submoduleManagement = submoduleManagementRaw.replace(/\{default_branch\}/g, config.git_preferences.default_branch);
        }
        resumeNotes = loadProtocolFile('task-startup/resume-notes-superrepo.md');
    } else {
        submoduleManagement = "";
        resumeNotes = loadProtocolFile('task-startup/resume-notes-standard.md');
    }

    // Set todos based on config
    const todoBranchContent = config.git_preferences.has_submodules
        ? 'Create/checkout task branch and matching submodule branches'
        : 'Create/checkout task branch';
    const todoBranchActive = config.git_preferences.has_submodules
        ? 'Creating/checking out task branches'
        : 'Creating/checking out task branch';

    const todos = [
        { content: todoBranchContent, status: "pending", activeForm: todoBranchActive },
        { content: "Gather context if task lacks context manifest", status: "pending", activeForm: "Gathering context for task" },
        { content: "Begin work on the task", status: "pending", activeForm: "Beginning work on task" }
    ];

    // Format protocol with template substitutions
    if (protocolContent) {
        protocolContent = protocolContent
            .replace(/\{task_reference\}/g, taskName)
            .replace(/\{submodule_management\}/g, submoduleManagement || '')
            .replace(/\{resume_notes\}/g, resumeNotes || '')
            .replace(/\{todo_branch\}/g, todos[0].content)
            .replace(/\{todo_branch_active\}/g, todos[0].activeForm);
    }

    // Update state with task and protocol
    editState(s => {
        const taskState = new TaskState({
            name: frontmatter.name,
            file: taskName,
            branch: frontmatter.branch,
            status: 'in-progress',
            created: frontmatter.created,
            started: frontmatter.started,
            updated: frontmatter.updated,
            dependencies: frontmatter.dependencies,
            submodules: frontmatter.submodules
        });
        s.current_task = taskState;
        s.active_protocol = SessionsProtocol.START;
        s.api.startup_load = true;
        s.todos.clearActive();
        s.todos.active = todos;
    });

    if (jsonOutput) {
        return {
            message: `Task '${frontmatter.name}' started`,
            task_file: taskName,
            protocol: protocolContent,
            todos: todos
        };
    }

    // Return protocol content for Claude to read
    let output = `Task startup initiated for: ${frontmatter.name}\n\n`;
    if (protocolContent) {
        output += protocolContent;
    } else {
        output += "Read sessions/protocols/task-startup/task-startup.md for startup protocol";
    }

    return output;
}
//!<

//!> Main task handler
function handleTaskCommand(args, jsonOutput = false, fromSlash = false) {
    /**
     * Handle task management commands.
     *
     * Usage:
     *     tasks idx list              - List all index files
     *     tasks idx <name>            - Show pending tasks in index
     *     tasks start <@task-name>    - Start a task
     */
    if (!args || args.length === 0 || args[0].toLowerCase() === 'help') {
        if (fromSlash) {
            return formatTaskHelp();
        }
        throw new Error("tasks command requires an action. Valid actions: idx, start");
    }

    const action = args[0].toLowerCase();

    if (action === 'idx') {
        if (args.length < 2) {
            if (fromSlash) {
                return "idx command requires an argument.\n\nUsage:\n  /sessions tasks idx list       - List all indexes\n  /sessions tasks idx <name>     - Show tasks in specific index";
            }
            throw new Error("idx command requires an argument: list or <index-name>");
        }

        const idxAction = args[1].toLowerCase();

        if (idxAction === 'list') {
            return handleIdxList(jsonOutput);
        } else {
            // Treat as index name
            return handleIdxShow(idxAction, jsonOutput);
        }

    } else if (action === 'start') {
        if (args.length < 2) {
            if (fromSlash) {
                return "start command requires a task name.\n\nUsage:\n  /sessions tasks start @<task-name>";
            }
            throw new Error("start command requires a task name");
        }

        const taskName = args[1];
        return handleTaskStart(taskName, jsonOutput, fromSlash);

    } else {
        if (fromSlash) {
            return `Unknown tasks action: ${action}\n\n${formatTaskHelp()}`;
        }
        throw new Error(`Unknown tasks action: ${action}. Valid actions: idx, start`);
    }
}

function formatTaskHelp() {
    /**Format help output for slash command.*/
    const lines = [
        "Sessions Task Commands:",
        "",
        "  /sessions tasks idx list        - List all available task indexes",
        "  /sessions tasks idx <name>      - Show pending tasks in specific index",
        "  /sessions tasks start @<name>   - Start working on a task",
        "",
        "Examples:",
        "  /sessions tasks idx list                    - See all indexes",
        "  /sessions tasks idx architecture            - View architecture tasks",
        "  /sessions tasks start @m-refactor-commands  - Start a task",
    ];
    return lines.join('\n');
}
//!<

//-#

// ==== EXPORTS ===== //
module.exports = {
    handleTaskCommand
};
//-#