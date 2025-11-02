#!/usr/bin/env node

// ==== IMPORTS ===== //

// ===== LOCAL ===== //
const { loadConfig, editConfig, TriggerCategory, GitAddPattern, GitCommitStyle, UserOS, UserShell, CCTools, IconStyle } = require('../hooks/shared_state.js');

//-#

// ==== GLOBALS ===== //
//-#

// ==== FUNCTIONS ===== //

//!> Main config handler
function handleConfigCommand(args, jsonOutput = false, fromSlash = false) {
    /**
     * Handle configuration commands.
     *
     * Usage:
     *     config                          - Show full config
     *     config phrases <operation>      - Manage trigger phrases
     *     config git <operation>          - Manage git preferences
     *     config env <operation>          - Manage environment settings
     *     config features <operation>     - Manage feature toggles
     *     config validate                 - Validate configuration
     */
    // Handle help command specially for slash command integration
    if (!args || args.length === 0 || (args.length > 0 && ['help', ''].includes(args[0].toLowerCase()))) {
        if (fromSlash && (!args || args.length === 0 || args[0].toLowerCase() === 'help')) {
            return formatConfigHelp();
        } else if (!args || args.length === 0) {
            // Show full config when no args
            const config = loadConfig();
            if (jsonOutput) {
                return config.toDict();
            }
            return formatConfigHuman(config);
        }
    }

    const section = args[0].toLowerCase();
    const sectionArgs = args.length > 1 ? args.slice(1) : [];

    // Handle show command
    if (section === 'show') {
        const config = loadConfig();
        if (jsonOutput) {
            return config.toDict();
        }
        return formatConfigHuman(config);
    } else if (['trigger', 'triggers', 'phrases'].includes(section)) {
        return handlePhrasesCommand(sectionArgs, jsonOutput, fromSlash);
    } else if (section === 'git') {
        return handleGitCommand(sectionArgs, jsonOutput, fromSlash);
    } else if (section === 'env') {
        return handleEnvCommand(sectionArgs, jsonOutput, fromSlash);
    } else if (section === 'features') {
        return handleFeaturesCommand(sectionArgs, jsonOutput, fromSlash);
    } else if (section === 'read') {
        return handleReadCommand(sectionArgs, jsonOutput, fromSlash);
    } else if (section === 'write') {
        return handleWriteCommand(sectionArgs, jsonOutput, fromSlash);
    } else if (section === 'tools') {
        return handleToolsCommand(sectionArgs, jsonOutput, fromSlash);
    } else if (['readonly', 'perms'].includes(section)) {
        return handleReadonlyCommand(sectionArgs, jsonOutput, fromSlash);
    } else if (section === 'validate') {
        return validateConfig(jsonOutput);
    } else {
        if (fromSlash) {
            return `Unknown command: ${section}\n\n${formatConfigHelp()}`;
        }
        throw new Error(`Unknown config section: ${section}. Valid sections: phrases, git, env, features, read, write, tools, readonly, validate`);
    }
}

function formatConfigHelp() {
    /**Format help output for slash command.*/
    const lines = [
        "Sessions Configuration Commands:",
        "",
        "  /sessions config show           - Display current configuration",
        "  /sessions config trigger ...    - Manage trigger phrases",
        "  /sessions config git ...        - Manage git preferences",
        "  /sessions config env ...        - Manage environment settings",
        "  /sessions config features ...   - Manage feature toggles",
        "  /sessions config read ...       - Manage bash read patterns",
        "  /sessions config write ...      - Manage bash write patterns",
        "  /sessions config tools ...      - Manage implementation-only tools",
        "  /sessions config read ...   - Manage readonly bash commands",
        "",
        "Use '/sessions config <section> help' for section-specific help"
    ];
    return lines.join('\n');
}

function formatConfigHuman(config) {
    /**Format full config for human reading.*/
    // Helper to safely get value from enum or string
    const getValue = (field) => {
        return typeof field === 'object' && field.value !== undefined ? field.value : field;
    };

    const lines = [
        "=== Sessions Configuration ===",
        "",
        "Trigger Phrases:"
    ];

    for (const category of Object.values(TriggerCategory)) {
        const phrases = config.trigger_phrases[category] || [];
        if (phrases.length > 0) {
            lines.push(`  ${category}: ${phrases.join(', ')}`);
        }
    }

    lines.push(...[
        "",
        "Git Preferences:",
        `  Add Pattern: ${getValue(config.git_preferences.add_pattern)}`,
        `  Default Branch: ${config.git_preferences.default_branch}`,
        `  Commit Style: ${getValue(config.git_preferences.commit_style)}`,
        `  Auto Merge: ${config.git_preferences.auto_merge}`,
        `  Auto Push: ${config.git_preferences.auto_push}`,
        `  Has Submodules: ${config.git_preferences.has_submodules}`,
        "",
        "Environment:",
        `  OS: ${getValue(config.environment.os)}`,
        `  Shell: ${getValue(config.environment.shell)}`,
        `  Developer Name: ${config.environment.developer_name}`,
        "",
        "Features:",
        `  Branch Enforcement: ${config.features.branch_enforcement}`,
        `  Task Detection: ${config.features.task_detection}`,
        `  Auto Ultrathink: ${config.features.auto_ultrathink}`,
        `  Icon Style: ${getValue(config.features.icon_style)}`,
        `  Context Warnings (85%): ${config.features.context_warnings.warn_85}`,
        `  Context Warnings (90%): ${config.features.context_warnings.warn_90}`,
    ]);

    return lines.join('\n');
}
//!<

//!> Trigger phrases handlers
function handlePhrasesCommand(args, jsonOutput = false, fromSlash = false) {
    /**
     * Handle trigger phrase commands.
     *
     * Usage:
     *     config phrases list [category]
     *     config phrases add <category> "<phrase>"
     *     config phrases remove <category> "<phrase>"
     *     config phrases clear <category>
     */
    if (!args || args.length === 0 || (args.length > 0 && args[0].toLowerCase() === 'help')) {
        if (fromSlash) {
            return formatPhrasesHelp();
        }
        // List all phrases
        const config = loadConfig();
        const phrases = config.trigger_phrases.listPhrases();
        if (jsonOutput) {
            return { phrases: phrases };
        }
        return formatPhrasesHuman(phrases);
    }

    const action = args[0].toLowerCase();

    // Map friendly category names for slash commands
    function mapCategory(cat) {
        if (!fromSlash) {
            return cat;
        }
        const mapping = {
            'go': 'implementation_mode',
            'no': 'discussion_mode',
            'create': 'task_creation',
            'start': 'task_startup',
            'complete': 'task_completion',
            'compact': 'context_compaction'
        };
        return mapping[cat] || cat;
    }

    if (action === 'list') {
        const config = loadConfig();
        let phrases;
        if (args.length > 1) {
            // List specific category
            const category = mapCategory(args[1]);
            phrases = config.trigger_phrases.listPhrases(category);
        } else {
            // List all
            phrases = config.trigger_phrases.listPhrases();
        }

        if (jsonOutput) {
            return { phrases: phrases };
        }
        return formatPhrasesHuman(phrases);

    } else if (action === 'add') {
        if (args.length < 2) {
            if (fromSlash) {
                return "Missing category for add command\nUsage: /sessions config trigger add <category> <phrase>\nValid categories: go, no, create, start, complete, compact\n\nExample: /sessions config trigger add go 'proceed'";
            }
            throw new Error('Usage: config phrases add <category> "<phrase>"');
        }

        const category = mapCategory(args[1]);

        // Check for valid category
        const validCategories = ['implementation_mode', 'discussion_mode', 'task_creation', 'task_startup', 'task_completion', 'context_compaction'];
        if (!validCategories.includes(category)) {
            if (fromSlash) {
                return `Invalid category '${args[1]}'\nValid categories: go, no, create, start, complete, compact\n\nUse '/sessions config trigger help' for more info`;
            }
            throw new Error(`Invalid category: ${category}`);
        }

        // Collect remaining args as phrase
        if (args.length < 3) {
            if (fromSlash) {
                return "Missing phrase to add\nUsage: /sessions config trigger add <category> <phrase>\n\nExample: /sessions config trigger add go 'proceed'";
            }
            throw new Error("Missing phrase");
        }

        const phrase = args.slice(2).join(' ');

        let added = false;
        editConfig(config => {
            added = config.trigger_phrases.addPhrase(category, phrase);
        });

        if (jsonOutput) {
            return { added: added, category: category, phrase: phrase };
        }
        if (added) {
            return `Added '${phrase}' to ${category}`;
        } else {
            return `'${phrase}' already exists in ${category}`;
        }

    } else if (action === 'remove') {
        if (args.length < 2) {
            if (fromSlash) {
                return "Missing category for remove command\nUsage: /sessions config trigger remove <category> <phrase>\nValid categories: go, no, create, start, complete, compact";
            }
            throw new Error('Usage: config phrases remove <category> "<phrase>"');
        }

        const category = mapCategory(args[1]);

        // Check for valid category
        const validCategories = ['implementation_mode', 'discussion_mode', 'task_creation', 'task_startup', 'task_completion', 'context_compaction'];
        if (!validCategories.includes(category)) {
            if (fromSlash) {
                return `Invalid category '${args[1]}'\nValid categories: go, no, create, start, complete, compact\n\nUse '/sessions config trigger help' for more info`;
            }
            throw new Error(`Invalid category: ${category}`);
        }

        // Collect remaining args as phrase
        if (args.length < 3) {
            if (fromSlash) {
                return "Missing phrase to remove\nUsage: /sessions config trigger remove <category> <phrase>";
            }
            throw new Error("Missing phrase");
        }

        const phrase = args.slice(2).join(' ');

        let removed = false;
        editConfig(config => {
            removed = config.trigger_phrases.removePhrase(category, phrase);
        });

        if (jsonOutput) {
            return { removed: removed, category: category, phrase: phrase };
        }
        if (removed) {
            return `Removed '${phrase}' from ${category}`;
        } else {
            return `'${phrase}' not found in ${category}`;
        }

    } else if (action === 'clear') {
        if (args.length < 2) {
            throw new Error("Usage: config phrases clear <category>");
        }

        const category = args[1];

        editConfig(config => {
            // Clear the category by setting it to empty list
            const categoryEnum = config.trigger_phrases._coaxPhraseType(category);
            config.trigger_phrases[categoryEnum] = [];
        });

        if (jsonOutput) {
            return { cleared: category };
        }
        return `Cleared all phrases in ${category}`;

    } else if (action === 'show') {
        // Show specific category
        if (args.length < 2) {
            throw new Error("Usage: config phrases show <category>");
        }

        const category = args[1];
        const config = loadConfig();
        const phrases = config.trigger_phrases.listPhrases(category);

        if (jsonOutput) {
            return { phrases: phrases };
        }
        return formatPhrasesHuman(phrases);

    } else {
        if (fromSlash) {
            return `Unknown trigger command: ${action}\n\n${formatPhrasesHelp()}`;
        }
        throw new Error(`Unknown phrases action: ${action}. Valid actions: list, add, remove, clear, show`);
    }
}

function formatPhrasesHelp() {
    /**Format phrases help for slash command.*/
    const lines = [
        "Trigger Phrase Commands:",
        "",
        "  /sessions config trigger list [category]           - List trigger phrases",
        "  /sessions config trigger add <category> <phrase>   - Add trigger phrase",
        "  /sessions config trigger remove <category> <phrase> - Remove trigger phrase",
        "",
        "Categories:",
        "  go       - implementation_mode triggers (yert, make it so, run that)",
        "  no       - discussion_mode triggers (stop, silence)",
        "  create   - task_creation triggers (mek:, mekdis)",
        "  start    - task_startup triggers (start^, begin task:)",
        "  complete - task_completion triggers (finito)",
        "  compact  - context_compaction triggers (lets compact, squish)"
    ];
    return lines.join('\n');
}

function formatPhrasesHuman(phrases) {
    /**Format phrases for human reading.*/
    const lines = ["Trigger Phrases:"];
    for (const [category, phraseList] of Object.entries(phrases)) {
        if (phraseList && phraseList.length > 0) {
            lines.push(`  ${category}:`);
            for (const phrase of phraseList) {
                lines.push(`    - ${phrase}`);
            }
        } else {
            lines.push(`  ${category}: (none)`);
        }
    }
    return lines.join('\n');
}
//!<

//!> Git preferences handlers
function handleGitCommand(args, jsonOutput = false, fromSlash = false) {
    /**
     * Handle git preference commands.
     *
     * Usage:
     *     config git show
     *     config git set <key> <value>
     */
    if (!args || args.length === 0 || ['help', ''].includes(args[0].toLowerCase())) {
        if (fromSlash) {
            return formatGitHelp();
        }
        // If not from slash and no args, show git preferences
        if (!args || args.length === 0) {
            return handleGitShow(jsonOutput);
        }
    }

    if (args && args[0] === 'show') {
        return handleGitShow(jsonOutput);
    }

    // Handle old 'set' command or direct subcommands
    const action = args[0].toLowerCase();
    let key, value;

    // Map direct subcommands to set operations
    if (['add', 'branch', 'commit', 'merge', 'push', 'repo'].includes(action)) {
        key = action;
        if (args.length < 2) {
            if (fromSlash) {
                return formatGitMissingValue(key);
            }
            throw new Error(`Missing value for git ${key}`);
        }
        value = action === 'branch' ? args.slice(1).join(' ') : args[1];
    } else if (action === 'set') {
        if (args.length < 3) {
            if (fromSlash) {
                return "Missing key and value\nUsage: /sessions config git <setting> <value>";
            }
            throw new Error("Usage: config git set <key> <value>");
        }
        key = args[1].toLowerCase();
        value = args[2];
    } else {
        if (fromSlash) {
            return `Unknown git command: ${action}\n\n${formatGitHelp()}`;
        }
        throw new Error(`Unknown git command: ${args[0]}`);
    }

    editConfig(config => {
        if (['add', 'add_pattern'].includes(key)) {
            // Validate value is valid
            if (!['ask', 'all'].includes(value)) {
                if (fromSlash) {
                    throw new Error(`Invalid value '${value}' for git add\nValid options: ask (prompt for files) or all (stage everything)\n\nUse '/sessions config git help' for more info`);
                }
                throw new Error(`Invalid add pattern: ${value}. Valid options: ask, all`);
            }
            config.git_preferences.add_pattern = value;

        } else if (['branch', 'default_branch'].includes(key)) {
            config.git_preferences.default_branch = value;

        } else if (['commit', 'commit_style'].includes(key)) {
            // Map friendly values
            let style = value;
            if (fromSlash) {
                const styleMap = { 'reg': 'conventional', 'simp': 'simple', 'op': 'detailed' };
                style = styleMap[value] || value;
            }
            // Validate style is valid
            if (!['conventional', 'simple', 'detailed'].includes(style)) {
                if (fromSlash) {
                    throw new Error(`Invalid style '${value}'\nValid styles: conventional, simple, detailed\n  conventional - feat: add feature (conventional commits)\n  simple       - Add feature (simple descriptions)\n  detailed     - Add feature with extended description\n\nUse '/sessions config git help' for more info`);
                }
                throw new Error(`Invalid commit style: ${value}. Valid options: conventional, simple, detailed`);
            }
            config.git_preferences.commit_style = style;

        } else if (['merge', 'auto_merge'].includes(key)) {
            if (fromSlash) {
                if (value === 'auto') {
                    config.git_preferences.auto_merge = true;
                } else if (value === 'ask') {
                    config.git_preferences.auto_merge = false;
                } else {
                    throw new Error(`Invalid value '${value}' for merge\nValid options: auto (merge automatically) or ask (prompt first)\n\nUse '/sessions config git help' for more info`);
                }
            } else {
                config.git_preferences.auto_merge = ['true', 'yes', '1', 'auto'].includes(value.toLowerCase());
            }

        } else if (['push', 'auto_push'].includes(key)) {
            if (fromSlash) {
                if (value === 'auto') {
                    config.git_preferences.auto_push = true;
                } else if (value === 'ask') {
                    config.git_preferences.auto_push = false;
                } else {
                    throw new Error(`Invalid value '${value}' for push\nValid options: auto (push automatically) or ask (prompt first)\n\nUse '/sessions config git help' for more info`);
                }
            } else {
                config.git_preferences.auto_push = ['true', 'yes', '1', 'auto'].includes(value.toLowerCase());
            }

        } else if (['repo', 'has_submodules'].includes(key)) {
            if (fromSlash) {
                if (value === 'super') {
                    config.git_preferences.has_submodules = true;
                } else if (value === 'mono') {
                    config.git_preferences.has_submodules = false;
                } else {
                    throw new Error(`Invalid value '${value}' for repo type\nValid options: super (has submodules) or mono (single repo)\n\nUse '/sessions config git help' for more info`);
                }
            } else {
                config.git_preferences.has_submodules = ['true', 'yes', '1', 'super'].includes(value.toLowerCase());
            }

        } else {
            if (fromSlash) {
                throw new Error(`Unknown git setting: ${key}\n\n${formatGitHelp()}`);
            }
            throw new Error(`Unknown git setting: ${key}`);
        }
    });

    if (jsonOutput) {
        return { updated: key, value: value };
    }
    return `Updated git ${key} to ${value}`;
}

function handleGitShow(jsonOutput = false) {
    /**Show git preferences.*/
    const config = loadConfig();
    const gitPrefs = config.git_preferences;

    // Helper to safely get value from enum or string
    const getValue = (field) => {
        return typeof field === 'object' && field.value !== undefined ? field.value : field;
    };

    if (jsonOutput) {
        return {
            git_preferences: {
                add_pattern: getValue(gitPrefs.add_pattern),
                default_branch: gitPrefs.default_branch,
                commit_style: getValue(gitPrefs.commit_style),
                auto_merge: gitPrefs.auto_merge,
                auto_push: gitPrefs.auto_push,
                has_submodules: gitPrefs.has_submodules,
            }
        };
    }

    const lines = [
        "Git Preferences:",
        `  Add Pattern: ${getValue(gitPrefs.add_pattern)}`,
        `  Default Branch: ${gitPrefs.default_branch}`,
        `  Commit Style: ${getValue(gitPrefs.commit_style)}`,
        `  Auto Merge: ${gitPrefs.auto_merge}`,
        `  Auto Push: ${gitPrefs.auto_push}`,
        `  Has Submodules: ${gitPrefs.has_submodules}`,
    ];
    return lines.join('\n');
}

function formatGitHelp() {
    /**Format git help for slash command.*/
    const lines = [
        "Git Preference Commands:",
        "",
        "  /sessions config git show                - Display git preferences",
        "  /sessions config git add <ask|all>       - Set staging behavior",
        "  /sessions config git branch <name>       - Set default branch",
        "  /sessions config git commit <style>      - Set commit style",
        "    Styles: conventional, simple, detailed",
        "  /sessions config git merge <auto|ask>    - Set merge behavior",
        "  /sessions config git push <auto|ask>     - Set push behavior",
        "  /sessions config git repo <super|mono>   - Set repository type"
    ];
    return lines.join('\n');
}

function formatGitMissingValue(key) {
    /**Format missing value error for git settings.*/
    const messages = {
        'add': "Missing value for git add\nOptions: ask (prompt for files) or all (stage everything)\n\nUsage: /sessions config git add <ask|all>",
        'branch': "Missing branch name\nUsage: /sessions config git branch <name>\n\nExample: /sessions config git branch main",
        'commit': "Missing commit style\nValid styles: conventional, simple, detailed\n\nUsage: /sessions config git commit <style>",
        'merge': "Missing merge preference\nOptions: auto (merge automatically) or ask (prompt first)\n\nUsage: /sessions config git merge <auto|ask>",
        'push': "Missing push preference\nOptions: auto (push automatically) or ask (prompt first)\n\nUsage: /sessions config git push <auto|ask>",
        'repo': "Missing repository type\nOptions: super (has submodules) or mono (single repo)\n\nUsage: /sessions config git repo <super|mono>"
    };
    return messages[key] || `Missing value for git ${key}`;
}
//!<

//!> Environment settings handlers
function handleEnvCommand(args, jsonOutput = false, fromSlash = false) {
    /**
     * Handle environment setting commands.
     *
     * Usage:
     *     config env show
     *     config env set <key> <value>
     */
    if (!args || args.length === 0 || ['help', ''].includes(args[0].toLowerCase())) {
        if (fromSlash) {
            return formatEnvHelp();
        }
        // If not from slash and no args, show env settings
        if (!args || args.length === 0) {
            return handleEnvShow(jsonOutput);
        }
    }

    if (args && args[0] === 'show') {
        return handleEnvShow(jsonOutput);
    }

    // Handle old 'set' command or direct subcommands
    const action = args[0].toLowerCase();
    let key, value;

    // Map direct subcommands to set operations
    if (['os', 'shell', 'name'].includes(action)) {
        key = action;
        if (args.length < 2) {
            if (fromSlash) {
                return formatEnvMissingValue(key);
            }
            throw new Error(`Missing value for env ${key}`);
        }
        value = action === 'name' ? args.slice(1).join(' ') : args[1];
    } else if (action === 'set') {
        if (args.length < 3) {
            if (fromSlash) {
                return "Missing key and value\nUsage: /sessions config env <setting> <value>";
            }
            throw new Error("Usage: config env set <key> <value>");
        }
        key = args[1].toLowerCase();
        value = ['developer_name', 'name'].includes(key) ? args.slice(2).join(' ') : args[2];
    } else {
        if (fromSlash) {
            return `Unknown env command: ${action}\n\n${formatEnvHelp()}`;
        }
        throw new Error(`Unknown env command: ${args[0]}`);
    }

    editConfig(config => {
        if (key === 'os') {
            // Map friendly values
            let osVal = value;
            if (fromSlash) {
                const osMap = { 'mac': 'macos', 'win': 'windows' };
                osVal = osMap[value.toLowerCase()] || value.toLowerCase();
            }
            // Validate OS value is valid
            if (!['linux', 'macos', 'windows'].includes(osVal)) {
                if (fromSlash) {
                    throw new Error(`Invalid OS '${value}'\nValid options: linux, macos, windows\n\nUse '/sessions config env help' for more info`);
                }
                throw new Error(`Invalid os: ${value}. Valid values: linux, macos, windows`);
            }
            config.environment.os = osVal;

        } else if (key === 'shell') {
            // Map friendly values
            let shellVal = value;
            if (fromSlash) {
                const shellMap = { 'pwr': 'powershell' };
                shellVal = shellMap[value.toLowerCase()] || value.toLowerCase();
            }
            // Validate shell value is valid
            if (!['bash', 'zsh', 'fish', 'powershell', 'cmd'].includes(shellVal)) {
                if (fromSlash) {
                    throw new Error(`Invalid shell '${value}'\nValid options: bash, zsh, fish, powershell, cmd\n\nUse '/sessions config env help' for more info`);
                }
                throw new Error(`Invalid shell: ${value}. Valid values: bash, zsh, fish, powershell, cmd`);
            }
            config.environment.shell = shellVal;

        } else if (['developer_name', 'name'].includes(key)) {
            config.environment.developer_name = value;

        } else {
            if (fromSlash) {
                throw new Error(`Unknown env setting: ${key}\n\n${formatEnvHelp()}`);
            }
            throw new Error(`Unknown environment setting: ${key}`);
        }
    });

    if (jsonOutput) {
        return { updated: key, value: value };
    }
    return `Updated env ${key} to ${value}`;
}

function handleEnvShow(jsonOutput = false) {
    /**Show environment settings.*/
    const config = loadConfig();
    const env = config.environment;

    // Helper to safely get value from enum or string
    const getValue = (field) => {
        return typeof field === 'object' && field.value !== undefined ? field.value : field;
    };

    if (jsonOutput) {
        return {
            environment: {
                os: getValue(env.os),
                shell: getValue(env.shell),
                developer_name: env.developer_name,
            }
        };
    }

    const lines = [
        "Environment Settings:",
        `  OS: ${getValue(env.os)}`,
        `  Shell: ${getValue(env.shell)}`,
        `  Developer Name: ${env.developer_name}`,
    ];
    return lines.join('\n');
}

function formatEnvHelp() {
    /**Format env help for slash command.*/
    const lines = [
        "Environment Commands:",
        "",
        "  /sessions config env show            - Display environment settings",
        "  /sessions config env os <os>         - Set operating system",
        "    Options: linux, macos, windows",
        "  /sessions config env shell <shell>   - Set shell preference",
        "    Options: bash, zsh, fish, powershell, cmd",
        "  /sessions config env name <name>     - Set developer name"
    ];
    return lines.join('\n');
}

function formatEnvMissingValue(key) {
    /**Format missing value error for env settings.*/
    const messages = {
        'os': "Missing operating system\nValid options: linux, macos, windows\n\nUsage: /sessions config env os <os>",
        'shell': "Missing shell preference\nValid options: bash, zsh, fish, powershell, cmd\n\nUsage: /sessions config env shell <shell>",
        'name': "Missing developer name\nUsage: /sessions config env name <name>\n\nExample: /sessions config env name John"
    };
    return messages[key] || `Missing value for env ${key}`;
}
//!<

//!> Feature toggles handlers
function handleFeaturesCommand(args, jsonOutput = false, fromSlash = false) {
    /**
     * Handle feature toggle commands.
     *
     * Usage:
     *     config features show
     *     config features set <key> <value>
     *     config features toggle <key>
     */
    // Handle help request and no args
    if (!args || args.length === 0) {
        return handleFeaturesCommand(['show'], jsonOutput, fromSlash);
    }

    if (args[0].toLowerCase() === 'help') {
        return formatFeaturesHelp();
    }

    const action = args[0].toLowerCase();

    if (action === 'show') {
        // Show feature toggles
        const config = loadConfig();
        const features = config.features;

        // Helper to safely get value from enum or string
        const getValue = (field) => {
            return typeof field === 'object' && field.value !== undefined ? field.value : field;
        };

        if (jsonOutput) {
            return {
                features: {
                    branch_enforcement: features.branch_enforcement,
                    task_detection: features.task_detection,
                    auto_ultrathink: features.auto_ultrathink,
                    icon_style: getValue(features.icon_style),
                    warn_85: features.context_warnings.warn_85,
                    warn_90: features.context_warnings.warn_90,
                }
            };
        }

        const lines = [
            "Feature Toggles:",
            `  branch_enforcement: ${features.branch_enforcement}`,
            `  task_detection: ${features.task_detection}`,
            `  auto_ultrathink: ${features.auto_ultrathink}`,
            `  icon_style: ${getValue(features.icon_style)}`,
            `  warn_85: ${features.context_warnings.warn_85}`,
            `  warn_90: ${features.context_warnings.warn_90}`,
        ];
        return lines.join('\n');
    }

    if (action === 'set') {
        if (args.length < 3) {
            throw new Error("Usage: config features set <key> <value>");
        }

        const key = args[1].toLowerCase();
        const value = args[2];
        let finalValue;

        editConfig(config => {
            if (['task_detection', 'auto_ultrathink', 'branch_enforcement'].includes(key)) {
                // Boolean features
                const boolValue = ['true', '1', 'yes', 'on'].includes(value.toLowerCase());
                config.features[key] = boolValue;
                finalValue = boolValue;

            } else if (key === 'icon_style') {
                // Enum feature - accepts nerd_fonts, emoji, ascii
                const lowerValue = value.toLowerCase();
                const validValues = Object.values(IconStyle);
                if (!validValues.includes(lowerValue)) {
                    throw new Error(`Invalid icon_style value: ${value}. Valid values: nerd_fonts, emoji, ascii`);
                }
                config.features.icon_style = lowerValue;
                finalValue = lowerValue;

            } else if (['warn_85', 'warn_90'].includes(key)) {
                // Context warning features
                const boolValue = ['true', '1', 'yes', 'on'].includes(value.toLowerCase());
                config.features.context_warnings[key] = boolValue;
                finalValue = boolValue;

            } else {
                throw new Error(`Unknown feature: ${key}`);
            }
        });

        if (jsonOutput) {
            return { updated: key, value: finalValue };
        }
        return `Updated features.${key} to ${finalValue}`;

    } else if (action === 'toggle') {
        if (args.length < 2) {
            throw new Error("Usage: config features toggle <key>");
        }

        const key = args[1].toLowerCase();

        // Get current value
        const config = loadConfig();
        let currentValue;
        if (['task_detection', 'auto_ultrathink', 'branch_enforcement'].includes(key)) {
            currentValue = config.features[key];
        } else if (key === 'icon_style') {
            currentValue = config.features.icon_style;
        } else if (['warn_85', 'warn_90'].includes(key)) {
            currentValue = config.features.context_warnings[key];
        } else {
            throw new Error(`Unknown feature: ${key}`);
        }

        // Toggle/cycle the value
        let newValue;
        if (key === 'icon_style') {
            // Cycle through enum values: nerd_fonts -> emoji -> ascii -> nerd_fonts
            const cycle = [IconStyle.NERD_FONTS, IconStyle.EMOJI, IconStyle.ASCII];
            const currentIdx = cycle.indexOf(currentValue);
            newValue = cycle[(currentIdx + 1) % cycle.length];
        } else {
            // Boolean toggle
            newValue = !currentValue;
        }

        // Save the toggled value
        editConfig(config => {
            if (['task_detection', 'auto_ultrathink', 'branch_enforcement'].includes(key)) {
                config.features[key] = newValue;
            } else if (key === 'icon_style') {
                config.features.icon_style = newValue;
            } else if (['warn_85', 'warn_90'].includes(key)) {
                config.features.context_warnings[key] = newValue;
            }
        });

        // Format values for display
        const oldDisplay = typeof currentValue === 'object' && currentValue.value !== undefined ? currentValue.value : currentValue;
        const newDisplay = typeof newValue === 'object' && newValue.value !== undefined ? newValue.value : newValue;

        if (jsonOutput) {
            return { toggled: key, old_value: oldDisplay, new_value: newDisplay };
        }
        return `Toggled ${key}: ${oldDisplay} → ${newDisplay}`;

    } else {
        if (fromSlash) {
            return `Unknown features action: ${action}\n\n${formatFeaturesHelp()}`;
        }
        throw new Error(`Unknown features action: ${action}. Valid actions: show, set, toggle`);
    }
}

function formatFeaturesHelp() {
    /**Format features help for slash command.*/
    const lines = [
        "Feature Toggle Commands:",
        "",
        "  /sessions config features show              - Display all feature flags",
        "  /sessions config features set <key> <value> - Set feature value",
        "  /sessions config features toggle <key>      - Toggle feature boolean",
        "",
        "Available Features:",
        "  branch_enforcement  - Git branch validation (default: true)",
        "  task_detection      - Task-based workflow automation (default: true)",
        "  auto_ultrathink     - Enhanced AI reasoning (default: true)",
        "  icon_style          - Statusline icon style: nerd_fonts, emoji, or ascii (default: nerd_fonts)",
        "  warn_85             - Context warning at 85% (default: true)",
        "  warn_90             - Context warning at 90% (default: true)",
        "",
        "Examples:",
        "  /sessions config features toggle icon_style          # Cycles through nerd_fonts -> emoji -> ascii",
        "  /sessions config features set icon_style emoji       # Set to emoji icons",
        "  /sessions config features set auto_ultrathink false",
        "  /sessions config features toggle branch_enforcement"
    ];
    return lines.join('\n');
}
//!<

//!> Readonly commands handlers
function handleReadonlyCommand(args, jsonOutput = false, fromSlash = false) {
    /**
     * Handle custom readonly command management.
     *
     * Usage:
     *     config read list              - List all custom readonly commands
     *     config read add <command>     - Add a command to readonly list
     *     config read remove <command>  - Remove a command from readonly list
     */
    if (!args || args.length === 0 || args[0] === 'list') {
        // List all readonly commands
        const config = loadConfig();
        const commands = config.blocked_actions.listReadonlyCommands();

        if (jsonOutput) {
            return { readonly_commands: commands };
        }

        if (commands.length > 0) {
            const lines = ["Custom Readonly Commands:"];
            for (const cmd of commands) {
                lines.push(`  - ${cmd}`);
            }
            return lines.join('\n');
        } else {
            return "No custom readonly commands configured";
        }
    }

    const action = args[0].toLowerCase();

    if (action === 'add') {
        if (args.length < 2) {
            throw new Error("Usage: config read add <command>");
        }

        const command = args[1];
        let added = false;

        editConfig(config => {
            added = config.blocked_actions.addReadonlyCommand(command);
        });

        if (jsonOutput) {
            return { added: added, command: command };
        }
        if (added) {
            return `Added '${command}' to readonly commands`;
        } else {
            return `'${command}' already exists in readonly commands`;
        }

    } else if (action === 'remove') {
        if (args.length < 2) {
            throw new Error("Usage: config read remove <command>");
        }

        const command = args[1];
        let removed = false;

        editConfig(config => {
            removed = config.blocked_actions.removeReadonlyCommand(command);
        });

        if (jsonOutput) {
            return { removed: removed, command: command };
        }
        if (removed) {
            return `Removed '${command}' from readonly commands`;
        } else {
            return `'${command}' not found in readonly commands`;
        }

    } else {
        throw new Error(`Unknown readonly action: ${action}. Valid actions: list, add, remove`);
    }
}
//!<

//!> Bash read patterns handlers
function handleReadCommand(args, jsonOutput = false, fromSlash = false) {
    /**
     * Handle bash read pattern management.
     *
     * Usage:
     *     config read list              - List all bash read patterns
     *     config read add <pattern>     - Add a pattern to read list
     *     config read remove <pattern>  - Remove a pattern from read list
     */
    if (!args || args.length === 0 || args[0] === 'list') {
        // List all read patterns
        const config = loadConfig();
        const patterns = config.blocked_actions.bash_read_patterns;

        if (jsonOutput) {
            return { bash_read_patterns: patterns };
        }

        if (patterns && patterns.length > 0) {
            const lines = ["Bash Read Patterns (allowed in discussion mode):"];
            for (const pattern of patterns) {
                lines.push(`  - ${pattern}`);
            }
            return lines.join('\n');
        } else {
            return "No custom bash read patterns configured";
        }
    }

    const action = args[0].toLowerCase();

    if (action === 'add') {
        if (args.length < 2) {
            if (fromSlash) {
                return "Missing pattern for add command\nUsage: /sessions config read add <pattern>\n\nExample: /sessions config read add 'docker ps'";
            }
            throw new Error("Usage: config read add <pattern>");
        }

        const pattern = args.slice(1).join(' ');
        let added = false;

        editConfig(config => {
            if (!config.blocked_actions.bash_read_patterns.includes(pattern)) {
                config.blocked_actions.bash_read_patterns.push(pattern);
                added = true;
            }
        });

        if (jsonOutput) {
            return { added: added, pattern: pattern };
        }
        if (added) {
            return `Added '${pattern}' to bash read patterns`;
        } else {
            return `'${pattern}' already exists in bash read patterns`;
        }

    } else if (action === 'remove') {
        if (args.length < 2) {
            if (fromSlash) {
                return "Missing pattern for remove command\nUsage: /sessions config read remove <pattern>";
            }
            throw new Error("Usage: config read remove <pattern>");
        }

        const pattern = args.slice(1).join(' ');
        let removed = false;

        editConfig(config => {
            const index = config.blocked_actions.bash_read_patterns.indexOf(pattern);
            if (index !== -1) {
                config.blocked_actions.bash_read_patterns.splice(index, 1);
                removed = true;
            }
        });

        if (jsonOutput) {
            return { removed: removed, pattern: pattern };
        }
        if (removed) {
            return `Removed '${pattern}' from bash read patterns`;
        } else {
            return `'${pattern}' not found in bash read patterns`;
        }

    } else {
        if (fromSlash) {
            return `Unknown read command: ${action}\n\nValid actions: list, add, remove\n\nUsage:\n  /sessions config read list\n  /sessions config read add <pattern>\n  /sessions config read remove <pattern>`;
        }
        throw new Error(`Unknown read action: ${action}. Valid actions: list, add, remove`);
    }
}
//!<

//!> Bash write patterns handlers
function handleWriteCommand(args, jsonOutput = false, fromSlash = false) {
    /**
     * Handle bash write pattern management.
     *
     * Usage:
     *     config write list              - List all bash write patterns
     *     config write add <pattern>     - Add a pattern to write list
     *     config write remove <pattern>  - Remove a pattern from write list
     */
    if (!args || args.length === 0 || args[0] === 'list') {
        // List all write patterns
        const config = loadConfig();
        const patterns = config.blocked_actions.bash_write_patterns;

        if (jsonOutput) {
            return { bash_write_patterns: patterns };
        }

        if (patterns && patterns.length > 0) {
            const lines = ["Bash Write Patterns (blocked in discussion mode):"];
            for (const pattern of patterns) {
                lines.push(`  - ${pattern}`);
            }
            return lines.join('\n');
        } else {
            return "No custom bash write patterns configured";
        }
    }

    const action = args[0].toLowerCase();

    if (action === 'add') {
        if (args.length < 2) {
            if (fromSlash) {
                return "Missing pattern for add command\nUsage: /sessions config write add <pattern>\n\nExample: /sessions config write add 'rm -rf'";
            }
            throw new Error("Usage: config write add <pattern>");
        }

        const pattern = args.slice(1).join(' ');
        let added = false;

        editConfig(config => {
            if (!config.blocked_actions.bash_write_patterns.includes(pattern)) {
                config.blocked_actions.bash_write_patterns.push(pattern);
                added = true;
            }
        });

        if (jsonOutput) {
            return { added: added, pattern: pattern };
        }
        if (added) {
            return `Added '${pattern}' to bash write patterns`;
        } else {
            return `'${pattern}' already exists in bash write patterns`;
        }

    } else if (action === 'remove') {
        if (args.length < 2) {
            if (fromSlash) {
                return "Missing pattern for remove command\nUsage: /sessions config write remove <pattern>";
            }
            throw new Error("Usage: config write remove <pattern>");
        }

        const pattern = args.slice(1).join(' ');
        let removed = false;

        editConfig(config => {
            const index = config.blocked_actions.bash_write_patterns.indexOf(pattern);
            if (index !== -1) {
                config.blocked_actions.bash_write_patterns.splice(index, 1);
                removed = true;
            }
        });

        if (jsonOutput) {
            return { removed: removed, pattern: pattern };
        }
        if (removed) {
            return `Removed '${pattern}' from bash write patterns`;
        } else {
            return `'${pattern}' not found in bash write patterns`;
        }

    } else {
        if (fromSlash) {
            return `Unknown write command: ${action}\n\nValid actions: list, add, remove\n\nUsage:\n  /sessions config write list\n  /sessions config write add <pattern>\n  /sessions config write remove <pattern>`;
        }
        throw new Error(`Unknown write action: ${action}. Valid actions: list, add, remove`);
    }
}
//!<

//!> Implementation-only tools handlers
function handleToolsCommand(args, jsonOutput = false, fromSlash = false) {
    /**
     * Handle implementation-only tools management.
     *
     * Usage:
     *     config tools list                - List all blocked tools
     *     config tools block <ToolName>    - Block a tool in discussion mode
     *     config tools unblock <ToolName>  - Unblock a tool
     */
    if (!args || args.length === 0 || args[0] === 'list') {
        // List all blocked tools
        const config = loadConfig();
        const tools = config.blocked_actions.implementation_only_tools;

        if (jsonOutput) {
            return { implementation_only_tools: tools };
        }

        if (tools && tools.length > 0) {
            const lines = ["Implementation-Only Tools (blocked in discussion mode):"];
            for (const tool of tools) {
                lines.push(`  - ${tool}`);
            }
            return lines.join('\n');
        } else {
            return "No tools configured as implementation-only";
        }
    }

    const action = args[0].toLowerCase();

    if (action === 'block') {
        if (args.length < 2) {
            if (fromSlash) {
                const validTools = Object.values(CCTools).join(', ');
                return `Missing tool name for block command\nUsage: /sessions config tools block <ToolName>\n\nValid tools: ${validTools}`;
            }
            throw new Error("Usage: config tools block <ToolName>");
        }

        const toolName = args[1];

        // Validate against CCTools enum
        const validToolValues = Object.values(CCTools);
        if (!validToolValues.includes(toolName)) {
            if (fromSlash) {
                return `Invalid tool name: ${toolName}\n\nValid tools: ${validToolValues.join(', ')}`;
            }
            throw new Error(`Invalid tool name: ${toolName}. Valid tools: ${validToolValues.join(', ')}`);
        }

        let added = false;
        editConfig(config => {
            if (!config.blocked_actions.implementation_only_tools.includes(toolName)) {
                config.blocked_actions.implementation_only_tools.push(toolName);
                added = true;
            }
        });

        if (jsonOutput) {
            return { blocked: added, tool: toolName };
        }
        if (added) {
            return `Blocked '${toolName}' in discussion mode`;
        } else {
            return `'${toolName}' is already blocked in discussion mode`;
        }

    } else if (action === 'unblock') {
        if (args.length < 2) {
            if (fromSlash) {
                return "Missing tool name for unblock command\nUsage: /sessions config tools unblock <ToolName>";
            }
            throw new Error("Usage: config tools unblock <ToolName>");
        }

        const toolName = args[1];
        let removed = false;

        editConfig(config => {
            const index = config.blocked_actions.implementation_only_tools.indexOf(toolName);
            if (index !== -1) {
                config.blocked_actions.implementation_only_tools.splice(index, 1);
                removed = true;
            }
        });

        if (jsonOutput) {
            return { unblocked: removed, tool: toolName };
        }
        if (removed) {
            return `Unblocked '${toolName}' (now allowed in discussion mode)`;
        } else {
            return `'${toolName}' was not blocked`;
        }

    } else {
        if (fromSlash) {
            return `Unknown tools command: ${action}\n\nValid actions: list, block, unblock\n\nUsage:\n  /sessions config tools list\n  /sessions config tools block <ToolName>\n  /sessions config tools unblock <ToolName>`;
        }
        throw new Error(`Unknown tools action: ${action}. Valid actions: list, block, unblock`);
    }
}
//!<

//!> Config validation
function validateConfig(jsonOutput = false) {
    /**
     * Validate the current configuration.
     */
    try {
        const config = loadConfig();
        // The load itself validates the structure

        // Additional validation checks
        const issues = [];

        // Check for empty required fields
        if (!config.git_preferences.default_branch) {
            issues.push("Git default_branch is empty");
        }

        if (!config.environment.developer_name) {
            issues.push("Developer name is empty");
        }

        // Check for at least one implementation trigger phrase
        if (!config.trigger_phrases.implementation_mode || config.trigger_phrases.implementation_mode.length === 0) {
            issues.push("No implementation mode trigger phrases defined");
        }

        if (issues.length > 0) {
            if (jsonOutput) {
                return { valid: false, issues: issues };
            }
            return "Configuration issues found:\n" + issues.map(issue => `  - ${issue}`).join('\n');
        }

        if (jsonOutput) {
            return { valid: true };
        }
        return "Configuration is valid";

    } catch (e) {
        if (jsonOutput) {
            return { valid: false, error: e.message };
        }
        return `Configuration error: ${e.message}`;
    }
}
//!<

//-#

// ==== EXPORTS ===== //
module.exports = {
    handleConfigCommand
};
//-#
