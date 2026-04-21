#!/usr/bin/env node
/**
 * validate-plugin-schema.js — Shape validator for Claude Code plugin manifests.
 *
 * Goes beyond JSON.parse: enforces the Claude Code plugin spec shape so
 * smoke-test failures surface before release. Added after v10.39.0 shipped
 * with `"author": "doobidoo"` (string) instead of `{"name": "doobidoo"}`
 * (object), which broke every `/plugin install`.
 *
 * Usage:
 *   node claude-hooks/scripts/validate-plugin-schema.js
 *
 * Exit codes:
 *   0 — all manifests valid
 *   1 — at least one manifest fails shape validation
 *
 * No npm dependencies — Node built-ins only.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const PLUGIN_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(PLUGIN_ROOT, '..');

// Known Claude Code v1 hook event names. Unknown names do NOT fail — we
// only fail on clearly invalid values (empty / non-CamelCase). This keeps
// the validator conservative and forward-compatible with new events.
const KNOWN_HOOK_EVENTS = new Set([
    'SessionStart',
    'SessionEnd',
    'UserPromptSubmit',
    'PreToolUse',
    'PostToolUse',
    'Stop',
    'SubagentStop',
    'Notification',
]);

function isPlainObject(v) {
    return v !== null && typeof v === 'object' && !Array.isArray(v);
}

function isNonEmptyString(v) {
    return typeof v === 'string' && v.length > 0;
}

function isCamelCase(s) {
    return typeof s === 'string' && /^[A-Z][A-Za-z0-9]*$/.test(s);
}

function typeName(v) {
    if (v === null) return 'null';
    if (Array.isArray(v)) return 'array';
    return typeof v;
}

// ---------------------------------------------------------------------------
// Per-manifest validators. Each returns {errors: [...], warnings: [...]}.
// `fileLabel` is the short path shown in error messages.
// ---------------------------------------------------------------------------

function validatePluginJson(obj, fileLabel = 'plugin.json') {
    const errors = [];
    const warnings = [];

    if (!isPlainObject(obj)) {
        errors.push(`${fileLabel}: root must be an object (got ${typeName(obj)})`);
        return { errors, warnings };
    }

    if (!isNonEmptyString(obj.name)) {
        errors.push(`${fileLabel}: "name" must be a non-empty string (got ${typeName(obj.name)})`);
    }
    if (!isNonEmptyString(obj.version)) {
        errors.push(`${fileLabel}: "version" must be a non-empty string (got ${typeName(obj.version)})`);
    }

    if ('description' in obj && typeof obj.description !== 'string') {
        errors.push(`${fileLabel}: "description" must be a string (got ${typeName(obj.description)})`);
    }

    if ('author' in obj) {
        // The bug this validator was created to catch: string instead of object.
        if (!isPlainObject(obj.author)) {
            errors.push(
                `${fileLabel}: "author" must be an object with a "name" field ` +
                `(got ${typeName(obj.author)})`,
            );
        } else {
            if (!isNonEmptyString(obj.author.name)) {
                errors.push(
                    `${fileLabel}: "author.name" must be a non-empty string ` +
                    `(got ${typeName(obj.author.name)})`,
                );
            }
            if ('url' in obj.author && typeof obj.author.url !== 'string') {
                errors.push(`${fileLabel}: "author.url" must be a string (got ${typeName(obj.author.url)})`);
            }
            if ('email' in obj.author && typeof obj.author.email !== 'string') {
                errors.push(`${fileLabel}: "author.email" must be a string (got ${typeName(obj.author.email)})`);
            }
        }
    }

    if ('homepage' in obj && typeof obj.homepage !== 'string') {
        errors.push(`${fileLabel}: "homepage" must be a string (got ${typeName(obj.homepage)})`);
    }

    if ('mcpServers' in obj) {
        const v = obj.mcpServers;
        if (typeof v === 'string') {
            // string path — external file, shape is validated when loaded
        } else if (isPlainObject(v)) {
            // inline object — same shape as the inner `mcpServers` value of
            // a .mcp.json file. Reuse validateMcpJson by wrapping.
            const nested = validateMcpJson({ mcpServers: v }, fileLabel);
            errors.push(...nested.errors);
            warnings.push(...nested.warnings);
        } else {
            errors.push(
                `${fileLabel}: "mcpServers" must be a string (path) or object (got ${typeName(v)})`,
            );
        }
    }

    if ('hooks' in obj) {
        const v = obj.hooks;
        if (typeof v === 'string') {
            // string path — external file, shape is validated when loaded
        } else if (isPlainObject(v)) {
            // inline object — same shape as a full hooks.json (which has a
            // top-level `hooks` key). Reuse validateHooksJson by wrapping.
            const nested = validateHooksJson({ hooks: v }, fileLabel);
            errors.push(...nested.errors);
            warnings.push(...nested.warnings);
        } else {
            errors.push(
                `${fileLabel}: "hooks" must be a string (path) or object (got ${typeName(v)})`,
            );
        }
    }

    return { errors, warnings };
}

function validateHooksJson(obj, fileLabel = 'hooks.json') {
    const errors = [];
    const warnings = [];

    if (!isPlainObject(obj)) {
        errors.push(`${fileLabel}: root must be an object (got ${typeName(obj)})`);
        return { errors, warnings };
    }

    if (!isPlainObject(obj.hooks)) {
        errors.push(`${fileLabel}: "hooks" must be an object (got ${typeName(obj.hooks)})`);
        return { errors, warnings };
    }

    for (const [eventName, entries] of Object.entries(obj.hooks)) {
        // Warn on unrecognized/invalid event names — don't hard-fail because
        // Claude Code may add new events before we update this list.
        if (!KNOWN_HOOK_EVENTS.has(eventName)) {
            if (!isCamelCase(eventName)) {
                errors.push(
                    `${fileLabel}: hook event "${eventName}" is not a valid event name ` +
                    `(must be CamelCase, e.g. SessionStart)`,
                );
                continue;
            }
            warnings.push(
                `${fileLabel}: unrecognized hook event "${eventName}" — allowed, but not in the known event list`,
            );
        }

        if (!Array.isArray(entries)) {
            errors.push(
                `${fileLabel}: "hooks.${eventName}" must be an array (got ${typeName(entries)})`,
            );
            continue;
        }

        entries.forEach((entry, i) => {
            const entryPath = `hooks.${eventName}[${i}]`;
            if (!isPlainObject(entry)) {
                errors.push(`${fileLabel}: "${entryPath}" must be an object (got ${typeName(entry)})`);
                return;
            }
            if ('matcher' in entry && typeof entry.matcher !== 'string') {
                errors.push(
                    `${fileLabel}: "${entryPath}.matcher" must be a string (got ${typeName(entry.matcher)})`,
                );
            }
            if (!Array.isArray(entry.hooks)) {
                errors.push(
                    `${fileLabel}: "${entryPath}.hooks" must be an array (got ${typeName(entry.hooks)})`,
                );
                return;
            }
            entry.hooks.forEach((hook, j) => {
                const hookPath = `${entryPath}.hooks[${j}]`;
                if (!isPlainObject(hook)) {
                    errors.push(`${fileLabel}: "${hookPath}" must be an object (got ${typeName(hook)})`);
                    return;
                }
                if (!isNonEmptyString(hook.type)) {
                    errors.push(
                        `${fileLabel}: "${hookPath}.type" must be a non-empty string (got ${typeName(hook.type)})`,
                    );
                }
                if (!isNonEmptyString(hook.command)) {
                    errors.push(
                        `${fileLabel}: "${hookPath}.command" must be a non-empty string (got ${typeName(hook.command)})`,
                    );
                }
            });
        });
    }

    return { errors, warnings };
}

function validateMcpJson(obj, fileLabel = '.mcp.json') {
    const errors = [];
    const warnings = [];

    if (!isPlainObject(obj)) {
        errors.push(`${fileLabel}: root must be an object (got ${typeName(obj)})`);
        return { errors, warnings };
    }

    if (!isPlainObject(obj.mcpServers)) {
        errors.push(
            `${fileLabel}: "mcpServers" must be an object (got ${typeName(obj.mcpServers)})`,
        );
        return { errors, warnings };
    }

    for (const [serverName, server] of Object.entries(obj.mcpServers)) {
        const base = `mcpServers.${serverName}`;
        if (!isPlainObject(server)) {
            errors.push(`${fileLabel}: "${base}" must be an object (got ${typeName(server)})`);
            continue;
        }
        if (!isNonEmptyString(server.command)) {
            errors.push(
                `${fileLabel}: "${base}.command" must be a non-empty string (got ${typeName(server.command)})`,
            );
        }
        if ('args' in server) {
            if (!Array.isArray(server.args)) {
                errors.push(`${fileLabel}: "${base}.args" must be an array (got ${typeName(server.args)})`);
            } else {
                server.args.forEach((a, i) => {
                    if (typeof a !== 'string') {
                        errors.push(
                            `${fileLabel}: "${base}.args[${i}]" must be a string (got ${typeName(a)})`,
                        );
                    }
                });
            }
        }
        if ('env' in server) {
            if (!isPlainObject(server.env)) {
                errors.push(`${fileLabel}: "${base}.env" must be an object (got ${typeName(server.env)})`);
            } else {
                for (const [k, v] of Object.entries(server.env)) {
                    if (typeof v !== 'string') {
                        errors.push(
                            `${fileLabel}: "${base}.env.${k}" must be a string (got ${typeName(v)})`,
                        );
                    }
                }
            }
        }
    }

    return { errors, warnings };
}

function validateMarketplaceJson(obj, fileLabel = 'marketplace.json') {
    const errors = [];
    const warnings = [];

    if (!isPlainObject(obj)) {
        errors.push(`${fileLabel}: root must be an object (got ${typeName(obj)})`);
        return { errors, warnings };
    }

    if (!isNonEmptyString(obj.name)) {
        errors.push(`${fileLabel}: "name" must be a non-empty string (got ${typeName(obj.name)})`);
    }

    // owner: object with required name. Same class of bug as plugin.author.
    if (!isPlainObject(obj.owner)) {
        errors.push(
            `${fileLabel}: "owner" must be an object with a "name" field ` +
            `(got ${typeName(obj.owner)})`,
        );
    } else {
        if (!isNonEmptyString(obj.owner.name)) {
            errors.push(
                `${fileLabel}: "owner.name" must be a non-empty string (got ${typeName(obj.owner.name)})`,
            );
        }
        if ('url' in obj.owner && typeof obj.owner.url !== 'string') {
            errors.push(`${fileLabel}: "owner.url" must be a string (got ${typeName(obj.owner.url)})`);
        }
    }

    if (!Array.isArray(obj.plugins)) {
        errors.push(`${fileLabel}: "plugins" must be an array (got ${typeName(obj.plugins)})`);
    } else {
        obj.plugins.forEach((p, i) => {
            const base = `plugins[${i}]`;
            if (!isPlainObject(p)) {
                errors.push(`${fileLabel}: "${base}" must be an object (got ${typeName(p)})`);
                return;
            }
            if (!isNonEmptyString(p.name)) {
                errors.push(`${fileLabel}: "${base}.name" must be a non-empty string (got ${typeName(p.name)})`);
            }
            if (!isNonEmptyString(p.source)) {
                errors.push(`${fileLabel}: "${base}.source" must be a non-empty string (got ${typeName(p.source)})`);
            }
            if (!isNonEmptyString(p.description)) {
                errors.push(
                    `${fileLabel}: "${base}.description" must be a non-empty string (got ${typeName(p.description)})`,
                );
            }
        });
    }

    return { errors, warnings };
}

// ---------------------------------------------------------------------------
// Orchestrator — reads real files from disk and runs all validators.
// ---------------------------------------------------------------------------

function readJson(filePath) {
    const raw = fs.readFileSync(filePath, 'utf8');
    try {
        return JSON.parse(raw);
    } catch (err) {
        const e = new Error(`Failed to parse ${filePath}: ${err.message}`);
        e.cause = err;
        throw e;
    }
}

function validateAll({ pluginRoot = PLUGIN_ROOT, repoRoot = REPO_ROOT } = {}) {
    const allErrors = [];
    const allWarnings = [];

    const files = [
        {
            filePath: path.join(pluginRoot, '.claude-plugin', 'plugin.json'),
            label: 'claude-hooks/.claude-plugin/plugin.json',
            validator: validatePluginJson,
        },
        {
            filePath: path.join(pluginRoot, '.claude-plugin', 'hooks.json'),
            label: 'claude-hooks/.claude-plugin/hooks.json',
            validator: validateHooksJson,
        },
        {
            filePath: path.join(pluginRoot, '.mcp.json'),
            label: 'claude-hooks/.mcp.json',
            validator: validateMcpJson,
        },
        {
            filePath: path.join(repoRoot, '.claude-plugin', 'marketplace.json'),
            label: '.claude-plugin/marketplace.json',
            validator: validateMarketplaceJson,
        },
    ];

    for (const { filePath, label, validator } of files) {
        let parsed;
        try {
            parsed = readJson(filePath);
        } catch (err) {
            const msg = err.code === 'ENOENT' ? 'file not found' : err.message;
            allErrors.push(`${label}: ${msg}`);
            continue;
        }
        const { errors, warnings } = validator(parsed, label);
        allErrors.push(...errors);
        allWarnings.push(...warnings);
    }

    return {
        ok: allErrors.length === 0,
        errors: allErrors,
        warnings: allWarnings,
    };
}

module.exports = {
    validateAll,
    validatePluginJson,
    validateHooksJson,
    validateMcpJson,
    validateMarketplaceJson,
};

// ---------------------------------------------------------------------------
// CLI entry — only runs when invoked directly.
// ---------------------------------------------------------------------------

if (require.main === module) {
    const result = validateAll();

    for (const w of result.warnings) {
        console.warn(`WARN: ${w}`);
    }

    if (!result.ok) {
        console.error('FAIL: plugin manifest schema validation failed:');
        for (const e of result.errors) {
            console.error(`  - ${e}`);
        }
        process.exit(1);
    }

    console.log('OK: all plugin manifests pass schema validation');
}
