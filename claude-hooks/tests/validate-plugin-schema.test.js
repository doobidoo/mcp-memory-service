#!/usr/bin/env node
/**
 * Tests for scripts/validate-plugin-schema.js
 * Run: node claude-hooks/tests/validate-plugin-schema.test.js
 */
'use strict';

const assert = require('assert');
const {
    validateAll,
    validatePluginJson,
    validateHooksJson,
    validateMcpJson,
    validateMarketplaceJson,
} = require('../scripts/validate-plugin-schema');

async function runTest(name, fn) {
    try {
        await fn();
        console.log(`PASS: ${name}`);
        return true;
    } catch (err) {
        console.error(`FAIL: ${name}`);
        console.error(err && err.stack ? err.stack : err);
        return false;
    }
}

function assertHasError(result, substring) {
    assert.strictEqual(result.errors.length > 0, true,
        `expected at least one error, got none. warnings: ${JSON.stringify(result.warnings)}`);
    const found = result.errors.some((e) => e.includes(substring));
    assert.strictEqual(found, true,
        `expected error containing "${substring}", got errors:\n  ${result.errors.join('\n  ')}`);
}

// --- Real-manifest smoke: current repo state must pass ---------------------

async function testCurrentPluginsPass() {
    const result = validateAll();
    assert.strictEqual(
        result.ok,
        true,
        `expected current manifests to validate, got errors:\n  ${result.errors.join('\n  ')}`,
    );
}

// --- plugin.json ------------------------------------------------------------

async function testRejectsStringAuthor() {
    // The exact v10.39.0 bug: author as string instead of object.
    const result = validatePluginJson({
        name: 'my-plugin',
        version: '1.0.0',
        author: 'doobidoo',
    });
    assertHasError(result, '"author" must be an object with a "name" field');
}

async function testAcceptsObjectAuthor() {
    const result = validatePluginJson({
        name: 'my-plugin',
        version: '1.0.0',
        author: { name: 'doobidoo' },
    });
    assert.deepStrictEqual(result.errors, [], `unexpected errors: ${result.errors.join(', ')}`);
}

async function testRejectsAuthorObjectMissingName() {
    const result = validatePluginJson({
        name: 'my-plugin',
        version: '1.0.0',
        author: { url: 'https://example.com' },
    });
    assertHasError(result, '"author.name" must be a non-empty string');
}

async function testRejectsMissingNameOrVersion() {
    const r1 = validatePluginJson({ version: '1.0.0' });
    assertHasError(r1, '"name" must be a non-empty string');
    const r2 = validatePluginJson({ name: 'x' });
    assertHasError(r2, '"version" must be a non-empty string');
}

async function testRejectsInlineMcpServersWithBadShape() {
    // Inline mcpServers in plugin.json should be validated recursively —
    // missing "command" must be caught just like it is in .mcp.json.
    const result = validatePluginJson({
        name: 'x',
        version: '1.0.0',
        mcpServers: {
            myServer: { args: ['foo'] }, // missing required "command"
        },
    });
    assertHasError(result, 'command');
}

async function testRejectsInlineHooksWithBadShape() {
    // Inline hooks in plugin.json should be validated recursively —
    // missing "type" must be caught just like it is in hooks.json.
    const result = validatePluginJson({
        name: 'x',
        version: '1.0.0',
        hooks: {
            SessionStart: [
                {
                    hooks: [
                        { command: 'node foo' }, // missing required "type"
                    ],
                },
            ],
        },
    });
    assertHasError(result, 'type');
}

// --- .mcp.json --------------------------------------------------------------

async function testRejectsMissingMcpServers() {
    const result = validateMcpJson({});
    assertHasError(result, '"mcpServers" must be an object');
}

async function testRejectsMcpServerWithoutCommand() {
    const result = validateMcpJson({ mcpServers: { memory: { args: ['-m'] } } });
    assertHasError(result, '"mcpServers.memory.command" must be a non-empty string');
}

async function testRejectsMcpArgsNotArray() {
    const result = validateMcpJson({
        mcpServers: { memory: { command: 'python', args: 'not-an-array' } },
    });
    assertHasError(result, '"mcpServers.memory.args" must be an array');
}

// --- hooks.json -------------------------------------------------------------

async function testRejectsHookCommandWithoutType() {
    const result = validateHooksJson({
        hooks: {
            SessionStart: [
                { hooks: [{ command: 'echo hi' }] },
            ],
        },
    });
    assertHasError(result, '.type" must be a non-empty string');
}

async function testRejectsHooksAsArrayAtRoot() {
    const result = validateHooksJson({ hooks: [] });
    assertHasError(result, '"hooks" must be an object');
}

async function testUnknownEventJustWarns() {
    const result = validateHooksJson({
        hooks: {
            // Not in KNOWN_HOOK_EVENTS but CamelCase → warning, no error.
            FutureEvent: [{ hooks: [{ type: 'command', command: 'echo' }] }],
        },
    });
    assert.deepStrictEqual(result.errors, [], `unexpected errors: ${result.errors.join(', ')}`);
    assert.strictEqual(
        result.warnings.some((w) => w.includes('FutureEvent')),
        true,
        `expected warning about FutureEvent, got: ${result.warnings.join(', ')}`,
    );
}

async function testRejectsInvalidEventName() {
    const result = validateHooksJson({
        hooks: {
            'not-camel-case': [{ hooks: [{ type: 'command', command: 'echo' }] }],
        },
    });
    assertHasError(result, 'not-camel-case');
}

// --- marketplace.json -------------------------------------------------------

async function testRejectsMarketplaceOwnerAsString() {
    const result = validateMarketplaceJson({
        name: 'mkt',
        owner: 'doobidoo',
        plugins: [],
    });
    assertHasError(result, '"owner" must be an object with a "name" field');
}

async function testRejectsMarketplacePluginMissingFields() {
    const result = validateMarketplaceJson({
        name: 'mkt',
        owner: { name: 'doobidoo' },
        plugins: [{ name: 'x' }], // missing source + description
    });
    assertHasError(result, '"plugins[0].source"');
    assertHasError(result, '"plugins[0].description"');
}

// --- runner -----------------------------------------------------------------

async function run() {
    const results = [];
    results.push(await runTest('testCurrentPluginsPass', testCurrentPluginsPass));
    results.push(await runTest('testRejectsStringAuthor', testRejectsStringAuthor));
    results.push(await runTest('testAcceptsObjectAuthor', testAcceptsObjectAuthor));
    results.push(await runTest('testRejectsAuthorObjectMissingName', testRejectsAuthorObjectMissingName));
    results.push(await runTest('testRejectsMissingNameOrVersion', testRejectsMissingNameOrVersion));
    results.push(await runTest('testRejectsInlineMcpServersWithBadShape', testRejectsInlineMcpServersWithBadShape));
    results.push(await runTest('testRejectsInlineHooksWithBadShape', testRejectsInlineHooksWithBadShape));
    results.push(await runTest('testRejectsMissingMcpServers', testRejectsMissingMcpServers));
    results.push(await runTest('testRejectsMcpServerWithoutCommand', testRejectsMcpServerWithoutCommand));
    results.push(await runTest('testRejectsMcpArgsNotArray', testRejectsMcpArgsNotArray));
    results.push(await runTest('testRejectsHookCommandWithoutType', testRejectsHookCommandWithoutType));
    results.push(await runTest('testRejectsHooksAsArrayAtRoot', testRejectsHooksAsArrayAtRoot));
    results.push(await runTest('testUnknownEventJustWarns', testUnknownEventJustWarns));
    results.push(await runTest('testRejectsInvalidEventName', testRejectsInvalidEventName));
    results.push(await runTest('testRejectsMarketplaceOwnerAsString', testRejectsMarketplaceOwnerAsString));
    results.push(await runTest('testRejectsMarketplacePluginMissingFields', testRejectsMarketplacePluginMissingFields));

    const passed = results.filter(Boolean).length;
    const total = results.length;
    console.log(`\n${passed}/${total} tests passed`);
    if (passed !== total) process.exit(1);
}

run().catch((err) => {
    console.error(err);
    process.exit(1);
});
