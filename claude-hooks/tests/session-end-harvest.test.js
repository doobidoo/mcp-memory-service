#!/usr/bin/env node
/**
 * Tests for claude-hooks/core/session-end-harvest.js
 *
 * Uses Node's built-in assert and http modules (no external deps).
 * Run: node claude-hooks/tests/session-end-harvest.test.js
 */

'use strict';

const assert = require('assert');
const http = require('http');
const fs = require('fs');
const fsp = require('fs').promises;
const path = require('path');
const os = require('os');

// Each test gets a fresh isolated HOME so the first-run flag is controlled.
function makeTempHome() {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'harvest-hook-'));
    return tmp;
}

function cleanupDir(dir) {
    try { fs.rmSync(dir, { recursive: true, force: true }); } catch (_) {}
}

// Run a mock HTTP server that records requests. Returns {server, url, requests}.
function startMockServer(handler) {
    return new Promise((resolve) => {
        const requests = [];
        const server = http.createServer((req, res) => {
            let body = '';
            req.on('data', (chunk) => { body += chunk; });
            req.on('end', () => {
                let parsed = null;
                try { parsed = JSON.parse(body); } catch (_) { parsed = null; }
                const rec = {
                    method: req.method,
                    url: req.url,
                    headers: req.headers,
                    body: parsed,
                    raw: body
                };
                requests.push(rec);
                handler(rec, res);
            });
        });
        server.listen(0, '127.0.0.1', () => {
            const { port } = server.address();
            resolve({ server, url: `http://127.0.0.1:${port}`, requests });
        });
    });
}

function stopServer(server) {
    return new Promise((resolve) => server.close(() => resolve()));
}

// Load the hook module fresh (important because config is read at call time,
// but we also stub loadConfig per-test).
function freshHook() {
    const p = require.resolve('../core/session-end-harvest.js');
    delete require.cache[p];
    return require('../core/session-end-harvest.js');
}

function buildContext({ numMessages = 20, cwd = '/Users/test/project-foo' } = {}) {
    const messages = [];
    for (let i = 0; i < numMessages; i++) {
        messages.push({ role: i % 2 === 0 ? 'user' : 'assistant', content: `msg ${i}` });
    }
    return { workingDirectory: cwd, conversation: { messages } };
}

// Override the loadConfig function on a freshly-required module.
function withConfig(hookModule, configObj) {
    hookModule._internal.loadConfig = async () => configObj;
    return hookModule;
}

async function runTest(name, fn) {
    const tmpHome = makeTempHome();
    const origHome = process.env.HOME;
    process.env.HOME = tmpHome;
    const origLog = console.log;
    const origWarn = console.warn;
    const logs = [];
    console.log = (...a) => logs.push(['log', a.join(' ')]);
    console.warn = (...a) => logs.push(['warn', a.join(' ')]);
    try {
        await fn({ tmpHome, logs });
        console.log = origLog;
        console.warn = origWarn;
        console.log(`PASS: ${name}`);
        return true;
    } catch (err) {
        console.log = origLog;
        console.warn = origWarn;
        console.error(`FAIL: ${name}`);
        console.error(err && err.stack ? err.stack : err);
        if (logs.length) {
            console.error('--- captured logs ---');
            for (const [lvl, msg] of logs) console.error(`[${lvl}] ${msg}`);
        }
        return false;
    } finally {
        process.env.HOME = origHome;
        cleanupDir(tmpHome);
    }
}

async function main() {
    const results = [];

    // 1. disabled_by_default
    results.push(await runTest('disabled_by_default', async () => {
        const { server, url, requests } = await startMockServer((_req, res) => {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ dry_run: true, results: [] }));
        });
        try {
            const hook = withConfig(freshHook(), { sessionHarvest: { enabled: false, endpoint: url } });
            await hook(buildContext());
            assert.strictEqual(requests.length, 0, 'no HTTP call when disabled');
        } finally {
            await stopServer(server);
        }
    }));

    // 2. skips_short_sessions
    results.push(await runTest('skips_short_sessions', async () => {
        const { server, url, requests } = await startMockServer((_req, res) => {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ dry_run: true, results: [] }));
        });
        try {
            const hook = withConfig(freshHook(), {
                sessionHarvest: { enabled: true, endpoint: url, minSessionMessages: 10, dryRunOnFirstUse: false }
            });
            await hook(buildContext({ numMessages: 3 }));
            assert.strictEqual(requests.length, 0, 'no HTTP call for short sessions');
        } finally {
            await stopServer(server);
        }
    }));

    // 3. first_run_forces_dry_run
    results.push(await runTest('first_run_forces_dry_run', async ({ tmpHome }) => {
        const { server, url, requests } = await startMockServer((_req, res) => {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ dry_run: true, results: [{ session_id: 's', total_messages: 10, found: 2, stored: 0, by_type: {}, candidates: [] }] }));
        });
        try {
            const hook = withConfig(freshHook(), {
                sessionHarvest: {
                    enabled: true,
                    endpoint: url,
                    dryRun: false,            // config says no dry-run
                    dryRunOnFirstUse: true,   // but first-run forces it
                    minSessionMessages: 5
                }
            });
            await hook(buildContext({ numMessages: 12 }));
            assert.strictEqual(requests.length, 1, 'HTTP call made');
            assert.strictEqual(requests[0].body.dry_run, true, 'first run must force dry_run=true');
            // Flag file should now exist
            const flagPath = path.join(tmpHome, '.claude', 'mcp-memory-harvest-first-run.done');
            assert.ok(fs.existsSync(flagPath), 'first-run flag file written');
        } finally {
            await stopServer(server);
        }
    }));

    // 4. subsequent_run_honors_config
    results.push(await runTest('subsequent_run_honors_config', async ({ tmpHome }) => {
        // Pre-create flag file
        const claudeDir = path.join(tmpHome, '.claude');
        fs.mkdirSync(claudeDir, { recursive: true });
        fs.writeFileSync(path.join(claudeDir, 'mcp-memory-harvest-first-run.done'), 'done\n');

        const { server, url, requests } = await startMockServer((_req, res) => {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ dry_run: false, results: [{ session_id: 's', total_messages: 10, found: 3, stored: 3, by_type: {}, candidates: [] }] }));
        });
        try {
            const hook = withConfig(freshHook(), {
                sessionHarvest: {
                    enabled: true,
                    endpoint: url,
                    dryRun: false,
                    dryRunOnFirstUse: true,
                    minSessionMessages: 5
                }
            });
            await hook(buildContext({ numMessages: 12 }));
            assert.strictEqual(requests.length, 1);
            assert.strictEqual(requests[0].body.dry_run, false, 'subsequent run honors dryRun=false');
        } finally {
            await stopServer(server);
        }
    }));

    // 5. timeout_is_non_fatal
    results.push(await runTest('timeout_is_non_fatal', async ({ tmpHome }) => {
        // Server that never responds within timeout.
        const server = http.createServer((_req, _res) => {
            // intentionally hang
        });
        await new Promise((r) => server.listen(0, '127.0.0.1', r));
        const { port } = server.address();
        const url = `http://127.0.0.1:${port}`;
        try {
            // Pre-create flag to skip first-run dry-run path (irrelevant here).
            fs.mkdirSync(path.join(tmpHome, '.claude'), { recursive: true });
            fs.writeFileSync(path.join(tmpHome, '.claude', 'mcp-memory-harvest-first-run.done'), 'x');

            const hook = withConfig(freshHook(), {
                sessionHarvest: { enabled: true, endpoint: url, minSessionMessages: 5, timeoutMs: 200 }
            });
            const start = Date.now();
            await hook(buildContext({ numMessages: 12 }));
            const elapsed = Date.now() - start;
            assert.ok(elapsed < 3000, `must resolve quickly after timeout (elapsed=${elapsed}ms)`);
        } finally {
            await new Promise((r) => server.close(r));
        }
    }));

    // 6. http_failure_is_non_fatal
    results.push(await runTest('http_failure_is_non_fatal', async ({ tmpHome }) => {
        const { server, url } = await startMockServer((_req, res) => {
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ detail: 'boom' }));
        });
        try {
            fs.mkdirSync(path.join(tmpHome, '.claude'), { recursive: true });
            fs.writeFileSync(path.join(tmpHome, '.claude', 'mcp-memory-harvest-first-run.done'), 'x');

            const hook = withConfig(freshHook(), {
                sessionHarvest: { enabled: true, endpoint: url, minSessionMessages: 5 }
            });
            // Should resolve without throwing.
            await hook(buildContext({ numMessages: 12 }));
        } finally {
            await stopServer(server);
        }
    }));

    // 7. api_key_precedence
    results.push(await runTest('api_key_precedence', async ({ tmpHome }) => {
        const { server, url, requests } = await startMockServer((_req, res) => {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ dry_run: true, results: [] }));
        });
        try {
            // Pre-flag to avoid first-run noise
            fs.mkdirSync(path.join(tmpHome, '.claude'), { recursive: true });
            fs.writeFileSync(path.join(tmpHome, '.claude', 'mcp-memory-harvest-first-run.done'), 'x');

            // Case A: context.apiKey wins
            process.env.MCP_API_KEY = 'env-key';
            let hook = withConfig(freshHook(), {
                sessionHarvest: { enabled: true, endpoint: url, minSessionMessages: 5, apiKey: 'config-key' }
            });
            const ctxA = buildContext({ numMessages: 12 });
            ctxA.apiKey = 'context-key';
            await hook(ctxA);
            assert.strictEqual(requests[0].headers['authorization'], 'Bearer context-key');

            // Case B: config wins when no context.apiKey
            hook = withConfig(freshHook(), {
                sessionHarvest: { enabled: true, endpoint: url, minSessionMessages: 5, apiKey: 'config-key' }
            });
            await hook(buildContext({ numMessages: 12 }));
            assert.strictEqual(requests[1].headers['authorization'], 'Bearer config-key');

            // Case C: env wins when neither context nor config
            hook = withConfig(freshHook(), {
                sessionHarvest: { enabled: true, endpoint: url, minSessionMessages: 5 }
            });
            await hook(buildContext({ numMessages: 12 }));
            assert.strictEqual(requests[2].headers['authorization'], 'Bearer env-key');

            delete process.env.MCP_API_KEY;
        } finally {
            await stopServer(server);
            delete process.env.MCP_API_KEY;
        }
    }));

    results.push(await runTest('tls_validation_default_and_opt_in', async () => {
        // The default for allowSelfSignedCerts must be undefined/false — we
        // verify by calling postHarvest directly. https.request receives the
        // options object; we intercept it by substituting a fake request fn.
        const fresh = freshHook();
        const { postHarvest } = fresh._internal;

        const capturedByRun = [];
        const originalHttps = require('https').request;

        // Monkey-patch https.request to record the options it was handed.
        require('https').request = function (opts, cb) {
            capturedByRun.push(opts);
            const fakeReq = {
                on: () => fakeReq,
                write: () => {},
                end: () => {
                    // Simulate an immediate non-network error so postHarvest resolves fast.
                    setImmediate(() => fakeReq.listeners.error && fakeReq.listeners.error({ message: 'mocked' }));
                }
            };
            fakeReq.listeners = {};
            const on = fakeReq.on;
            fakeReq.on = (event, handler) => { fakeReq.listeners[event] = handler; return fakeReq; };
            return fakeReq;
        };

        try {
            // Case A: allowSelfSignedCerts omitted → TLS validation stays enabled.
            await postHarvest('https://example.invalid', null, { sessions: 1 }, 500, {});
            const optA = capturedByRun[0];
            assert.strictEqual(optA.rejectUnauthorized, undefined,
                'TLS validation must stay enabled by default');

            // Case B: explicit allowSelfSignedCerts=true → rejectUnauthorized=false.
            await postHarvest('https://example.invalid', null, { sessions: 1 }, 500,
                { allowSelfSignedCerts: true });
            const optB = capturedByRun[1];
            assert.strictEqual(optB.rejectUnauthorized, false,
                'allowSelfSignedCerts=true must disable TLS validation');
        } finally {
            require('https').request = originalHttps;
        }
    }));

    results.push(await runTest('count_transcript_messages', async () => {
        const { countTranscriptMessages } = freshHook()._internal;
        const fs = require('fs');
        const os = require('os');
        const path = require('path');
        const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'harvest-test-'));
        const transcript = path.join(tmp, 'session.jsonl');
        fs.writeFileSync(transcript, [
            JSON.stringify({ type: 'user', message: { content: 'hi' } }),
            JSON.stringify({ type: 'assistant', message: { content: 'hello' } }),
            '',  // empty line — must be skipped
            'not json',  // malformed — must be skipped
            JSON.stringify({ type: 'user', message: { content: 'third' } })
        ].join('\n'));
        const count = await countTranscriptMessages(transcript);
        assert.strictEqual(count, 3, 'must count valid JSONL lines only');

        // Missing file is non-fatal → returns 0.
        const missing = await countTranscriptMessages(path.join(tmp, 'does-not-exist.jsonl'));
        assert.strictEqual(missing, 0, 'missing file must return 0 (non-fatal)');

        fs.rmSync(tmp, { recursive: true, force: true });
    }));

    const passed = results.filter(Boolean).length;
    const total = results.length;
    console.log(`\n${passed}/${total} tests passed`);
    if (passed !== total) process.exit(1);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
