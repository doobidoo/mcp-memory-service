#!/usr/bin/env node
/**
 * Tests for scripts/ensure-server.js
 * Run: node claude-hooks/tests/ensure-server.test.js
 */
'use strict';

const assert = require('assert');
const http = require('http');
const path = require('path');
const { spawn } = require('child_process');

const SCRIPT = path.join(__dirname, '..', 'scripts', 'ensure-server.js');

function runScript(env = {}) {
    return new Promise((resolve) => {
        const proc = spawn('node', [SCRIPT], {
            env: { ...process.env, ...env },
            stdio: ['ignore', 'pipe', 'pipe'],
        });
        let stdout = '';
        let stderr = '';
        proc.stdout.on('data', (d) => (stdout += d));
        proc.stderr.on('data', (d) => (stderr += d));
        proc.on('close', (code) => resolve({ code, stdout, stderr }));
    });
}

function startMockHealthServer(statusCode = 200) {
    return new Promise((resolve) => {
        const server = http.createServer((req, res) => {
            if (req.url === '/api/health') {
                res.writeHead(statusCode, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ status: 'ok' }));
            } else {
                res.writeHead(404);
                res.end();
            }
        });
        server.listen(0, '127.0.0.1', () => {
            const { port } = server.address();
            resolve({ server, port });
        });
    });
}

function stopServer(server) {
    return new Promise((resolve) => server.close(() => resolve()));
}

async function runTest(name, fn) {
    try {
        await fn();
        console.log(`PASS: ${name}`);
        return true;
    } catch (err) {
        console.error(`FAIL: ${name}`);
        console.error(err);
        return false;
    }
}

async function testHealthyServerExitsZero() {
    const { server, port } = await startMockHealthServer(200);
    try {
        const { code, stderr } = await runScript({
            MCP_MEMORY_ENDPOINT: `http://127.0.0.1:${port}`,
            ENSURE_SERVER_NO_SPAWN: '1',
        });
        assert.strictEqual(code, 0, `expected exit 0, got ${code}. stderr: ${stderr}`);
    } finally {
        await stopServer(server);
    }
}

async function testUnreachableServerExitsZeroWithWarning() {
    const { code, stderr } = await runScript({
        MCP_MEMORY_ENDPOINT: 'http://127.0.0.1:1',
        ENSURE_SERVER_NO_SPAWN: '1',
    });
    assert.strictEqual(code, 0, 'must never block session start');
    assert.ok(
        /unreach|could not|failed/i.test(stderr),
        `expected warning in stderr, got: ${stderr}`,
    );
}

async function run() {
    const results = [];
    results.push(await runTest('testHealthyServerExitsZero', testHealthyServerExitsZero));
    results.push(await runTest('testUnreachableServerExitsZeroWithWarning', testUnreachableServerExitsZeroWithWarning));
    const passed = results.filter(Boolean).length;
    console.log(`${passed}/${results.length} tests passed`);
    if (passed !== results.length) process.exit(1);
}

run().catch((err) => {
    console.error(err);
    process.exit(1);
});
