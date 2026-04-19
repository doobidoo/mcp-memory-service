#!/usr/bin/env node
/**
 * Tests for MemoryClient.storeMemory()
 * Run: node claude-hooks/tests/memory-client-store.test.js
 */
'use strict';

const assert = require('assert');
const http = require('http');
const { MemoryClient } = require('../utilities/memory-client');

function startMockServer(handler) {
    return new Promise((resolve) => {
        const server = http.createServer(handler);
        server.listen(0, '127.0.0.1', () => {
            const { port } = server.address();
            resolve({ server, endpoint: `http://127.0.0.1:${port}` });
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
        console.error(err && err.stack ? err.stack : err);
        return false;
    }
}

async function testHttpStoreSuccess() {
    let receivedBody = null;
    let receivedHeaders = null;
    const { server, endpoint } = await startMockServer((req, res) => {
        receivedHeaders = req.headers;
        let data = '';
        req.on('data', (chunk) => (data += chunk));
        req.on('end', () => {
            receivedBody = JSON.parse(data);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: true, content_hash: 'abc123' }));
        });
    });

    try {
        const client = new MemoryClient({
            protocol: 'http',
            http: { endpoint, apiKey: 'test-key' },
        });
        client.activeProtocol = 'http';

        const result = await client.storeMemory('hello world', {
            tags: ['test', 'unit'],
            memoryType: 'note',
            metadata: { source: 'test' },
        });

        assert.strictEqual(result.success, true, 'should return success');
        assert.strictEqual(result.contentHash, 'abc123', 'should expose contentHash');
        assert.strictEqual(receivedBody.content, 'hello world');
        assert.deepStrictEqual(receivedBody.tags, ['test', 'unit']);
        assert.strictEqual(receivedBody.memory_type, 'note');
        assert.deepStrictEqual(receivedBody.metadata, { source: 'test' });
        assert.strictEqual(receivedHeaders['x-api-key'], 'test-key');
    } finally {
        await stopServer(server);
    }
}

async function testMcpStoreSuccess() {
    const calls = [];
    const fakeMcpClient = {
        storeMemory: async (content, opts) => {
            calls.push({ content, opts });
            return { success: true, content_hash: 'mcp-hash-xyz' };
        },
    };

    const client = new MemoryClient({ protocol: 'mcp' });
    client.activeProtocol = 'mcp';
    client.mcpClient = fakeMcpClient;

    const result = await client.storeMemory('via mcp', {
        tags: ['mcp'],
        memoryType: 'note',
        metadata: {},
    });

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.contentHash, 'mcp-hash-xyz');
    assert.strictEqual(calls.length, 1);
    assert.strictEqual(calls[0].content, 'via mcp');
    assert.deepStrictEqual(calls[0].opts.tags, ['mcp']);
}

async function testStoreNoActiveProtocol() {
    const client = new MemoryClient({ protocol: 'auto' });
    // activeProtocol is null — not connected

    await assert.rejects(
        () => client.storeMemory('x', {}),
        /No active connection available/,
    );
}

async function run() {
    const results = [];
    results.push(await runTest('testHttpStoreSuccess', testHttpStoreSuccess));
    results.push(await runTest('testMcpStoreSuccess', testMcpStoreSuccess));
    results.push(await runTest('testStoreNoActiveProtocol', testStoreNoActiveProtocol));

    const passed = results.filter(Boolean).length;
    const total = results.length;
    console.log(`\n${passed}/${total} tests passed`);
    if (passed !== total) process.exit(1);
}

run().catch((err) => {
    console.error(err);
    process.exit(1);
});
