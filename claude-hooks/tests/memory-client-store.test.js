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

    server.close();
    console.log('  ✓ testHttpStoreSuccess');
}

async function run() {
    await testHttpStoreSuccess();
    console.log('All tests passed.');
}

run().catch((err) => {
    console.error(err);
    process.exit(1);
});
