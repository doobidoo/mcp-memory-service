'use strict';

/**
 * Tests for MemoryClient.findMemoriesByTags() and updateMemory(), which the
 * SessionEnd hook uses to keep one summary per session (#2150).
 *
 * Run: node --test claude-hooks/tests/memory-client-tag-update.test.js
 *
 * A local http server on 127.0.0.1 stands in for the memory service, as in
 * memory-client-store.test.js.
 */

const { test } = require('node:test');
const assert = require('node:assert');
const http = require('http');
const { MemoryClient } = require('../utilities/memory-client');

async function withServer(handler, fn) {
    const requests = [];
    const server = http.createServer((req, res) => {
        let data = '';
        req.setEncoding('utf8');
        req.on('data', (chunk) => (data += chunk));
        req.on('end', () => {
            const request = { method: req.method, url: req.url, headers: req.headers, body: data ? JSON.parse(data) : null };
            requests.push(request);
            handler(request, res);
        });
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const client = new MemoryClient({
        protocol: 'http',
        http: { endpoint: `http://127.0.0.1:${server.address().port}`, apiKey: 'test-key' },
    });
    client.activeProtocol = 'http';
    try {
        await fn(client, requests);
    } finally {
        server.closeAllConnections();
        await new Promise((resolve) => server.close(resolve));
    }
}

function json(res, status, body) {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(body));
}

test('findMemoriesByTags posts an all-tags search and returns the memories', async () => {
    const memory = { content_hash: 'h1', tags: ['sys:claude-session:abc'], metadata: { session_id: 'abc' } };
    await withServer((req, res) => json(res, 200, { results: [{ memory, similarity_score: 1 }, { memory: null }] }), async (client, requests) => {
        const result = await client.findMemoriesByTags(['sys:claude-session:abc'], { limit: 5, timeoutMs: 1000 });

        assert.deepStrictEqual(result, { success: true, memories: [memory] });
        assert.strictEqual(requests[0].method, 'POST');
        assert.strictEqual(requests[0].url, '/api/search/by-tag');
        assert.deepStrictEqual(requests[0].body, { tags: ['sys:claude-session:abc'], match_all: true, limit: 5 });
        assert.strictEqual(requests[0].headers['x-api-key'], 'test-key');
    });
});

test('findMemoriesByTags reports an HTTP error as a failure, not as "no memories"', async () => {
    await withServer((req, res) => json(res, 500, { detail: 'boom' }), async (client) => {
        const result = await client.findMemoriesByTags(['x']);
        assert.strictEqual(result.success, false);
        assert.strictEqual(result.error, 'HTTP 500: {"detail":"boom"}');
    });
});

test('findMemoriesByTags decodes a multibyte character split across chunks', async () => {
    const memory = { content_hash: 'h1', content: 'Résumé of the session', metadata: {} };
    const bytes = Buffer.from(JSON.stringify({ results: [{ memory }] }), 'utf8');
    const split = bytes.indexOf(Buffer.from('é', 'utf8')) + 1;  // inside the two-byte 'é'
    await withServer((req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.write(bytes.subarray(0, split));
        setTimeout(() => res.end(bytes.subarray(split)), 10);
    }, async (client) => {
        const result = await client.findMemoriesByTags(['x']);
        assert.strictEqual(result.memories[0].content, 'Résumé of the session');
    });
});

test('findMemoriesByTags gives up after timeoutMs when the server sends headers and stalls', async () => {
    await withServer((req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.write('{"results": [');
    }, async (client) => {
        const started = Date.now();
        const result = await client.findMemoriesByTags(['x'], { timeoutMs: 50 });
        assert.deepStrictEqual(result, { success: false, memories: [], error: 'Request timeout' });
        assert.ok(Date.now() - started < 2000);
    });
});

test('findMemoriesByTags gives up after timeoutMs', async () => {
    await withServer(() => { /* never answers */ }, async (client) => {
        const started = Date.now();
        const result = await client.findMemoriesByTags(['x'], { timeoutMs: 50 });
        assert.deepStrictEqual(result, { success: false, memories: [], error: 'Request timeout' });
        assert.ok(Date.now() - started < 2000);
    });
});

test('findMemoriesByTags and updateMemory report failure over MCP without a request', async () => {
    const client = new MemoryClient({ protocol: 'mcp' });
    client.activeProtocol = 'mcp';
    client._requestJson = () => {
        throw new Error('no HTTP request may be sent over MCP');
    };
    assert.strictEqual((await client.findMemoriesByTags(['x'])).success, false);
    assert.strictEqual((await client.updateMemory('h1', { content: 'c' })).success, false);
});

test('an invalid header value is reported as a failure, not thrown', async () => {
    const client = new MemoryClient({ protocol: 'http', http: { endpoint: 'http://127.0.0.1:1', apiKey: 'bad\r\nkey' } });
    client.activeProtocol = 'http';
    const result = await client.findMemoriesByTags(['x'], { timeoutMs: 1000 });
    assert.strictEqual(result.success, false);
    assert.match(result.error, /header/i);
});

test('updateMemory PUTs the new fields and returns the new version hash', async () => {
    await withServer((req, res) => json(res, 200, { success: true, message: 'ok', content_hash: 'h2' }), async (client, requests) => {
        const result = await client.updateMemory('h1', {
            content: 'new summary',
            tags: ['a'],
            memoryType: 'session',
            metadata: { session_id: 'abc' },
        });

        assert.deepStrictEqual(result, { success: true, contentHash: 'h2' });
        assert.strictEqual(requests[0].method, 'PUT');
        assert.strictEqual(requests[0].url, '/api/memories/h1');
        assert.deepStrictEqual(requests[0].body, {
            content: 'new summary',
            tags: ['a'],
            memory_type: 'session',
            metadata: { session_id: 'abc' },
        });
    });
});

test('updateMemory reports a rejected update (success: false in a 200 body)', async () => {
    await withServer((req, res) => json(res, 200, { success: false, message: 'Content exceeds the limit', content_hash: 'h1' }), async (client) => {
        const result = await client.updateMemory('h1', { content: 'x' });
        assert.deepStrictEqual(result, { success: false, rejected: true, error: 'Content exceeds the limit' });
    });
});

test('updateMemory reports 404 for a hash the server no longer has', async () => {
    await withServer((req, res) => json(res, 404, { detail: 'not found' }), async (client) => {
        const result = await client.updateMemory('gone', { content: 'x' });
        assert.deepStrictEqual(result, { success: false, rejected: true, error: 'HTTP 404: {"detail":"not found"}' });
    });
});

test('updateMemory reports a timeout as an unknown outcome, not a rejection', async () => {
    await withServer(() => { /* never answers */ }, async (client) => {
        const result = await client.updateMemory('h1', { content: 'x' }, { timeoutMs: 50 });
        assert.deepStrictEqual(result, { success: false, rejected: false, error: 'Request timeout' });
    });
});
