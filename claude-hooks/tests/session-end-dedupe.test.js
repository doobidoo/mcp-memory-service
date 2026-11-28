'use strict';

/**
 * Tests for one-summary-per-session in the SessionEnd hook (#2150).
 *
 * Run: node --test claude-hooks/tests/session-end-dedupe.test.js
 *
 * Every SessionEnd re-summarizes the whole transcript, and the server does not
 * dedupe memory_type 'session' (#898), so exiting and resuming one session used
 * to add a near-identical summary each time. The hook now tags the summary with
 * the session id, skips a repeat end whose transcript has not grown, updates
 * the stored summary when it has, and stores as before when the lookup fails.
 *
 * MemoryClient's network methods are replaced by an in-memory fake. Only the
 * onSessionEnd test opens a socket, to a local server on 127.0.0.1 that
 * receives the quality-evaluation requests.
 */

const { test, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const { MemoryClient } = require('../utilities/memory-client');
const { _internal, handler: onSessionEnd } = require('../core/session-end');
const {
    storeSessionMemory,
    decideSessionWrite,
    normalizeSessionId,
    sessionTag,
    sessionEndDeadline,
    lookupTimeoutMs,
} = _internal;

const SESSION_ID = '00893aaf-19fa-41d2-8238-13269b9b3ca0';
const PATCHED = ['connect', 'disconnect', 'storeMemory', 'findMemoriesByTags', 'updateMemory'];

const projectContext = { name: 'widget-repo', language: 'JavaScript', frameworks: [] };

function analysisOfLength(sessionLength) {
    return {
        topics: ['testing'],
        decisions: ['decided to keep one summary per session'],
        insights: [],
        codeChanges: [],
        nextSteps: [],
        sessionLength,
        confidence: 0.2,
    };
}

/**
 * In-memory stand-in for the HTTP API: POST /api/memories, POST
 * /api/search/by-tag (newest first) and PUT /api/memories/{hash} (new hash,
 * old version no longer listed).
 */
function installFakeServer() {
    const state = { memories: [], calls: [], nextId: 1, lookup: null, update: null, protocol: 'http' };
    const originals = Object.fromEntries(PATCHED.map((name) => [name, MemoryClient.prototype[name]]));

    MemoryClient.prototype.connect = async function () {
        this.activeProtocol = state.protocol;
        return { protocol: state.protocol };
    };
    MemoryClient.prototype.disconnect = async function () {
        this.activeProtocol = null;
    };
    MemoryClient.prototype.storeMemory = async function (content, { tags, memoryType, metadata }) {
        state.calls.push({ op: 'store' });
        const memory = { content_hash: `hash-${state.nextId++}`, content, tags, memory_type: memoryType, metadata };
        state.memories.unshift(memory);
        return { success: true, contentHash: memory.content_hash };
    };
    MemoryClient.prototype.findMemoriesByTags = async function (tags, opts) {
        state.calls.push({ op: 'lookup', tags, opts });
        if (state.lookup) return state.lookup();
        const memories = state.memories.filter((m) => tags.every((t) => m.tags.includes(t)));
        return { success: true, memories };
    };
    MemoryClient.prototype.updateMemory = async function (hash, fields, opts) {
        state.calls.push({ op: 'update', hash, opts });
        if (state.update) return state.update();
        const index = state.memories.findIndex((m) => m.content_hash === hash);
        if (index < 0) return { success: false, error: 'HTTP 404' };
        const memory = {
            content_hash: `hash-${state.nextId++}`,
            content: fields.content,
            tags: fields.tags,
            memory_type: fields.memoryType,
            metadata: fields.metadata,
        };
        state.memories.splice(index, 1);
        state.memories.unshift(memory);
        return { success: true, contentHash: memory.content_hash };
    };

    state.restore = () => {
        for (const [name, fn] of Object.entries(originals)) MemoryClient.prototype[name] = fn;
    };
    return state;
}

function ops(state) {
    return state.calls.map((call) => call.op);
}

function store(sessionLength, sessionId = SESSION_ID, deadline = null) {
    return storeSessionMemory(
        'http://127.0.0.1:1', 'key', `summary at ${sessionLength}`,
        projectContext, analysisOfLength(sessionLength), false, { sessionId, deadline }
    );
}

let server;
beforeEach(() => {
    server = installFakeServer();
});
afterEach(() => {
    server.restore();
});

test('first summary of a session is stored with session_id metadata and a session tag', async () => {
    const result = await store(500);

    assert.strictEqual(result.success, true);
    assert.deepStrictEqual(ops(server), ['lookup', 'store']);
    assert.strictEqual(server.memories.length, 1);
    const [memory] = server.memories;
    assert.strictEqual(memory.metadata.session_id, SESSION_ID);
    assert.ok(memory.tags.includes(`sys:claude-session:${SESSION_ID}`), memory.tags.join(', '));
    assert.strictEqual(memory.metadata.session_analysis.session_length, 500);
    assert.deepStrictEqual(server.calls[0].tags, [sessionTag(SESSION_ID)]);
    assert.strictEqual(server.calls[0].opts.timeoutMs, 1000);
    assert.strictEqual(server.calls[0].opts.matchAll, true);
});

test('repeated session ends on an unchanged transcript leave one summary', async () => {
    await store(500);
    const second = await store(500);
    const third = await store(500);

    assert.strictEqual(second.skipped, true);
    assert.strictEqual(third.skipped, true);
    assert.strictEqual(second.contentHash, 'hash-1');
    assert.strictEqual(server.memories.length, 1);
    assert.deepStrictEqual(ops(server), ['lookup', 'store', 'lookup', 'lookup']);
});

test('a grown transcript updates the stored summary instead of adding one', async () => {
    await store(500);
    const result = await store(900);

    assert.strictEqual(result.updated, true);
    assert.strictEqual(result.previousHash, 'hash-1');
    assert.strictEqual(result.contentHash, 'hash-2');
    assert.deepStrictEqual(ops(server), ['lookup', 'store', 'lookup', 'update']);
    assert.strictEqual(server.calls[3].hash, 'hash-1');
    assert.strictEqual(server.memories.length, 1);
    assert.strictEqual(server.memories[0].content, 'summary at 900');
    assert.strictEqual(server.memories[0].metadata.session_analysis.session_length, 900);
    assert.strictEqual(server.memories[0].metadata.session_id, SESSION_ID);

    // The updated summary is the baseline for the next end.
    assert.strictEqual((await store(900)).skipped, true);
});

test('a failed lookup falls back to storing a new summary', async () => {
    await store(500);
    server.lookup = () => ({ success: false, memories: [], error: 'Request timeout' });

    const result = await store(500);

    assert.strictEqual(result.success, true);
    assert.ok(!result.skipped);
    assert.strictEqual(server.memories.length, 2);
    assert.deepStrictEqual(ops(server), ['lookup', 'store', 'lookup', 'store']);
});

test('a lookup that throws falls back to storing a new summary', async () => {
    server.lookup = () => {
        throw new Error('socket hang up');
    };

    const result = await store(500);

    assert.strictEqual(result.success, true);
    assert.strictEqual(server.memories.length, 1);
});

test('a rejected update falls back to storing a new summary', async () => {
    await store(500);
    server.update = () => ({ success: false, rejected: true, error: 'HTTP 500' });

    const result = await store(900);

    assert.strictEqual(result.success, true);
    assert.ok(!result.updated);
    assert.deepStrictEqual(ops(server), ['lookup', 'store', 'lookup', 'update', 'store']);
    assert.strictEqual(server.memories.length, 2);
});

test('an update with an unknown outcome is not followed by a store', async () => {
    // The server may have applied an update whose response never arrived; a
    // store after it would leave two summaries. At worst the stored one is stale.
    await store(500);
    server.update = () => ({ success: false, rejected: false, error: 'Request timeout' });
    const timedOut = await store(900);
    assert.strictEqual(timedOut.success, false);
    assert.strictEqual(timedOut.updateUnconfirmed, true);

    server.update = () => {
        throw new Error('socket hang up');
    };
    const threw = await store(900);
    assert.strictEqual(threw.updateUnconfirmed, true);

    assert.deepStrictEqual(ops(server), ['lookup', 'store', 'lookup', 'update', 'lookup', 'update']);
    assert.strictEqual(server.memories.length, 1);
});

test('an update keeps the server-added source: and agent: tags of the stored summary', async () => {
    await store(500);
    server.memories[0].tags = [...server.memories[0].tags, 'source:build-host', 'agent:reviewer-1', 'stale-topic'];

    await store(900);

    const [memory] = server.memories;
    assert.ok(memory.tags.includes('source:build-host'), memory.tags.join(', '));
    assert.ok(memory.tags.includes('agent:reviewer-1'), memory.tags.join(', '));
    assert.ok(!memory.tags.includes('stale-topic'), 'other tags are regenerated, not carried over');
    assert.ok(memory.tags.includes(sessionTag(SESSION_ID)));
});

test('without a usable session id the summary is stored as before, with no lookup', async () => {
    for (const sessionId of [null, undefined, '', 'unknown', 'has spaces,commas']) {
        await storeSessionMemory(
            'http://127.0.0.1:1', 'key', 'summary', projectContext, analysisOfLength(500), false, { sessionId }
        );
    }
    // Callers that predate the options argument.
    await storeSessionMemory('http://127.0.0.1:1', 'key', 'summary', projectContext, analysisOfLength(500));

    assert.deepStrictEqual([...new Set(ops(server))], ['store']);
    assert.strictEqual(server.memories.length, 6);
    for (const memory of server.memories) {
        assert.ok(!('session_id' in memory.metadata));
        assert.ok(!memory.tags.some((tag) => tag.startsWith('sys:claude-session:')));
    }
});

test('decideSessionWrite ignores summaries of other sessions and compares with the longest', () => {
    const memory = (hash, sessionId, sessionLength) => ({
        content_hash: hash,
        metadata: { session_id: sessionId, session_analysis: { session_length: sessionLength } },
    });

    assert.deepStrictEqual(decideSessionWrite([], SESSION_ID, 10), { action: 'store' });
    // A server that ignored the tag filter must not make us skip or overwrite.
    assert.deepStrictEqual(decideSessionWrite([memory('other', 'another-session', 999)], SESSION_ID, 10), { action: 'store' });

    const existing = [memory('newest', SESSION_ID, 400), memory('longest', SESSION_ID, 600), memory('old', SESSION_ID, 600)];
    assert.strictEqual(decideSessionWrite(existing, SESSION_ID, 600).action, 'skip');
    assert.strictEqual(decideSessionWrite(existing, SESSION_ID, 600).target.content_hash, 'longest');
    assert.strictEqual(decideSessionWrite(existing, SESSION_ID, 601).action, 'update');
    assert.strictEqual(decideSessionWrite(existing, SESSION_ID, 601).target.content_hash, 'longest');

    // A stored summary without session_length is treated as length 0.
    const bare = [{ content_hash: 'bare', metadata: { session_id: SESSION_ID } }];
    assert.strictEqual(decideSessionWrite(bare, SESSION_ID, 1).action, 'update');
});

test('normalizeSessionId accepts UUIDs and rejects placeholders and unsafe values', () => {
    assert.strictEqual(normalizeSessionId(SESSION_ID.toUpperCase()), SESSION_ID);
    assert.strictEqual(normalizeSessionId(`  ${SESSION_ID}  `), SESSION_ID);
    assert.strictEqual(normalizeSessionId('unknown'), null);
    assert.strictEqual(normalizeSessionId(''), null);
    assert.strictEqual(normalizeSessionId(42), null);
    assert.strictEqual(normalizeSessionId('a,b'), null);
    assert.strictEqual(normalizeSessionId('x'.repeat(129)), null);
});

test('the lookup leaves part of the SessionEnd budget for the fallback store', async () => {
    const now = Date.now();
    assert.strictEqual(lookupTimeoutMs(null, now), 1000);
    assert.strictEqual(lookupTimeoutMs(now + 1500, now), 1000);
    assert.strictEqual(lookupTimeoutMs(now + 1200, now), 700);
    assert.ok(lookupTimeoutMs(now + 300, now) < 100);

    await store(500, SESSION_ID, Date.now() + 1200);
    const { timeoutMs } = server.calls[0].opts;
    assert.ok(timeoutMs > 0 && timeoutMs <= 700, `lookup timeout ${timeoutMs}`);

    // Too little time left: no lookup, store as before.
    await store(900, SESSION_ID, Date.now() + 300);
    assert.deepStrictEqual(ops(server), ['lookup', 'store', 'store']);

    // The update gets what is left of the budget minus the store reserve.
    await store(1200, SESSION_ID, Date.now() + 5000);
    const update = server.calls.find((call) => call.op === 'update');
    assert.ok(update.opts.timeoutMs > 3500 && update.opts.timeoutMs <= 4500, `update timeout ${update.opts.timeoutMs}`);
});

test('sessionEndDeadline honours CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS', () => {
    assert.strictEqual(sessionEndDeadline({}, 1000), 2500);
    assert.strictEqual(sessionEndDeadline({ CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS: '5000' }, 1000), 6000);
    assert.strictEqual(sessionEndDeadline({ CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS: 'soon' }, 1000), 2500);
    assert.strictEqual(sessionEndDeadline({ CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS: '0' }, 1000), 2500);
});

test('over an MCP connection the summary is stored as before, with its session id', async () => {
    server.protocol = 'mcp';
    await store(500);
    await store(500);

    assert.deepStrictEqual(ops(server), ['store', 'store']);
    assert.strictEqual(server.memories[0].metadata.session_id, SESSION_ID);
});

/** Local stand-in for POST /api/quality/memories/{hash}/evaluate. */
async function startQualityServer() {
    const hashes = [];
    const waiters = [];
    const httpServer = http.createServer((req, res) => {
        const match = req.url.match(/^\/api\/quality\/memories\/([^/]+)\/evaluate$/);
        if (match) hashes.push(match[1]);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false }));
        req.resume();
        for (const waiter of waiters.splice(0)) waiter();
    });
    await new Promise((resolve) => httpServer.listen(0, '127.0.0.1', resolve));
    const waitFor = async (count) => {
        const deadline = Date.now() + 5000;
        while (hashes.length < count) {
            if (Date.now() > deadline) throw new Error(`expected ${count} quality evaluations, got ${hashes.length}`);
            await new Promise((resolve) => {
                waiters.push(resolve);
                setTimeout(resolve, 100);
            });
        }
    };
    return {
        endpoint: `http://127.0.0.1:${httpServer.address().port}`,
        hashes,
        waitFor,
        close: () => new Promise((resolve) => {
            httpServer.closeAllConnections();
            httpServer.close(resolve);
        }),
    };
}

test('onSessionEnd: second end is skipped without a quality evaluation, a grown one updates and is evaluated', async () => {
    const quality = await startQualityServer();
    const home = fs.mkdtempSync(path.join(os.tmpdir(), 'session-end-dedupe-home-'));
    const workdir = fs.mkdtempSync(path.join(os.tmpdir(), 'session-end-dedupe-proj-'));
    fs.mkdirSync(path.join(home, '.claude', 'hooks'), { recursive: true });
    fs.writeFileSync(path.join(home, '.claude', 'hooks', 'config.json'), JSON.stringify({
        memoryService: { http: { endpoint: quality.endpoint, apiKey: 'key' }, enableSessionConsolidation: true },
        sessionAnalysis: { minSessionLength: 100 },
    }));
    const saved = { HOME: process.env.HOME, USERPROFILE: process.env.USERPROFILE };
    process.env.HOME = home;
    process.env.USERPROFILE = home;

    const messages = [
        { role: 'user', content: 'How should the session end hook avoid storing the same summary twice?' },
        { role: 'assistant', content: 'We decided to look the summary up by its session tag before storing a new one.' },
    ];
    const end = (msgs) => onSessionEnd({
        workingDirectory: workdir,
        sessionId: SESSION_ID,
        deadline: Date.now() + 60000,
        conversation: { messages: msgs },
    });

    try {
        await end(messages);
        await quality.waitFor(1);
        await end(messages);
        await end([...messages, { role: 'user', content: 'Also cover the case where the lookup itself fails.' }]);
        await quality.waitFor(2);
    } finally {
        await quality.close();
        process.env.HOME = saved.HOME;
        if (saved.USERPROFILE === undefined) delete process.env.USERPROFILE;
        else process.env.USERPROFILE = saved.USERPROFILE;
        fs.rmSync(home, { recursive: true, force: true });
        fs.rmSync(workdir, { recursive: true, force: true });
    }

    assert.deepStrictEqual(ops(server), ['lookup', 'store', 'lookup', 'lookup', 'update']);
    assert.strictEqual(server.memories.length, 1);
    assert.strictEqual(server.memories[0].metadata.session_id, SESSION_ID);
    assert.deepStrictEqual(quality.hashes, ['hash-1', 'hash-2']);
});
