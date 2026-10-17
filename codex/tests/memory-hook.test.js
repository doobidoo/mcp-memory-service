'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const crypto = require('node:crypto');
const { spawn } = require('node:child_process');
const { parseTranscript, runHook, request } = require('../memory-hook');

const decision = 'We decided to use SQLite for local storage because it keeps deployment simple and supports the required transactions.';
const learning = 'We discovered that the timeout must cover the entire request, including a server that keeps sending partial responses.';
const message = (role, text, phase) => ({ type: 'response_item', payload: {
    type: 'message', role, phase, content: [{ type: role === 'user' ? 'input_text' : 'output_text', text }]
} });

test('transcript parser reads Codex messages, excludes instructions, tools and reasoning, tolerates partial JSONL', () => {
    const rows = [message('developer', 'secret instruction'),
        message('user', '# AGENTS.md instructions for /workspace\n' + decision),
        message('user', '<environment_context>private environment</environment_context>'), message('user', 'Please fix this.'),
        message('assistant', 'Working on it.', 'commentary'), message('assistant', decision, 'final_answer'),
        { type: 'event_msg', payload: { type: 'agent_message', message: decision } },
        { type: 'response_item', payload: { type: 'reasoning', content: 'private reasoning' } }];
    const result = parseTranscript(rows.map(JSON.stringify).join('\n') + '\n{"partial":');
    assert.deepEqual(result.map(m => m.content), ['Please fix this.', decision]);
});

test('event-only transcripts are supported without capturing tool output', () => {
    const result = parseTranscript([
        { type: 'event_msg', payload: { type: 'user_message', message: 'hello' } },
        { type: 'event_msg', payload: { type: 'agent_message', message: learning } },
        { type: 'event_msg', payload: { type: 'exec_command_end', aggregated_output: 'secret' } }
    ].map(JSON.stringify).join('\n'));
    assert.deepEqual(result.map(m => m.content), ['hello', learning]);
});

async function fixture(t) {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'codex-memory-test-'));
    const requests = [];
    const memories = new Map();
    let rejectStores = false;
    let rejectSearch = false;
    const server = http.createServer(async (req, res) => {
        let body = '';
        for await (const chunk of req) body += chunk;
        requests.push({ url: req.url, headers: req.headers, body: JSON.parse(body || '{}') });
        res.setHeader('Content-Type', 'application/json');
        if (req.method === 'GET' && req.url.startsWith('/api/memories/')) {
            const memory = memories.get(req.url.split('/').pop());
            if (!memory) { res.statusCode = 404; res.end('{}'); return; }
            res.end(JSON.stringify(memory));
        } else if (req.url === '/api/search') {
            if (rejectSearch) { res.statusCode = 503; res.end('{}'); return; }
            res.end(JSON.stringify({ results: [
                { memory: { content: 'unrelated project', tags: ['elsewhere'] } },
                { memory: { content: 'Use transactions for writes.', tags: ['project'] } }
            ] }));
        } else if (rejectStores && typeof rejectStores === 'object') {
            res.end(JSON.stringify(rejectStores));
        } else if (rejectStores === true || requests.length === rejectStores) {
            res.statusCode = 503;
            res.end('{"success":false}');
        } else {
            const payload = requests.at(-1).body;
            const contentHash = crypto.createHash('sha256').update(payload.content.trim().toLowerCase()).digest('hex');
            if (memories.has(contentHash)) {
                // Match the SQLite-backed HTTP API's exact-duplicate response.
                res.end(JSON.stringify({ success: false, message: 'Duplicate content detected (exact match)', content_hash: null }));
            } else {
                memories.set(contentHash, { ...payload, content_hash: contentHash });
                res.end(JSON.stringify({ success: true, content_hash: contentHash }));
            }
        }
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    t.after(async () => { await new Promise(resolve => server.close(resolve)); await fs.rm(dir, { recursive: true }); });
    const config = { endpoint: `http://127.0.0.1:${server.address().port}`, apiKey: 'test-only',
        projectName: 'project', stateDir: path.join(dir, 'state'), minCaptureLength: 40 };
    const transcript = path.join(dir, 'rollout.jsonl');
    const event = { session_id: '../../untrusted-session', cwd: dir, transcript_path: transcript, hook_event_name: 'Stop' };
    const write = rows => fs.writeFile(transcript, rows.map(JSON.stringify).join('\n'));
    return { dir, requests, memories, config, event, write, reject: value => { rejectStores = value; }, failSearch: () => { rejectSearch = true; } };
}

test('startup injects bounded project-scoped context and sends the configured API key', async t => {
    const f = await fixture(t);
    const result = await runHook({ ...f.event, hook_event_name: 'SessionStart', source: 'compact' }, f.config);
    assert.equal(result.hookSpecificOutput.hookEventName, 'SessionStart');
    assert.match(result.hookSpecificOutput.additionalContext, /Use transactions/);
    assert.doesNotMatch(result.hookSpecificOutput.additionalContext, /unrelated project/);
    assert.equal(f.requests[0].headers['x-api-key'], 'test-only');
    assert.equal(f.requests[0].body.n_results, 32);
});

test('capture and incremental summaries survive process restarts without duplicate writes', async t => {
    const f = await fixture(t);
    await f.write([message('user', 'Choose storage.'), message('assistant', decision, 'final_answer')]);
    await runHook(f.event, f.config);
    assert.equal(f.requests.length, 2);
    assert.deepEqual(f.requests.map(r => r.body.memory_type), ['decision', 'session']);
    assert.equal(f.requests[0].body.tags[0], 'project');
    assert.equal(f.requests[0].body.metadata.session_id, f.event.session_id);
    await runHook({ ...f.event, hook_event_name: 'SessionEnd' }, f.config);
    assert.equal(f.requests.length, 2);
    await f.write([message('user', 'Choose storage.'), message('assistant', decision, 'final_answer'),
        message('user', 'What did you find?'), message('assistant', learning, 'final_answer')]);
    await runHook(f.event, f.config);
    assert.equal(f.requests.length, 4);
    assert.doesNotMatch(f.requests[3].body.content, /decided to use SQLite/);
    const files = await fs.readdir(f.config.stateDir);
    assert.ok(files.every(name => /^[a-f0-9]{64}\.json$/.test(name)));
});

test('#skip excludes a whole turn from capture and summaries; #remember forces a short note', async t => {
    const f = await fixture(t);
    await f.write([message('user', '#skip private discussion'), message('assistant', decision, 'final_answer'),
        message('user', '#remember My preferred editor is Vim.'), message('assistant', 'Understood.', 'final_answer')]);
    await runHook(f.event, f.config);
    assert.ok(f.requests.length > 0);
    assert.ok(f.requests.every(r => !r.body.content.includes('SQLite')));
    assert.ok(f.requests.some(r => r.body.content.includes('preferred editor is Vim')));
    assert.ok(f.requests.every(r => !r.body.content.includes('Understood.')));
});

test('failed storage is retried and never marked as captured', async t => {
    const f = await fixture(t);
    await f.write([message('user', 'Choose storage.'), message('assistant', decision, 'final_answer')]);
    f.reject(true);
    await assert.rejects(runHook(f.event, f.config), /HTTP 503/);
    f.reject(false);
    await runHook(f.event, f.config);
    assert.ok(f.requests.some(r => r.body.memory_type === 'session'));
});

test('a failed summary retries without repeating an individual capture that already succeeded', async t => {
    const f = await fixture(t);
    await f.write([message('user', 'Choose storage.'), message('assistant', decision, 'final_answer')]);
    f.reject(2);
    await assert.rejects(runHook(f.event, f.config), /HTTP 503/);
    f.reject(false);
    await runHook(f.event, f.config);
    assert.deepEqual(f.requests.map(r => r.body.memory_type), ['decision', 'session', 'session']);
});

test('exact duplicates from earlier sessions do not block new captures or checkpoint progress', async t => {
    const f = await fixture(t);
    await f.write([message('user', 'Choose storage.'), message('assistant', decision, 'final_answer')]);
    await runHook({ ...f.event, session_id: 'previous-session' }, f.config);
    await f.write([message('user', 'Choose storage.'), message('assistant', decision, 'final_answer'),
        message('user', 'What did you find?'), message('assistant', learning, 'final_answer')]);
    await runHook(f.event, f.config);
    const freshWrites = f.requests.slice(2);
    assert.ok(freshWrites.some(r => r.url.startsWith('/api/memories/')));
    assert.ok(freshWrites.some(r => r.body.memory_type === 'learning'));
    assert.ok(freshWrites.some(r => r.body.memory_type === 'session'));
    const count = f.requests.length;
    await runHook(f.event, f.config);
    assert.equal(f.requests.length, count);
});

test('an exact-duplicate session summary is confirmed and checkpointed', async t => {
    const f = await fixture(t);
    await f.write([message('user', 'Choose storage.'), message('assistant', decision, 'final_answer')]);
    await runHook({ ...f.event, session_id: 'previous-session' }, f.config);
    await runHook(f.event, f.config);
    const count = f.requests.length;
    assert.equal(f.requests.filter(r => r.url.startsWith('/api/memories/')).length, 2);
    await runHook(f.event, f.config);
    assert.equal(f.requests.length, count);
});

test('an unconfirmed exact duplicate remains retryable instead of suppressing an unsaved memory', async t => {
    const f = await fixture(t);
    await f.write([message('user', 'Choose storage.'), message('assistant', decision, 'final_answer')]);
    f.reject({ success: false, message: 'Duplicate content detected (exact match); already exists in store \'docs\'' });
    await assert.rejects(runHook(f.event, f.config), /HTTP 404/);
    const contentHash = crypto.createHash('sha256').update(`Project: project\n\n${decision}`.toLowerCase()).digest('hex');
    f.memories.set(contentHash, { content_hash: contentHash, content: 'A different memory' });
    await assert.rejects(runHook(f.event, f.config), /Memory service rejected request/);
    f.memories.clear();
    f.reject(false);
    await runHook(f.event, f.config);
    assert.equal(f.requests.filter(r => r.body.memory_type === 'decision').length, 3);
    assert.ok(f.requests.some(r => r.body.memory_type === 'session'));
});

test('semantic duplicate and other application rejections remain failures without an exact-match lookup', async t => {
    const f = await fixture(t);
    await f.write([message('user', 'Choose storage.'), message('assistant', decision, 'final_answer')]);
    for (const rejection of ['Storage temporarily unavailable', 'Duplicate content detected (semantically similar to abc)']) {
        f.reject({ success: false, message: rejection });
        await assert.rejects(runHook(f.event, f.config), /Memory service rejected request/);
    }
    assert.equal(f.requests.filter(r => r.url.startsWith('/api/memories/')).length, 0);
    f.reject(false);
    await runHook(f.event, f.config);
    assert.ok(f.requests.some(r => r.body.memory_type === 'session'));
});

test('repeated pending outcomes occupy one summary slot before the eight-item limit', async t => {
    const f = await fixture(t);
    await f.write([message('user', 'Choose storage.'),
        ...Array.from({ length: 9 }, () => message('assistant', decision, 'final_answer')),
        message('assistant', learning, 'final_answer')]);
    await runHook(f.event, f.config);
    const summary = f.requests.find(r => r.body.memory_type === 'session').body.content;
    assert.equal(summary.split(decision).length - 1, 1);
    assert.equal(summary.split(learning).length - 1, 1);
});

test('parallel Stop and PreCompact hooks serialize writes and cached context survives an outage', async t => {
    const f = await fixture(t);
    const start = { ...f.event, hook_event_name: 'SessionStart' };
    const first = await runHook(start, f.config);
    f.failSearch();
    assert.deepEqual(await runHook({ ...start, source: 'compact' }, f.config), first);
    await f.write([message('user', 'Choose storage.'), message('assistant', decision, 'final_answer')]);
    await Promise.all([runHook(f.event, f.config), runHook({ ...f.event, hook_event_name: 'PreCompact' }, f.config)]);
    assert.equal(f.requests.filter(r => r.url === '/api/memories').length, 2);
});

async function lockSession(f, ageMs, directory = false) {
    await runHook({ ...f.event, hook_event_name: 'SessionStart' }, f.config);
    const stateFile = (await fs.readdir(f.config.stateDir)).find(name => name.endsWith('.json'));
    const lockname = path.join(f.config.stateDir, `${stateFile}.lock`);
    if (directory) await fs.mkdir(lockname);
    const ownerPath = directory ? path.join(lockname, 'owner-deadbeef') : lockname;
    await fs.writeFile(ownerPath, 'lock-owner');
    const modified = new Date(Date.now() - ageMs);
    await fs.utimes(ownerPath, modified, modified);
    f.requests.length = 0;
    return lockname;
}

test('SessionStart recovers a stale lock and loads memories in the same callback', async t => {
    const f = await fixture(t);
    const lockname = await lockSession(f, 61000);
    const result = await runHook({ ...f.event, hook_event_name: 'SessionStart', source: 'resume' }, f.config);
    assert.match(result?.hookSpecificOutput?.additionalContext || '', /Use transactions/);
    assert.equal(f.requests.length, 1);
    await assert.rejects(fs.stat(lockname), { code: 'ENOENT' });
});

test('the final SessionEnd recovers a stale lock and checkpoints pending outcomes', async t => {
    const f = await fixture(t);
    const lockname = await lockSession(f, 61000);
    await f.write([message('user', 'Choose storage.'), message('assistant', decision, 'final_answer')]);
    await runHook({ ...f.event, hook_event_name: 'SessionEnd' }, f.config);
    assert.deepEqual(f.requests.map(r => r.body.memory_type), ['decision', 'session']);
    await assert.rejects(fs.stat(lockname), { code: 'ENOENT' });
});

test('a fresh lock is preserved and its competing callback does no work', async t => {
    const f = await fixture(t);
    const lockname = await lockSession(f, 0);
    const result = await runHook({ ...f.event, hook_event_name: 'SessionStart' }, f.config);
    assert.equal(result, null);
    assert.equal(f.requests.length, 0);
    assert.equal(await fs.readFile(lockname, 'utf8'), 'lock-owner');
});

test('two callbacks observing the same stale owner cannot delete the winner\'s replacement lock', async t => {
    const f = await fixture(t);
    const lockname = await lockSession(f, 61000, true);
    const ownerPath = path.join(lockname, 'owner-deadbeef');
    const originalStat = fs.stat;
    let observations = 0, release;
    const bothObserved = new Promise(resolve => { release = resolve; });
    t.mock.method(fs, 'stat', async (...args) => {
        const result = await originalStat(...args);
        if (args[0] === ownerPath) {
            if (++observations === 2) release();
            await bothObserved;
        }
        return result;
    });
    await f.write([message('user', 'Choose storage.'), message('assistant', decision, 'final_answer')]);
    await Promise.all([runHook(f.event, f.config), runHook({ ...f.event, hook_event_name: 'PreCompact' }, f.config)]);
    assert.equal(observations, 2);
    assert.deepEqual(f.requests.map(r => r.body.memory_type), ['decision', 'session']);
    await assert.rejects(originalStat(lockname), { code: 'ENOENT' });
    assert.ok((await fs.readdir(f.config.stateDir)).every(name => name.endsWith('.json')));
});

test('legacy stale-lock recovery cannot unlink a replacement acquired by another callback', async t => {
    const f = await fixture(t);
    const lockname = await lockSession(f, 61000);
    const originalStat = fs.stat, originalLstat = fs.lstat;
    const originalOpen = fs.open, originalRename = fs.rename, originalUnlink = fs.unlink;
    let observations = 0, removals = 0, releaseSnapshots, acquired;
    const bothObserved = new Promise(resolve => { releaseSnapshots = resolve; });
    const winnerAcquired = new Promise(resolve => { acquired = resolve; });
    const observe = original => async (...args) => {
        const result = await original(...args);
        if (args[0] === lockname && result.isFile()) {
            if (++observations === 2) releaseSnapshots();
            await bothObserved;
        }
        return result;
    };
    t.mock.method(fs, 'stat', observe(originalStat));
    t.mock.method(fs, 'lstat', observe(originalLstat));
    t.mock.method(fs, 'open', async (...args) => {
        const result = await originalOpen(...args);
        if (args[0] === lockname && args[1] === 'wx') acquired();
        return result;
    });
    t.mock.method(fs, 'rename', async (...args) => {
        const result = await originalRename(...args);
        if (args[1] === lockname) acquired();
        return result;
    });
    t.mock.method(fs, 'unlink', async (...args) => {
        if (args[0] === lockname && ++removals === 2) await winnerAcquired;
        return originalUnlink(...args);
    });
    await f.write([message('user', 'Choose storage.'), message('assistant', decision, 'final_answer')]);
    await Promise.all([runHook(f.event, f.config), runHook({ ...f.event, hook_event_name: 'PreCompact' }, f.config)]);
    assert.equal(observations, 2);
    assert.deepEqual(f.requests.map(r => r.body.memory_type), ['decision', 'session']);
});

test('absolute request timeout stops a server that trickles response bytes', async t => {
    const server = http.createServer((req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        const timer = setInterval(() => res.write(' '), 5);
        res.on('close', () => clearInterval(timer));
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    t.after(() => new Promise(resolve => server.close(resolve)));
    await assert.rejects(request({ endpoint: `http://127.0.0.1:${server.address().port}`, timeoutMs: 40 },
        '/api/search', {}, Date.now() + 1000), /timed out/);
});

test('plain chatter and fenced code do not create memories', async t => {
    const f = await fixture(t);
    await f.write([message('user', 'hello'), message('assistant', 'Hello!'),
        message('user', 'Here is sample text'), message('assistant', '```text\n' + decision + '\n```', 'final_answer')]);
    await runHook(f.event, f.config);
    assert.equal(f.requests.length, 0);
});

test('CLI errors are advisory, emit no JSON fragments or secrets, and exit successfully', async t => {
    const f = await fixture(t);
    const configPath = path.join(f.dir, 'config.json');
    await fs.writeFile(configPath, JSON.stringify({ ...f.config, endpoint: 'http://127.0.0.1:1' }));
    const child = spawn(process.execPath, [path.resolve(__dirname, '../memory-hook.js')], {
        env: { ...process.env, CODEX_MEMORY_CONFIG: configPath, CODEX_MEMORY_ENDPOINT: '', CODEX_MEMORY_API_KEY: '' }
    });
    let stdout = '', stderr = '';
    child.stdout.on('data', data => { stdout += data; });
    child.stderr.on('data', data => { stderr += data; });
    child.stdin.end(JSON.stringify({ ...f.event, hook_event_name: 'SessionStart' }));
    const code = await new Promise(resolve => child.on('close', resolve));
    assert.equal(code, 0);
    assert.equal(stdout, '');
    assert.match(stderr, /Memory hook unavailable/);
    assert.doesNotMatch(stderr, /test-only/);
});
