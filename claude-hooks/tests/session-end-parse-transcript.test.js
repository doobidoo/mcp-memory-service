/**
 * session-end parseTranscript must not feed harness-injected text into the
 * conversation used for session summaries. Fixtures are synthetic.
 * Run: node --test claude-hooks/tests/session-end-parse-transcript.test.js
 */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');

const { _internal } = require('../core/session-end');
const { parseTranscript } = _internal;

const REMINDER = '<system-reminder>\nSynthetic harness reminder.\n</system-reminder>';

async function parse(entries) {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'se-parse-'));
    const file = path.join(dir, 'transcript.jsonl');
    await fs.writeFile(file, entries.map((o) => JSON.stringify(o)).join('\n') + '\n', 'utf8');
    try {
        return await parseTranscript(file);
    } finally {
        await fs.rm(dir, { recursive: true, force: true });
    }
}

const user = (content, extra = {}) => ({ type: 'user', ...extra, message: { role: 'user', content } });
const assistant = (text) => ({ type: 'assistant', message: { role: 'assistant', content: [{ type: 'text', text }] } });

test('drops isMeta, reminder-only, task-notification and peer user entries', async () => {
    const { messages } = await parse([
        user('real prompt', { origin: { kind: 'human' } }),
        user('meta text', { isMeta: true }),
        user(REMINDER),
        user('<task-notification><task-id>x</task-id></task-notification>', { origin: { kind: 'task-notification' } }),
        user('Another Claude session sent a message: synthetic', { origin: { kind: 'peer' } }),
        user('[SYSTEM NOTIFICATION - NOT USER INPUT] synthetic'),
        assistant('answer'),
    ]);

    assert.deepEqual(messages, [
        { role: 'user', content: 'real prompt' },
        { role: 'assistant', content: 'answer' },
    ]);
});

test('strips reminder spans from a typed prompt, in blocks or inline', async () => {
    const { messages } = await parse([
        user([{ type: 'text', text: REMINDER }, { type: 'text', text: 'prompt one' }]),
        user(`prompt two ${REMINDER}`),
    ]);

    assert.deepEqual(messages.map((m) => m.content), ['prompt one', 'prompt two']);
});

test('keeps assistant text that mentions a reminder tag', async () => {
    const { messages } = await parse([assistant(`quoting ${REMINDER}`)]);

    assert.equal(messages[0].content, `quoting ${REMINDER}`);
});
