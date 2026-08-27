/**
 * Unit tests for auto-capture parseTranscript pairing (#1069).
 * Run: node --test claude-hooks/tests/auto-capture-parse-transcript.test.js
 */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');

const { parseTranscript } = require('../core/auto-capture-hook');

async function withTranscript(lines, fn) {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'ac-parse-'));
    const file = path.join(dir, 'transcript.jsonl');
    await fs.writeFile(file, lines.map((o) => JSON.stringify(o)).join('\n') + '\n', 'utf8');
    try {
        return await fn(file);
    } finally {
        await fs.rm(dir, { recursive: true, force: true });
    }
}

test('pairs last assistant with preceding user, not a later follow-up', async () => {
    const result = await withTranscript(
        [
            { message: { role: 'user', content: 'Q1' } },
            { message: { role: 'assistant', content: 'A1 Decision: ship it' } },
            { message: { role: 'user', content: 'Q2 follow-up' } },
        ],
        (file) => parseTranscript(file),
    );

    assert.equal(result.userMessage, 'Q1');
    assert.equal(result.assistantMessage, 'A1 Decision: ship it');
});

test('skips tool_result-only user envelopes when pairing', async () => {
    const result = await withTranscript(
        [
            { message: { role: 'user', content: 'real question' } },
            {
                message: {
                    role: 'user',
                    content: [{ type: 'tool_result', content: 'ok' }],
                },
            },
            { message: { role: 'assistant', content: 'the answer' } },
        ],
        (file) => parseTranscript(file),
    );

    assert.equal(result.userMessage, 'real question');
    assert.equal(result.assistantMessage, 'the answer');
});
