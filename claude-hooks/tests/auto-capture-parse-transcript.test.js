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

// Harness-injected text must not become "the user prompt". Fixtures are
// synthetic; shapes mirror what Claude Code writes (isMeta / origin.kind /
// <system-reminder> as its own text block or inline in a string).

const REMINDER = '<system-reminder>\nSynthetic harness reminder.\n</system-reminder>';

test('strips a <system-reminder> text block next to the typed prompt', async () => {
    const result = await withTranscript(
        [
            {
                message: {
                    role: 'user',
                    content: [
                        { type: 'text', text: REMINDER },
                        { type: 'text', text: 'typed prompt' },
                    ],
                },
            },
            { message: { role: 'assistant', content: 'the answer' } },
        ],
        (file) => parseTranscript(file),
    );

    assert.equal(result.userMessage, 'typed prompt');
});

test('strips an inline <system-reminder> span from a string prompt', async () => {
    const result = await withTranscript(
        [
            { message: { role: 'user', content: `typed prompt\n${REMINDER}\nmore typed text` } },
            { message: { role: 'assistant', content: 'the answer' } },
        ],
        (file) => parseTranscript(file),
    );

    assert.equal(result.userMessage, 'typed prompt\n\nmore typed text');
});

test('scans past isMeta and reminder-only user entries to the real prompt', async () => {
    const result = await withTranscript(
        [
            { type: 'user', message: { role: 'user', content: 'real prompt' } },
            { type: 'user', isMeta: true, message: { role: 'user', content: 'meta injected text' } },
            { type: 'user', message: { role: 'user', content: REMINDER } },
            { type: 'assistant', message: { role: 'assistant', content: 'the answer' } },
        ],
        (file) => parseTranscript(file),
    );

    assert.equal(result.userMessage, 'real prompt');
});

test('stops at a task-notification or peer entry instead of pairing with an older prompt', async () => {
    const notification = {
        type: 'user',
        origin: { kind: 'task-notification' },
        message: { role: 'user', content: '<task-notification>\n<task-id>x</task-id>\n</task-notification>' },
    };
    const peer = {
        type: 'user',
        origin: { kind: 'peer', from: 'synthetic' },
        message: { role: 'user', content: 'Another Claude session sent a message: synthetic' },
    };

    for (const starter of [notification, peer]) {
        const result = await withTranscript(
            [
                { type: 'user', origin: { kind: 'human' }, message: { role: 'user', content: 'Q1' } },
                { type: 'assistant', message: { role: 'assistant', content: 'A1 final' } },
                starter,
                { type: 'assistant', message: { role: 'assistant', content: 'A2' } },
            ],
            (file) => parseTranscript(file),
        );

        assert.equal(result.userMessage, '');
        assert.equal(result.assistantMessage, 'A2');
    }
});

test('stops at a legacy entry with no origin recognised by its notification prefix', async () => {
    const result = await withTranscript(
        [
            { type: 'user', message: { role: 'user', content: 'Q1' } },
            { type: 'assistant', message: { role: 'assistant', content: 'A1 final' } },
            { type: 'user', message: { role: 'user', content: '[SYSTEM NOTIFICATION - NOT USER INPUT] synthetic' } },
            { type: 'assistant', message: { role: 'assistant', content: 'A2' } },
        ],
        (file) => parseTranscript(file),
    );

    assert.equal(result.userMessage, '');
});

test('scans past a reminder followed by a local-command caveat to the real prompt', async () => {
    const result = await withTranscript(
        [
            { type: 'user', message: { role: 'user', content: 'real prompt' } },
            {
                type: 'user',
                message: {
                    role: 'user',
                    content: `${REMINDER}\n<local-command-caveat>Caveat: synthetic</local-command-caveat>`,
                },
            },
            { type: 'user', message: { role: 'user', content: '<bash-stdout>synthetic</bash-stdout>' } },
            { type: 'assistant', message: { role: 'assistant', content: 'the answer' } },
        ],
        (file) => parseTranscript(file),
    );

    assert.equal(result.userMessage, 'real prompt');
});

test('keeps a human-origin prompt that starts with a notification sentence', async () => {
    const result = await withTranscript(
        [
            {
                type: 'user',
                origin: { kind: 'human' },
                message: { role: 'user', content: 'Another Claude session told me the build is broken' },
            },
            { type: 'assistant', message: { role: 'assistant', content: 'the answer' } },
        ],
        (file) => parseTranscript(file),
    );

    assert.equal(result.userMessage, 'Another Claude session told me the build is broken');
});

test('a reminder-only entry that starts a harness turn stops the scan', async () => {
    const result = await withTranscript(
        [
            { type: 'user', turnOrigin: 'human', message: { role: 'user', content: 'Q1' } },
            { type: 'assistant', turnOrigin: 'human', message: { role: 'assistant', content: 'A1 final' } },
            { type: 'user', turnOrigin: 'host_synthetic', message: { role: 'user', content: REMINDER } },
            { type: 'assistant', turnOrigin: 'host_synthetic', message: { role: 'assistant', content: 'A2' } },
        ],
        (file) => parseTranscript(file),
    );

    assert.equal(result.userMessage, '');
});

test('a task-notification inside a user-started turn does not hide the prompt', async () => {
    const result = await withTranscript(
        [
            { type: 'user', turnOrigin: 'human', message: { role: 'user', content: 'Q1' } },
            {
                type: 'user',
                turnOrigin: 'human',
                origin: { kind: 'task-notification' },
                message: { role: 'user', content: '<task-notification>x</task-notification>' },
            },
            { type: 'assistant', turnOrigin: 'human', message: { role: 'assistant', content: 'A1' } },
        ],
        (file) => parseTranscript(file),
    );

    assert.equal(result.userMessage, 'Q1');
});

test('strips a fork-boilerplate preamble from a subagent first message', async () => {
    const result = await withTranscript(
        [
            {
                type: 'user',
                message: {
                    role: 'user',
                    content: [
                        { type: 'tool_result', content: 'ok' },
                        { type: 'text', text: '<fork-boilerplate>synthetic preamble</fork-boilerplate>\n\nthe forked task' },
                    ],
                },
            },
            { type: 'assistant', message: { role: 'assistant', content: 'the answer' } },
        ],
        (file) => parseTranscript(file),
    );

    assert.equal(result.userMessage, 'the forked task');
});

test('#skip typed after a reminder is still detected, #skip inside a reminder is not', async () => {
    const { hasUserOverride } = require('../utilities/auto-capture-patterns');
    const typed = await withTranscript(
        [
            { message: { role: 'user', content: `${REMINDER}\nfix it #skip` } },
            { message: { role: 'assistant', content: 'the answer' } },
        ],
        (file) => parseTranscript(file),
    );
    const injected = await withTranscript(
        [
            { message: { role: 'user', content: 'fix it' } },
            { message: { role: 'user', content: '<system-reminder>#skip</system-reminder>' } },
            { message: { role: 'assistant', content: 'the answer' } },
        ],
        (file) => parseTranscript(file),
    );

    assert.equal(hasUserOverride(typed.userMessage).forceSkip, true);
    assert.equal(injected.userMessage, 'fix it');
    assert.equal(hasUserOverride(injected.userMessage).forceSkip, false);
});

test('leaves a normal prompt untouched', async () => {
    const result = await withTranscript(
        [
            { type: 'user', origin: { kind: 'human' }, message: { role: 'user', content: 'plain <b>prompt</b>' } },
            { type: 'assistant', message: { role: 'assistant', content: 'the answer' } },
        ],
        (file) => parseTranscript(file),
    );

    assert.equal(result.userMessage, 'plain <b>prompt</b>');
});
