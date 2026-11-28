/**
 * Unit tests for the transcript-filter helper. Fixtures are synthetic.
 * Run: node --test claude-hooks/tests/transcript-filter.test.js
 */
const { test } = require('node:test');
const assert = require('node:assert/strict');

const { classifyUserEntry, cleanUserText, stripHarnessTags } = require('../utilities/transcript-filter');

const kindOf = (entry, text) => classifyUserEntry(entry, text).kind;

test('non-human origin and compact summaries are turn starters', () => {
    assert.equal(kindOf({ origin: { kind: 'task-notification' } }, 'x'), 'turnStart');
    assert.equal(kindOf({ origin: { kind: 'peer' }, isMeta: true }, 'x'), 'turnStart');
    assert.equal(kindOf({ origin: { kind: 'auto-continuation' }, isMeta: true }, 'x'), 'turnStart');
    assert.equal(kindOf({ isCompactSummary: true }, 'x'), 'turnStart');
});

test('isMeta and reminder-only entries are attachments', () => {
    assert.equal(kindOf({ isMeta: true }, 'text'), 'attachment');
    assert.equal(kindOf({}, '<system-reminder>only</system-reminder>'), 'attachment');
    assert.equal(kindOf({ origin: { kind: 'human' } }, '<system-reminder>only</system-reminder>'), 'attachment');
});

test('legacy prefixes: notifications start a turn, command output is an attachment', () => {
    assert.equal(kindOf({}, '<task-notification>x</task-notification>'), 'turnStart');
    assert.equal(kindOf({}, 'Another Claude session sent a message'), 'turnStart');
    assert.equal(kindOf({}, '[SYSTEM NOTIFICATION - NOT USER INPUT] x'), 'turnStart');
    for (const tag of ['local-command-caveat', 'local-command-stdout', 'local-command-stderr',
        'bash-stdout', 'bash-stderr', 'side-sessions-event']) {
        assert.equal(kindOf({}, `<${tag}>x</${tag}>`), 'attachment', tag);
    }
});

test('a reminder in front of a notification or caveat does not hide it', () => {
    const r = '<system-reminder>x</system-reminder>\n';
    assert.equal(kindOf({}, `${r}<local-command-caveat>x</local-command-caveat>`), 'attachment');
    assert.equal(kindOf({}, `${r}<task-notification>x</task-notification>`), 'turnStart');
});

test('turnOrigin decides first: harness turn starts, user turn downgrades turn starters', () => {
    const reminder = '<system-reminder>x</system-reminder>';
    for (const turnOrigin of ['host_synthetic', 'system', 'task_notification', 'peer', 'scheduled']) {
        assert.equal(kindOf({ turnOrigin }, reminder), 'turnStart', turnOrigin);
        assert.equal(kindOf({ turnOrigin }, 'plain text'), 'turnStart', turnOrigin);
    }
    for (const turnOrigin of ['human', 'sdk']) {
        assert.equal(kindOf({ turnOrigin, origin: { kind: 'peer' } }, 'x'), 'attachment', turnOrigin);
        assert.equal(kindOf({ turnOrigin, isCompactSummary: true }, 'x'), 'attachment', turnOrigin);
        assert.equal(kindOf({ turnOrigin }, '<task-notification>x</task-notification>'), 'attachment', turnOrigin);
        assert.equal(classifyUserEntry({ turnOrigin }, 'typed').text, 'typed', turnOrigin);
    }
});

test('a notification wrapped entirely in a reminder is a turn start', () => {
    assert.equal(kindOf({}, '<system-reminder>[SYSTEM NOTIFICATION - NOT USER INPUT] x</system-reminder>'), 'turnStart');
    assert.equal(kindOf({}, '<system-reminder>ordinary reminder</system-reminder>'), 'attachment');
});

test('origin human is trusted: the prefix heuristic does not apply', () => {
    const entry = { origin: { kind: 'human' } };
    assert.deepEqual(classifyUserEntry(entry, 'Another Claude session told me X'),
        { kind: 'prompt', text: 'Another Claude session told me X' });
    // without origin the same text is treated as injected (legacy transcripts)
    assert.equal(kindOf({}, 'Another Claude session told me X'), 'turnStart');
});

test('user-typed markers and normal prompts are prompts', () => {
    assert.equal(kindOf({}, '<command-name>/foo</command-name>'), 'prompt');
    assert.equal(kindOf({}, '<bash-input>ls</bash-input>'), 'prompt');
    assert.equal(cleanUserText({}, 'what is a task-notification?'), 'what is a task-notification?');
    assert.equal(cleanUserText({ origin: { kind: 'human' } }, 'hello'), 'hello');
});

test('cleanUserText returns empty for anything injected', () => {
    assert.equal(cleanUserText({ isMeta: true }, 'text'), '');
    assert.equal(cleanUserText({ origin: { kind: 'peer' } }, 'text'), '');
    assert.equal(cleanUserText({}, undefined), '');
});

test('stripHarnessTags removes every span, multi-line, and trims', () => {
    const text = 'a <system-reminder>x\ny</system-reminder> b <system-reminder>z</system-reminder>';
    assert.equal(stripHarnessTags(text), 'a  b');
    assert.equal(stripHarnessTags(undefined), '');
});

test('a fork-boilerplate span is stripped and the task after it is the prompt', () => {
    const text = '<fork-boilerplate>synthetic preamble\nmore</fork-boilerplate>\n\nthe forked task';
    assert.equal(stripHarnessTags(text), 'the forked task');
    assert.deepEqual(classifyUserEntry({}, text), { kind: 'prompt', text: 'the forked task' });
    assert.equal(kindOf({}, '<fork-boilerplate>only preamble</fork-boilerplate>'), 'attachment');
});

test('stripHarnessTags removes nested spans without a stray closing tag', () => {
    const text = 'a <system-reminder>x <system-reminder>y</system-reminder> z</system-reminder> b';
    assert.equal(stripHarnessTags(text), 'a  b');
});

test('stripHarnessTags leaves an unclosed tag alone', () => {
    assert.equal(stripHarnessTags('see <system-reminder> docs'), 'see <system-reminder> docs');
});
