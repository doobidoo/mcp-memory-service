/**
 * Filters harness-injected text out of Claude Code transcript user entries.
 *
 * Claude Code stores more than typed prompts under the `user` role: hook and
 * harness <system-reminder> blocks, background-task notifications, messages from
 * other Claude sessions, compaction summaries and local command output. The
 * transcript parsers used to take the nearest user-role entry as "the user
 * prompt", so that text reached #skip/#remember detection and was stored as part
 * of auto-captured memories and session summaries.
 *
 * Observed transcript shapes this module handles (all synthetic examples in
 * tests/transcript-filter.test.js):
 *   - flag:   `isMeta: true` or `isCompactSummary: true` on the envelope
 *   - turn:   `turnOrigin` (human, sdk, task_notification, peer, system,
 *             host_synthetic, scheduled, ...), identical for every entry of a
 *             turn. A string other than human/sdk means the harness started the
 *             whole turn, even when its first entry is only a reminder.
 *   - origin: `origin.kind` other than "human" (task-notification, peer,
 *             auto-continuation); older transcripts carry no `origin` at all
 *   - prefix: legacy entries (no `origin`) whose text starts with a harness tag
 *             or sentence
 *   - inline: <system-reminder>...</system-reminder> and
 *             <fork-boilerplate>...</fork-boilerplate> spans inside the text of
 *             a prompt that is otherwise genuine, as their own text block or
 *             embedded in a string
 *
 * Injected entries come in two kinds that callers must treat differently:
 *   - "attachment": context glued to the current turn (reminders, local command
 *     caveats and output). Skip it and keep looking for the typed prompt.
 *   - "turnStart": something that starts a new turn on its own (a background
 *     task notification, a message from another session, a compaction
 *     summary). There is no typed prompt for the reply that follows, so a
 *     backward scan must stop here rather than pair the reply with an older,
 *     already answered prompt.
 *
 * Inside a user-started turn (`turnOrigin` human/sdk) a notification or
 * compaction summary is mid-turn, so it is only an attachment there.
 *
 * Conservative on purpose: an entry that matches none of these is returned
 * unchanged apart from trimming, so ordinary prompts behave as before.
 */

'use strict';

// Tags whose spans are removed from the text of an otherwise genuine prompt.
// <fork-boilerplate> is the preamble Claude Code puts in front of the task text
// of a forked subagent's first message; in every local instance it was closed
// and followed by the task, so only the span is dropped.
const STRIPPED_TAGS = ['system-reminder', 'fork-boilerplate'];

// Legacy entries (no `origin`) are recognised by how their text starts. These
// start a turn of their own.
const TURN_START_PREFIX = new RegExp(
    '^\\s*(?:<task-notification[\\s>]|Another Claude session|\\[SYSTEM NOTIFICATION)'
);

// ... and these are attachments to the current turn.
const ATTACHMENT_PREFIX = new RegExp(
    '^\\s*<(?:local-command-caveat|local-command-stdout|local-command-stderr' +
    '|bash-stdout|bash-stderr|side-sessions-event)[\\s>]'
);

const STRIPPED_SPANS = STRIPPED_TAGS.map(
    (tag) => // Innermost span first (no opening tag inside), so nesting resolves by looping.
    new RegExp(`<${tag}>(?:(?!<${tag}>)[\\s\\S])*?</${tag}>`, 'g')
);

/**
 * Remove harness tag spans (e.g. <system-reminder>...</system-reminder>),
 * repeating until nothing changes so nested spans are removed whole.
 * @param {string} text
 * @returns {string}
 */
function stripHarnessTags(text) {
    if (typeof text !== 'string') return '';
    let current = text;
    for (;;) {
        const next = STRIPPED_SPANS.reduce((acc, span) => acc.replace(span, ''), current);
        if (next === current) return current.trim();
        current = next;
    }
}

// Text of the <system-reminder> spans themselves, for entries that are only a
// reminder wrapping a notification.
function reminderInnerText(text) {
    const inner = [];
    for (const m of text.matchAll(/<system-reminder>([\s\S]*?)<\/system-reminder>/g)) inner.push(m[1]);
    return inner.join('\n').trim();
}

// Classification from the entry alone, ignoring the turn it belongs to.
function classifyEntry(entry, rawText) {
    const none = (kind) => ({ kind, text: '' });
    const origin = entry && entry.origin && entry.origin.kind;
    if (typeof origin === 'string' && origin !== 'human') return none('turnStart');
    if (entry && entry.isCompactSummary === true) return none('turnStart');
    if (entry && entry.isMeta === true) return none('attachment');

    // Strip first: a reminder in front of a notification must not hide it.
    const text = stripHarnessTags(rawText);
    if (!text) {
        return TURN_START_PREFIX.test(reminderInnerText(rawText)) ? none('turnStart') : none('attachment');
    }
    if (origin === 'human') return { kind: 'prompt', text };
    if (TURN_START_PREFIX.test(text)) return none('turnStart');
    if (ATTACHMENT_PREFIX.test(text)) return none('attachment');
    return { kind: 'prompt', text };
}

/**
 * Classify a user-role transcript entry.
 *
 * When `turnOrigin` is present it decides first: a harness-started turn is a
 * turn start whatever its first entry looks like; in a user-started turn
 * (human/sdk) a would-be turn start is only an attachment. Otherwise, when `origin.kind` is present it is trusted: "human" entries are typed
 * prompts (only reminder spans are stripped), any other kind starts a turn.
 * The prefix heuristics apply to legacy transcripts without `origin` only, so
 * a typed prompt that merely begins with e.g. "Another Claude session" is kept.
 *
 * @param {object} entry - one parsed transcript line
 * @param {string} rawText - the entry's text content, already joined
 * @returns {{kind: 'prompt'|'attachment'|'turnStart', text: string}}
 *   `text` is the cleaned prompt for kind "prompt" and '' otherwise.
 */
function classifyUserEntry(entry, rawText) {
    if (typeof rawText !== 'string') return { kind: 'attachment', text: '' };
    const turnOrigin = entry && entry.turnOrigin;
    const userTurn = turnOrigin === 'human' || turnOrigin === 'sdk';
    if (typeof turnOrigin === 'string' && !userTurn) return { kind: 'turnStart', text: '' };

    const result = classifyEntry(entry, rawText);
    return userTurn && result.kind === 'turnStart' ? { kind: 'attachment', text: '' } : result;
}

/**
 * Text the user actually typed in a user-role entry, or '' for anything the
 * harness injected. For callers that only need the text (no turn pairing).
 * @param {object} entry
 * @param {string} rawText
 * @returns {string}
 */
function cleanUserText(entry, rawText) {
    return classifyUserEntry(entry, rawText).text;
}

module.exports = { classifyUserEntry, cleanUserText, stripHarnessTags };
