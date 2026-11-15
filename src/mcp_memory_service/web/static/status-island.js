/**
 * Status island: one capsule in the header for what the dashboard is doing.
 *
 * Several things run at once here (the live connection, a sync, an import), so the
 * island keeps one entry per named source and shows the one that matters most:
 * an error before work in progress, before a confirmation, before "ready".
 * A confirmation from one source therefore never covers a failure of another.
 *
 * A source reports text as a catalog key plus values, not as a finished string,
 * so a language switch re-renders it.
 *
 *   island.report('sync', {
 *     state: 'busy',                                  // ready | busy | success | error
 *     label: { key: 'sync.title', fallback: 'Sync' }, // names the source in the panel
 *     text: { key: 'sync.status.syncing', fallback: 'Syncing', vars: {} },
 *     progress: 40,                                   // 0-100 when measured, 'unknown', or null
 *     actions: [{ id: 'pause', label: { key: '...', fallback: 'Pause' }, run: () => {} }],
 *     dismissible: true,                              // an error the reader may close
 *     settle: { state: 'ready', text: {...} },        // what a success turns into after a while
 *   });
 *   island.clear('upload-42');                        // a source that is gone
 *
 * This file has two parts: a model without any DOM (tested in node) and the view.
 */
(function (root) {
    'use strict';

    const RANK = { error: 4, busy: 3, success: 2, ready: 1 };
    const SETTLE_MS = 5000;

    /** A percentage only when it is a real number from 0 to 100; anything else is "unknown" or nothing. */
    function normalizeProgress(value) {
        if (value === null || value === undefined) return null;
        if (typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 100) return value;
        return 'unknown';
    }

    function createModel() {
        const sources = new Map();
        let seq = 0;
        let registered = 0;

        // What a reader can see of an entry. Two reports with the same signature are the same report.
        const signature = (entry) => JSON.stringify([
            entry.state, entry.label, entry.text, entry.progress, entry.dismissible,
            entry.actions.map((action) => [action.id, action.label]),
        ]);
        // A caller may keep one text object and change its values between reports. The
        // entry therefore holds its own copy of everything that is displayed, and its
        // signature is fixed when the report is accepted, not recomputed from live objects.
        const copy = (value) => (value === null || value === undefined ? value : JSON.parse(JSON.stringify(value)));

        function put(id, entry) {
            if (!RANK[entry.state]) throw new Error(`status island: unknown state "${entry.state}"`);
            const previous = sources.get(id);
            const stored = {
                id,
                state: entry.state,
                label: copy(entry.label) || null,
                text: copy(entry.text) || { key: '', fallback: '' },
                progress: entry.state === 'busy' ? normalizeProgress(entry.progress) : null,
                actions: (Array.isArray(entry.actions) ? entry.actions : [])
                    .map((action) => ({ id: action.id, label: copy(action.label), run: action.run })),
                dismissible: entry.state === 'error' && entry.dismissible === true,
                settle: entry.settle || null,
                // The position a source got when it first appeared; it decides who leads when idle.
                order: previous ? previous.order : ++registered,
            };
            // A poll that reports what is already shown changes nothing: no new rank, no
            // redraw, and a running "steps back" timer keeps running. Only the callbacks
            // are taken over, because they may close over newer data.
            stored.signature = signature(stored);
            if (previous && previous.signature === stored.signature) {
                previous.actions = stored.actions;
                previous.settle = stored.settle;
                previous.unchanged = true;
                return previous;
            }
            stored.seq = ++seq;
            sources.set(id, stored);
            return stored;
        }

        /** What a finished or dismissed entry turns into: its settle entry, or nothing. */
        function settle(id) {
            const current = sources.get(id);
            if (!current) return;
            if (current.settle) put(id, { ...current.settle, label: current.settle.label || current.label });
            else sources.delete(id);
        }

        return {
            report: put,
            clear(id) { return sources.delete(id); },
            get(id) { return sources.get(id) || null; },
            /**
             * Sorted by importance. Among sources that need attention the newest report
             * leads; among idle ones the source that appeared first, so the capsule does
             * not flip between two "ready" texts whenever one of them reports.
             */
            list() {
                return [...sources.values()].sort((a, b) => RANK[b.state] - RANK[a.state]
                    || (a.state === 'ready' ? a.order - b.order : b.seq - a.seq));
            },
            lead() { return this.list()[0] || null; },
            /**
             * A confirmation steps back by itself, but only the report that set the timer
             * may do it: a newer report of the same source, and any error, stays.
             */
            expire(id, atSeq) {
                const current = sources.get(id);
                if (!current || current.seq !== atSeq || current.state !== 'success') return false;
                settle(id);
                return true;
            },
            /** Only an error that was reported as dismissible can be closed by the reader. */
            dismiss(id) {
                const current = sources.get(id);
                if (!current || !current.dismissible) return false;
                settle(id);
                return true;
            },
        };
    }

    /**
     * The view. `options.root` is the empty container in the header, `options.t` the
     * dashboard's translate function. Nothing in here may break the dashboard: every
     * public method catches and logs.
     */
    function StatusIsland(options) {
        const doc = options.root.ownerDocument;
        const t = options.t || ((key, fallback) => fallback || key);
        const model = createModel();
        const timers = new Map();
        const host = options.root;
        let announced = '';

        const el = (tag, className, parent) => {
            const node = doc.createElement(tag);
            if (className) node.className = className;
            if (parent) parent.appendChild(node);
            return node;
        };
        const say = (text) => (text && text.key ? t(text.key, text.fallback || '', text.vars || {}) : (text && text.fallback) || '');

        const toggle = el('button', 'status-island-toggle', host);
        toggle.type = 'button';
        toggle.setAttribute('aria-expanded', 'false');
        toggle.setAttribute('aria-controls', 'statusIslandPanel');
        el('span', 'status-island-dot', toggle).setAttribute('aria-hidden', 'true');
        const text = el('span', 'status-island-text', toggle);
        const more = el('span', 'status-island-count', toggle);
        const hint = el('span', 'sr-only', toggle);
        el('span', 'status-island-chevron', toggle).setAttribute('aria-hidden', 'true');
        const panel = el('div', 'status-island-panel', host);
        panel.id = 'statusIslandPanel';
        panel.hidden = true;
        const rows = el('ul', 'status-island-rows', panel);
        const live = el('span', 'sr-only', host);
        live.setAttribute('role', 'status');

        const open = (yes) => {
            panel.hidden = !yes;
            toggle.setAttribute('aria-expanded', String(yes));
        };
        toggle.addEventListener('click', () => open(panel.hidden));
        // Escape closes the panel from anywhere; the focus goes back to the capsule only
        // if it was inside the island, so the reader is not pulled away from another control.
        doc.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && !panel.hidden) {
                const inside = host.contains(doc.activeElement);
                open(false);
                if (inside) toggle.focus();
            }
        });
        // The path is taken when the click happens. A report made by the clicked action
        // redraws the rows, and the button is then no longer inside the island.
        doc.addEventListener('click', (event) => {
            const path = event.composedPath ? event.composedPath() : [];
            if (!panel.hidden && !path.includes(host) && !host.contains(event.target)) open(false);
        });

        function button(label, parent, focusKey, onClick) {
            const node = el('button', 'status-island-action', parent);
            node.type = 'button';
            node.textContent = label;
            node.dataset.focusKey = focusKey;
            node.addEventListener('click', onClick);
            return node;
        }

        function row(source) {
            const item = el('li', 'status-island-row', rows);
            item.dataset.state = source.state;
            item.dataset.source = source.id;
            const head = el('div', 'status-island-row-head', item);
            el('span', 'status-island-dot', head).setAttribute('aria-hidden', 'true');
            if (source.label) el('span', 'status-island-row-label', head).textContent = say(source.label);
            el('span', 'status-island-row-text', head).textContent = say(source.text);
            if (source.progress !== null) {
                const track = el('div', 'status-island-track', item);
                track.dataset.progress = source.progress === 'unknown' ? 'unknown' : 'known';
                const bar = el('div', 'status-island-bar', track);
                if (source.progress !== 'unknown') bar.style.width = source.progress + '%';
            }
            if (source.actions.length || source.dismissible) {
                const actions = el('div', 'status-island-actions', item);
                source.actions.forEach((action) => button(say(action.label), actions, `${source.id}:${action.id}`, () => {
                    // Looked up now: an unchanged report may have replaced the callback.
                    const current = (model.get(source.id) || source).actions.find((item) => item.id === action.id) || action;
                    try { current.run(); } catch (error) { console.warn('status island action', error); }
                }));
                if (source.dismissible) {
                    button(t('island.dismiss', 'Close'), actions, `${source.id}:dismiss`, () => {
                        model.dismiss(source.id);
                        render();
                        toggle.focus();
                    });
                }
            }
        }

        function render(silent) {
            const list = model.list();
            const lead = list[0];
            host.hidden = !lead;
            if (!lead) return;
            host.dataset.state = lead.state;
            const leadText = say(lead.text);
            text.textContent = leadText;
            toggle.title = leadText;
            hint.textContent = ', ' + t('island.details', 'Details');
            // Other sources that are not simply "ready" are counted next to the lead.
            const others = list.slice(1).filter((source) => source.state !== 'ready').length;
            more.textContent = others ? '+' + others : '';
            more.hidden = !others;
            // The rows are rebuilt on every change. If the focus was on one of their
            // buttons it goes back to the same button, or to the capsule if that is gone.
            const focused = rows.contains(doc.activeElement) ? doc.activeElement.dataset.focusKey : null;
            rows.textContent = '';
            list.forEach(row);
            if (focused) {
                const again = [...rows.querySelectorAll('.status-island-action')].find((node) => node.dataset.focusKey === focused);
                (again || toggle).focus();
            }
            // Announce results and failures of the lead, never work in progress or its ticks.
            const key = lead.state === 'busy' ? announced : `${lead.id}:${lead.state}:${leadText}`;
            if (silent) announced = key;
            else if (key !== announced) {
                announced = key;
                live.textContent = lead.label ? `${say(lead.label)}: ${leadText}` : leadText;
            }
        }

        function schedule(source) {
            clearTimeout(timers.get(source.id));
            timers.delete(source.id);
            if (source.state !== 'success') return;
            timers.set(source.id, setTimeout(() => {
                timers.delete(source.id);
                if (model.expire(source.id, source.seq)) render();
            }, options.settleMs || SETTLE_MS));
        }

        const guarded = (fn) => (...args) => {
            try { return fn(...args); } catch (error) { console.warn('status island', error); return undefined; }
        };

        this.report = guarded((id, entry) => {
            const stored = model.report(id, entry);
            if (stored.unchanged) { stored.unchanged = false; return; }
            schedule(stored);
            render();
        });
        this.clear = guarded((id) => { clearTimeout(timers.get(id)); timers.delete(id); model.clear(id); render(); });
        /** Call after a language switch. */
        this.refresh = guarded(() => render(true));
        this.get = (id) => model.get(id);
    }

    StatusIsland.createModel = createModel;
    StatusIsland.normalizeProgress = normalizeProgress;

    if (typeof module === 'object' && module.exports) module.exports = StatusIsland;
    else root.StatusIsland = StatusIsland;
})(typeof window !== 'undefined' ? window : globalThis);
