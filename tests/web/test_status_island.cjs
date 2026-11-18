// Run with: node --test tests/web/test_status_island.cjs
// The island's model has no DOM, so its rules are tested here directly. The second
// half loads app.js the way test_dashboard_live_refresh.cjs does and checks what the
// dashboard reports for the live connection.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const STATIC = 'src/mcp_memory_service/web/static/';
const StatusIsland = require('../../' + STATIC + 'status-island.js');

const text = (fallback) => ({ key: '', fallback });
const ready = { state: 'ready', text: text('Ready') };

test('the lead is the most important source: error, then busy, then success, then ready', () => {
    const model = StatusIsland.createModel();
    model.report('a', ready);
    assert.equal(model.lead().id, 'a');
    model.report('b', { state: 'success', text: text('Done') });
    assert.equal(model.lead().id, 'b');
    model.report('c', { state: 'busy', text: text('Working') });
    assert.equal(model.lead().id, 'c');
    model.report('d', { state: 'error', text: text('Failed') });
    assert.equal(model.lead().id, 'd');
    assert.deepEqual(model.list().map((source) => source.id), ['d', 'c', 'b', 'a']);
});

test('a confirmation from one source never covers a failure of another', () => {
    const model = StatusIsland.createModel();
    model.report('store', { state: 'error', text: text('Storing failed') });
    model.report('search', { state: 'busy', text: text('Searching') });
    model.report('search', { state: 'success', text: text('4 results') });
    assert.equal(model.lead().id, 'store');
    assert.equal(model.lead().state, 'error');
});

test('a report that changes nothing is not a new report', () => {
    const model = StatusIsland.createModel();
    const first = model.report('sync', { state: 'success', text: text('Synced'), settle: ready });
    model.report('other', { state: 'success', text: text('Done') });
    const repeated = model.report('sync', { state: 'success', text: text('Synced'), settle: ready });
    assert.equal(repeated.seq, first.seq, 'a poll with the same answer keeps its place');
    assert.equal(repeated.unchanged, true);
    assert.equal(model.lead().id, 'other');
    assert.equal(model.expire('sync', first.seq), true, 'and its timer still counts');

    const changed = model.report('other', { state: 'success', text: text('Done, 3 items') });
    assert.notEqual(changed.unchanged, true);
});

test('a caller that reuses one object and changes its values gets a new report', () => {
    const model = StatusIsland.createModel();
    const shared = { key: 'sync.progress', fallback: '{done} of {total}', vars: { done: 1, total: 10 } };
    const label = { key: 'sync.title', fallback: 'Sync' };
    const action = { id: 'pause', label: { key: '', fallback: 'Pause' }, run() {} };
    const first = model.report('sync', { state: 'busy', text: shared, label, actions: [action] });
    assert.equal(first.text.vars.done, 1);

    shared.vars.done = 2;
    assert.equal(model.get('sync').text.vars.done, 1, 'the stored entry is not changed from outside');
    const second = model.report('sync', { state: 'busy', text: shared, label, actions: [action] });
    assert.notEqual(second.unchanged, true);
    assert.ok(second.seq > first.seq);
    assert.equal(model.get('sync').text.vars.done, 2);

    label.fallback = 'Delta sync';
    assert.notEqual(model.report('sync', { state: 'busy', text: shared, label, actions: [action] }).unchanged, true);
    action.label.fallback = 'Resume';
    assert.notEqual(model.report('sync', { state: 'busy', text: shared, label, actions: [action] }).unchanged, true);
    assert.equal(model.get('sync').actions[0].label.fallback, 'Resume');

    assert.equal(model.report('sync', { state: 'busy', text: shared, label, actions: [action] }).unchanged, true, 'and the same values again are still no report');
});

test('an unchanged report still takes over the newer callbacks', () => {
    const model = StatusIsland.createModel();
    const action = (run) => ({ state: 'error', text: text('Failed'), actions: [{ id: 'retry', label: text('Retry'), run }] });
    let ran = '';
    model.report('backup', action(() => { ran = 'old'; }));
    model.report('backup', action(() => { ran = 'new'; }));
    model.get('backup').actions[0].run();
    assert.equal(ran, 'new');
});

test('when everything is idle the source that appeared first leads, whoever reported last', () => {
    const model = StatusIsland.createModel();
    model.report('connection', { state: 'ready', text: text('Connected') });
    model.report('sync', { state: 'ready', text: text('Synced') });
    assert.equal(model.lead().id, 'connection');
    model.report('sync', { state: 'busy', text: text('Syncing') });
    assert.equal(model.lead().id, 'sync');
    model.report('sync', { state: 'ready', text: text('Synced') });
    assert.equal(model.lead().id, 'connection', 'back to the first source, not to the last reporter');
    model.report('connection', { state: 'error', text: text('Disconnected') });
    model.report('connection', { state: 'ready', text: text('Connected') });
    assert.equal(model.lead().id, 'connection');
});

test('among sources that need attention the newest report leads', () => {
    const model = StatusIsland.createModel();
    model.report('first', { state: 'busy', text: text('one') });
    model.report('second', { state: 'busy', text: text('two') });
    assert.equal(model.lead().id, 'second');
    model.report('first', { state: 'busy', text: text('one again') });
    assert.equal(model.lead().id, 'first');
});

test('a confirmation steps back only for the report that set it', () => {
    const model = StatusIsland.createModel();
    const done = model.report('sync', { state: 'success', text: text('Synced'), settle: ready });
    const again = model.report('sync', { state: 'busy', text: text('Syncing') });
    assert.equal(model.expire('sync', done.seq), false, 'a newer report of the same source stays');
    assert.equal(model.get('sync').state, 'busy');
    assert.equal(model.expire('sync', again.seq), false, 'only a success expires');

    const finished = model.report('sync', { state: 'success', text: text('Synced'), settle: ready });
    assert.equal(model.expire('sync', finished.seq), true);
    assert.equal(model.get('sync').state, 'ready');
});

test('a success without a settle entry disappears, which is how a finished upload leaves', () => {
    const model = StatusIsland.createModel();
    const done = model.report('upload-1', { state: 'success', text: text('Imported') });
    assert.equal(model.expire('upload-1', done.seq), true);
    assert.equal(model.get('upload-1'), null);
    assert.equal(model.lead(), null);
});

test('an error never expires and ends only through its own source', () => {
    const model = StatusIsland.createModel();
    const failed = model.report('connection', { state: 'error', text: text('Disconnected') });
    assert.equal(model.expire('connection', failed.seq), false);
    assert.equal(model.dismiss('connection'), false, 'not reported as dismissible');
    model.report('other', { state: 'success', text: text('Done') });
    assert.equal(model.lead().id, 'connection');
    model.report('connection', ready);
    assert.equal(model.get('connection').state, 'ready');
});

test('a dismissible error can be closed and settles', () => {
    const model = StatusIsland.createModel();
    model.report('backup', { state: 'error', text: text('Backup failed'), dismissible: true, settle: ready });
    assert.equal(model.dismiss('backup'), true);
    assert.equal(model.get('backup').state, 'ready');
    assert.equal(model.dismiss('backup'), false, 'only an error is dismissible');
});

test('dismissible is ignored on anything but an error', () => {
    const model = StatusIsland.createModel();
    assert.equal(model.report('x', { state: 'busy', text: text('Working'), dismissible: true }).dismissible, false);
});

test('a percentage is kept only when it was measured', () => {
    const progress = (value) => StatusIsland.createModel().report('x', { state: 'busy', text: text(''), progress: value }).progress;
    assert.equal(progress(0), 0);
    assert.equal(progress(42.5), 42.5);
    assert.equal(progress(100), 100);
    assert.equal(progress('unknown'), 'unknown');
    assert.equal(progress(NaN), 'unknown');
    assert.equal(progress(Infinity), 'unknown');
    assert.equal(progress(140), 'unknown');
    assert.equal(progress(-1), 'unknown');
    assert.equal(progress(undefined), null);
    assert.equal(progress(null), null);
});

test('only work in progress carries progress', () => {
    const model = StatusIsland.createModel();
    assert.equal(model.report('x', { state: 'success', text: text('Done'), progress: 100 }).progress, null);
    assert.equal(model.report('x', { state: 'error', text: text('Failed'), progress: 30 }).progress, null);
});

test('an unknown state is refused instead of shown as something else', () => {
    assert.throws(() => StatusIsland.createModel().report('x', { state: 'syncing', text: text('') }), /unknown state/);
});

test('clear removes a source', () => {
    const model = StatusIsland.createModel();
    model.report('upload-1', { state: 'busy', text: text('Importing') });
    assert.equal(model.clear('upload-1'), true);
    assert.equal(model.lead(), null);
});

// ---- What the dashboard reports for the live connection ----

function dashboard({ island, authenticated = true, requiresAuth = false } = {}) {
    const reports = [];
    const context = vm.createContext({
        console: { log() {}, error() {}, warn() {} },
        document: { hidden: false, addEventListener() {}, getElementById: () => null },
        window: { addEventListener() {} },
        setTimeout, clearTimeout, setInterval() { return 0; }, clearInterval() {},
    });
    vm.runInContext(fs.readFileSync(STATIC + 'app.js', 'utf8') + '\nthis.Dashboard = MemoryDashboard;', context);
    const app = Object.create(context.Dashboard.prototype);
    app.t = (key, fallback) => fallback;
    app.authState = { requiresAuth, isAuthenticated: authenticated };
    app.setupSSE = () => { app.reconnects = (app.reconnects || 0) + 1; };
    app.statusIsland = island === undefined ? { report: (id, entry) => reports.push([id, entry]) } : island;
    return { app, reports, context };
}

test('every connection state maps to an island state and an existing catalog key', () => {
    const { app, reports } = dashboard();
    for (const status of ['connected', 'connecting', 'disconnected', 'unauthorized', 'something-new']) app.updateConnectionStatus(status);
    assert.deepEqual(
        reports.map(([id, entry]) => [id, entry.state, entry.text.key]),
        [
            ['connection', 'ready', 'status.connected'],
            ['connection', 'busy', 'status.connecting'],
            ['connection', 'error', 'status.disconnected'],
            ['connection', 'error', 'error.authRequired'],
            ['connection', 'error', 'status.unknown'],
        ],
    );
    assert.equal(reports[1][1].progress, 'unknown', 'connecting has no measurable progress');
    assert.ok(reports.every(([, entry]) => entry.dismissible !== true), 'a connection failure cannot be clicked away');
});

test('a dropped stream offers a new attempt; a refused ticket does not', () => {
    const { app, reports } = dashboard();
    app.updateConnectionStatus('disconnected');
    app.updateConnectionStatus('unauthorized');
    const [dropped, refused] = reports.map(([, entry]) => entry);
    assert.equal(dropped.actions.length, 1);
    dropped.actions[0].run();
    assert.equal(app.reconnects, 1);
    assert.equal(refused.actions, undefined);
});

test('no reconnect is offered while signed out', () => {
    const { app, reports } = dashboard({ requiresAuth: true, authenticated: false });
    app.updateConnectionStatus('disconnected');
    assert.equal(reports[0][1].actions.length, 0);
});

test('the catalog keys the connection uses exist in en.json', () => {
    const catalog = JSON.parse(fs.readFileSync(STATIC + 'i18n/en.json', 'utf8'));
    const { app, reports } = dashboard();
    for (const status of ['connected', 'connecting', 'disconnected', 'unauthorized', 'x']) app.updateConnectionStatus(status);
    const keys = reports.flatMap(([, entry]) => [entry.text.key, ...(entry.actions || []).map((action) => action.label.key)]);
    assert.deepEqual(keys.filter((key) => !catalog[key]), []);
});

test('only a refused ticket says "authentication required"; a failed exchange can be retried', () => {
    // 401/403 and "network or server trouble after four attempts" used to share one status.
    const source = fs.readFileSync(STATIC + 'app.js', 'utf8');
    const exchange = source.slice(source.indexOf('async _openSSEWithTicket('), source.indexOf('updateConnectionStatus(status) {'));
    const statuses = [...exchange.matchAll(/updateConnectionStatus\('([a-z]+)'\)/g)].map((match) => match[1]);
    assert.deepEqual(statuses, ['unauthorized', 'disconnected']);
    assert.ok(exchange.indexOf("resp.status === 401") < exchange.indexOf("'unauthorized'"));
});

test('the dashboard works without the island', () => {
    const { app } = dashboard({ island: null });
    assert.doesNotThrow(() => app.updateConnectionStatus('connected'));
});

test('a missing script or a throwing constructor leaves the dashboard without an island, not broken', () => {
    const { app, context } = dashboard();
    context.document.getElementById = () => ({});
    app.setupStatusIsland();
    assert.equal(app.statusIsland, null, 'no StatusIsland global');

    context.StatusIsland = function () { throw new Error('boom'); };
    assert.doesNotThrow(() => app.setupStatusIsland());
    assert.equal(app.statusIsland, null);

    context.StatusIsland = function (options) { this.options = options; };
    app.setupStatusIsland();
    assert.equal(app.statusIsland.options.t('status.connected', 'Connected'), 'Connected');
});

// ---- What the dashboard reports for hybrid sync (#2002) ----

// A dashboard wired to a real island model, so the poll sees what the events reported.
function syncDashboard() {
    const { app } = dashboard();
    const model = StatusIsland.createModel();
    app.statusIsland = {
        report: (id, entry) => model.report(id, entry),
        clear: (id) => model.clear(id),
        get: (id) => model.get(id),
    };
    app.toasts = [];
    app.showToast = (message, type) => app.toasts.push([message, type]);
    app._scheduleDashboardRefresh = () => {};
    app.checkSyncStatus = async () => {};
    return { app, model };
}

const hybrid = (status, extra = {}) => ({ is_hybrid: true, is_running: true, is_paused: false, status, ...extra });

test('sync progress shows its percentage on the island, without a toast per event', () => {
    const { app, model } = syncDashboard();
    for (const progress_percentage of [10, 40]) {
        app.handleSyncProgress({ sync_type: 'manual', synced_count: 10, total_count: 100, progress_percentage });
    }
    const entry = model.get('sync');
    assert.equal(entry.state, 'busy');
    assert.equal(entry.progress, 40);
    assert.equal(entry.text.key, 'sync.status.syncing');
    assert.equal(entry.label.key, 'island.source.sync');
    assert.deepEqual(app.toasts, []);
});

test('a finished sync confirms on the island and steps back to "Synced"', () => {
    const { app, model } = syncDashboard();
    app.handleSyncProgress({ synced_count: 90, total_count: 100, progress_percentage: 90 });
    app.handleSyncCompleted({ synced_count: 100, total_count: 100, time_taken_seconds: 1 });
    const entry = model.get('sync');
    assert.equal(entry.state, 'success');
    assert.deepEqual([entry.settle.state, entry.settle.text.key], ['ready', 'sync.status.synced']);
    assert.ok(model.expire('sync', entry.seq));
    assert.equal(model.get('sync').state, 'ready');
    assert.equal(model.get('sync').label.key, 'island.source.sync', 'the settled row keeps its name');
    assert.deepEqual(app.toasts, [], 'the island is the confirmation; no toast says it again');
});

test('every polled sync status maps to an island state and a sync.status key', () => {
    const { app, model } = syncDashboard();
    const cases = [
        [hybrid('synced'), 'ready', 'sync.status.synced'],
        [hybrid('syncing'), 'busy', 'sync.status.syncing'],
        [hybrid('pending'), 'ready', 'sync.status.pending'],
        [hybrid('error'), 'error', 'sync.status.error'],
        [hybrid('synced', { is_paused: true }), 'ready', 'sync.status.paused'],
        [hybrid('syncing', { is_running: false }), 'ready', 'sync.status.paused'],
    ];
    for (const [status, state, key] of cases) {
        model.clear('sync');
        app._reportSyncStatus(status);
        assert.deepEqual([model.get('sync').state, model.get('sync').text.key], [state, key], JSON.stringify(status));
    }
    model.clear('sync');
    app._reportSyncStatus(hybrid('syncing'));
    assert.equal(model.get('sync').progress, 'unknown', 'the poll has no percentage');
});

test('a sync error stays on the island until sync reports again', () => {
    const { app, model } = syncDashboard();
    app._reportSyncStatus(hybrid('error'));
    assert.equal(model.dismiss('sync'), false, 'not something the reader clicks away');
    assert.equal(model.lead().id, 'sync');
    app._reportSyncStatus(hybrid('synced'));
    assert.equal(model.get('sync').state, 'ready');
});

// What GET /api/sync/status answers while a pull runs (SyncStatusResponse in
// web/api/sync.py). `status` is "syncing" only for `actively_syncing`, which the
// storage never reports, so a running pull reads "synced", or "pending" while
// writes wait in the queue.
const pollDuringPull = (status) => ({
    is_hybrid: true, is_running: true, is_paused: false, last_sync_time: 1700000000,
    operations_pending: status === 'pending' ? 3 : 0, operations_processed: 120, operations_failed: 0,
    consecutive_failures: status === 'error' ? 1 : 0, sync_interval_seconds: 300,
    time_since_last_sync_seconds: 12, next_sync_eta_seconds: 288, status,
    delta_sync_peers: [], delta_sync: {},
});

test('the poll does not erase a running sync, whatever it answers', () => {
    const { app, model } = syncDashboard();
    app.handleSyncProgress({ synced_count: 30, total_count: 100, progress_percentage: 30 });
    for (const status of ['synced', 'pending', 'syncing']) {
        app._reportSyncStatus(pollDuringPull(status));
        assert.deepEqual([model.get('sync').state, model.get('sync').progress], ['busy', 30], `poll answered ${status}`);
    }
    app._reportSyncStatus({ ...pollDuringPull('synced'), is_paused: true });
    assert.equal(model.get('sync').progress, 30, 'paused');

    app.handleSyncProgress({ synced_count: 60, total_count: 100, progress_percentage: 60 });
    assert.equal(model.get('sync').progress, 60, 'the next event still moves the bar');

    app._reportSyncStatus(pollDuringPull('error'));
    assert.equal(model.get('sync').state, 'error', 'but a failure is always shown');
});

test('the poll keeps the confirmation for its time', () => {
    const { app, model } = syncDashboard();
    app.handleSyncProgress({ synced_count: 90, total_count: 100, progress_percentage: 90 });
    app.handleSyncCompleted({ synced_count: 100, total_count: 100, time_taken_seconds: 1 });
    app._reportSyncStatus(pollDuringPull('synced'));
    assert.equal(model.get('sync').state, 'success');
    app._reportSyncStatus(pollDuringPull('error'));
    assert.equal(model.get('sync').state, 'error');
});

test('a sync whose events stopped a minute ago goes back to the poll', () => {
    // A dropped stream never delivers sync_completed; the bar must not stay forever.
    const { app, model } = syncDashboard();
    app.handleSyncProgress({ synced_count: 30, total_count: 100, progress_percentage: 30 });
    app._syncProgressAt -= 61000;
    app._reportSyncStatus(pollDuringPull('synced'));
    assert.equal(model.get('sync').state, 'ready');
    assert.equal(model.get('sync').text.key, 'sync.status.synced');
});

test('a backend without hybrid sync reports nothing, and a stale sync row goes', () => {
    const { app, model } = syncDashboard();
    app._reportSyncStatus({ is_hybrid: false, status: 'synced' });
    assert.equal(model.get('sync'), null);
    app._reportSyncStatus(hybrid('pending'));
    app._reportSyncStatus({ is_hybrid: false });
    assert.equal(model.get('sync'), null);
});

test('sync reporting works without the island', () => {
    const { app } = dashboard({ island: null });
    app.showToast = () => {};
    app._scheduleDashboardRefresh = () => {};
    app.checkSyncStatus = async () => {};
    assert.doesNotThrow(() => {
        app.handleSyncProgress({ progress_percentage: 10 });
        app.handleSyncCompleted({ synced_count: 1 });
        app._reportSyncStatus(hybrid('error'));
    });
});

test('the catalog keys the sync source uses exist in en.json, and #syncStatus is gone', () => {
    const catalog = JSON.parse(fs.readFileSync(STATIC + 'i18n/en.json', 'utf8'));
    const { app, model } = syncDashboard();
    const keys = new Set();
    const collect = () => { const entry = model.get('sync'); keys.add(entry.label.key).add(entry.text.key); };
    for (const status of ['synced', 'syncing', 'pending', 'error']) { model.clear('sync'); app._reportSyncStatus(hybrid(status)); collect(); }
    model.clear('sync'); app._reportSyncStatus(hybrid('synced', { is_paused: true })); collect();
    assert.deepEqual([...keys].filter((key) => !catalog[key]), []);
    assert.ok(!fs.readFileSync(STATIC + 'app.js', 'utf8').includes("getElementById('syncStatus')"));
});

// ---- What the dashboard reports for document imports (#2004) ----

// A dashboard on a real island model whose upload poll runs by hand: `apiCall` answers
// from a list, and `next()` runs the timer the poll set last.
function uploadDashboard() {
    const { app, context } = dashboard();
    const model = StatusIsland.createModel();
    app.statusIsland = {
        report: (id, entry) => model.report(id, entry),
        clear: (id) => model.clear(id),
        get: (id) => model.get(id),
    };
    const timers = [];
    context.setTimeout = (run, delay) => { timers.push([run, delay]); return timers.length; };
    const answers = {};
    app.apiCall = async (endpoint) => {
        const next = answers[endpoint].shift();
        if (next instanceof Error) throw next;
        return next;
    };
    app.historyLoads = 0;
    app.loadUploadHistory = () => { app.historyLoads += 1; };
    const next = async () => {
        const [run, delay] = timers.shift();
        await run();
        return delay;
    };
    const answer = (uploadId, ...list) => { answers[`/documents/status/${uploadId}`] = list; };
    return { app, model, timers, next, answer };
}

// What GET /api/documents/status/{upload_id} answers (UploadStatus in web/api/documents.py).
const upload = (status, extra = {}) => ({
    upload_id: 'u1', filename: 'notes.pdf', status, chunks_processed: 0, chunks_stored: 0,
    total_chunks: 0, progress: 0, errors: [], created_at: '2026-11-16T10:00:00', ...extra,
});

test('an import shows on the island under its file name, with a rising percentage', async () => {
    const { app, model, timers, next, answer } = uploadDashboard();
    answer('u1',
        upload('queued'),
        upload('processing', { progress: 25, chunks_processed: 3 }),
        upload('processing', { progress: 75, chunks_processed: 9 }),
        upload('completed', { progress: 100, chunks_processed: 12, chunks_stored: 12 }));
    app.monitorUploadProgress('u1');

    await next();
    let entry = model.get('upload-u1');
    assert.deepEqual([entry.state, entry.progress, entry.text.key], ['busy', 'unknown', 'island.upload.queued']);
    assert.equal(JSON.stringify(entry.label), JSON.stringify({ fallback: 'notes.pdf' }), 'a file name has no catalog key');

    await next();
    assert.deepEqual([model.get('upload-u1').state, model.get('upload-u1').progress], ['busy', 25]);
    assert.equal(model.get('upload-u1').text.key, 'island.upload.processing');
    await next();
    assert.equal(model.get('upload-u1').progress, 75);
    assert.equal(model.lead().id, 'upload-u1');

    await next();
    entry = model.get('upload-u1');
    assert.equal(entry.state, 'success');
    assert.equal(entry.text.key, 'island.upload.completed');
    assert.equal(entry.settle, null, 'no settle: the row leaves by itself');
    assert.ok(model.expire('upload-u1', entry.seq));
    assert.equal(model.get('upload-u1'), null);
    assert.equal(timers.length, 0, 'the poll stops after the end state');
    assert.equal(app.historyLoads, 1);
});

test('a batch at 100% that is not completed yet keeps being polled', async () => {
    // process_batch_upload sets progress to 100 after the last file, then the status.
    const { app, model, timers, next, answer } = uploadDashboard();
    answer('b1', upload('processing', { progress: 100 }), upload('completed', { progress: 100 }));
    app.monitorUploadProgress('b1');
    await next();
    assert.equal(model.get('upload-b1').progress, 100);
    assert.equal(timers.length, 1);
    await next();
    assert.equal(model.get('upload-b1').state, 'success');
});

test('a failed import stays on the island with its first error until it is closed', async () => {
    const { app, model, timers, next, answer } = uploadDashboard();
    answer('u2', upload('processing', { progress: 50 }),
        upload('failed', { errors: ['notes.pdf: No loader available', 'Processing error'] }));
    app.monitorUploadProgress('u2');
    await next();
    await next();
    const entry = model.get('upload-u2');
    assert.equal(entry.state, 'error');
    assert.equal(entry.dismissible, true);
    assert.equal(entry.text.fallback, 'notes.pdf: No loader available');
    assert.equal(entry.label.fallback, 'notes.pdf');
    assert.equal(timers.length, 0);
    assert.equal(model.expire('upload-u2', entry.seq), false, 'an error does not leave by itself');
    assert.equal(model.dismiss('upload-u2'), true);
    assert.equal(model.get('upload-u2'), null);
});

test('a completed import with errors stays on the island as a closable warning', async () => {
    // process_batch_upload() answers "completed" once any chunk was stored, even when
    // other files or chunks failed; that must not look like a clean import.
    const { app, model, timers, next, answer } = uploadDashboard();
    answer('p1', upload('completed', {
        progress: 100, chunks_processed: 12, chunks_stored: 9,
        errors: ['broken.docx: No loader available', 'Chunk 4: processing error'],
    }));
    app.monitorUploadProgress('p1');
    await next();
    const entry = model.get('upload-p1');
    assert.equal(entry.state, 'error');
    assert.equal(entry.dismissible, true);
    assert.equal(entry.text.key, 'island.upload.partial');
    assert.equal(entry.text.vars.error, 'broken.docx: No loader available');
    assert.equal(entry.label.fallback, 'notes.pdf');
    assert.equal(model.expire('upload-p1', entry.seq), false, 'it does not leave by itself');
    assert.equal(timers.length, 0, 'the poll still stops at the end state');
    assert.equal(model.dismiss('upload-p1'), true);
    assert.equal(model.get('upload-p1'), null);
});

test('a failed import without an error message uses the catalog text', () => {
    const { app, model } = uploadDashboard();
    app._reportUpload('u3', upload('failed', { errors: [] }));
    assert.equal(model.get('upload-u3').text.key, 'island.upload.failed');
});

test('two imports at once are two rows', async () => {
    const { app, model, next, answer } = uploadDashboard();
    answer('a', upload('processing', { filename: 'a.md', progress: 10 }));
    answer('b', upload('processing', { filename: 'b.txt', progress: 60 }));
    app.monitorUploadProgress('a');
    app.monitorUploadProgress('b');
    await next();
    await next();
    const rows = model.list().map((source) => [source.id, source.label.fallback, source.progress]);
    assert.deepEqual(rows.sort(), [['upload-a', 'a.md', 10], ['upload-b', 'b.txt', 60]]);
});

test('a failed poll keeps the last state and reports no error', async () => {
    const { app, model, timers, next, answer } = uploadDashboard();
    answer('u4', upload('processing', { progress: 40 }), new Error('network'), upload('processing', { progress: 80 }));
    app.monitorUploadProgress('u4');
    await next();
    await next();
    assert.deepEqual([model.get('upload-u4').state, model.get('upload-u4').progress], ['busy', 40]);
    assert.equal(timers[0][1], 5000, 'it tries again, later');
    await next();
    assert.equal(model.get('upload-u4').progress, 80);
});

test('import reporting works without the island', () => {
    const { app } = dashboard({ island: null });
    assert.doesNotThrow(() => app._reportUpload('u5', upload('failed', { errors: ['x'] })));
});

test('the catalog keys document imports use exist in en.json', () => {
    const catalog = JSON.parse(fs.readFileSync(STATIC + 'i18n/en.json', 'utf8'));
    const { app, model } = uploadDashboard();
    const keys = [];
    for (const status of ['queued', 'processing', 'completed', 'failed']) {
        app._reportUpload(status, upload(status));
        keys.push(model.get(`upload-${status}`).text.key);
    }
    app._reportUpload('partial', upload('completed', { errors: ['x'] }));
    keys.push(model.get('upload-partial').text.key);
    assert.deepEqual(keys, ['island.upload.queued', 'island.upload.processing', 'island.upload.completed', 'island.upload.failed', 'island.upload.partial']);
    assert.deepEqual(keys.filter((key) => !catalog[key]), []);
});

// ---- What the dashboard reports for a finished consolidation (#2006) ----

// A dashboard whose stream is a stub, so the scheduler's event goes through the real
// consolidation_completed listener in _connectSSE() and lands in a real island model.
function consolidationDashboard() {
    const { app, context } = dashboard();
    const model = StatusIsland.createModel();
    app.statusIsland = {
        report: (id, entry) => model.report(id, entry),
        get: (id) => model.get(id),
        clear: (id) => model.clear(id),
    };
    app.refreshes = 0;
    app._scheduleDashboardRefresh = () => { app.refreshes += 1; };
    context.EventSource = class {
        constructor() { this.listeners = {}; }
        addEventListener(type, listener) { this.listeners[type] = listener; }
    };
    app._connectSSE('http://localhost:8000/api/events');
    // What consolidate() broadcasts (#2011), as web/sse.py puts it on the wire.
    const send = (type, data) => app.eventSource.listeners[type]({
        data: JSON.stringify({ timestamp: '2026-11-16T03:00:00Z', ...data }),
    });
    const finish = (time_horizon, success, run_id) => send('consolidation_completed', { run_id, time_horizon, success });
    const start = (run_id, time_horizon) => send('consolidation_started', { run_id, time_horizon });
    const phase = (run_id, time_horizon, name) => send('consolidation_phase', { run_id, time_horizon, phase: name });
    // Reconnect: setupSSE() opens a new EventSource, without Last-Event-ID.
    const reconnect = () => {
        app._connectSSE('http://localhost:8000/api/events');
        app.eventSource.onopen();
    };
    return { app, model, finish, start, phase, reconnect };
}

test('a finished consolidation is confirmed on the island with its horizon, then goes', () => {
    const { app, model, finish } = consolidationDashboard();
    finish('weekly', true);
    const entry = model.get('consolidation');
    assert.equal(entry.state, 'success');
    assert.equal(entry.text.key, 'island.consolidation.completed');
    assert.equal(entry.text.vars.horizon, 'weekly');
    assert.equal(entry.label.key, 'island.source.consolidation');
    assert.equal(entry.settle, null);
    assert.ok(model.expire('consolidation', entry.seq));
    assert.equal(model.get('consolidation'), null, 'nothing is left behind once the confirmation goes');
    assert.equal(app.refreshes, 1, 'the numbers are still refreshed');
});

test('a failed consolidation stays on the island until the reader closes it', () => {
    const { app, model, finish } = consolidationDashboard();
    finish('monthly', false);
    const entry = model.get('consolidation');
    assert.equal(entry.state, 'error');
    assert.equal(entry.text.key, 'island.consolidation.failed');
    assert.equal(entry.text.vars.horizon, 'monthly');
    assert.equal(model.lead().id, 'consolidation');
    assert.equal(model.expire('consolidation', entry.seq), false, 'a failure does not step back by itself');
    assert.ok(model.dismiss('consolidation'));
    assert.equal(model.get('consolidation'), null);
    assert.equal(app.refreshes, 1, 'a failed run can still have changed the counts');
});

test('the next run replaces a failure that was not closed', () => {
    const { model, finish } = consolidationDashboard();
    finish('daily', false);
    finish('daily', true);
    assert.equal(model.get('consolidation').state, 'success');
});

test('a consolidation report works without the island', () => {
    const { app } = dashboard({ island: null });
    app._scheduleDashboardRefresh = () => {};
    assert.doesNotThrow(() => {
        app.handleConsolidationCompleted({ time_horizon: 'daily', success: true });
        app.handleConsolidationCompleted({ time_horizon: 'daily', success: false });
    });
});

test('the catalog keys a consolidation uses exist in every catalog', () => {
    const { model, finish, start, phase } = consolidationDashboard();
    const keys = new Set();
    for (const success of [true, false]) {
        finish('daily', success);
        const entry = model.get('consolidation');
        keys.add(entry.label.key).add(entry.text.key);
    }
    model.clear('consolidation');
    start('k', 'daily');
    keys.add(model.get('consolidation').text.key);
    for (const name of ['relevance', 'clustering', 'associations', 'compression', 'forgetting', 'finalizing']) {
        phase('k', 'daily', name);
        keys.add(model.get('consolidation').text.key);
    }
    assert.equal(keys.size, 10, 'label, the two ends, running and six phases');
    for (const file of fs.readdirSync(STATIC + 'i18n').filter((name) => name.endsWith('.json'))) {
        const catalog = JSON.parse(fs.readFileSync(STATIC + 'i18n/' + file, 'utf8'));
        assert.deepEqual([...keys].filter((key) => !catalog[key]), [], file);
        for (const key of [...keys].filter((key) => key !== 'island.source.consolidation')) {
            assert.ok(catalog[key].includes('{horizon}'), `${file}: ${key} names the horizon`);
        }
    }
});

// ---- A run while it is going (#2011) ----

test('a started run is busy without a percentage, then names each phase', () => {
    const { model, start, phase } = consolidationDashboard();
    start('r1', 'weekly');
    let entry = model.get('consolidation');
    assert.equal(entry.state, 'busy');
    assert.equal(entry.progress, 'unknown');
    assert.equal(entry.text.key, 'island.consolidation.running');
    assert.equal(entry.text.vars.horizon, 'weekly');
    phase('r1', 'weekly', 'clustering');
    entry = model.get('consolidation');
    assert.equal(entry.state, 'busy');
    assert.equal(entry.progress, 'unknown', 'a phase is not a percentage');
    assert.equal(entry.text.key, 'island.consolidation.phase.clustering');
    assert.equal(entry.text.vars.horizon, 'weekly');
});

test('a run stays busy through finalization, then turns into a confirmation', () => {
    const { app, model, start, phase, finish } = consolidationDashboard();
    start('r1', 'weekly');
    phase('r1', 'weekly', 'finalizing');
    assert.equal(model.get('consolidation').state, 'busy');
    assert.equal(model.get('consolidation').text.key, 'island.consolidation.phase.finalizing');
    finish('weekly', true, 'r1');
    assert.equal(model.get('consolidation').state, 'success');
    assert.equal(model.get('consolidation').text.key, 'island.consolidation.completed');
    assert.equal(app._consolidationRuns().size, 0);
});

test('a run that fails goes from busy to a failure that stays', () => {
    const { model, start, phase, finish } = consolidationDashboard();
    start('r1', 'monthly');
    phase('r1', 'monthly', 'forgetting');
    finish('monthly', false, 'r1');
    const entry = model.get('consolidation');
    assert.equal(entry.state, 'error');
    assert.equal(entry.text.key, 'island.consolidation.failed');
    assert.equal(entry.dismissible, true);
    assert.equal(entry.settle, null, 'nothing is running behind it');
});

test('the end of an older run does not clear a newer run that is still going', () => {
    const { app, model, start, phase, finish } = consolidationDashboard();
    start('old', 'daily');
    start('new', 'weekly');
    phase('new', 'weekly', 'associations');
    phase('old', 'daily', 'finalizing');
    assert.equal(model.get('consolidation').text.vars.horizon, 'weekly', 'the newest run is shown');
    finish('daily', true, 'old');
    let entry = model.get('consolidation');
    assert.equal(entry.state, 'busy');
    assert.equal(entry.text.key, 'island.consolidation.phase.associations');
    assert.equal(app.refreshes, 1, 'the numbers of the finished run are still refreshed');
    finish('weekly', true, 'new');
    entry = model.get('consolidation');
    assert.equal(entry.state, 'success');
    assert.equal(entry.text.vars.horizon, 'weekly');
});

test('an older run that fails is shown in front of the newer one, and closing it reveals the run', () => {
    const { model, start, phase, finish } = consolidationDashboard();
    start('old', 'daily');
    start('new', 'weekly');
    finish('daily', false, 'old');
    assert.equal(model.get('consolidation').state, 'error');
    phase('new', 'weekly', 'compression');
    assert.equal(model.get('consolidation').state, 'error', 'a phase does not hide a failure the reader has not closed');
    assert.ok(model.dismiss('consolidation'));
    const entry = model.get('consolidation');
    assert.equal(entry.state, 'busy');
    assert.equal(entry.text.key, 'island.consolidation.phase.compression');
    assert.equal(entry.text.vars.horizon, 'weekly');
});

test('a dashboard opened mid-run learns of the run from its next phase', () => {
    const { model, phase, finish } = consolidationDashboard();
    phase('r7', 'quarterly', 'compression');
    assert.equal(model.get('consolidation').state, 'busy');
    assert.equal(model.get('consolidation').text.vars.horizon, 'quarterly');
    finish('quarterly', true, 'r7');
    assert.equal(model.get('consolidation').state, 'success');
});

test('a run that ended while the stream was down does not stay busy after a reconnect', () => {
    const { app, model, start, phase, finish, reconnect } = consolidationDashboard();
    start('r1', 'weekly');
    phase('r1', 'weekly', 'clustering');
    // The stream drops here; r1's completion is sent while no stream is open.
    reconnect();
    assert.equal(app._consolidationRuns().size, 0);
    assert.equal(model.get('consolidation'), null, 'the stale busy entry is gone');
    start('r2', 'daily');
    finish('daily', true, 'r2');
    assert.equal(model.get('consolidation').state, 'success', 'a later run ends in its confirmation');
    assert.equal(model.get('consolidation').text.vars.horizon, 'daily');
});

test('a run still going after a reconnect comes back with its next phase', () => {
    const { model, start, phase, finish, reconnect } = consolidationDashboard();
    start('r1', 'monthly');
    reconnect();
    phase('r1', 'monthly', 'forgetting');
    assert.equal(model.get('consolidation').state, 'busy');
    assert.equal(model.get('consolidation').text.key, 'island.consolidation.phase.forgetting');
    finish('monthly', true, 'r1');
    assert.equal(model.get('consolidation').state, 'success');
});

test('a failure in front of a run no longer reveals that run after a reconnect', () => {
    const { model, start, finish, reconnect } = consolidationDashboard();
    start('new', 'weekly');
    start('old', 'daily');
    finish('daily', false, 'old');
    assert.ok(model.get('consolidation').settle);
    reconnect();
    const entry = model.get('consolidation');
    assert.equal(entry.state, 'error', 'the failure the reader has not closed stays');
    assert.equal(entry.settle, null);
    assert.ok(model.dismiss('consolidation'));
    assert.equal(model.get('consolidation'), null);
});

test("the browser's own retry of the same stream keeps the run", () => {
    const { app, model, start } = consolidationDashboard();
    app.eventSource.onopen();
    start('r1', 'weekly');
    app.eventSource.onopen();
    assert.equal(app._consolidationRuns().size, 1);
    assert.equal(model.get('consolidation').state, 'busy');
});

test('a phase name the dashboard does not know shows as running', () => {
    const { model, start, phase } = consolidationDashboard();
    start('r1', 'daily');
    phase('r1', 'daily', 'toString');
    assert.equal(model.get('consolidation').text.key, 'island.consolidation.running');
});

test('a run reported without the island does not throw', () => {
    const { app } = dashboard({ island: null });
    app._scheduleDashboardRefresh = () => {};
    assert.doesNotThrow(() => {
        app.handleConsolidationStarted({ run_id: 'r1', time_horizon: 'daily' });
        app.handleConsolidationPhase({ run_id: 'r1', time_horizon: 'daily', phase: 'clustering' });
        app.handleConsolidationCompleted({ run_id: 'r1', time_horizon: 'daily', success: true });
    });
});
