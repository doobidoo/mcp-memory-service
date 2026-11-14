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
