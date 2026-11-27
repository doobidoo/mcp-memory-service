// Run with: node --test tests/web/test_dashboard_sync_island_actions.cjs
// #2007: pause, resume and "sync now" are actions of the island's sync row, and
// the compact dashboard control is gone. The action mapping is tested on a real
// island model, the harness is the one test_status_island.cjs uses.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const STATIC = 'src/mcp_memory_service/web/static/';
const StatusIsland = require('../../' + STATIC + 'status-island.js');

function dashboard() {
    const context = vm.createContext({
        console: { log() {}, error() {}, warn() {} },
        document: { hidden: false, addEventListener() {}, getElementById: () => null },
        window: { addEventListener() {} },
        setTimeout, clearTimeout, setInterval() { return 0; }, clearInterval() {},
    });
    vm.runInContext(fs.readFileSync(STATIC + 'app.js', 'utf8') + '\nthis.Dashboard = MemoryDashboard;', context);
    const app = Object.create(context.Dashboard.prototype);
    app.t = (key, fallback) => fallback;
    app.authState = { requiresAuth: false, isAuthenticated: true };
    app.setupSSE = () => {};
    app.statusIsland = null;
    return { app, context };
}

// A dashboard on a real island model whose network calls answer from a list.
function syncDashboard() {
    const { app } = dashboard();
    const model = StatusIsland.createModel();
    const reports = [];
    app.statusIsland = {
        report: (id, entry) => { reports.push([id, entry]); return model.report(id, entry); },
        clear: (id) => model.clear(id),
        get: (id) => model.get(id),
    };
    app.toasts = [];
    app.showToast = (message, type) => app.toasts.push([message, type]);
    app._scheduleDashboardRefresh = () => {};
    app.checkSyncStatus = async () => {};
    const answers = {};
    app.apiCall = async (endpoint) => {
        const next = answers[endpoint].shift();
        if (next instanceof Error) throw next;
        return next;
    };
    return { app, model, reports, answers };
}

const hybrid = (status, extra = {}) => ({ is_hybrid: true, is_running: true, is_paused: false, status, ...extra });
// The actions array is built inside the vm context, so comparing it against a
// literal trips deepStrictEqual's realm check; compare joined ids instead.
const actionIds = (entry) => entry.actions.map((action) => action.id).join(',');

test('the sync row offers pause while running, resume while paused, sync now otherwise', () => {
    const { app, model } = syncDashboard();
    const cases = [
        [hybrid('syncing'), 'pause', 'no resume while running'],
        [hybrid('synced', { is_paused: true }), 'resume', 'no pause while paused'],
        [hybrid('syncing', { is_running: false }), 'resume', 'a stopped sync counts as paused'],
        [hybrid('synced'), 'sync-now', ''],
        [hybrid('pending'), 'pause', 'queued uploads keep Pause'],
        [hybrid('error'), 'sync-now', ''],
    ];
    for (const [status, ids, message] of cases) {
        model.clear('sync');
        app._reportSyncStatus(status);
        assert.equal(actionIds(model.get('sync')), ids, message);
    }
});

test('the row actions call the dashboard methods', () => {
    const { app, model } = syncDashboard();
    const calls = [];
    app.pauseSync = () => calls.push('pause');
    app.resumeSync = () => calls.push('resume');
    app.forceSync = () => calls.push('force');
    app._reportSyncStatus(hybrid('syncing'));
    model.get('sync').actions[0].run();
    app._reportSyncStatus(hybrid('synced', { is_paused: true }));
    model.get('sync').actions[0].run();
    app._reportSyncStatus(hybrid('pending'));
    model.get('sync').actions[0].run();
    assert.deepEqual(calls, ['pause', 'resume', 'pause']);
});

test('the action labels are catalog keys that exist in every catalog', () => {
    const { app, model } = syncDashboard();
    const keys = new Set();
    for (const status of [hybrid('syncing'), hybrid('synced', { is_paused: true }), hybrid('synced'), hybrid('pending'), hybrid('error')]) {
        model.clear('sync');
        app._reportSyncStatus(status);
        for (const action of model.get('sync').actions) keys.add(action.label.key);
    }
    assert.deepEqual([...keys].sort(), ['sync.nowTitle', 'sync.pauseTitle', 'sync.resumeTitle']);
    for (const file of fs.readdirSync(STATIC + 'i18n').filter((name) => name.endsWith('.json'))) {
        const catalog = JSON.parse(fs.readFileSync(STATIC + 'i18n/' + file, 'utf8'));
        assert.deepEqual([...keys].filter((key) => !catalog[key]), [], file);
    }
});

test('a running sync keeps its pause action through SSE progress events', () => {
    const { app, model } = syncDashboard();
    // The poll ran first and cached the status, as it does in the real flow.
    app._lastSyncStatus = hybrid('syncing');
    app._reportSyncStatus(hybrid('syncing'));
    assert.equal(actionIds(model.get('sync')), 'pause');
    app.handleSyncProgress({ synced_count: 30, total_count: 100, progress_percentage: 30 });
    assert.deepEqual([model.get('sync').state, model.get('sync').progress], ['busy', 30]);
    assert.equal(actionIds(model.get('sync')), 'pause', 'the progress event does not strip the action');
});

test('a paused or non-hybrid node gets no pause from a stray progress event', () => {
    const { app, model } = syncDashboard();
    app._lastSyncStatus = hybrid('syncing', { is_paused: true });
    app.handleSyncProgress({ progress_percentage: 10 });
    assert.equal(actionIds(model.get('sync')), '', 'no pause while paused');
    app._lastSyncStatus = { is_hybrid: false, is_running: true, is_paused: false, status: 'synced' };
    app.handleSyncProgress({ progress_percentage: 20 });
    assert.equal(actionIds(model.get('sync')), '', 'and none without hybrid sync');
});

test('an explicit pause replaces a running row and exposes Resume', () => {
    const { app, model } = syncDashboard();
    app._lastSyncStatus = hybrid('syncing');
    app._reportSyncStatus(hybrid('syncing'));
    app.handleSyncProgress({ progress_percentage: 40 });
    assert.equal(model.get('sync').state, 'busy');
    app._reportSyncStatus(hybrid('syncing', { is_paused: true }));
    assert.equal(model.get('sync').state, 'ready');
    assert.equal(model.get('sync').text.key, 'sync.status.paused');
    assert.equal(actionIds(model.get('sync')), 'resume');
});

test('a backend without hybrid sync still reports nothing', () => {
    const { app, model } = syncDashboard();
    app._reportSyncStatus({ is_hybrid: false, status: 'synced' });
    assert.equal(model.get('sync'), null, 'no row, so no actions either');
});

// ---- The removed compact control's duties ----

test('force sync reports the run at once and settles from the poll afterwards', async () => {
    const { app, model, reports, answers } = syncDashboard();
    // The real poll settles the row: undo the harness stub.
    delete app.checkSyncStatus;
    answers['/sync/status'] = [hybrid('pending'), hybrid('synced')];
    answers['/sync/force'] = [{ success: true, operations_synced: 3, time_taken_seconds: 2 }];
    app.currentView = 'dashboard';
    app.loadDashboardData = async () => {};
    await app.forceSync();
    const busy = reports.filter(([id, entry]) => id === 'sync' && entry.state === 'busy');
    assert.ok(busy.length >= 1, 'the run is on the island while it goes');
    assert.equal(busy[0][1].progress, 'unknown');
    assert.equal(busy[0][1].actions[0].id, 'pause', 'the initial report keeps Pause');
    assert.deepEqual([model.get('sync').state, model.get('sync').text.key], ['ready', 'sync.status.synced'],
        'the poll that follows the run settles the row');
});

test('force sync reports a failure on the island, without any card button', async () => {
    const { app, model, answers } = syncDashboard();
    answers['/sync/status'] = [hybrid('pending')];
    answers['/sync/force'] = [{ success: false, message: 'nope' }];
    app.currentView = 'browse';
    await app.forceSync();
    assert.equal(model.get('sync-request').state, 'error');
    // The island is the failure message; only the start toast goes beside it.
    assert.equal(JSON.stringify(app.toasts), JSON.stringify([['Starting sync...', 'info']]));
});

test('pause and resume work without the removed card elements', async () => {
    const { app, answers } = syncDashboard();
    answers['/sync/pause'] = [{ success: true }];
    await app.pauseSync();
    assert.deepEqual(app.toasts, [['Sync paused', 'success']]);
    answers['/sync/resume'] = [{ success: true }];
    await app.resumeSync();
    assert.deepEqual(app.toasts[1], ['Sync resumed', 'success']);
});

test('an explicit pause refreshes the row while a force sync is in flight', async () => {
    const { app, model, answers } = syncDashboard();
    app.updateDeltaSyncStatus = () => {};
    app._isForceSyncing = true;
    answers['/sync/status'] = [hybrid('syncing', { is_paused: true })];
    await Object.getPrototypeOf(app).checkSyncStatus.call(app, true);
    assert.equal(model.get('sync').state, 'ready');
    assert.equal(model.get('sync').text.key, 'sync.status.paused');
    assert.equal(actionIds(model.get('sync')), 'resume');
});

test('the compact control is gone from the markup and the code; the delta panel stays', () => {
    const html = fs.readFileSync(STATIC + 'index.html', 'utf8');
    for (const gone of ['id="syncControl"', 'pauseSyncButton', 'resumeSyncButton', 'forceSyncButton',
        'syncStatusText', 'syncStatusDot', 'id="syncProgress"']) {
        assert.ok(!html.includes(gone), `${gone} left in index.html`);
    }
    assert.ok(html.includes('deltaSyncControl'), 'the delta sync panel stays');
    const source = fs.readFileSync(STATIC + 'app.js', 'utf8');
    for (const gone of ["getElementById('syncControl')", "getElementById('syncStatusText')",
        "getElementById('pauseSyncButton')", "getElementById('resumeSyncButton')",
        "getElementById('forceSyncButton')"]) {
        assert.ok(!source.includes(gone), `${gone} still looked up in app.js`);
    }
});
