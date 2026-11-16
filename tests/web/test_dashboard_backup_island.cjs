// Run with: node --test tests/web/test_dashboard_backup_island.cjs
// createBackup() reports its run to the status island (#2005): busy while the request
// is in flight, a confirmation with name and size on success, and a dismissible
// failure with a retry action otherwise. The backup toasts are gone. Because the
// island lives in the header below the Settings modal, starting a backup from
// Settings closes the modal so the run stays visible.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function dashboard(apiCall) {
    const context = vm.createContext({
        console: { log() {}, warn() {}, error() {} },
        document: { getElementById: () => null, addEventListener() {} },
        window: { addEventListener() {} },
        localStorage: { getItem: () => null },
    });
    vm.runInContext(fs.readFileSync('src/mcp_memory_service/web/static/app.js', 'utf8') + '\nthis.Dashboard = MemoryDashboard;', context);
    context.Dashboard.prototype.init = () => {};
    const app = new context.Dashboard();
    const island = { reports: [], report(id, entry) { this.reports.push([id, entry]); }, get() { return null; }, clear() {} };
    app.statusIsland = island;
    app.showToast = (...args) => { throw new Error('a backup toast fired: ' + JSON.stringify(args)); };
    app.apiCall = apiCall;
    app.checkBackupStatus = async () => {};
    return { app, island };
}

test('a started backup is busy on the island without a percentage', async () => {
    let release;
    const { app, island } = dashboard(() => new Promise((resolve) => { release = resolve; }));
    const pending = app.createBackup();
    await new Promise((r) => setTimeout(r, 0));
    assert.equal(island.reports.length, 1);
    const [id, entry] = island.reports[0];
    assert.equal(id, 'backup');
    assert.equal(entry.state, 'busy');
    assert.equal(entry.progress, 'unknown');
    assert.equal(JSON.stringify(entry.label), JSON.stringify({ key: 'island.source.backup', fallback: 'Backup' }));
    assert.equal(entry.text.key, 'toast.backupCreating');
    release({ success: true, filename: 'b.db.gz', size_bytes: 1024 * 1024 });
    await pending;
});

test('a finished backup confirms with file name and size, without a settle entry', async () => {
    const { app, island } = dashboard(async () => ({ success: true, filename: 'backup-1.db.gz', size_bytes: 2.5 * 1024 * 1024 }));
    await app.createBackup();
    const [, entry] = island.reports.at(-1);
    assert.equal(entry.state, 'success');
    assert.equal(entry.settle, undefined);
    assert.equal(entry.text.key, 'toast.backupCreated');
    assert.equal(JSON.stringify(entry.text.vars), JSON.stringify({ name: 'backup-1.db.gz', size: '2.50' }));
});

test('a failed backup stays as a dismissible error with a retry action', async () => {
    const { app, island } = dashboard(async () => ({ success: false, error: 'disk full' }));
    await app.createBackup();
    const [, entry] = island.reports.at(-1);
    assert.equal(entry.state, 'error');
    assert.equal(entry.dismissible, true);
    assert.equal(entry.text.key, 'toast.backupFailedWithReason');
    assert.equal(JSON.stringify(entry.text.vars), JSON.stringify({ reason: 'disk full' }));
    assert.equal(entry.actions.length, 1);
    assert.equal(entry.actions[0].id, 'retry');
    assert.equal(JSON.stringify(entry.actions[0].label), JSON.stringify({ key: 'island.action.retry', fallback: 'Retry' }));
});

test('the retry action runs the backup again', async () => {
    let calls = 0;
    const { app, island } = dashboard(async () => { calls += 1; return { success: false, error: 'boom' }; });
    await app.createBackup();
    assert.equal(calls, 1);
    const [, entry] = island.reports.at(-1);
    await entry.actions[0].run();
    assert.equal(calls, 2);
});

test('a request exception reports the generic failure text, still dismissible with retry', async () => {
    const { app, island } = dashboard(async () => { throw new Error('network down'); });
    await app.createBackup();
    const [, entry] = island.reports.at(-1);
    assert.equal(entry.state, 'error');
    assert.equal(entry.text.key, 'toast.backupFailed');
    assert.equal(entry.dismissible, true);
    assert.equal(entry.actions[0].id, 'retry');
});

test('a backup started from Settings closes the modal so the island stays visible', async () => {
    const classes = new Set(['active']);
    const modal = { classList: { contains: (c) => classes.has(c), remove: (c) => classes.delete(c) } };
    const context = vm.createContext({
        console: { log() {}, warn() {}, error() {} },
        document: {
            getElementById: (id) => (id === 'settingsModal' ? modal : null),
            addEventListener() {},
            body: { style: {} },
        },
        window: { addEventListener() {} },
        localStorage: { getItem: () => null },
    });
    vm.runInContext(fs.readFileSync('src/mcp_memory_service/web/static/app.js', 'utf8') + '\nthis.Dashboard = MemoryDashboard;', context);
    context.Dashboard.prototype.init = () => {};
    const app = new context.Dashboard();
    const island = { reports: [], report(id, entry) { this.reports.push([id, entry]); }, get() { return null; }, clear() {} };
    app.statusIsland = island;
    app.apiCall = async () => ({ success: true, filename: 'b.db.gz', size_bytes: 1024 });
    app.checkBackupStatus = async () => {};

    await app.createBackup();
    assert.equal(classes.has('active'), false);
    assert.equal(island.reports.length, 2);
    assert.equal(island.reports[0][1].state, 'busy');
    assert.equal(island.reports.at(-1)[1].state, 'success');
});

test('without a status island the backup still runs and refreshes the settings modal', async () => {
    const { app } = dashboard(async () => ({ success: true, filename: 'b.db.gz', size_bytes: 1024 }));
    app.statusIsland = null;
    let refreshed = false;
    app.checkBackupStatus = async () => { refreshed = true; };
    await app.createBackup();
    assert.equal(refreshed, true);
});
