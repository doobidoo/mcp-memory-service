// Run with: node --test tests/web/test_dashboard_backup_island.cjs
// createBackup() reports its run to the status island (#2005): busy while the request
// is in flight, a confirmation with name and size on success, and a dismissible
// failure with a retry action otherwise. The backup toasts are gone. Because the
// island lives in the header below the Settings modal, starting a backup from
// Settings closes the modal so the run stays visible; unsaved preference edits
// are stashed at that point and restored when Settings reopens.
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
    const island = { reports: [], report(id, entry) { this.reports.push([id, entry]); }, get(id) { return this.reports.findLast(([source]) => source === id)?.[1] || null; }, clear() {} };
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
    const island = { reports: [], report(id, entry) { this.reports.push([id, entry]); }, get(id) { return this.reports.findLast(([source]) => source === id)?.[1] || null; }, clear() {} };
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

function settingsDashboard(fields, modalActive, apiCall) {
    const classes = new Set(modalActive ? ['active'] : []);
    const modal = { classList: { contains: (c) => classes.has(c), remove: (c) => classes.delete(c) } };
    const context = vm.createContext({
        console: { log() {}, warn() {}, error() {} },
        document: {
            getElementById: (id) => (id === 'settingsModal' ? modal : fields[id] || null),
            addEventListener() {},
            body: { style: {} },
        },
        window: { addEventListener() {} },
        localStorage: { getItem: () => null },
    });
    vm.runInContext(fs.readFileSync('src/mcp_memory_service/web/static/app.js', 'utf8') + '\nthis.Dashboard = MemoryDashboard;', context);
    context.Dashboard.prototype.init = () => {};
    const app = new context.Dashboard();
    const island = { reports: [], report(id, entry) { this.reports.push([id, entry]); }, get(id) { return this.reports.findLast(([source]) => source === id)?.[1] || null; }, clear() {} };
    app.statusIsland = island;
    app.apiCall = apiCall;
    app.checkBackupStatus = async () => {};
    app.resetSystemInfoLoadingState = () => {};
    app.loadSystemInfo = async () => {};
    app.openModal = () => {};
    return { app, island, modal, classes };
}

test('a backup closing Settings preserves unsaved preference edits for the next open', async () => {
    const fields = {
        themeSelect: { value: 'light' },
        viewDensity: { value: 'compact' },
        previewLines: { value: '7' },
    };
    const { app, classes } = settingsDashboard(fields, true, async () => ({ success: true, filename: 'b.db.gz', size_bytes: 1024 }));
    app.settings = { theme: 'dark', viewDensity: 'comfortable', previewLines: 3 };

    await app.createBackup();
    assert.equal(classes.has('active'), false);

    // Reopening Settings used to overwrite the fields from this.settings; the
    // stashed draft is restored instead, and nothing is persisted to settings.
    await app.openSettingsModal();
    assert.equal(fields.themeSelect.value, 'light');
    assert.equal(fields.viewDensity.value, 'compact');
    assert.equal(fields.previewLines.value, '7');
    assert.equal(JSON.stringify(app.settings), JSON.stringify({ theme: 'dark', viewDensity: 'comfortable', previewLines: 3 }));
});

test('without a stashed draft, Settings still opens with the saved preferences', async () => {
    const fields = {
        themeSelect: { value: '' },
        viewDensity: { value: '' },
        previewLines: { value: '' },
    };
    const { app } = settingsDashboard(fields, false, async () => ({ success: true }));
    app.settings = { theme: 'dark', viewDensity: 'comfortable', previewLines: 3 };

    await app.openSettingsModal();
    assert.equal(fields.themeSelect.value, 'dark');
    assert.equal(fields.viewDensity.value, 'comfortable');
    assert.equal(fields.previewLines.value, 3);
});

test('an untouched preference is not stashed, so a newer saved choice survives', async () => {
    // Every field still shows the saved value: nothing counts as an edit, so the
    // backup stashes no draft at all. A theme change saved while the backup runs
    // (the header toggle) must still be there when Settings reopens - restoring
    // the stale value would silently undo it on the next save.
    const fields = {
        themeSelect: { value: 'dark' },
        viewDensity: { value: 'comfortable' },
        previewLines: { value: '3' },
    };
    const { app, classes } = settingsDashboard(fields, true, async () => ({ success: true, filename: 'b.db.gz', size_bytes: 1024 }));
    app.settings = { theme: 'dark', viewDensity: 'comfortable', previewLines: 3 };

    await app.createBackup();
    assert.equal(classes.has('active'), false);
    assert.equal(app._settingsDraft, null);

    app.settings.theme = 'light';
    await app.openSettingsModal();
    assert.equal(fields.themeSelect.value, 'light');
    assert.equal(fields.viewDensity.value, 'comfortable');
    assert.equal(fields.previewLines.value, 3);
});

test('a backup stashes only the edited preference fields', async () => {
    // Only viewDensity was edited; the other two fields match the saved
    // preferences and stay out of the draft, so a newer saved theme survives
    // while the genuine edit is still restored.
    const fields = {
        themeSelect: { value: 'dark' },
        viewDensity: { value: 'compact' },
        previewLines: { value: '3' },
    };
    const { app, classes } = settingsDashboard(fields, true, async () => ({ success: true, filename: 'b.db.gz', size_bytes: 1024 }));
    app.settings = { theme: 'dark', viewDensity: 'comfortable', previewLines: 3 };

    await app.createBackup();
    assert.equal(classes.has('active'), false);
    assert.equal(JSON.stringify(app._settingsDraft), JSON.stringify({ viewDensity: 'compact' }));

    app.settings.theme = 'light';
    await app.openSettingsModal();
    assert.equal(fields.themeSelect.value, 'light');
    assert.equal(fields.viewDensity.value, 'compact');
    assert.equal(fields.previewLines.value, 3);
    assert.equal(JSON.stringify(app.settings), JSON.stringify({ theme: 'light', viewDensity: 'comfortable', previewLines: 3 }));
});

test('overlapping Settings opens keep the stashed draft', async () => {
    // The modal opens only after a loading wait; a second open during that wait
    // repopulates the form from this.settings. The draft must still be there to
    // re-apply, so the restored edits survive the double open.
    const fields = {
        themeSelect: { value: 'light' },
        viewDensity: { value: 'compact' },
        previewLines: { value: '7' },
    };
    const { app } = settingsDashboard(fields, true, async () => ({ success: true, filename: 'b.db.gz', size_bytes: 1024 }));
    app.settings = { theme: 'dark', viewDensity: 'comfortable', previewLines: 3 };

    await app.createBackup();

    const releases = [];
    app.loadSystemInfo = () => new Promise((resolve) => { releases.push(resolve); });
    const first = app.openSettingsModal();
    await new Promise((r) => setTimeout(r, 0));
    const second = app.openSettingsModal();
    await new Promise((r) => setTimeout(r, 0));
    releases.forEach((resolve) => resolve());
    await first;
    await second;

    assert.equal(fields.themeSelect.value, 'light');
    assert.equal(fields.viewDensity.value, 'compact');
    assert.equal(fields.previewLines.value, '7');
    assert.equal(app._settingsDraft, null);
});
