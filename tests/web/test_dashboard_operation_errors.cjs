// Run with node --test tests/web/test_dashboard_operation_errors.cjs.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const StatusIsland = require('../../src/mcp_memory_service/web/static/status-island.js');

function dashboard() {
    const elements = new Map();
    const context = vm.createContext({
        console: { log() {}, warn() {}, error() {} },
        document: {
            getElementById: (id) => elements.get(id) || null,
            addEventListener() {},
            createElement: () => ({ click() {} }),
            body: { appendChild() {}, removeChild() {} },
        },
        window: { addEventListener() {} },
        localStorage: { getItem: () => null },
        confirm: () => true,
        setTimeout: (fn) => { fn(); return 1; },
        clearTimeout() {},
        Blob, URL,
    });
    vm.runInContext(fs.readFileSync('src/mcp_memory_service/web/static/app.js', 'utf8') + '\nthis.Dashboard = MemoryDashboard;', context);
    context.Dashboard.prototype.init = () => {};
    const app = new context.Dashboard();
    const model = StatusIsland.createModel();
    app.statusIsland = model;
    const toasts = [];
    app.showToast = (...args) => toasts.push(args);
    app.setLoading = () => {};
    app.loadUploadHistory = async () => {};
    app.updateUploadButton = () => {};
    app.loadManageData = async () => {};
    app.loadDashboardData = async () => {};
    app.checkSyncStatus = async () => {};
    app.checkBackupStatus = async () => {};
    app.confirmBulkOperation = async () => true;
    app.currentView = 'documents';
    elements.set('forceSyncButton', { innerHTML: 'Sync', disabled: false });
    elements.set('deleteTagSelect', { value: 'work', querySelector: () => ({ dataset: { count: '2' } }) });
    elements.set('deleteDateInput', { value: '2026-01-01' });
    elements.set('untaggedCount', { dataset: { count: '2' } });
    return { app, model, toasts, elements, context };
}

const operations = [
    ['export', (a) => a.handleExportData(), null],
    ['sync-request', (a) => a.forceSync(), { success: false, message: 'sync rejected' }],
    ['server-update', (a) => a.updateAndRestart(), null],
    ['server-restart', (a) => a.restartServer(), null],
    ['document-remove-42', (a) => a.removeDocument('42', 'report.pdf'), { status: 'error' }],
    ['bulk-delete-tag', (a) => a.handleBulkDeleteByTag(), { success: false, message: 'delete rejected' }],
    ['cleanup-duplicates', (a) => a.handleCleanupDuplicates(), { success: false, message: 'cleanup rejected' }],
    ['bulk-delete-date', (a) => a.handleBulkDeleteByDate(), { success: false, message: 'date rejected' }],
    ['delete-untagged', (a) => a.handleDeleteUntagged(), { success: false, message: 'untagged rejected' }],
];

for (const [id, run, failure] of operations) {
    for (const kind of failure ? ['exception', 'rejected'] : ['exception']) {
        test(`${id}: ${kind} stays on the island without an error toast`, async () => {
            const { app, model, toasts } = dashboard();
            app.apiCall = async (path) => {
                if (path === '/sync/status') return { is_paused: false };
                if (kind === 'exception') throw new Error('network unavailable');
                return failure;
            };
            await run(app);
            const entry = model.get(id);
            assert.ok(entry, `${id} has no persistent error`);
            assert.equal(entry.state, 'error');
            assert.equal(entry.dismissible, true);
            assert.ok(entry.label);
            assert.ok(entry.text.key || entry.text.fallback);
            assert.equal(model.expire(id, entry.seq), false);
            assert.equal(toasts.filter(([, type]) => type === 'error').length, 0);
            assert.equal(model.dismiss(id), true);
            assert.equal(model.get(id), null);
        });
    }
}

test('failed individual uploads keep separate rows and continue to the next file', async () => {
    const { app, model, toasts } = dashboard();
    app.selectedFiles = [{ name: 'a.pdf' }, { name: 'b.pdf' }];
    app.processingMode = 'individual';
    let calls = 0;
    app.uploadSingleDocument = async () => { calls++; throw new Error('upload refused'); };
    await app.handleDocumentUpload();
    assert.equal(calls, 2);
    const errors = model.list().filter((e) => e.state === 'error');
    assert.equal(errors.length, 2);
    assert.deepEqual(errors.map((e) => e.label.fallback).sort(), ['a.pdf', 'b.pdf']);
    assert.equal(toasts.filter(([, type]) => type === 'error').length, 0);
});

test('batch submission errors stay visible', async () => {
    const { app, model } = dashboard();
    app.selectedFiles = [{ name: 'a.pdf' }, { name: 'b.pdf' }];
    app.processingMode = 'batch';
    app.uploadBatchDocuments = async () => { throw new Error('batch refused'); };
    await app.handleDocumentUpload();
    assert.equal(model.get('upload-batch').state, 'error');
});

test('input errors remain short toasts and create no island row', async () => {
    const { app, model, toasts } = dashboard();
    app.selectedFiles = [];
    await app.handleDocumentUpload();
    assert.equal(model.list().length, 0);
    assert.equal(toasts.length, 1);
    assert.equal(toasts[0][1], 'error');
    assert.notEqual(toasts[0][2], 0);
});

test('an export retry clears only its own error, preserving another failed operation', async () => {
    const { app, model } = dashboard();
    app.apiCall = async () => { throw new Error('offline'); };
    await app.handleExportData();
    await app.handleCleanupDuplicates();
    assert.equal(model.list().length, 2);
    app.apiCall = async () => ({ total: 0, memories: [] });
    await app.handleExportData();
    assert.equal(model.get('export'), null);
    assert.equal(model.get('cleanup-duplicates').state, 'error');
});

test('a subsequent sync status poll cannot erase a failed explicit sync request', async () => {
    const { app, model } = dashboard();
    app.apiCall = async (path) => path === '/sync/status' ? { is_paused: false } : { success: false, message: 'failed' };
    await app.forceSync();
    app._reportSyncStatus({ is_hybrid: true, is_running: true, status: 'synced' });
    assert.equal(model.get('sync-request').state, 'error');
});

for (const mode of ['missing', 'throwing', 'silent']) {
    test(`an unavailable island (${mode}) falls back to a persistent, dismissible toast`, async () => {
        const { app, toasts } = dashboard();
        app.statusIsland = mode === 'missing' ? null : {
            report() { if (mode === 'throwing') throw new Error('broken view'); },
            get() { return null; }, clear() {},
        };
        app.apiCall = async () => { throw new Error('export offline'); };
        await app.handleExportData();
        const errors = toasts.filter(([, type]) => type === 'error');
        assert.equal(errors.length, 1);
        assert.equal(errors[0][2], 0);
    });
}

test('backup failures also retain a fallback when the island is missing', async () => {
    const { app, toasts } = dashboard();
    app.statusIsland = null;
    app.apiCall = async () => ({ success: false, error: 'disk full' });
    await app.createBackup();
    assert.equal(toasts.filter(([, type, duration]) => type === 'error' && duration === 0).length, 1);
});

test('persistent fallback toasts do not schedule expiry but can still be clicked away', () => {
    const { app, context, elements } = dashboard();
    const calls = [];
    let click, removed = false;
    const toast = { addEventListener(event, fn) { if (event === 'click') click = fn; }, remove() { removed = true; } };
    context.document.createElement = () => toast;
    elements.set('toastContainer', { appendChild() {} });
    context.setTimeout = (...args) => calls.push(args);
    context.Dashboard.prototype.showToast.call(app, 'persistent failure', 'error', 0);
    assert.equal(calls.length, 0);
    click();
    assert.equal(removed, true);
});

test('a different file does not erase the previous failed file in the same selection slot', async () => {
    const { app, model } = dashboard();
    const first = { name: 'a.pdf', size: 12, lastModified: 100 };
    app.selectedFiles = [first];
    app.uploadSingleDocument = async () => { throw new Error('upload refused'); };
    await app.handleDocumentUpload();
    app.selectedFiles = [{ name: 'b.pdf', size: 20, lastModified: 101 }];
    app.uploadSingleDocument = async () => ({});
    await app.handleDocumentUpload();
    assert.equal(model.list().length, 1);
    assert.equal(model.list()[0].label.fallback, 'a.pdf');
    app.selectedFiles = [first];
    await app.handleDocumentUpload();
    assert.equal(model.list().length, 0);
});

test('retrying an operation removes its persistent fallback toast', async () => {
    const { app } = dashboard();
    app.statusIsland = null;
    let removed = 0;
    app.showToast = () => ({ remove() { removed++; } });
    app.apiCall = async () => { throw new Error('offline'); };
    await app.handleExportData();
    assert.equal(removed, 0);
    app.apiCall = async () => ({ total: 0, memories: [] });
    await app.handleExportData();
    assert.equal(removed, 1);
});

test('server errors reveal the island while keeping unsaved Settings preferences', async () => {
    const { app, model, elements } = dashboard();
    const classes = new Set(['active']);
    const modal = { classList: { contains: (c) => classes.has(c) } };
    elements.set('settingsModal', modal);
    elements.set('themeSelect', { value: 'dark' });
    app.closeModal = () => classes.delete('active');
    app.apiCall = async () => { throw new Error('update failed'); };
    await app.updateAndRestart();
    assert.equal(classes.has('active'), false);
    assert.equal(app._settingsDraft.themeSelect, 'dark');
    assert.equal(model.get('server-update').state, 'error');
});

for (const [id, run] of operations) {
    test(`${id}: an explicit successful retry removes its failure only`, async () => {
        const { app, model } = dashboard();
        app.apiCall = async (path) => {
            if (path === '/sync/status') return { is_paused: false };
            throw new Error('offline');
        };
        await run(app);
        assert.equal(model.get(id)?.state, 'error');
        model.report('other-operation', { state: 'error', dismissible: true, text: { fallback: 'Other failure' } });
        app.showRestartOverlay = () => {};
        app.apiCall = async (path) => path === '/sync/status' ? { is_paused: false } : {
            success: true, status: 'success', total: 0, memories: [],
            operations_synced: 3, time_taken_seconds: 1, memories_deleted: 2,
        };
        await run(app);
        assert.equal(model.get(id), null);
        assert.equal(model.get('other-operation').state, 'error');
    });
}

test('a successful backup retry removes the missing-island fallback too', async () => {
    const { app } = dashboard();
    app.statusIsland = null;
    let removed = 0;
    app.showToast = () => ({ remove() { removed++; } });
    app.apiCall = async () => ({ success: false, error: 'disk full' });
    await app.createBackup();
    assert.equal(removed, 0);
    app.apiCall = async () => ({ success: true, filename: 'backup.db.gz', size_bytes: 1024 });
    await app.createBackup();
    assert.equal(removed, 1);
});

for (const succeeds of [true, false]) {
    test(`batch retry clears only selected-file failures (success=${succeeds})`, async () => {
        const { app, model } = dashboard();
        const selected = { name: 'retry.pdf', size: 21, lastModified: 7 };
        const other = { name: 'later.pdf', size: 22, lastModified: 8 };
        const id = (file) => `upload-request-${JSON.stringify([file.name, file.size, file.lastModified])}`;
        app.selectedFiles = [selected, other];
        app.processingMode = 'individual';
        app.uploadSingleDocument = async () => { throw new Error('first attempt failed'); };
        await app.handleDocumentUpload();
        assert.equal(model.get(id(selected)).state, 'error');
        assert.equal(model.get(id(other)).state, 'error');
        app.selectedFiles = [selected, { name: 'new.pdf', size: 2, lastModified: 9 }];
        app.processingMode = 'batch';
        app.uploadBatchDocuments = async () => {
            assert.equal(model.get(id(selected)), null);
            assert.equal(model.get(id(other)).state, 'error');
            if (!succeeds) throw new Error('batch unavailable');
        };
        await app.handleDocumentUpload();
        assert.equal(model.get(id(selected)), null);
        assert.equal(model.get(id(other)).state, 'error');
        assert.equal(model.get('upload-batch')?.state, succeeds ? undefined : 'error');
    });
}

for (const choice of ['accept', 'decline', 'retry-fails']) {
    test(`dirty-tree update ${choice} keeps progress visible or shows its final error`, async () => {
        const { app, model, elements, context } = dashboard();
        const classes = new Set(['active']);
        elements.set('settingsModal', { classList: { contains: (c) => classes.has(c) } });
        elements.set('themeSelect', { value: 'dark' });
        elements.set('serverActionStatus', { style: {} });
        app.closeModal = () => classes.delete('active');
        let confirmations = 0, requests = 0, overlay = false;
        context.confirm = () => ++confirmations === 1 || choice !== 'decline';
        app.showRestartOverlay = () => { overlay = true; };
        app.apiCall = async (path, method, payload) => {
            assert.equal(classes.has('active'), true, 'update progress is hidden');
            assert.equal(elements.get('serverActionStatus').style.display, 'flex');
            assert.equal(payload.force, requests++ > 0);
            if (requests === 1) throw Object.assign(new Error('local changes'), { status: 409 });
            if (choice === 'retry-fails') throw new Error('retry unavailable');
            return {};
        };
        await app.updateAndRestart();
        assert.equal(requests, choice === 'decline' ? 1 : 2);
        if (choice === 'accept') {
            assert.equal(overlay, true);
            assert.equal(model.get('server-update'), null);
            assert.equal(classes.has('active'), true);
        } else {
            assert.equal(overlay, false);
            assert.equal(model.get('server-update').state, 'error');
            assert.equal(classes.has('active'), false);
            assert.equal(app._settingsDraft.themeSelect, 'dark');
        }
    });
}
