// Run with: node --test tests/web/test_dashboard_no_overlay_manage.cjs
// The long manage operations no longer cover the page with the loading overlay (#2009):
// removeDocument() and the four bulk operations report busy to the island on their
// existing sources, disable the triggering button while the request runs, and keep
// their failure on the island. View loads that the island does not report keep the
// overlay, asserted here through loadBrowseData().
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function dashboard(elements) {
    const context = vm.createContext({
        console: { log() {}, warn() {}, error() {} },
        confirm: () => true,
        document: {
            getElementById: (id) => elements[id] || null,
            addEventListener() {},
            querySelectorAll: () => [],
        },
        window: { addEventListener() {} },
        localStorage: { getItem: () => null },
    });
    vm.runInContext(fs.readFileSync('src/mcp_memory_service/web/static/app.js', 'utf8') + '\nthis.Dashboard = MemoryDashboard;', context);
    context.Dashboard.prototype.init = () => {};
    const app = new context.Dashboard();
    const island = {
        reports: [],
        cleared: [],
        report(id, entry) { this.reports.push([id, entry]); },
        get(id) { return this.reports.findLast(([source]) => source === id)?.[1] || null; },
        clear(id) { this.cleared.push(id); },
    };
    app.statusIsland = island;
    app.t = (key, fallback) => (fallback !== undefined ? fallback : key);
    app.showToast = () => {};
    app.loadManageData = async () => {};
    app.loadDashboardData = async () => {};
    app.loadUploadHistory = async () => {};
    app.confirmBulkOperation = async () => true;
    app.setLoading = () => { throw new Error('the page overlay was shown'); };
    return { app, island };
}

function button() {
    return { disabled: false };
}

function overlay() {
    const calls = [];
    return { calls, classList: { remove: (c) => calls.push(`show:${c}`), add: (c) => calls.push(`hide:${c}`) } };
}

test('removeDocument shows no overlay, disables the button and reports busy to the island', async () => {
    const removeBtn = button();
    const { app, island } = dashboard({
        'document-remove-btn': removeBtn,
    });
    let release;
    app.apiCall = () => new Promise((resolve) => { release = resolve; });
    const pending = app.removeDocument('u1', 'notes.pdf', removeBtn);
    await new Promise((r) => setTimeout(r, 0));
    assert.equal(removeBtn.disabled, true);
    assert.equal(island.reports.length, 1);
    assert.equal(island.reports[0][0], 'document-remove-u1');
    assert.equal(island.reports[0][1].state, 'busy');
    assert.equal(island.reports[0][1].text.key, 'island.document.removing');
    release({ status: 'success', memories_deleted: 3 });
    await pending;
    assert.equal(removeBtn.disabled, false);
    assert.ok(island.cleared.includes('document-remove-u1'));
});

test('a second removal of the same upload is ignored while the first is in flight', async () => {
    const removeBtn = button();
    const { app, island } = dashboard({ 'document-remove-btn': removeBtn });
    let calls = 0;
    let release;
    app.apiCall = () => {
        calls += 1;
        return new Promise((resolve) => { release = resolve; });
    };
    const first = app.removeDocument('u3', 'a.txt', removeBtn);
    await new Promise((r) => setTimeout(r, 0));
    // The dashboard re-renders on a language switch and wires a fresh, enabled button.
    // Not awaited: without the guard this second call would never settle.
    const clearsBefore = island.cleared.length;
    app.removeDocument('u3', 'a.txt', button());
    await new Promise((r) => setTimeout(r, 0));
    assert.equal(calls, 1);
    assert.equal(island.reports.filter(([id]) => id === 'document-remove-u3').length, 1);
    // The ignored click must not hide the running removal from the island.
    assert.equal(island.cleared.length, clearsBefore);
    release({ status: 'success', memories_deleted: 1 });
    await first;
    // Once the request finished, the same upload can be removed again.
    app.apiCall = async () => { calls += 1; return { status: 'success', memories_deleted: 1 }; };
    await app.removeDocument('u3', 'a.txt', button());
    assert.equal(calls, 2);
});

test('a failed removal re-enables the button that a re-render put on screen', async () => {
    const removeBtn = button();
    const { app, island } = dashboard({ 'document-remove-btn': removeBtn });
    let reject;
    app.apiCall = () => new Promise((_resolve, rejectFn) => { reject = rejectFn; });
    const pending = app.removeDocument('u4', 'a.txt', removeBtn);
    await new Promise((r) => setTimeout(r, 0));
    // A language switch re-renders the list and wires a fresh button for the upload
    // while the DELETE is still running.
    const rerendered = button();
    app._trackRemovalButton('u4', rerendered);
    assert.equal(rerendered.disabled, true);
    reject(new Error('network down'));
    await pending;
    assert.equal(rerendered.disabled, false);
    assert.equal(app._pendingRemovals.has('u4'), false);
    assert.equal(island.reports.at(-1)[1].state, 'error');
});

test('a failed removal stays on the island as a dismissible error and releases the button', async () => {
    const removeBtn = button();
    const { app, island } = dashboard({ 'document-remove-btn': removeBtn });
    app.apiCall = async () => ({ status: 'error' });
    await app.removeDocument('u2', 'a.txt', removeBtn);
    const [, entry] = island.reports.at(-1);
    assert.equal(entry.state, 'error');
    assert.equal(entry.dismissible, true);
    assert.equal(island.reports.at(-1)[0], 'document-remove-u2');
    assert.equal(removeBtn.disabled, false);
});

test('a removal that throws reports the error and releases the button', async () => {
    const removeBtn = button();
    const { app, island } = dashboard({ 'document-remove-btn': removeBtn });
    app.apiCall = async () => { throw new Error('network down'); };
    await app.removeDocument('u3', 'a.txt', removeBtn);
    const [, entry] = island.reports.at(-1);
    assert.equal(entry.state, 'error');
    assert.equal(removeBtn.disabled, false);
});

test('bulk delete by tag shows no overlay, disables its button and reports busy', async () => {
    const deleteByTagBtn = button();
    const select = { value: 'topic', querySelector: () => ({ dataset: { count: '5' } }) };
    const { app, island } = dashboard({ deleteByTagBtn, deleteTagSelect: select });
    let release;
    app.apiCall = () => new Promise((resolve) => { release = resolve; });
    const pending = app.handleBulkDeleteByTag();
    await new Promise((r) => setTimeout(r, 0));
    assert.equal(deleteByTagBtn.disabled, true);
    assert.equal(island.reports[0][0], 'bulk-delete-tag');
    assert.equal(island.reports[0][1].state, 'busy');
    assert.equal(island.reports[0][1].text.key, 'manage.bulk.deleting');
    release({ success: true, message: 'gone' });
    await pending;
    assert.equal(deleteByTagBtn.disabled, false);
    assert.ok(island.cleared.includes('bulk-delete-tag'));
});

test('a failed bulk delete by tag keeps the failure on the island and releases the button', async () => {
    const deleteByTagBtn = button();
    const select = { value: 'topic', querySelector: () => ({ dataset: { count: '5' } }) };
    const { app, island } = dashboard({ deleteByTagBtn, deleteTagSelect: select });
    app.apiCall = async () => ({ success: false, message: 'server said no' });
    await app.handleBulkDeleteByTag();
    const [, entry] = island.reports.at(-1);
    assert.equal(entry.state, 'error');
    assert.equal(entry.text.fallback, 'server said no');
    assert.equal(deleteByTagBtn.disabled, false);
});

test('cleanup duplicates shows no overlay, disables its button and reports busy', async () => {
    const cleanupDuplicatesBtn = button();
    const { app, island } = dashboard({ cleanupDuplicatesBtn });
    let release;
    app.apiCall = () => new Promise((resolve) => { release = resolve; });
    const pending = app.handleCleanupDuplicates();
    await new Promise((r) => setTimeout(r, 0));
    assert.equal(cleanupDuplicatesBtn.disabled, true);
    assert.equal(island.reports[0][0], 'cleanup-duplicates');
    assert.equal(island.reports[0][1].state, 'busy');
    release({ success: true, message: 'cleaned' });
    await pending;
    assert.equal(cleanupDuplicatesBtn.disabled, false);
    assert.ok(island.cleared.includes('cleanup-duplicates'));
});

test('bulk delete by date shows no overlay, disables its button and reports busy', async () => {
    const deleteByDateBtn = button();
    const { app, island } = dashboard({ deleteByDateBtn, deleteDateInput: { value: '2026-01-01' } });
    let release;
    app.apiCall = () => new Promise((resolve) => { release = resolve; });
    const pending = app.handleBulkDeleteByDate();
    await new Promise((r) => setTimeout(r, 0));
    assert.equal(deleteByDateBtn.disabled, true);
    assert.equal(island.reports[0][0], 'bulk-delete-date');
    assert.equal(island.reports[0][1].state, 'busy');
    release({ success: true, message: 'gone' });
    await pending;
    assert.equal(deleteByDateBtn.disabled, false);
    assert.ok(island.cleared.includes('bulk-delete-date'));
});

test('delete untagged shows no overlay, disables its button and reports busy', async () => {
    const deleteUntaggedBtn = button();
    const { app, island } = dashboard({ deleteUntaggedBtn, untaggedCount: { dataset: { count: '4' } } });
    let release;
    app.apiCall = () => new Promise((resolve) => { release = resolve; });
    const pending = app.handleDeleteUntagged();
    await new Promise((r) => setTimeout(r, 0));
    assert.equal(deleteUntaggedBtn.disabled, true);
    assert.equal(island.reports[0][0], 'delete-untagged');
    assert.equal(island.reports[0][1].state, 'busy');
    release({ success: true, message: 'gone' });
    await pending;
    assert.equal(deleteUntaggedBtn.disabled, false);
    assert.ok(island.cleared.includes('delete-untagged'));
});

test('a view load the island does not report still covers the page', async () => {
    const indicator = overlay();
    const { app } = dashboard({ loadingOverlay: indicator });
    delete app.setLoading; // restore the real prototype method
    let release;
    app.apiCall = () => new Promise((resolve) => { release = resolve; });
    const pending = app.loadBrowseData();
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(indicator.calls.some((c) => c.startsWith('show:')));
    release({});
    await pending;
    assert.ok(indicator.calls.some((c) => c.startsWith('hide:')));
});
