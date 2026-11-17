// Run with: node --test tests/web/test_dashboard_upload_no_overlay.cjs
// handleDocumentUpload() no longer covers the page with the loading overlay (#2009):
// the island reports the transfer, the upload button is disabled while it runs, and a
// second trigger is ignored.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function dashboard() {
    const button = { disabled: false, textContent: '' };
    const context = vm.createContext({
        console: { log() {}, warn() {}, error() {} },
        document: { getElementById: (id) => (id === 'uploadBtn' ? button : null), addEventListener() {}, querySelectorAll: () => [] },
        window: { addEventListener() {} },
        localStorage: { getItem: () => null },
    });
    vm.runInContext(fs.readFileSync('src/mcp_memory_service/web/static/app.js', 'utf8') + '\nthis.Dashboard = MemoryDashboard;', context);
    context.Dashboard.prototype.init = () => {};
    const app = new context.Dashboard();
    const island = { reports: [], cleared: [], report(id, entry) { this.reports.push([id, entry]); }, clear(id) { this.cleared.push(id); } };
    app.statusIsland = island;
    app.selectedFiles = [{ name: 'a.txt', size: 1, lastModified: 1 }];
    app.processingMode = 'individual';
    app.loadUploadHistory = async () => {};
    app.setupDocumentsEventListeners = () => {};
    app.setLoading = () => { throw new Error('the page overlay was shown'); };
    return { app, island, button };
}

test('an upload shows no overlay, disables the button and reports busy to the island', async () => {
    const { app, island, button } = dashboard();
    let release;
    let calls = 0;
    app.uploadSingleDocument = () => { calls += 1; return new Promise((resolve) => { release = resolve; }); };
    const first = app.handleDocumentUpload();
    await new Promise((r) => setTimeout(r, 0));
    assert.equal(button.disabled, true);
    app.updateUploadButton(); // e.g. a chunk slider change mid-upload
    assert.equal(button.disabled, true);
    assert.equal(island.reports[0][0], 'upload-batch');
    assert.equal(island.reports[0][1].state, 'busy');
    await app.handleDocumentUpload(); // second trigger while running
    assert.equal(calls, 1);
    release();
    await first;
    assert.ok(island.cleared.includes('upload-batch'));
    assert.equal(app._uploading, false);
});

test('a failed upload reports the file error and releases the guard', async () => {
    const { app } = dashboard();
    const errors = [];
    app.uploadSingleDocument = async () => { throw new Error('boom'); };
    app._reportOperationError = (source) => errors.push(source);
    await app.handleDocumentUpload();
    assert.equal(errors.length, 1);
    assert.equal(app._uploading, false);
});
