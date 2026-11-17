// Run with: node --test tests/web/test_dashboard_memory_edit.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function dashboard(fields, editingMemory) {
    const context = vm.createContext({
        console: { log() {}, error() {} },
        document: { getElementById: id => fields[id], addEventListener() {} },
        navigator: { userAgent: 'node-test' },
        window: { addEventListener() {} },
    });
    vm.runInContext(
        fs.readFileSync('src/mcp_memory_service/web/static/app.js', 'utf8') + '\nthis.Dashboard = MemoryDashboard;',
        context,
    );
    const app = Object.create(context.Dashboard.prototype);
    const calls = [];
    const events = [];
    app.editingMemory = editingMemory;
    app.currentView = '';
    app.apiCall = async (path, method, payload) => {
        calls.push({ path, method, payload });
        return { success: true, content_hash: 'updated-hash' };
    };
    app.t = (_key, fallback, vars) => fallback.replace('{reason}', vars?.reason ?? '');
    app.showToast = (message, kind) => events.push({ message, kind });
    app.closeModal = () => events.push({ kind: 'close' });
    app.resetAddMemoryModal = () => events.push({ kind: 'reset' });
    return { app, calls, events };
}

function fields(content, tags = 'memory, test', type = 'note') {
    return {
        memoryContent: { value: content },
        memoryTags: { value: tags },
        memoryType: { value: type },
    };
}

test('content edits use a versioned PUT for the original memory', async () => {
    const original = {
        content: 'A copper kite hangs beside the workshop window.',
        content_hash: 'original-hash',
    };
    const { app, calls } = dashboard(fields(original.content + '!'), original);

    await app.handleSaveMemory();

    assert.equal(calls.length, 1);
    assert.equal(calls[0].method, 'PUT');
    assert.equal(calls[0].path, '/memories/original-hash');
    assert.equal(calls[0].payload.content, original.content + '!');
    assert.equal(calls[0].payload.metadata.updated_via, 'dashboard_edit');
});

test('tag-only edits ignore outer whitespace and do not create a new version', async () => {
    const original = {
        content: 'A copper kite hangs beside the workshop window.\n',
        content_hash: 'original-hash',
    };
    const { app, calls } = dashboard(fields(original.content, 'memory, edited'), original);

    await app.handleSaveMemory();

    assert.equal(calls.length, 1);
    assert.equal(calls[0].method, 'PUT');
    assert.equal(calls[0].path, '/memories/original-hash');
    assert.equal('content' in calls[0].payload, false);
    assert.equal(JSON.stringify(calls[0].payload.tags), JSON.stringify(['memory', 'edited']));
});

test('a rejected update keeps the editor open and shows the error', async () => {
    const original = { content: 'Original memory', content_hash: 'original-hash' };
    const { app, events } = dashboard(fields('Edited memory'), original);
    app.apiCall = async () => ({ success: false, message: 'Memory already exists' });

    await app.handleSaveMemory();

    assert.equal(app.editingMemory, original);
    assert.equal(events.some(event => event.kind === 'close' || event.kind === 'reset'), false);
    assert(events.some(event => event.kind === 'error' && event.message.includes('Memory already exists')));
});

test('new memories still use the create endpoint', async () => {
    const { app, calls } = dashboard(fields('A new memory'), null);

    await app.handleSaveMemory();

    assert.equal(calls.length, 1);
    assert.equal(calls[0].method, 'POST');
    assert.equal(calls[0].path, '/memories');
});
