const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function dashboard() {
    const parts = {};
    const container = { innerHTML: '', querySelector: key => parts[key] ??= { innerHTML: '', value: '' } };
    const context = vm.createContext({
        console: { log() {} },
        document: { getElementById: () => container, addEventListener() {} },
        window: { addEventListener() {} },
        localStorage: { getItem: () => null },
    });
    vm.runInContext(fs.readFileSync('src/mcp_memory_service/web/static/app.js', 'utf8') + '\nthis.Dashboard = MemoryDashboard;', context);
    context.Dashboard.prototype.init = () => {};
    const app = new context.Dashboard();
    const data = { tags: [{ tag: 'once', count: 1 }, { tag: 'twice', count: 2 }, { tag: 'often', count: 10 }] };
    app.renderTagManagementTable(data);
    const input = parts['#tagManagementMinCount'];
    function set(value, event = 'oninput') { input.value = value; input[event](); }
    return { app, data, parts, container, input, set, rows: () => parts.tbody.innerHTML, summary: () => parts['.tag-management-count'].textContent };
}

test('default minimum 2 hides single-use tags and reports total', () => {
    const d = dashboard();
    assert.equal(d.app.tagManagementMinCount, 2);
    assert.doesNotMatch(d.rows(), />once</);
    assert.match(d.rows(), />twice</);
    assert.match(d.rows(), />often</);
    assert.equal(d.summary(), 'Showing 2 of 3 tags');
    assert.match(d.container.innerHTML, /type="number" min="1" step="1" value="2"/);
});

test('filter changes are client-side, preserve input, and handle empty results', () => {
    const d = dashboard();
    const html = d.container.innerHTML;
    d.app.apiCall = () => assert.fail('filter must not request data');
    d.set('1');
    assert.match(d.rows(), />once</);
    assert.equal(d.summary(), 'Showing 3 of 3 tags');
    assert.equal(d.container.innerHTML, html, 'typing must not rebuild the focused input');
    d.set('11');
    assert.equal(d.rows(), '');
    assert.equal(d.summary(), 'Showing 0 of 3 tags');
    d.app.renderTagManagementTable({ tags: [] });
    assert.equal(d.summary(), 'Showing 0 of 0 tags');
});

test('language rerender preserves minimum and translates count with values', () => {
    const d = dashboard();
    d.set('1');
    d.app.translations = JSON.parse(fs.readFileSync('src/mcp_memory_service/web/static/i18n/de.json', 'utf8'));
    d.app.renderTagManagementTable(d.data);
    assert.match(d.container.innerHTML, /value="1"/);
    assert.equal(d.summary(), '3 von 3 Tags werden angezeigt');
    assert.equal(d.app.tagManagementMinCount, 1);
});

test('invalid values cannot remove the minimum bound', () => {
    const d = dashboard();
    for (const value of ['', '0', '-3', '1.5', 'NaN']) {
        d.set(value);
        assert.equal(d.app.tagManagementMinCount, 2);
    }
    for (const value of ['', '0', '-3', 'NaN']) {
        d.set(value, 'onchange');
        assert.equal(d.app.tagManagementMinCount, 1);
        assert.equal(d.input.value, '1');
    }
});

test('visible rename and delete buttons still delegate the exact tag and count', () => {
    const d = dashboard();
    const calls = [];
    d.app.renameTag = tag => calls.push(['rename', tag]);
    d.app.deleteTag = (tag, count) => calls.push(['delete', tag, count]);
    for (const action of ['rename-tag', 'delete-tag']) {
        d.container.onclick({ target: { closest: () => ({ dataset: { action, tag: 'twice', count: '2' } }) } });
    }
    assert.deepEqual(calls, [['rename', 'twice'], ['delete', 'twice', 2]]);
});
