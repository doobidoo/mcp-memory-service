// Run with: node --test tests/web/test_type_only_filter.cjs
//
// Regression test for #1971: the Search tab's type-only filter (a content type
// and nothing else) asked GET /api/memories for page_size=1000, which the route
// caps at 100, so every type-only search was a 422 and a "Filter search failed"
// toast. The fake API below enforces the same cap and filters by memory_type
// the way the route does.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const PAGE_SIZE_CAP = 100;

function store() {
    const memories = [];
    for (let i = 0; i < 150; i++) memories.push({ content_hash: `n${i}`, memory_type: 'note', tags: ['seed'] });
    for (let i = 0; i < 20; i++) memories.push({ content_hash: `r${i}`, memory_type: 'reference', tags: ['seed'] });
    return memories;
}

function dashboard({ type = '', tags = '', date = '', query = '' } = {}) {
    const fields = {
        typeFilter: { value: type },
        tagFilter: { value: tags },
        dateFilter: { value: date },
        quickSearch: { value: query },
    };
    const context = vm.createContext({
        console: { log() {}, error() {}, warn() {} },
        document: {
            addEventListener() {},
            getElementById: id => fields[id] || null,
            querySelector: () => null,
            querySelectorAll: () => [],
        },
        window: { addEventListener() {} },
        localStorage: { getItem: () => null, setItem() {} },
        URLSearchParams,
    });
    vm.runInContext(
        fs.readFileSync('src/mcp_memory_service/web/static/app.js', 'utf8') + '\nthis.Dashboard = MemoryDashboard;',
        context
    );
    const app = Object.create(context.Dashboard.prototype);
    const memories = store();
    const calls = [];
    const toasts = [];
    let rendered = null;

    app.t = (key, fallback) => fallback;
    app.showToast = (message, kind) => toasts.push({ message, kind });
    app.renderSearchResults = results => { rendered = results; };
    app.updateResultsCount = () => {};
    app.updateActiveFilters = () => {};
    app.searchMemories = async (q, filters) => {
        calls.push({ endpoint: 'searchMemories', q, filters });
        return memories.filter(m => !filters.type || m.memory_type === filters.type).map(memory => ({ memory }));
    };
    app.apiCall = async (endpoint, method = 'GET', data = null) => {
        calls.push({ endpoint, method, data });
        if (endpoint.startsWith('/memories?')) {
            const params = new URLSearchParams(endpoint.split('?')[1]);
            const page = Number(params.get('page') || 1);
            const pageSize = Number(params.get('page_size') || 10);
            if (pageSize > PAGE_SIZE_CAP) {
                const err = new Error('HTTP 422');
                err.status = 422;
                throw err;
            }
            const type = params.get('memory_type');
            const matching = memories.filter(m => !type || m.memory_type === type);
            const start = (page - 1) * pageSize;
            return {
                memories: matching.slice(start, start + pageSize),
                total: matching.length,
                page,
                page_size: pageSize,
                has_more: start + pageSize < matching.length,
            };
        }
        if (endpoint === '/search/by-tag' || endpoint === '/search/by-time') {
            return { results: memories.slice(0, 5).concat(memories.slice(150, 153)).map(memory => ({ memory })) };
        }
        throw new Error(`unexpected endpoint ${endpoint}`);
    };
    return { app, calls, toasts, rendered: () => rendered };
}

test('type-only filter returns every memory of a type with more than 100', async () => {
    const d = dashboard({ type: 'note' });
    await d.app.handleFilterChange();
    assert.deepEqual(d.toasts, [], 'no error toast');
    assert.equal(d.rendered().length, 150);
    assert.ok(d.rendered().every(r => r.memory.memory_type === 'note'));
    assert.equal(new Set(d.rendered().map(r => r.memory.content_hash)).size, 150, 'no page is fetched twice');
});

test('type-only filter asks the server for the type, within the page-size cap', async () => {
    const d = dashboard({ type: 'note' });
    await d.app.handleFilterChange();
    const listCalls = d.calls.filter(c => c.endpoint.startsWith('/memories?'));
    assert.equal(listCalls.length, 2, 'two pages of 100 for 150 memories');
    for (const call of listCalls) {
        const params = new URLSearchParams(call.endpoint.split('?')[1]);
        assert.equal(params.get('memory_type'), 'note');
        assert.ok(Number(params.get('page_size')) <= PAGE_SIZE_CAP);
    }
});

test('type-only filter for a small type and for an empty type', async () => {
    const small = dashboard({ type: 'reference' });
    await small.app.handleFilterChange();
    assert.deepEqual(small.toasts, []);
    assert.equal(small.rendered().length, 20);

    const empty = dashboard({ type: 'decision' });
    await empty.app.handleFilterChange();
    assert.deepEqual(empty.toasts, []);
    assert.equal(empty.rendered().length, 0);
});

test('the other filter branches still apply the type the same way', async () => {
    const byTag = dashboard({ type: 'reference', tags: 'seed' });
    await byTag.app.handleFilterChange();
    assert.equal(byTag.calls[0].endpoint, '/search/by-tag');
    assert.equal(byTag.rendered().length, 3);

    const byDate = dashboard({ type: 'note', date: 'last week' });
    await byDate.app.handleFilterChange();
    assert.equal(byDate.calls[0].endpoint, '/search/by-time');
    assert.equal(byDate.rendered().length, 5);

    const byQuery = dashboard({ type: 'reference', query: 'anything' });
    await byQuery.app.handleFilterChange();
    assert.equal(byQuery.calls[0].endpoint, 'searchMemories');
    assert.equal(byQuery.calls[0].filters.type, 'reference');
    assert.equal(byQuery.rendered().length, 20);
});
