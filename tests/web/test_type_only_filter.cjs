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

function store(notes = 150) {
    const memories = [];
    for (let i = 0; i < notes; i++) memories.push({ content_hash: `n${i}`, memory_type: 'note', tags: ['seed'] });
    for (let i = 0; i < 20; i++) memories.push({ content_hash: `r${i}`, memory_type: 'reference', tags: ['seed'] });
    return memories;
}

function dashboard({ type = '', tags = '', date = '', query = '', notes = 150, gate = null } = {}) {
    const fields = {
        typeFilter: { value: type },
        tagFilter: { value: tags },
        dateFilter: { value: date },
        quickSearch: { value: query },
        applyFiltersBtn: {
            disabled: false,
            loading: false,
            classList: {
                add(name) { if (name === 'loading') fields.applyFiltersBtn.loading = true; },
                remove(name) { if (name === 'loading') fields.applyFiltersBtn.loading = false; },
            },
        },
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
    const memories = store(notes);
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
        if (gate) await gate('searchMemories');
        return memories.filter(m => !filters.type || m.memory_type === filters.type).map(memory => ({ memory }));
    };
    app.apiCall = async (endpoint, method = 'GET', data = null) => {
        calls.push({ endpoint, method, data });
        if (gate) await gate(endpoint);
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
    return { app, calls, toasts, fields, rendered: () => rendered };
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

// Greptile on #1984: a type-only search that pages can finish after a newer
// search (or a Clear All) and overwrite what that one showed. A held-back page
// stands in for a slow response.
function held() {
    const waiting = new Map();
    const gate = endpoint => {
        if (!endpoint.includes('page=2&')) return Promise.resolve();
        return new Promise(resolve => waiting.set(endpoint, resolve));
    };
    const release = () => { for (const resolve of waiting.values()) resolve(); waiting.clear(); };
    const isHeld = () => waiting.size > 0;
    return { gate, release, isHeld };
}

async function untilHeld(h) {
    for (let i = 0; i < 50 && !h.isHeld(); i++) await new Promise(r => setImmediate(r));
    assert.ok(h.isHeld(), 'the first search should be waiting on page 2');
}

test('a newer type-only search is not overwritten by an older one still paging', async () => {
    const h = held();
    const d = dashboard({ type: 'note', notes: 250, gate: h.gate });
    const older = d.app.handleFilterChange();
    await untilHeld(h);

    d.fields.typeFilter.value = 'reference';
    await d.app.handleFilterChange();
    assert.equal(d.rendered().length, 20);

    h.release();
    await older;
    assert.equal(d.rendered().length, 20, 'the reference results stay on screen');
    assert.ok(d.rendered().every(r => r.memory.memory_type === 'reference'));
    assert.ok(!d.calls.some(c => c.endpoint.includes('page=3&')), 'the outdated search stops paging');
    assert.equal(d.toasts.length, 0);
});

test('an empty newer type is not overwritten by an older search', async () => {
    const h = held();
    const d = dashboard({ type: 'note', notes: 250, gate: h.gate });
    const older = d.app.handleFilterChange();
    await untilHeld(h);

    d.fields.typeFilter.value = 'decision';
    await d.app.handleFilterChange();
    assert.equal(d.rendered().length, 0);

    h.release();
    await older;
    assert.equal(d.rendered().length, 0);
});

test('Clear All cancels a type-only search that is still paging', async () => {
    const h = held();
    const d = dashboard({ type: 'note', notes: 250, gate: h.gate });
    const older = d.app.handleFilterChange();
    await untilHeld(h);

    d.app.clearAllFilters();
    assert.equal(d.fields.typeFilter.value, '');
    assert.equal(d.rendered().length, 0);

    h.release();
    await older;
    assert.equal(d.rendered().length, 0, 'cleared results stay cleared');
    assert.equal(d.app.searchResults.length, 0);
    assert.ok(!d.calls.some(c => c.endpoint.includes('page=3&')));
});

test('a slow query search does not overwrite a newer type-only search', async () => {
    let releaseQuery;
    const gate = endpoint => endpoint === 'searchMemories'
        ? new Promise(resolve => { releaseQuery = resolve; })
        : Promise.resolve();
    const d = dashboard({ type: 'note', query: 'anything', gate });
    const older = d.app.handleFilterChange();
    for (let i = 0; i < 50 && !releaseQuery; i++) await new Promise(r => setImmediate(r));

    d.fields.quickSearch.value = '';
    d.fields.typeFilter.value = 'reference';
    await d.app.handleFilterChange();
    assert.equal(d.rendered().length, 20);

    releaseQuery();
    await older;
    assert.equal(d.rendered().length, 20);
});

test('an outdated search that fails shows no error, and the button is not left loading', async () => {
    let failOlder;
    const gate = endpoint => endpoint.includes('page=2&') && !failOlder
        ? new Promise((_, reject) => { failOlder = () => reject(new Error('HTTP 500')); })
        : Promise.resolve();
    const d = dashboard({ type: 'note', gate });
    const older = d.app.handleFilterChange();
    for (let i = 0; i < 50 && !failOlder; i++) await new Promise(r => setImmediate(r));
    assert.equal(d.fields.applyFiltersBtn.loading, true);

    d.app.clearAllFilters();
    assert.equal(d.fields.applyFiltersBtn.loading, false, 'Clear All drops the loading state');
    assert.equal(d.fields.applyFiltersBtn.disabled, false);

    failOlder();
    await older;
    assert.equal(d.toasts.filter(t => t.kind === 'error').length, 0, 'no error toast for a cancelled search');
    assert.equal(d.fields.applyFiltersBtn.loading, false);
    assert.equal(d.rendered().length, 0);
});

test('the current search still clears the loading state and reports its own error', async () => {
    const d = dashboard({ type: 'note' });
    d.app.apiCall = async () => { throw new Error('HTTP 500'); };
    await d.app.handleFilterChange();
    assert.equal(d.toasts.filter(t => t.kind === 'error').length, 1);
    assert.equal(d.fields.applyFiltersBtn.loading, false);
    assert.equal(d.fields.applyFiltersBtn.disabled, false);
});
