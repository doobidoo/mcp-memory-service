// Run with: node --test tests/web/test_tag_usage_chart.cjs
//
// Regression test for #1736: the minor-tag bucket of the tag usage chart summed the
// per-tag percentages. Those percentages are `count / total_memories`, so a memory
// carrying three tags is counted in three of them, and the bucket passed 100% as
// soon as anything was multiply tagged - which then drew a bar wider than the
// 200px scale it is measured against.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function dashboard() {
    const context = vm.createContext({
        console: { log() {}, error() {}, warn() {} },
        document: {
            hidden: false,
            addEventListener() {},
            getElementById: () => null,
            querySelector: () => null,
            querySelectorAll: () => [],
        },
        window: { addEventListener() {} },
        localStorage: { getItem: () => null, setItem() {} },
        setTimeout: () => 0,
        clearTimeout: () => {},
        setInterval: () => 0,
        clearInterval: () => {},
    });
    vm.runInContext(
        fs.readFileSync('src/mcp_memory_service/web/static/app.js', 'utf8') + '\nthis.Dashboard = MemoryDashboard;',
        context
    );
    return Object.create(context.Dashboard.prototype);
}

// The shape reported in the issue: 12 memories, every tag below the 10-memory
// significance threshold, and eleven of the memories carrying three tags each.
// Summing the per-tag percentages gives 225%.
const SMALL_MULTI_TAGGED_DB = {
    total_memories: 12,
    tags: [
        { tag: 'work', count: 3, percentage: 25 },
        { tag: 'diverse', count: 3, percentage: 25 },
        { tag: 'idea', count: 3, percentage: 25 },
        { tag: 'todo', count: 3, percentage: 25 },
        { tag: 'bug', count: 3, percentage: 25 },
        { tag: 'note', count: 3, percentage: 25 },
        { tag: 'later', count: 3, percentage: 25 },
        { tag: 'reading', count: 3, percentage: 25 },
        { tag: 'meeting', count: 3, percentage: 25 },
    ],
};

function render(data) {
    const app = dashboard();
    const container = { innerHTML: '' };
    app.renderTagUsageChart(container, data);
    return container.innerHTML;
}

function barWidths(html) {
    return [...html.matchAll(/width:\s*([\d.]+)px/g)].map(m => parseFloat(m[1]));
}

test('the minor-tag bucket reports a tag count, not a percentage', () => {
    const html = render(SMALL_MULTI_TAGGED_DB);

    assert.match(html, /more tags \(≤10 memories each\)/, 'aggregate row is missing');

    // The value cell of the aggregate row: the first chart-value that is not a
    // "count (percentage%)" pair.
    const values = [...html.matchAll(/<span class="chart-value">([^<]*)<\/span>/g)].map(m => m[1]);
    assert.equal(values.length, 1, 'the whole fixture is minor, so there is one value cell');
    assert.equal(values[0], '9 tags');
    assert.doesNotMatch(values[0], /%/, 'a set of overlapping tags has no honest percentage');
});

test('no bar is drawn wider than the 200px scale', () => {
    // A fixture with both a significant tag and the minor bucket, so the clamp on
    // the per-tag bar is exercised as well.
    const data = {
        total_memories: 12,
        tags: [
            { tag: 'work', count: 11, percentage: 91.7 },
            ...SMALL_MULTI_TAGGED_DB.tags,
            // A backend that reports a percentage above 100 for one tag.
            { tag: 'broken', count: 30, percentage: 250 },
        ],
    };
    const widths = barWidths(render(data));
    assert.ok(widths.length >= 2, 'expected a bar per significant tag');
    for (const width of widths) {
        assert.ok(width <= 200, `bar width ${width}px exceeds the 200px scale`);
    }
});

test('a tag below the threshold is not drawn as its own bar', () => {
    const html = render(SMALL_MULTI_TAGGED_DB);
    assert.equal(barWidths(html).length, 0, 'minor tags must not get individual bars');
});
