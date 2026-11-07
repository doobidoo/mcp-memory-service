// Run with: node --test tests/web/test_dashboard_live_refresh.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function dashboard() {
    const timers = new Map();
    let id = 0;
    const elements = {};
    const context = vm.createContext({
        console: { log() {}, error() {} },
        document: { hidden: false, addEventListener() {}, getElementById: key => elements[key] },
        window: { addEventListener() {} },
        setTimeout: fn => { timers.set(++id, fn); return id; },
        clearTimeout: key => timers.delete(key),
        setInterval: fn => { timers.set(++id, fn); return id; },
        clearInterval: key => timers.delete(key),
    });
    vm.runInContext(fs.readFileSync('src/mcp_memory_service/web/static/app.js', 'utf8') + '\nthis.Dashboard = MemoryDashboard;', context);
    const app = Object.create(context.Dashboard.prototype);
    app.currentView = 'dashboard';
    app.showToast = () => {};
    return { app, timers, elements, context };
}

test('delta-sync bursts refresh dashboard and sync status once, quietly', async () => {
    const { app, timers } = dashboard();
    let loads = 0, checks = 0;
    app.loadDashboardData = async options => { assert.equal(options.silent, true); loads++; };
    app.checkSyncStatus = async () => { checks++; };
    for (let i = 0; i < 20; i++) app.handleSyncDeltaCompleted({});
    assert.equal(timers.size, 1);
    await [...timers.values()][0]();
    assert.equal(loads, 1);
    assert.equal(checks, 1);
});

test('hidden pages skip event-driven network refreshes', async () => {
    const { app, timers, context } = dashboard();
    context.document.hidden = true;
    app.loadDashboardData = app.checkSyncStatus = () => assert.fail('hidden page refreshed');
    app._scheduleDashboardRefresh();
    await [...timers.values()][0]();
});

test('live refresh cleanup clears scheduled refresh, fallback, and countdown', () => {
    const { app, timers } = dashboard();
    app.startSyncStatusMonitoring();
    app._scheduleDashboardRefresh();
    app.destroy();
    assert.equal(timers.size, 0);
});

test('overdue schedule does not claim a running sync or leak a timer', () => {
    const { app, timers, elements } = dashboard();
    elements.deltaSyncSchedule = {};
    app._startSyncCountdown(new Date(Date.now() - 10000).toISOString());
    assert.equal(elements.deltaSyncSchedule.textContent, 'Scheduled sync due');
    assert.equal(timers.size, 0);
});
