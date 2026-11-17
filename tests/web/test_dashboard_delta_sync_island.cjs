// Run with: node --test tests/web/test_dashboard_delta_sync_island.cjs
// The scheduled delta-sync cycles report to the status island as their own source
// (#2003). The server announces a cycle only when it ends, so the source is `ready`
// or `error` and never `busy`: there is no start event and no percentage. A peer
// counts as failed by the rule the delta-sync panel already uses, so the panel and
// the island cannot disagree about the same peer.
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
    const island = {
        reports: [],
        cleared: [],
        report(id, entry) { this.reports.push([id, entry]); },
        get() { return null; },
        clear(id) { this.cleared.push(id); },
    };
    app.statusIsland = island;
    app.apiCall = apiCall;
    app.updateDeltaSyncStatus = () => {};   // the panel needs a DOM, the island does not
    app.loadDashboardData = async () => {};
    return { app, island };
}

const peersReady = [{ peer_id: 'hub-a', status: 'healthy', last_cycle_at: '2026-11-17T06:00:00Z' }];

function delta(overrides = {}) {
    return { configured: true, enabled: true, status: 'enabled', peers: peersReady, ...overrides };
}

test('every peer succeeded: the island shows a ready delta-sync source', async () => {
    const { app, island } = dashboard(async () => ({ is_hybrid: false, delta_sync: delta() }));
    await app.checkSyncStatus();
    const [id, entry] = island.reports.at(-1);
    assert.equal(id, 'delta-sync');
    assert.equal(entry.state, 'ready');
    assert.equal(entry.text.key, 'sync.status.synced');
    assert.equal(JSON.stringify(entry.label), JSON.stringify({ key: 'sync.deltaSync', fallback: 'Delta sync' }));
    assert.equal(entry.progress, undefined);
});

test('a failing peer makes the source an error that names it', async () => {
    const peers = [{ ...peersReady[0], status: 'error', last_error: 'connection refused' }];
    const { app, island } = dashboard(async () => ({ is_hybrid: false, delta_sync: delta({ peers, status: 'error' }) }));
    await app.checkSyncStatus();
    const [, entry] = island.reports.at(-1);
    assert.equal(entry.state, 'error');
    assert.equal(entry.dismissible, true);
    assert.equal(entry.text.key, 'sync.deltaSync.failed');
    assert.equal(JSON.stringify(entry.text.vars), JSON.stringify({ peer: 'hub-a' }));
});

test('the panel\'s failure rule is reused: last_error alone is a failure', async () => {
    const peers = [{ ...peersReady[0], status: 'healthy', last_error: 'timeout' }];
    const { app, island } = dashboard(async () => ({ is_hybrid: false, delta_sync: delta({ peers }) }));
    await app.checkSyncStatus();
    assert.equal(island.reports.at(-1)[1].state, 'error');
});

test('incompatible and unavailable peers count as failures too', async () => {
    for (const status of ['incompatible', 'unavailable']) {
        const peers = [{ ...peersReady[0], status }];
        const { app, island } = dashboard(async () => ({ is_hybrid: false, delta_sync: delta({ peers }) }));
        await app.checkSyncStatus();
        assert.equal(island.reports.at(-1)[1].state, 'error', `${status} must be a failure`);
    }
});

test('a peer waiting for its first cycle is pending, not synced', async () => {
    const peers = [{ peer_id: 'hub-b', status: 'pending' }];
    const { app, island } = dashboard(async () => ({ is_hybrid: false, delta_sync: delta({ peers }) }));
    await app.checkSyncStatus();
    const [, entry] = island.reports.at(-1);
    assert.equal(entry.state, 'ready');
    assert.equal(entry.text.key, 'sync.status.pending');
});

test('no configured peers report nothing and clear the stale source', async () => {
    for (const payload of [delta({ peers: [] }), delta({ configured: false })]) {
        const { app, island } = dashboard(async () => ({ is_hybrid: false, delta_sync: payload }));
        await app.checkSyncStatus();
        assert.equal(island.reports.length, 0);
        assert.deepEqual(island.cleared, ['delta-sync']);
    }
});

test('a completed cycle is a confirmation with a settle entry', () => {
    const { app, island } = dashboard(async () => ({}));
    app._scheduleDashboardRefresh = () => {};
    app.handleSyncDeltaCompleted({ peer_id: 'hub-a', success: true, pulled: 3, pushed: 1 });
    const [id, entry] = island.reports.at(-1);
    assert.equal(id, 'delta-sync');
    assert.equal(entry.state, 'success');
    assert.equal(entry.text.key, 'island.deltaSync.completed');
    assert.equal(JSON.stringify(entry.settle), JSON.stringify({ state: 'ready', text: { key: 'sync.status.synced', fallback: 'Synced' } }));
});

test('a failed cycle reports no confirmation: the poll owns the error', () => {
    const { app, island } = dashboard(async () => ({}));
    app._scheduleDashboardRefresh = () => {};
    app.handleSyncDeltaCompleted({ peer_id: 'hub-a', success: false });
    assert.equal(island.reports.length, 0);
});

test('the ten-second poll does not replace a fresh confirmation', async () => {
    const { app, island } = dashboard(async () => ({ is_hybrid: false, delta_sync: delta() }));
    island.get = (id) => (id === 'delta-sync' ? { state: 'success' } : null);
    await app.checkSyncStatus();
    assert.equal(island.reports.length, 0);
});
