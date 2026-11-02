// Run with: node --test tests/web/test_dashboard_system_theme.cjs
//
// #1737: the dashboard always started in light mode. The default theme was a
// hardcoded 'light' and nothing read `prefers-color-scheme`. A user who never
// picked a theme now follows the OS, live; an explicit light or dark choice
// still wins, and the header toggle only ever stores light or dark.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function fakeColorScheme(dark) {
    const listeners = [];
    return {
        matches: dark,
        media: '(prefers-color-scheme: dark)',
        addEventListener(type, fn) {
            if (type === 'change') listeners.push(fn);
        },
        // Simulate the OS switching between light and dark.
        set(nextDark) {
            this.matches = nextDark;
            listeners.forEach(fn => fn({ matches: nextDark }));
        },
    };
}

function fakeClassList() {
    const classes = new Set();
    return {
        contains: name => classes.has(name),
        toggle(name, force) {
            const on = force === undefined ? !classes.has(name) : force;
            if (on) classes.add(name); else classes.delete(name);
            return on;
        },
        add: name => classes.add(name),
        remove: name => classes.delete(name),
    };
}

// Builds the real MemoryDashboard from app.js, running its constructor (so the
// default settings are the shipped ones) with only the async init() stubbed out,
// then runs the same theme steps init() runs.
function dashboard({ osDark, saved }) {
    const scheme = fakeColorScheme(osDark);
    const store = {};
    if (saved !== undefined) store.memoryDashboardSettings = JSON.stringify(saved);
    const body = { classList: fakeClassList() };
    const context = vm.createContext({
        console: { log() {}, error() {}, warn() {} },
        document: {
            body,
            hidden: false,
            addEventListener() {},
            getElementById: () => null,
            querySelector: () => null,
            querySelectorAll: () => [],
        },
        window: {
            addEventListener() {},
            matchMedia: query => {
                assert.equal(query, '(prefers-color-scheme: dark)');
                return scheme;
            },
        },
        localStorage: {
            getItem: key => (key in store ? store[key] : null),
            setItem: (key, value) => { store[key] = value; },
        },
        setTimeout: () => 0,
        clearTimeout: () => {},
        setInterval: () => 0,
        clearInterval: () => {},
    });
    vm.runInContext(
        fs.readFileSync('src/mcp_memory_service/web/static/app.js', 'utf8') + '\nthis.Dashboard = MemoryDashboard;',
        context
    );
    context.Dashboard.prototype.init = function () {};
    const app = new context.Dashboard();
    app.t = (_key, fallback) => fallback;
    app.showToast = () => {};
    app.loadSettings();
    app.applyTheme();
    app.watchSystemColorScheme();
    const isDark = () => body.classList.contains('dark-mode');
    const savedTheme = () => JSON.parse(store.memoryDashboardSettings || '{}').theme;
    return { app, scheme, isDark, savedTheme };
}

test('a first visit on a dark OS starts in dark mode', () => {
    const { app, isDark } = dashboard({ osDark: true });
    assert.equal(app.settings.theme, 'system');
    assert.equal(isDark(), true);
});

test('a first visit on a light OS starts in light mode', () => {
    const { isDark } = dashboard({ osDark: false });
    assert.equal(isDark(), false);
});

test('the system setting follows the OS when it switches', () => {
    const { scheme, isDark } = dashboard({ osDark: false, saved: { theme: 'system' } });
    assert.equal(isDark(), false);
    scheme.set(true);
    assert.equal(isDark(), true);
    scheme.set(false);
    assert.equal(isDark(), false);
});

test('an explicit light choice wins over a dark OS, and ignores OS changes', () => {
    const { scheme, isDark } = dashboard({ osDark: true, saved: { theme: 'light' } });
    assert.equal(isDark(), false);
    scheme.set(false);
    scheme.set(true);
    assert.equal(isDark(), false);
});

test('an explicit dark choice wins over a light OS', () => {
    const { isDark } = dashboard({ osDark: false, saved: { theme: 'dark' } });
    assert.equal(isDark(), true);
});

test('the header toggle flips what is shown and stores an explicit theme', () => {
    const { app, scheme, isDark, savedTheme } = dashboard({ osDark: true });
    assert.equal(isDark(), true);

    app.toggleTheme();
    assert.equal(isDark(), false);
    assert.equal(savedTheme(), 'light');

    // Explicit now: the OS no longer drives the page.
    scheme.set(false);
    scheme.set(true);
    assert.equal(isDark(), false);

    app.toggleTheme();
    assert.equal(isDark(), true);
    assert.equal(savedTheme(), 'dark');
});

test('without matchMedia the system setting falls back to light', () => {
    const { app } = dashboard({ osDark: true });
    assert.equal(app.resolveTheme('system'), 'dark');
    app.systemColorSchemeQuery = () => null;
    assert.equal(app.resolveTheme('system'), 'light');
    assert.equal(app.resolveTheme('dark'), 'dark');
});
