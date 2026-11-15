'use strict';

/**
 * Tests for utilities/version-checker.js readLocalVersion().
 *
 * Run: node --test claude-hooks/tests/version-checker.test.js
 *
 * Regression guard: __init__.py imports __version__ from _version.py and keeps an
 * old literal ("10.13.0") only as a last-resort fallback. The session-start hook
 * read that literal and reported every install as 10.13.0 (outdated).
 */

const { test, afterEach } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { readLocalVersion } = require('../utilities/version-checker');

const roots = [];

// Runs after each test, also when its assertion failed.
afterEach(() => {
    for (const root of roots.splice(0)) {
        fs.rmSync(root, { recursive: true, force: true });
    }
});

function makeProject(files) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'version-checker-'));
    roots.push(root);
    const pkg = path.join(root, 'src', 'mcp_memory_service');
    fs.mkdirSync(pkg, { recursive: true });
    for (const [name, content] of Object.entries(files)) {
        fs.writeFileSync(path.join(pkg, name), content);
    }
    return root;
}

const INIT_WITH_FALLBACK = [
    'try:',
    '    from ._version import __version__',
    'except ImportError:',
    '    __version__ = "10.13.0"',
    ''
].join('\n');

test('reads _version.py, not the fallback literal in __init__.py', async () => {
    const root = makeProject({
        '__init__.py': INIT_WITH_FALLBACK,
        '_version.py': '__version__ = "12.10.0"\n'
    });
    assert.strictEqual(await readLocalVersion(root), '12.10.0');
});

test('falls back to __init__.py when _version.py is missing', async () => {
    const root = makeProject({ '__init__.py': '__version__ = "9.1.0"\n' });
    assert.strictEqual(await readLocalVersion(root), '9.1.0');
});

test('returns null when neither file has a version', async () => {
    const root = makeProject({});
    assert.strictEqual(await readLocalVersion(root), null);
});
