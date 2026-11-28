'use strict';

/**
 * Tests for project naming in utilities/project-detector.js (#2150).
 *
 * Run: node --test claude-hooks/tests/project-detector.test.js
 *
 * A manifest without a name used to yield a placeholder ('rust-project',
 * 'node-project', 'python-project', 'go-project', 'java-maven-project',
 * 'java-gradle-project'). The placeholder is truthy, so it won over the git
 * repository and directory names, and every Cargo workspace root was tagged
 * 'rust-project'. Names a manifest does declare must keep winning.
 *
 * Fixtures are temporary directories; the git ones get a local 'git init', an
 * empty commit and optionally an origin URL that is never contacted.
 */

const { test, after } = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const {
    detectProjectContext,
    detectFramework,
    cargoPackageName,
} = require('../utilities/project-detector');

const PLACEHOLDERS = ['rust-project', 'node-project', 'python-project', 'go-project', 'java-maven-project', 'java-gradle-project'];
const roots = [];
// GIT_DIR and friends (set when this runs inside a git hook) would point the
// fixture's git commands at the outer repository.
const gitEnv = Object.fromEntries(Object.entries(process.env).filter(([name]) => !name.startsWith('GIT_')));

after(() => {
    for (const root of roots) fs.rmSync(root, { recursive: true, force: true });
});

/** A fresh directory named `name` holding `files`; optionally a git repo with an origin. */
function fixture(name, files, { git = false, origin = null } = {}) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'project-detector-'));
    roots.push(root);
    const dir = path.join(root, name);
    fs.mkdirSync(dir);
    for (const [file, content] of Object.entries(files)) {
        fs.writeFileSync(path.join(dir, file), content);
    }
    if (git) {
        execFileSync('git', ['init', '-q'], { cwd: dir, stdio: 'ignore', env: gitEnv });
        if (origin) execFileSync('git', ['remote', 'add', 'origin', origin], { cwd: dir, stdio: 'ignore', env: gitEnv });
        // getGitInfo reads the last commit, which fails in a repository without one.
        execFileSync('git', [
            '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid',
            '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null',
            'commit', '-q', '--allow-empty', '--no-verify', '-m', 'fixture',
        ], { cwd: dir, stdio: 'ignore', env: gitEnv });
    }
    return dir;
}

async function quietly(fn) {
    const log = console.log;
    console.log = () => {};
    try {
        return await fn();
    } finally {
        console.log = log;
    }
}

const nameOf = (dir) => quietly(async () => (await detectProjectContext(dir)).name);

const VIRTUAL_WORKSPACE = [
    '[workspace]',
    'members = ["crates/core", "crates/cli"]',
    'resolver = "2"',
    '',
    '[workspace.package]',
    'version = "0.3.0"',
    'edition = "2021"',
    '',
    '[workspace.dependencies]',
    'tokio = { version = "1", features = ["full"] }',
    '',
].join('\n');

test('Cargo workspace root in a git repository is named after the repository', async () => {
    const dir = fixture('checkout-dir', { 'Cargo.toml': VIRTUAL_WORKSPACE }, {
        git: true,
        origin: 'https://example.invalid/acme/widget-repo.git',
    });
    assert.strictEqual(await nameOf(dir), 'widget-repo');
});

test('Cargo workspace root in a git repository without a remote is named after the repository root', async () => {
    const dir = fixture('widget-root', { 'Cargo.toml': VIRTUAL_WORKSPACE }, { git: true });
    assert.strictEqual(await nameOf(dir), 'widget-root');
});

test('Cargo workspace root outside git is named after the directory', async () => {
    assert.strictEqual(await nameOf(fixture('plain-workspace', { 'Cargo.toml': VIRTUAL_WORKSPACE })), 'plain-workspace');
});

test('named Cargo package keeps its package name, also inside a git repository', async () => {
    const cargo = '[package]\nname = "widget-core"\nversion = "0.1.0"\n\n[dependencies]\ntokio = "1"\n';
    const dir = fixture('checkout-dir', { 'Cargo.toml': cargo }, {
        git: true,
        origin: 'git@example.invalid:acme/widget-repo.git',
    });
    assert.strictEqual(await nameOf(dir), 'widget-core');
});

test('cargoPackageName reads only the [package] table', () => {
    assert.strictEqual(cargoPackageName(VIRTUAL_WORKSPACE), null);
    assert.strictEqual(cargoPackageName('[package]\nname = "foo"\n'), 'foo');
    assert.strictEqual(cargoPackageName("[package]\r\nversion = '1'\r\nname = 'crlf'\r\n"), 'crlf');
    assert.strictEqual(cargoPackageName('[[bin]]\nname = "tool"\n\n[package]\nname = "foo"\n'), 'foo');
    assert.strictEqual(cargoPackageName('[workspace]\n\n[workspace.metadata.release]\nname = "not-a-package"\n'), null);
    assert.strictEqual(cargoPackageName('[workspace]\nmembers = ["a"]\n\n[package] # root crate\nname = "root-crate"\n'), 'root-crate');
});

test('manifests without a declared name yield null, never a placeholder', async () => {
    const cases = {
        'Cargo.toml': VIRTUAL_WORKSPACE,
        'package.json': JSON.stringify({ private: true, workspaces: ['packages/*'] }),
        'pyproject.toml': '[tool.black]\nline-length = 88\n',
        'go.mod': 'go 1.22\n',
        'pom.xml': '<project></project>',
        'build.gradle': "plugins { id 'java' }",
    };
    for (const [file, content] of Object.entries(cases)) {
        const dir = fixture('unnamed-project', { [file]: content });
        const framework = await detectFramework(dir);
        assert.strictEqual(framework.projectName, null, `${file} gave ${framework.projectName}`);
        const name = await nameOf(dir);
        assert.strictEqual(name, 'unnamed-project', `${file} gave ${name}`);
        assert.ok(!PLACEHOLDERS.includes(name));
    }
});

test('named package.json, pyproject.toml and go.mod keep their names', async () => {
    assert.strictEqual(await nameOf(fixture('d1', { 'package.json': JSON.stringify({ name: 'web-app' }) })), 'web-app');
    assert.strictEqual(await nameOf(fixture('d2', { 'pyproject.toml': '[project]\nname = "py-tool"\n' })), 'py-tool');
    assert.strictEqual(await nameOf(fixture('d3', { 'go.mod': 'module github.com/acme/go-svc\n\ngo 1.22\n' })), 'go-svc');
});
