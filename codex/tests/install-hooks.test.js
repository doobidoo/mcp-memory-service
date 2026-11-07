'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');

const installer = path.resolve(__dirname, '../install_hooks.py');
const python = process.env.MCS_PYTHON_BIN || (fs.existsSync('.venv/bin/python') ? path.resolve('.venv/bin/python') : 'python3');

function fixture(t) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "codex-install space 'quote-"));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    const home = path.join(root, '.codex');
    fs.mkdirSync(home);
    const run = (...extra) => spawnSync(python, [installer, '--codex-home', home, ...extra], { encoding: 'utf8' });
    return { root, home, run };
}

test('installer preserves unrelated hooks, is idempotent and creates a runnable independent copy', async t => {
    const f = fixture(t);
    const old = { description: 'Keep this', hooks: { Stop: [{ matcher: '*', hooks: [{ type: 'command', command: 'echo existing' }] }] } };
    const hooksFile = path.join(f.home, 'hooks.json');
    fs.writeFileSync(hooksFile, JSON.stringify(old));
    assert.equal(f.run().status, 0);
    const installed = fs.readFileSync(hooksFile, 'utf8');
    const config = JSON.parse(installed);
    assert.equal(config.description, old.description);
    assert.deepEqual(config.hooks.Stop[0], old.hooks.Stop[0]);
    assert.equal(config.hooks.SessionEnd[0].hooks[0].timeout, 3);
    assert.equal(config.hooks.SessionStart[0].hooks[0].additionalContextLimit, 2000);
    assert.ok(!('additionalContextLimit' in config.hooks.Stop[1].hooks[0]));
    assert.ok(fs.readdirSync(f.home).some(name => name.endsWith('.bak')));
    const runtimeConfig = path.join(f.home, 'memory-service', 'config.json');
    fs.writeFileSync(runtimeConfig, '{"endpoint":"http://localhost:9999","apiKey":"keep-me"}');
    assert.equal(f.run().status, 0);
    assert.equal(fs.readFileSync(hooksFile, 'utf8'), installed);
    assert.match(fs.readFileSync(runtimeConfig, 'utf8'), /keep-me/);
    const command = config.hooks.Stop[1].hooks[0].command;
    const child = spawnSync(command, { shell: true, input: '{"hook_event_name":"Unsupported"}', encoding: 'utf8', env: { ...process.env, CODEX_HOME: f.home } });
    assert.equal(child.status, 0, child.stderr);
    assert.equal(child.stderr, '');
    assert.equal(child.stdout, '');
});

test('dry-run writes nothing and malformed hook files remain untouched', t => {
    const f = fixture(t);
    assert.equal(f.run('--dry-run').status, 0);
    assert.deepEqual(fs.readdirSync(f.home), []);
    const hooksFile = path.join(f.home, 'hooks.json');
    for (const invalid of ['not json', '{"hooks":{"Stop":{}}}', '{"hooks":{"Stop":[{}]}}']) {
        fs.writeFileSync(hooksFile, invalid);
        assert.equal(f.run().status, 1);
        assert.equal(fs.readFileSync(hooksFile, 'utf8'), invalid);
        assert.deepEqual(fs.readdirSync(f.home), ['hooks.json']);
    }
});

test('project installation reads its own configuration and respects CODEX_HOME for user installs', t => {
    const f = fixture(t);
    const project = path.join(f.root, 'project');
    const result = spawnSync(python, [installer, '--project', project], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    assert.ok(fs.existsSync(path.join(project, '.codex', 'hooks.json')));
    const configPath = path.join(project, '.codex', 'memory-service', 'config.json');
    fs.writeFileSync(configPath, '{"endpoint":"http://127.0.0.1:9997"}');
    const installedScript = path.join(project, '.codex', 'memory-service', 'codex', 'memory-hook.js');
    const loaded = spawnSync(process.execPath, ['-e',
        'require(process.argv[1]).loadConfig().then(c => console.log(c.endpoint))', installedScript], {
        encoding: 'utf8', env: { ...process.env, CODEX_MEMORY_CONFIG: '', CODEX_MEMORY_ENDPOINT: '', CODEX_MEMORY_API_KEY: '' }
    });
    assert.equal(loaded.status, 0, loaded.stderr);
    assert.equal(loaded.stdout.trim(), 'http://127.0.0.1:9997');
    const home = path.join(f.root, 'custom-home');
    assert.equal(spawnSync(python, [installer], { encoding: 'utf8', env: { ...process.env, CODEX_HOME: home } }).status, 0);
    assert.ok(fs.existsSync(path.join(home, 'hooks.json')));
});
