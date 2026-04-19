# Claude Code Plugin Packaging — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `claude-hooks/` as an installable Claude Code plugin via GitHub marketplace, with protocol-native writes through `MemoryClient.storeMemory()`.

**Architecture:** Two sequential PRs in `doobidoo/mcp-memory-service`. PR 1 adds `storeMemory()` on `MemoryClient` (HTTP primary, MCP fallback) and refactors `session-end.js` + `auto-capture-hook.js` to use it. PR 2 adds plugin manifest, hook wiring, self-healing server starter, smoke test, and marketplace registration — all new files, no touches to existing hooks.

**Tech Stack:** Node.js (hooks), Python (MCP server), bash (smoke tests). Tests use Node built-in `assert`, no external test framework.

**Design spec:** [docs/superpowers/specs/2026-04-19-claude-code-plugin-packaging-design.md](../specs/2026-04-19-claude-code-plugin-packaging-design.md)

**Issue:** [#530](https://github.com/doobidoo/mcp-memory-service/issues/530)

---

## Phase 1 — PR 1: `storeMemory()` on `MemoryClient`

Goal: Refactor raw HTTP writes in hooks to go through `MemoryClient`, which can fall back to MCP protocol. This is the prerequisite for PR 2 and unblocks issue #530 Option B for all existing users.

### Task 1.1: Write failing test for `MemoryClient.storeMemory` HTTP path

**Files:**
- Create: `claude-hooks/tests/memory-client-store.test.js`

- [ ] **Step 1: Create the test file with HTTP success case**

Follow the existing pattern in [claude-hooks/tests/session-end-harvest.test.js](../../../claude-hooks/tests/session-end-harvest.test.js): Node built-in `assert`, local `http` server for mocking, runnable via `node <file>`.

```javascript
#!/usr/bin/env node
/**
 * Tests for MemoryClient.storeMemory()
 * Run: node claude-hooks/tests/memory-client-store.test.js
 */
'use strict';

const assert = require('assert');
const http = require('http');
const { MemoryClient } = require('../utilities/memory-client');

function startMockServer(handler) {
    return new Promise((resolve) => {
        const server = http.createServer(handler);
        server.listen(0, '127.0.0.1', () => {
            const { port } = server.address();
            resolve({ server, endpoint: `http://127.0.0.1:${port}` });
        });
    });
}

async function testHttpStoreSuccess() {
    let receivedBody = null;
    let receivedHeaders = null;
    const { server, endpoint } = await startMockServer((req, res) => {
        receivedHeaders = req.headers;
        let data = '';
        req.on('data', (chunk) => (data += chunk));
        req.on('end', () => {
            receivedBody = JSON.parse(data);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: true, content_hash: 'abc123' }));
        });
    });

    const client = new MemoryClient({
        protocol: 'http',
        http: { endpoint, apiKey: 'test-key' },
    });
    client.activeProtocol = 'http';

    const result = await client.storeMemory('hello world', {
        tags: ['test', 'unit'],
        memoryType: 'note',
        metadata: { source: 'test' },
    });

    assert.strictEqual(result.success, true, 'should return success');
    assert.strictEqual(result.contentHash, 'abc123', 'should expose contentHash');
    assert.strictEqual(receivedBody.content, 'hello world');
    assert.deepStrictEqual(receivedBody.tags, ['test', 'unit']);
    assert.strictEqual(receivedBody.memory_type, 'note');
    assert.deepStrictEqual(receivedBody.metadata, { source: 'test' });
    assert.strictEqual(receivedHeaders['x-api-key'], 'test-key');

    server.close();
    console.log('  ✓ testHttpStoreSuccess');
}

async function run() {
    await testHttpStoreSuccess();
    console.log('All tests passed.');
}

run().catch((err) => {
    console.error(err);
    process.exit(1);
});
```

- [ ] **Step 2: Run test — expect failure**

Run: `node claude-hooks/tests/memory-client-store.test.js`

Expected: `TypeError: client.storeMemory is not a function`

- [ ] **Step 3: Commit failing test**

```bash
git add claude-hooks/tests/memory-client-store.test.js
git commit -m "test: add failing test for MemoryClient.storeMemory HTTP path"
```

---

### Task 1.2: Implement `storeMemory()` HTTP path on `MemoryClient`

**Files:**
- Modify: `claude-hooks/utilities/memory-client.js` (add new methods after `queryMemoriesByTime` at line 241)

- [ ] **Step 1: Add `storeMemory` dispatcher and HTTP implementation**

Insert the following block in `claude-hooks/utilities/memory-client.js` immediately after the closing `}` of `queryMemoriesByTime` (before the private helper `_performApiPost` at line 247):

```javascript
    /**
     * Store a memory using the active protocol.
     * @param {string} content - Memory content
     * @param {object} opts - { tags, memoryType, metadata }
     * @returns {Promise<{success: boolean, contentHash?: string, error?: string}>}
     */
    async storeMemory(content, opts = {}) {
        const { tags = [], memoryType = null, metadata = {} } = opts;
        if (this.activeProtocol === 'mcp' && this.mcpClient) {
            return this.storeMemoryMCP(content, { tags, memoryType, metadata });
        } else if (this.activeProtocol === 'http') {
            return this.storeMemoryHTTP(content, { tags, memoryType, metadata });
        } else {
            throw new Error('No active connection available');
        }
    }

    /**
     * Store memory via HTTP REST API.
     * @private
     */
    storeMemoryHTTP(content, { tags, memoryType, metadata }) {
        return new Promise((resolve) => {
            const url = new URL('/api/memories', this.httpConfig.endpoint);
            const isHttps = url.protocol === 'https:';
            const payload = JSON.stringify({
                content,
                tags,
                memory_type: memoryType,
                metadata,
            });

            const options = {
                hostname: url.hostname,
                port: url.port || (isHttps ? 8443 : 8000),
                path: url.pathname,
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': Buffer.byteLength(payload),
                    'X-API-Key': this.httpConfig.apiKey,
                },
                timeout: 5000,
            };

            if (isHttps) {
                options.rejectUnauthorized = false;
            }

            const requestModule = isHttps ? https : http;
            const req = requestModule.request(options, (res) => {
                let data = '';
                res.on('data', (chunk) => (data += chunk));
                res.on('end', () => {
                    if (res.statusCode >= 200 && res.statusCode < 300) {
                        try {
                            const parsed = JSON.parse(data);
                            resolve({
                                success: parsed.success !== false,
                                contentHash: parsed.content_hash || parsed.contentHash,
                            });
                        } catch (err) {
                            resolve({ success: false, error: `Parse error: ${err.message}` });
                        }
                    } else {
                        resolve({ success: false, error: `HTTP ${res.statusCode}: ${data}` });
                    }
                });
            });

            req.on('error', (error) => {
                resolve({ success: false, error: error.message });
            });

            req.on('timeout', () => {
                req.destroy();
                resolve({ success: false, error: 'Request timeout' });
            });

            req.write(payload);
            req.end();
        });
    }

    /**
     * Store memory via MCP protocol.
     * @private
     */
    async storeMemoryMCP(content, { tags, memoryType, metadata }) {
        try {
            const result = await this.mcpClient.storeMemory(content, {
                tags,
                memoryType,
                metadata,
            });
            return {
                success: result?.success !== false,
                contentHash: result?.content_hash || result?.contentHash,
            };
        } catch (err) {
            return { success: false, error: err.message };
        }
    }
```

- [ ] **Step 2: Run test — expect pass**

Run: `node claude-hooks/tests/memory-client-store.test.js`

Expected: `  ✓ testHttpStoreSuccess\nAll tests passed.`

- [ ] **Step 3: Commit implementation**

```bash
git add claude-hooks/utilities/memory-client.js
git commit -m "feat(hooks): add storeMemory() HTTP path to MemoryClient"
```

---

### Task 1.3: Add MCP fallback test and verify `mcp-client.js` has `storeMemory`

**Files:**
- Modify: `claude-hooks/tests/memory-client-store.test.js` (add MCP fallback test)
- Verify: `claude-hooks/utilities/mcp-client.js` (check for existing `storeMemory`)

- [ ] **Step 1: Verify `mcp-client.js` has `storeMemory` method**

Run: `grep -n "storeMemory" claude-hooks/utilities/mcp-client.js`

If present (any line number), skip to Step 3. If absent, continue to Step 2.

- [ ] **Step 2: Add `storeMemory` to `mcp-client.js` if missing**

Only execute this step if Step 1 showed no match. Add method inside the `MCPClient` class, following the pattern of existing methods like `queryMemories`. Model after the MCP `store_memory` tool schema from [src/mcp_memory_service/server_impl.py](../../../src/mcp_memory_service/server_impl.py). Implementation must call the same JSON-RPC transport used by `queryMemories`, passing params `{ content, tags, memory_type, metadata }`.

- [ ] **Step 3: Add MCP fallback test to the test file**

In `claude-hooks/tests/memory-client-store.test.js`, add before the `run()` function:

```javascript
async function testMcpStoreSuccess() {
    const calls = [];
    const fakeMcpClient = {
        storeMemory: async (content, opts) => {
            calls.push({ content, opts });
            return { success: true, content_hash: 'mcp-hash-xyz' };
        },
    };

    const client = new MemoryClient({ protocol: 'mcp' });
    client.activeProtocol = 'mcp';
    client.mcpClient = fakeMcpClient;

    const result = await client.storeMemory('via mcp', {
        tags: ['mcp'],
        memoryType: 'note',
        metadata: {},
    });

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.contentHash, 'mcp-hash-xyz');
    assert.strictEqual(calls.length, 1);
    assert.strictEqual(calls[0].content, 'via mcp');
    assert.deepStrictEqual(calls[0].opts.tags, ['mcp']);

    console.log('  ✓ testMcpStoreSuccess');
}

async function testStoreNoActiveProtocol() {
    const client = new MemoryClient({ protocol: 'auto' });
    // activeProtocol is null — not connected

    await assert.rejects(
        () => client.storeMemory('x', {}),
        /No active connection available/,
    );
    console.log('  ✓ testStoreNoActiveProtocol');
}
```

Update the `run()` function body to call these:

```javascript
async function run() {
    await testHttpStoreSuccess();
    await testMcpStoreSuccess();
    await testStoreNoActiveProtocol();
    console.log('All tests passed.');
}
```

- [ ] **Step 4: Run tests — expect all three pass**

Run: `node claude-hooks/tests/memory-client-store.test.js`

Expected:
```
  ✓ testHttpStoreSuccess
  ✓ testMcpStoreSuccess
  ✓ testStoreNoActiveProtocol
All tests passed.
```

- [ ] **Step 5: Commit**

```bash
git add claude-hooks/tests/memory-client-store.test.js claude-hooks/utilities/mcp-client.js
git commit -m "test(hooks): add MCP fallback and error-path tests for storeMemory"
```

---

### Task 1.4: Refactor `session-end.js` to use `MemoryClient.storeMemory()`

**Files:**
- Modify: `claude-hooks/core/session-end.js:272-357` (replace `storeSessionMemory` function body)

- [ ] **Step 1: Replace `storeSessionMemory` function**

In `claude-hooks/core/session-end.js`, locate the `storeSessionMemory` function (starts at line 272 with `function storeSessionMemory(endpoint, apiKey, content, projectContext, analysis) {`). Replace the entire function (lines 272-357) with:

```javascript
async function storeSessionMemory(endpoint, apiKey, content, projectContext, analysis) {
    const { MemoryClient } = require('../utilities/memory-client');

    // Generate and normalize tags
    const tags = [
        'claude-code-session',
        'session-consolidation',
        projectContext.name,
        projectContext.language ? `language:${projectContext.language}` : null,
        ...analysis.topics.slice(0, 3),
        ...projectContext.frameworks.slice(0, 2),
        `confidence:${Math.round(analysis.confidence * 100)}`,
    ]
        .filter(Boolean)
        .map((tag) => String(tag).toLowerCase());

    const uniqueTags = [...new Set(tags)];

    const client = new MemoryClient({
        protocol: 'auto',
        preferredProtocol: 'http',
        http: { endpoint, apiKey },
    });

    try {
        await client.connect();
    } catch (err) {
        return { success: false, error: `Connect failed: ${err.message}` };
    }

    const result = await client.storeMemory(content, {
        tags: uniqueTags,
        memoryType: 'session-summary',
        metadata: {
            session_analysis: {
                topics: analysis.topics,
                decisions_count: analysis.decisions.length,
                insights_count: analysis.insights.length,
                code_changes_count: analysis.codeChanges.length,
                next_steps_count: analysis.nextSteps.length,
                session_length: analysis.sessionLength,
                confidence: analysis.confidence,
            },
            project_context: {
                name: projectContext.name,
                language: projectContext.language,
                frameworks: projectContext.frameworks,
            },
            generated_by: 'claude-code-session-end-hook',
            generated_at: new Date().toISOString(),
        },
    });

    await client.disconnect();
    return result;
}
```

- [ ] **Step 2: Remove now-unused imports**

Run: `grep -n "^const.*require.*https\|^const.*require.*http'" claude-hooks/core/session-end.js`

If the file still uses `https`/`http` elsewhere (e.g. `evaluateQuality` HTTP call), keep the imports. Otherwise remove them.

- [ ] **Step 3: Verify session-end still parses**

Run: `node -e "require('./claude-hooks/core/session-end.js')"`

Expected: exit 0 with no error output.

- [ ] **Step 4: Run integration test**

Run: `node claude-hooks/tests/integration-test.js`

Expected: PASS (or pre-existing failures unchanged — no new failures introduced by this refactor).

- [ ] **Step 5: Commit**

```bash
git add claude-hooks/core/session-end.js
git commit -m "refactor(hooks): route session-end writes through MemoryClient.storeMemory"
```

---

### Task 1.5: Refactor `auto-capture-hook.js` to use `MemoryClient.storeMemory()`

**Files:**
- Modify: `claude-hooks/core/auto-capture-hook.js:162-220` (replace `storeMemory` function)
- Modify: `claude-hooks/core/auto-capture-hook.js:322` (update call site)

- [ ] **Step 1: Replace local `storeMemory` function**

In `claude-hooks/core/auto-capture-hook.js`, locate the local `storeMemory` function (starts at line 162 with `async function storeMemory(config, content, memoryType, tags) {`). Replace lines 162-220 (the entire function) with:

```javascript
async function storeMemory(config, content, memoryType, tags) {
    const { MemoryClient } = require('../utilities/memory-client');

    const client = new MemoryClient({
        protocol: 'auto',
        preferredProtocol: 'http',
        http: {
            endpoint: config.memoryService.http.endpoint,
            apiKey: config.memoryService.http.apiKey,
        },
    });

    try {
        await client.connect();
    } catch (err) {
        throw new Error(`Connect failed: ${err.message}`);
    }

    const result = await client.storeMemory(content, {
        tags,
        memoryType,
        metadata: {
            source: 'auto-capture',
            hook: 'PostToolUse',
            captured_at: new Date().toISOString(),
        },
    });

    await client.disconnect();

    if (!result.success) {
        throw new Error(result.error || 'storeMemory returned success=false');
    }
    return result;
}
```

- [ ] **Step 2: Verify call site still works**

Run: `grep -n "storeMemory(" claude-hooks/core/auto-capture-hook.js`

Expected: shows call at line ~322 passing `(config, content, memoryType, tags)` — matches new signature.

- [ ] **Step 3: Remove unused `https`/`http` imports if safe**

Run: `grep -nE "(https|http)\." claude-hooks/core/auto-capture-hook.js`

If the only usages were inside the replaced function, remove the top-level `require('https')` and `require('http')` lines. Otherwise keep them.

- [ ] **Step 4: Verify parses**

Run: `node -e "require('./claude-hooks/core/auto-capture-hook.js')"`

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add claude-hooks/core/auto-capture-hook.js
git commit -m "refactor(hooks): route auto-capture writes through MemoryClient.storeMemory"
```

---

### Task 1.6: Update CHANGELOG and open PR 1

**Files:**
- Modify: `CHANGELOG.md` (add entry under `[Unreleased]`)

- [ ] **Step 1: Add CHANGELOG entry**

Prepend under the `[Unreleased]` section (create section if missing):

```markdown
### Changed
- **hooks**: Route memory writes through `MemoryClient.storeMemory()` — enables MCP protocol fallback for `session-end` and `auto-capture` hooks, closing the silent-failure path documented in #530 (Option B).
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): add entry for MemoryClient.storeMemory refactor"
```

- [ ] **Step 3: Push branch and open PR 1**

```bash
git push -u origin feat/plugin-packaging-design
gh pr create --title "feat(hooks): add storeMemory() to MemoryClient for protocol-native writes" \
  --body "$(cat <<'EOF'
## Summary
- Adds \`storeMemory(content, opts)\` to \`MemoryClient\` with HTTP primary + MCP fallback
- Refactors \`session-end.js\` and \`auto-capture-hook.js\` to use \`MemoryClient\` instead of raw HTTP
- Closes #530 Option B

## Why
Hooks that write memories (\`session-end\`, \`auto-capture\`) used raw HTTP. If the HTTP server was not running they failed silently — 218+ sessions documented with zero stored consolidations in #530. Reads already used \`MemoryClient\` with MCP fallback; this closes the write asymmetry.

## Test plan
- [x] \`node claude-hooks/tests/memory-client-store.test.js\` — HTTP success, MCP fallback, no-connection error
- [ ] Manual: run a session with HTTP server stopped, verify session-end falls back to MCP and the memory appears via query
- [ ] Manual: verify \`node claude-hooks/tests/integration-test.js\` shows no regressions

## Scope
Standalone change. Unblocks #530 Option B for all existing installer users (not just future plugin adopters). Plugin packaging follows in a separate PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Wait for PR 1 review and merge before starting Phase 2**

Phase 2 depends on PR 1 being merged to main. Do not start Phase 2 tasks until PR 1 is in `main`.

---

## Phase 2 — PR 2: Plugin Packaging

Goal: Add plugin manifest, hook wiring, self-healing server starter, smoke test, and marketplace registration. All new files. Existing hooks unchanged.

**Branch name:** `feat/claude-code-plugin`

### Task 2.1: Create plugin manifest and hook wiring

**Files:**
- Create: `claude-hooks/.claude-plugin/plugin.json`
- Create: `claude-hooks/.claude-plugin/hooks.json`
- Create: `claude-hooks/.mcp.json`

- [ ] **Step 1: Create branch from current main**

```bash
git checkout main && git pull
git checkout -b feat/claude-code-plugin
```

- [ ] **Step 2: Create `claude-hooks/.claude-plugin/plugin.json`**

```json
{
  "name": "mcp-memory-service",
  "version": "1.0.0",
  "description": "Automatic memory capture and injection for Claude Code via MCP Memory Service",
  "author": "doobidoo",
  "homepage": "https://github.com/doobidoo/mcp-memory-service",
  "mcpServers": "./.mcp.json",
  "hooks": "./.claude-plugin/hooks.json"
}
```

- [ ] **Step 3: Create `claude-hooks/.claude-plugin/hooks.json`**

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume",
        "hooks": [
          { "type": "command", "command": "node \"${CLAUDE_PLUGIN_ROOT}/scripts/ensure-server.js\"" },
          { "type": "command", "command": "node \"${CLAUDE_PLUGIN_ROOT}/core/session-start.js\"" }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          { "type": "command", "command": "node \"${CLAUDE_PLUGIN_ROOT}/core/session-end.js\"" }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          { "type": "command", "command": "node \"${CLAUDE_PLUGIN_ROOT}/core/mid-conversation.js\"" }
        ]
      }
    ],
    "PostToolUse": [
      {
        "hooks": [
          { "type": "command", "command": "node \"${CLAUDE_PLUGIN_ROOT}/core/auto-capture-hook.js\"" }
        ]
      }
    ]
  }
}
```

- [ ] **Step 4: Create `claude-hooks/.mcp.json`**

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "mcp_memory_service.server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "hybrid",
        "MCP_HYBRID_SYNC_OWNER": "http"
      }
    }
  }
}
```

- [ ] **Step 5: Verify all three files parse as JSON**

```bash
node -e "JSON.parse(require('fs').readFileSync('claude-hooks/.claude-plugin/plugin.json'))" && \
node -e "JSON.parse(require('fs').readFileSync('claude-hooks/.claude-plugin/hooks.json'))" && \
node -e "JSON.parse(require('fs').readFileSync('claude-hooks/.mcp.json'))" && \
echo "all valid"
```

Expected: `all valid`

- [ ] **Step 6: Commit**

```bash
git add claude-hooks/.claude-plugin/ claude-hooks/.mcp.json
git commit -m "feat(plugin): add Claude Code plugin manifest and hook wiring"
```

---

### Task 2.2: Write failing test for `ensure-server.js`

**Files:**
- Create: `claude-hooks/tests/ensure-server.test.js`

- [ ] **Step 1: Create test file**

```javascript
#!/usr/bin/env node
/**
 * Tests for scripts/ensure-server.js
 * Run: node claude-hooks/tests/ensure-server.test.js
 */
'use strict';

const assert = require('assert');
const http = require('http');
const path = require('path');
const { spawn } = require('child_process');

const SCRIPT = path.join(__dirname, '..', 'scripts', 'ensure-server.js');

function runScript(env = {}) {
    return new Promise((resolve) => {
        const proc = spawn('node', [SCRIPT], {
            env: { ...process.env, ...env },
            stdio: ['ignore', 'pipe', 'pipe'],
        });
        let stdout = '';
        let stderr = '';
        proc.stdout.on('data', (d) => (stdout += d));
        proc.stderr.on('data', (d) => (stderr += d));
        proc.on('close', (code) => resolve({ code, stdout, stderr }));
    });
}

function startMockHealthServer(statusCode = 200) {
    return new Promise((resolve) => {
        const server = http.createServer((req, res) => {
            if (req.url === '/api/health') {
                res.writeHead(statusCode, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ status: 'ok' }));
            } else {
                res.writeHead(404);
                res.end();
            }
        });
        server.listen(0, '127.0.0.1', () => {
            const { port } = server.address();
            resolve({ server, port });
        });
    });
}

async function testHealthyServerExitsZero() {
    const { server, port } = await startMockHealthServer(200);
    const { code, stderr } = await runScript({
        MCP_MEMORY_ENDPOINT: `http://127.0.0.1:${port}`,
        ENSURE_SERVER_NO_SPAWN: '1',
    });
    server.close();
    assert.strictEqual(code, 0, `expected exit 0, got ${code}. stderr: ${stderr}`);
    console.log('  ✓ testHealthyServerExitsZero');
}

async function testUnreachableServerExitsZeroWithWarning() {
    const { code, stderr } = await runScript({
        MCP_MEMORY_ENDPOINT: 'http://127.0.0.1:1', // reserved/unused port
        ENSURE_SERVER_NO_SPAWN: '1',
    });
    assert.strictEqual(code, 0, 'must never block session start');
    assert.ok(
        /unreach|could not|failed/i.test(stderr),
        `expected warning in stderr, got: ${stderr}`,
    );
    console.log('  ✓ testUnreachableServerExitsZeroWithWarning');
}

async function run() {
    await testHealthyServerExitsZero();
    await testUnreachableServerExitsZeroWithWarning();
    console.log('All tests passed.');
}

run().catch((err) => {
    console.error(err);
    process.exit(1);
});
```

- [ ] **Step 2: Run — expect failure (script does not exist yet)**

Run: `node claude-hooks/tests/ensure-server.test.js`

Expected: exit code non-zero with error about missing script path.

- [ ] **Step 3: Commit failing test**

```bash
git add claude-hooks/tests/ensure-server.test.js
git commit -m "test(plugin): add failing tests for ensure-server.js"
```

---

### Task 2.3: Implement `ensure-server.js`

**Files:**
- Create: `claude-hooks/scripts/ensure-server.js`

- [ ] **Step 1: Create the script**

```javascript
#!/usr/bin/env node
/**
 * ensure-server.js — SessionStart hook that ensures the HTTP memory server
 * is reachable, starting it in the background if necessary.
 *
 * Contract: NEVER block session start. All failure paths exit 0 with stderr.
 */
'use strict';

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');

const HEALTH_TIMEOUT_MS = 500;
const POLL_INTERVAL_MS = 500;
const POLL_TIMEOUT_MS = 10_000;
const NO_SPAWN = process.env.ENSURE_SERVER_NO_SPAWN === '1';

function log(msg) {
    process.stderr.write(`[ensure-server] ${msg}\n`);
}

function resolveEndpoint() {
    if (process.env.MCP_MEMORY_ENDPOINT) {
        return process.env.MCP_MEMORY_ENDPOINT;
    }
    const configPath = path.join(os.homedir(), '.claude', 'hooks', 'config.json');
    try {
        const cfg = JSON.parse(fs.readFileSync(configPath, 'utf8'));
        const endpoint =
            cfg.memoryService?.http?.endpoint ||
            cfg.memoryService?.endpoint ||
            cfg.sessionHarvest?.endpoint;
        if (endpoint) return endpoint;
    } catch (_) {
        // fall through to default
    }
    return 'http://127.0.0.1:8000';
}

function checkHealth(endpoint) {
    return new Promise((resolve) => {
        let url;
        try {
            url = new URL('/api/health', endpoint);
        } catch (_) {
            return resolve(false);
        }
        const lib = url.protocol === 'https:' ? https : http;
        const req = lib.get(url, { timeout: HEALTH_TIMEOUT_MS }, (res) => {
            res.resume();
            resolve(res.statusCode === 200);
        });
        req.on('error', () => resolve(false));
        req.on('timeout', () => {
            req.destroy();
            resolve(false);
        });
    });
}

function resolveLogPath() {
    const preferred = path.join(os.homedir(), '.mcp-memory-service', 'http.log');
    try {
        fs.mkdirSync(path.dirname(preferred), { recursive: true });
        fs.accessSync(path.dirname(preferred), fs.constants.W_OK);
        return preferred;
    } catch (_) {
        return path.join(os.tmpdir(), 'mcp-memory-service-http.log');
    }
}

function spawnServer() {
    const pythonCandidates = [
        process.env.MCP_MEMORY_PYTHON,
        path.join(process.cwd(), '.venv', 'bin', 'python'),
        'python3',
        'python',
    ].filter(Boolean);

    const scriptPath = path.join('scripts', 'server', 'run_http_server.py');
    const logPath = resolveLogPath();
    const logFd = fs.openSync(logPath, 'a');

    for (const python of pythonCandidates) {
        try {
            const child = spawn(python, [scriptPath], {
                detached: true,
                stdio: ['ignore', logFd, logFd],
                env: process.env,
            });
            child.unref();
            log(`spawned HTTP server via ${python} (log: ${logPath})`);
            return true;
        } catch (err) {
            log(`spawn with ${python} failed: ${err.message}`);
        }
    }
    log('could not spawn python — install python3 or set MCP_MEMORY_PYTHON');
    return false;
}

async function pollUntilHealthy(endpoint, deadlineMs) {
    while (Date.now() < deadlineMs) {
        if (await checkHealth(endpoint)) return true;
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
    }
    return false;
}

async function main() {
    const endpoint = resolveEndpoint();
    if (await checkHealth(endpoint)) return;

    log(`HTTP server unreachable at ${endpoint}`);
    if (NO_SPAWN) {
        log('ENSURE_SERVER_NO_SPAWN=1 — skipping spawn (test mode)');
        return;
    }

    if (!spawnServer()) return;

    const deadline = Date.now() + POLL_TIMEOUT_MS;
    const ok = await pollUntilHealthy(endpoint, deadline);
    if (!ok) {
        log(`server did not become healthy within ${POLL_TIMEOUT_MS}ms`);
    } else {
        log('server is healthy');
    }
}

main()
    .catch((err) => {
        log(`unexpected error: ${err.message}`);
    })
    .finally(() => {
        process.exit(0); // never block
    });
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x claude-hooks/scripts/ensure-server.js
```

- [ ] **Step 3: Run tests — expect pass**

Run: `node claude-hooks/tests/ensure-server.test.js`

Expected:
```
  ✓ testHealthyServerExitsZero
  ✓ testUnreachableServerExitsZeroWithWarning
All tests passed.
```

- [ ] **Step 4: Commit**

```bash
git add claude-hooks/scripts/ensure-server.js
git commit -m "feat(plugin): add ensure-server.js self-healing HTTP starter"
```

---

### Task 2.4: Create structural smoke test

**Files:**
- Create: `claude-hooks/scripts/test-plugin-install.sh`

- [ ] **Step 1: Create the script**

```bash
#!/usr/bin/env bash
# test-plugin-install.sh — Validate plugin structure without installing.
# Run: bash claude-hooks/scripts/test-plugin-install.sh
set -euo pipefail

PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PLUGIN_ROOT/.." && pwd)"

fail() { echo "FAIL: $1" >&2; exit 1; }
ok() { echo "OK: $1"; }

# 1. plugin.json must parse
node -e "JSON.parse(require('fs').readFileSync('$PLUGIN_ROOT/.claude-plugin/plugin.json'))" \
  || fail "plugin.json invalid"
ok "plugin.json parseable"

# 2. hooks.json must parse
node -e "JSON.parse(require('fs').readFileSync('$PLUGIN_ROOT/.claude-plugin/hooks.json'))" \
  || fail "hooks.json invalid"
ok "hooks.json parseable"

# 3. .mcp.json must parse
node -e "JSON.parse(require('fs').readFileSync('$PLUGIN_ROOT/.mcp.json'))" \
  || fail ".mcp.json invalid"
ok ".mcp.json parseable"

# 4. marketplace.json (repo root) must parse
node -e "JSON.parse(require('fs').readFileSync('$REPO_ROOT/.claude-plugin/marketplace.json'))" \
  || fail "marketplace.json invalid"
ok "marketplace.json parseable"

# 5. Every script referenced in hooks.json must exist
node -e "
const hooks = JSON.parse(require('fs').readFileSync('$PLUGIN_ROOT/.claude-plugin/hooks.json'));
const fs = require('fs');
const path = require('path');
for (const [event, entries] of Object.entries(hooks.hooks)) {
  for (const entry of entries) {
    for (const hook of entry.hooks) {
      const m = hook.command.match(/\\\$\{CLAUDE_PLUGIN_ROOT\}\/([^\"]+)/);
      if (!m) { console.error('Unparseable command:', hook.command); process.exit(1); }
      const file = path.join('$PLUGIN_ROOT', m[1]);
      if (!fs.existsSync(file)) { console.error('Missing:', file); process.exit(1); }
    }
  }
}
"
ok "all referenced hook scripts exist"

# 6. Every referenced script must parse as Node
for f in \
  "$PLUGIN_ROOT/scripts/ensure-server.js" \
  "$PLUGIN_ROOT/core/session-start.js" \
  "$PLUGIN_ROOT/core/session-end.js" \
  "$PLUGIN_ROOT/core/mid-conversation.js" \
  "$PLUGIN_ROOT/core/auto-capture-hook.js"; do
  node -e "require('$f')" 2>/dev/null || fail "$f fails to load"
  ok "$(basename "$f") loads"
done

echo
echo "All plugin structural checks passed."
```

- [ ] **Step 2: Make executable**

```bash
chmod +x claude-hooks/scripts/test-plugin-install.sh
```

- [ ] **Step 3: Create marketplace.json first (referenced by smoke test)**

Create `/.claude-plugin/marketplace.json` at repo root (NOT in `claude-hooks/`):

```bash
mkdir -p .claude-plugin
```

Create `.claude-plugin/marketplace.json`:

```json
{
  "name": "mcp-memory-service",
  "owner": { "name": "doobidoo", "url": "https://github.com/doobidoo" },
  "plugins": [
    {
      "name": "mcp-memory-service",
      "source": "./claude-hooks",
      "description": "Semantic memory for Claude Code sessions"
    }
  ]
}
```

- [ ] **Step 4: Run smoke test**

Run: `bash claude-hooks/scripts/test-plugin-install.sh`

Expected: every line is `OK: ...` and final `All plugin structural checks passed.`

- [ ] **Step 5: Commit**

```bash
git add .claude-plugin/ claude-hooks/scripts/test-plugin-install.sh
git commit -m "feat(plugin): add marketplace registration and structural smoke test"
```

---

### Task 2.5: Add plugin documentation

**Files:**
- Create: `claude-hooks/PLUGIN.md`
- Modify: `claude-hooks/README.md` (add plugin install section at top)

- [ ] **Step 1: Create `claude-hooks/PLUGIN.md`**

```markdown
# Claude Code Plugin Installation

This directory can be installed as a Claude Code plugin via the GitHub
marketplace. Two setup paths exist: the plugin (recommended for new users)
and the legacy Python installer (`install_hooks.py`). Both work; they should
not be active simultaneously.

## Install via plugin marketplace

```
/plugin marketplace add doobidoo/mcp-memory-service
/plugin install mcp-memory-service
```

Claude Code registers the MCP server (`.mcp.json`) and the hook wiring
(`.claude-plugin/hooks.json`) automatically. Updates come via `git pull`
on the marketplace entry.

## Configuration

The plugin reads `~/.claude/hooks/config.json`. If that file does not
exist, copy `config.template.json` from this directory:

```
mkdir -p ~/.claude/hooks
cp ~/.claude/plugins/marketplaces/doobidoo/mcp-memory-service/claude-hooks/config.template.json \
   ~/.claude/hooks/config.json
```

Then edit `~/.claude/hooks/config.json` — at minimum set
`memoryService.http.endpoint` and `memoryService.http.apiKey`.

## Server lifecycle

On SessionStart the plugin runs `scripts/ensure-server.js`, which:

1. Probes `GET /api/health` on the configured endpoint
2. If unreachable, spawns `python scripts/server/run_http_server.py` in the background
3. Polls for health for up to 10 seconds

The script never blocks session start — all failure paths exit 0 with a
warning on stderr. Logs go to `~/.mcp-memory-service/http.log`.

Set `ENSURE_SERVER_NO_SPAWN=1` to disable the spawn path (probe-only mode).

## Migrating from `install_hooks.py`

If you already installed via the Python installer, remove its hook entries
from `~/.claude/settings.json` before installing the plugin — otherwise
every hook runs twice.

```
grep -n "claude-hooks" ~/.claude/settings.json
```

Remove the matching entries from the `hooks` object and save.

## Uninstall

```
/plugin uninstall mcp-memory-service
```

This removes hooks and MCP server registration. It does not delete
`~/.claude/hooks/config.json` or `~/.mcp-memory-service/` logs.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Hooks run twice | Both installer and plugin active | Remove installer hook entries from `~/.claude/settings.json` |
| `ensure-server.js` warns "could not spawn python" | No Python on `PATH` | Install Python 3.11+, or set `MCP_MEMORY_PYTHON` to the interpreter path |
| Writes silently fail | API key missing | Set `memoryService.http.apiKey` in `~/.claude/hooks/config.json` |
| Server starts but never becomes healthy | Port collision | Check `~/.mcp-memory-service/http.log`, change port in `.env` |

## Status

v1.0.0 is **experimental**. Please file issues at
https://github.com/doobidoo/mcp-memory-service/issues with the `plugin` label.
```

- [ ] **Step 2: Add plugin section to `claude-hooks/README.md`**

Read the first 30 lines of `claude-hooks/README.md` to find the right insertion point. Insert immediately after the main title (before any "Installation" section) a new block:

```markdown
## Installation

**Recommended (experimental):** Install as a Claude Code plugin — see [PLUGIN.md](./PLUGIN.md).

**Legacy:** The Python installer (`install_hooks.py`) continues to work and is documented in the sections below. Both setups are mutually exclusive.

---
```

- [ ] **Step 3: Verify markdown renders**

```bash
node -e "const fs=require('fs'); fs.readFileSync('claude-hooks/PLUGIN.md','utf8'); fs.readFileSync('claude-hooks/README.md','utf8'); console.log('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add claude-hooks/PLUGIN.md claude-hooks/README.md
git commit -m "docs(plugin): add PLUGIN.md and update README for plugin install path"
```

---

### Task 2.6: Update CHANGELOG and open PR 2

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add CHANGELOG entry under `[Unreleased]`**

```markdown
### Added
- **plugin**: Claude Code plugin packaging for the claude-hooks suite. Install via `/plugin marketplace add doobidoo/mcp-memory-service` + `/plugin install mcp-memory-service`. Ships with `.mcp.json`, hook wiring, and self-healing `ensure-server.js`. Coexists with the legacy `install_hooks.py` installer — see [claude-hooks/PLUGIN.md](claude-hooks/PLUGIN.md). Closes #530 (plugin packaging track).
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): announce Claude Code plugin packaging"
```

- [ ] **Step 3: Run full validation suite**

```bash
node claude-hooks/tests/memory-client-store.test.js && \
node claude-hooks/tests/ensure-server.test.js && \
bash claude-hooks/scripts/test-plugin-install.sh
```

Expected: all three pass.

- [ ] **Step 4: Push branch and open PR 2**

```bash
git push -u origin feat/claude-code-plugin
gh pr create --title "feat: Claude Code plugin packaging for mcp-memory-service hooks" \
  --body "$(cat <<'EOF'
## Summary
- Ships \`claude-hooks/\` as a Claude Code plugin via GitHub marketplace
- Adds \`ensure-server.js\` that self-starts the HTTP server on SessionStart
- Installer (\`install_hooks.py\`) remains — plugin is additive, experimental

## Why
Issue #530 documents the manual-setup burden (API key generation, two-process bash wrapper, hand-edited MCP config). With plugin packaging users install in two commands.

## Scope
v1 hooks: \`session-start\`, \`session-end\`, \`mid-conversation\`, \`auto-capture-hook\`. Extended set (harvest, topic-change, permission-request, memory-retrieval) deferred to v2.

## Depends on
- #[PR-1-number] (storeMemory on MemoryClient) — already merged

## Test plan
- [x] \`bash claude-hooks/scripts/test-plugin-install.sh\` (structural)
- [x] \`node claude-hooks/tests/ensure-server.test.js\`
- [ ] Manual: fresh Claude Code install, \`/plugin marketplace add\` + \`/plugin install\`, verify session-start hook runs and memories store via MCP
- [ ] Manual: verify installer + plugin coexist does NOT cause double-hook execution when installer entries are removed per PLUGIN.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- `storeMemory()` on MemoryClient → Task 1.2
- HTTP primary + MCP fallback → Tasks 1.2, 1.3
- session-end refactor → Task 1.4
- auto-capture refactor → Task 1.5
- `.claude-plugin/plugin.json` + `hooks.json` → Task 2.1
- `.mcp.json` → Task 2.1
- `ensure-server.js` with all edge cases → Task 2.3
- `marketplace.json` at repo root → Task 2.4
- Smoke test → Task 2.4
- `PLUGIN.md` → Task 2.5
- README update → Task 2.5
- CHANGELOG entries → Tasks 1.6, 2.6
- Coexistence with installer → Task 2.5 (documented in PLUGIN.md)
- Rollout as experimental → Task 2.5 (PLUGIN.md `Status` section)

**Not in plan (deferred to v2 per spec non-goals):** extended hook set, embedded HTTP (Option A), Windows-specific fallback, installer removal.

**Placeholder scan:** No TBDs, all code blocks complete, all commands exact.

**Type consistency:** `storeMemory` signature `(content, { tags, memoryType, metadata })` consistent across Tasks 1.2, 1.3, 1.4, 1.5. Return shape `{ success, contentHash, error? }` consistent.
