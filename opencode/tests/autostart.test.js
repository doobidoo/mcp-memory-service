/**
 * Tests for the optional memory-service auto-start in memory-plugin.js.
 *
 * Run: node opencode/tests/autostart.test.js
 *
 * autoStart changes who owns the server process, and it spawns a process from
 * inside the editor, so its decisions (start or not, which command, whether to
 * stop) are pinned here with injected dependencies instead of a live service.
 * The failure that matters in production is a wrong guess about ownership: a
 * second instance started next to a healthy one, or a stop that leaves the
 * server running after the editor is gone.
 */

import { test } from "node:test"
import assert from "node:assert"

import { _internal } from "../memory-plugin.js"

const { DEFAULT_CONFIG, ensureMemoryService, stopMemoryService } = _internal

function configWith(autoStart) {
  return {
    memoryService: {
      ...DEFAULT_CONFIG.memoryService,
      autoStart: { ...DEFAULT_CONFIG.memoryService.autoStart, ...autoStart },
    },
  }
}

function fakeChild(pid = 4242) {
  const child = {
    pid,
    exitCode: null,
    killed: false,
    on() {},
    unref() {},
    kill() { child.killed = true },
  }
  return child
}

test("autoStart is off unless the configuration enables it", async () => {
  let spawnCalls = 0
  const started = await ensureMemoryService(configWith({ enabled: false }), {
    getHealth: async () => { throw new Error("down") },
    spawn: () => { spawnCalls += 1 },
  })
  assert.strictEqual(started, null)
  assert.strictEqual(spawnCalls, 0)
})

test("a reachable service is reused instead of starting a second instance", async () => {
  let spawnCalls = 0
  const started = await ensureMemoryService(configWith({ enabled: true }), {
    getHealth: async () => ({ status: "healthy" }),
    spawn: () => { spawnCalls += 1 },
  })
  assert.strictEqual(started, null)
  assert.strictEqual(spawnCalls, 0)
})

test("an unreachable service is started with the configured command and env", async () => {
  const child = fakeChild()
  const spawnCalls = []
  let healthCalls = 0

  const started = await ensureMemoryService(configWith({
    enabled: true,
    command: ["memory", "server", "--http"],
    env: { MCP_ALLOW_ANONYMOUS_ACCESS: "true" },
    startTimeoutMs: 1000,
  }), {
    getHealth: async () => {
      healthCalls += 1
      if (healthCalls === 1) throw new Error("down")
      return { status: "healthy" }
    },
    sleep: async () => {},
    platform: "linux",
    spawn: (command, args, options) => {
      spawnCalls.push({ command, args, options })
      return child
    },
  })

  assert.strictEqual(started, child)
  assert.deepStrictEqual(spawnCalls, [{
    command: "memory",
    args: ["server", "--http"],
    options: {
      stdio: "ignore",
      windowsHide: true,
      detached: true,
      env: { ...process.env, MCP_ALLOW_ANONYMOUS_ACCESS: "true" },
    },
  }])
})

test("Windows stops the launcher tree with taskkill", async () => {
  const child = fakeChild()
  const getHealth = () => { throw new Error("down") }
  await ensureMemoryService(configWith({ enabled: true, startTimeoutMs: 0 }), {
    getHealth,
    sleep: async () => {},
    platform: "win32",
    spawn: () => child,
  })

  const calls = []
  stopMemoryService({
    platform: "win32",
    execFile: (file, args, options) => { calls.push({ file, args, options }) },
  })

  assert.deepStrictEqual(calls, [{
    file: "taskkill",
    args: ["/PID", "4242", "/T", "/F"],
    options: { windowsHide: true },
  }])
})

test("POSIX signals the process group so the server child stops too", async () => {
  const child = fakeChild(5151)
  await ensureMemoryService(configWith({ enabled: true, startTimeoutMs: 0 }), {
    getHealth: () => { throw new Error("down") },
    sleep: async () => {},
    platform: "linux",
    spawn: () => child,
  })

  const signals = []
  stopMemoryService({
    platform: "linux",
    kill: (pid, signal) => { signals.push({ pid, signal }) },
  })

  assert.deepStrictEqual(signals, [{ pid: -5151, signal: "SIGTERM" }])
})

test("stopOnExit: false leaves the service running", async () => {
  const child = fakeChild()
  await ensureMemoryService(configWith({ enabled: true, startTimeoutMs: 0, stopOnExit: false }), {
    getHealth: () => { throw new Error("down") },
    sleep: async () => {},
    platform: "win32",
    spawn: () => child,
  })

  let execCalls = 0
  stopMemoryService({ platform: "win32", execFile: () => { execCalls += 1 } })

  assert.strictEqual(execCalls, 0)
  assert.strictEqual(child.killed, false)
})

test("a service that never answers is reported and does not block forever", async () => {
  const messages = []
  const started = await ensureMemoryService(configWith({
    enabled: true,
    command: [""],
    startTimeoutMs: 0,
  }), {
    getHealth: () => { throw new Error("down") },
    log: async (message) => { messages.push(message) },
  })

  assert.strictEqual(started, null)
  assert.match(messages[0], /autoStart\.command is empty/)
})
