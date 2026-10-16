/**
 * End-to-end wiring test for the OpenCode inter-agent inbox (#1527).
 */

import { test } from "node:test"
import assert from "node:assert"
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import http from "node:http"
import { tmpdir } from "node:os"
import path from "node:path"

function startServer() {
  const requests = []
  const server = http.createServer((request, response) => {
    let body = ""
    request.on("data", (chunk) => { body += chunk })
    request.on("end", () => {
      requests.push({ url: request.url, body: body ? JSON.parse(body) : null })
      response.setHeader("Content-Type", "application/json")
      if (request.url === "/api/health/detailed") {
        response.end(JSON.stringify({ status: "ok", storage_backend: "test" }))
        return
      }
      if (request.url === "/api/search/by-tag") {
        response.end(JSON.stringify({
          results: [
            {
              memory: {
                content_hash: "message-1",
                content: "Rate limit on provider X is 50 RPM",
                tags: ["msg:cluster", "agent:cluster-agent-3"],
                created_at_iso: new Date().toISOString(),
              },
            },
          ],
        }))
        return
      }
      response.statusCode = 404
      response.end(JSON.stringify({ detail: "not found" }))
    })
  })

  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const address = server.address()
      resolve({ server, requests, endpoint: `http://127.0.0.1:${address.port}` })
    })
  })
}

test("injects inbox messages and updates toast and sidebar status", async () => {
  const tempDirectory = await mkdtemp(path.join(tmpdir(), "opencode-inbox-"))
  const configPath = path.join(tempDirectory, "memory-plugin.json")
  const statusPath = path.join(tempDirectory, "memory-status.json")
  const inboxStatePath = path.join(tempDirectory, "memory-inbox-state.json")
  const { server, requests, endpoint } = await startServer()

  process.env.OPENCODE_MEMORY_PLUGIN_CONFIG = configPath
  process.env.OPENCODE_MEMORY_STATUS_FILE = statusPath
  process.env.OPENCODE_MEMORY_INBOX_STATE_FILE = inboxStatePath
  process.env.OPENCODE_MEMORY_INBOX_ENABLED = "true"
  process.env.OPENCODE_MEMORY_INBOX_TAGS = "msg:cluster, msg:handoff"
  process.env.OPENCODE_MEMORY_INBOX_MAX_ITEMS = "7"
  process.env.OPENCODE_MEMORY_INBOX_CHECK_INTERVAL_MINUTES = "3"
  process.env.OPENCODE_MEMORY_INBOX_AGENT_ID = "local-agent"

  await writeFile(configPath, JSON.stringify({
    memoryService: {
      endpoint,
      projectQueries: [],
    },
    output: { verbose: false },
    autoCapture: { enabled: false },
    sessionEnd: { enabled: false },
    inbox: {
      enabled: false,
      tags: ["config-value-is-overridden"],
    },
  }))

  const toasts = []
  try {
    const { OpenCodeMemoryPlugin } = await import(`../memory-plugin.js?integration=${Date.now()}`)
    const plugin = await OpenCodeMemoryPlugin({
      directory: tempDirectory,
      client: {
        app: { log: async () => {} },
        tui: { showToast: async (toast) => { toasts.push(toast) } },
      },
    })
    const output = { system: [] }

    await plugin["experimental.chat.system.transform"](
      { sessionID: "session-1" },
      output,
    )

    assert.strictEqual(output.system.length, 1)
    assert.match(output.system[0], /^Messages from other agents/)
    assert.match(output.system[0], /agent:cluster-agent-3/)
    assert.match(output.system[0], /Rate limit on provider X is 50 RPM/)
    assert.strictEqual(toasts.length, 1)
    assert.strictEqual(toasts[0].body.message, "1 new message from other agents")
    assert.strictEqual(
      JSON.parse(await readFile(statusPath, "utf8")).lastAction,
      "1 new message from other agents",
    )
    assert.strictEqual(
      requests.filter((request) => request.url === "/api/search/by-tag").length,
      1,
    )
    assert.deepStrictEqual(
      requests.find((request) => request.url === "/api/search/by-tag").body,
      {
        tags: ["msg:cluster", "msg:handoff"],
        match_all: false,
        time_filter: "last 1 days",
        limit: 7,
      },
    )
    const inboxState = JSON.parse(await readFile(inboxStatePath, "utf8"))
    assert.strictEqual(Object.keys(inboxState.markers).length, 1)
    assert.ok(Number.isFinite(Date.parse(Object.values(inboxState.markers)[0])))
  } finally {
    await new Promise((resolve) => server.close(resolve))
    delete process.env.OPENCODE_MEMORY_PLUGIN_CONFIG
    delete process.env.OPENCODE_MEMORY_STATUS_FILE
    delete process.env.OPENCODE_MEMORY_INBOX_STATE_FILE
    delete process.env.OPENCODE_MEMORY_INBOX_ENABLED
    delete process.env.OPENCODE_MEMORY_INBOX_TAGS
    delete process.env.OPENCODE_MEMORY_INBOX_MAX_ITEMS
    delete process.env.OPENCODE_MEMORY_INBOX_CHECK_INTERVAL_MINUTES
    delete process.env.OPENCODE_MEMORY_INBOX_AGENT_ID
    await rm(tempDirectory, { recursive: true, force: true })
  }
})
