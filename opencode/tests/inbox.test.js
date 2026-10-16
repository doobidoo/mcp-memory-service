/**
 * Tests for the OpenCode inter-agent inbox (#1527).
 *
 * Run: node opencode/tests/inbox.test.js
 */

import { test } from "node:test"
import assert from "node:assert"

import { _internal } from "../memory-plugin.js"

const {
  DEFAULT_CONFIG,
  createInboxChecker,
  formatInboxBlock,
  searchInboxMemories,
} = _internal

function inboxConfig(overrides = {}) {
  return {
    ...DEFAULT_CONFIG,
    memoryService: { ...DEFAULT_CONFIG.memoryService },
    output: { ...DEFAULT_CONFIG.output },
    inbox: {
      ...DEFAULT_CONFIG.inbox,
      enabled: true,
      ...overrides,
    },
  }
}

test("the default configuration disables the inbox", async () => {
  assert.strictEqual(DEFAULT_CONFIG.inbox.enabled, false)

  let searches = 0
  const checker = createInboxChecker({
    config: { ...DEFAULT_CONFIG, inbox: { ...DEFAULT_CONFIG.inbox } },
    search: async () => { searches += 1; return [] },
  })

  assert.strictEqual(await checker.check("session-1"), null)
  assert.strictEqual(searches, 0)
})

test("the marker advances only after a successful query", async () => {
  const previous = "2026-10-14T08:00:00.000Z"
  const current = new Date("2026-10-14T09:00:00.000Z")
  const writes = []
  const successful = createInboxChecker({
    config: inboxConfig(),
    now: () => current,
    readMarker: async () => previous,
    writeMarker: async (_key, value) => { writes.push(value) },
    search: async () => [],
  })

  await successful.check("session-1")
  assert.deepStrictEqual(writes, [current.toISOString()])

  let attempts = 0
  const failedWrites = []
  const failing = createInboxChecker({
    config: inboxConfig(),
    now: () => current,
    readMarker: async () => previous,
    writeMarker: async (_key, value) => { failedWrites.push(value) },
    search: async () => {
      attempts += 1
      throw new Error("server unavailable")
    },
  })

  await assert.rejects(failing.check("session-2"), /server unavailable/)
  await assert.rejects(failing.check("session-2"), /server unavailable/)
  assert.strictEqual(attempts, 2, "a failed first check remains retryable")
  assert.deepStrictEqual(failedWrites, [])
})

test("tag search excludes the configured agent and exact-old results", async () => {
  const calls = []
  const config = inboxConfig({ agentId: "local-agent", maxItems: 10 })
  const since = new Date("2026-10-14T08:00:00.000Z")
  const now = new Date("2026-10-15T10:00:00.000Z")
  const request = async (receivedConfig, pathname, init) => {
    calls.push({ receivedConfig, pathname, body: JSON.parse(init.body) })
    return {
      results: [
        {
          memory: {
            content_hash: "own",
            content: "my own message",
            tags: ["msg:cluster", "agent:local-agent"],
            created_at_iso: "2026-10-15T09:30:00.000Z",
          },
        },
        {
          memory: {
            content_hash: "other",
            content: "Rate limit is 50 RPM",
            tags: ["msg:cluster", "agent:cluster-agent-3"],
            created_at_iso: "2026-10-15T09:42:00.000Z",
          },
        },
        {
          memory: {
            content_hash: "old",
            content: "already seen",
            tags: ["msg:cluster", "agent:cluster-agent-2"],
            created_at_iso: "2026-10-14T07:59:59.000Z",
          },
        },
      ],
    }
  }

  const memories = await searchInboxMemories(config, since, now, request)

  assert.deepStrictEqual(memories.map((memory) => memory.id), ["other"])
  assert.strictEqual(calls.length, 1)
  assert.strictEqual(calls[0].pathname, "/api/search/by-tag")
  assert.deepStrictEqual(calls[0].body, {
    tags: ["msg:cluster"],
    match_all: false,
    time_filter: "last 2 days",
    limit: 10,
  })
})

test("checks the first prompt per session and throttles later prompts", async () => {
  let current = new Date("2026-10-14T09:00:00.000Z")
  let searches = 0
  const checker = createInboxChecker({
    config: inboxConfig({ checkIntervalMinutes: 5 }),
    now: () => current,
    readMarker: async () => "2026-10-14T08:00:00.000Z",
    writeMarker: async () => {},
    search: async () => { searches += 1; return [] },
  })

  await checker.check("session-1")
  assert.strictEqual(searches, 1)

  current = new Date("2026-10-14T09:01:00.000Z")
  assert.strictEqual(await checker.check("session-1"), null)
  assert.strictEqual(searches, 1)

  await checker.check("session-2")
  assert.strictEqual(searches, 2, "a new session always checks its first prompt")

  current = new Date("2026-10-14T09:06:00.000Z")
  await checker.check("session-2")
  assert.strictEqual(searches, 3)
})

test("formats the shared Claude Code and OpenCode inbox block", () => {
  const block = formatInboxBlock(
    [
      {
        content: "Rate limit on provider X is 50 RPM",
        tags: ["msg:cluster", "agent:cluster-agent-3"],
        createdAt: "2026-10-14T09:42:00.000Z",
      },
    ],
    new Date("2026-10-14T08:10:00.000Z"),
    inboxConfig(),
  )

  assert.strictEqual(
    block,
    "Messages from other agents (since 2026-10-14 08:10):\n"
      + "- [agent:cluster-agent-3, 09:42] Rate limit on provider X is 50 RPM",
  )
})
