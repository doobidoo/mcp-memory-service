/**
 * Tests for the OpenCode V2 adapter in memory-plugin.js.
 *
 * Run: node opencode/tests/v2-plugin-adapter.test.js
 *
 * OpenCode V2 loads a plugin through `default.setup()` and renamed every hook
 * and event the V1 implementation listens for, so the adapter translates V2
 * events into the V1 shapes at the call site. These tests pin that translation:
 * the V1 branches dereference `properties.info.id` and
 * `properties.sessionID` unguarded, so a wrong or half-built mapping either
 * throws inside the event loop (where the subscription swallows it) or writes
 * session state under `undefined`. Neither shows up as a visible failure in a
 * running OpenCode, which is why the mapping is covered here instead.
 */

import { test } from "node:test"
import assert from "node:assert"

import { _internal } from "../memory-plugin.js"

const { sessionIdentity, translateEvent, transcriptCaptureEvents } = _internal

const DIRECTORY = "C:\\Users\\example\\project"

function v2Event(type, data, location = { directory: DIRECTORY }) {
  return { id: "evt_1", created: 1, type, location, data }
}

test("session step that ends a tool round maps to nothing", () => {
  const event = v2Event("session.step.ended", { sessionID: "ses_1", finish: "tool-calls" })
  assert.strictEqual(translateEvent(event, DIRECTORY), null)
})

test("session step that ends a turn maps to session.idle", () => {
  const event = v2Event("session.step.ended", { sessionID: "ses_1", finish: "stop" })
  assert.deepStrictEqual(translateEvent(event, DIRECTORY), {
    type: "session.idle",
    properties: { info: { id: "ses_1", directory: DIRECTORY } },
  })
})

test("session.created keeps its own type when it carries an id", () => {
  const event = v2Event("session.created", { info: { id: "ses_1", directory: DIRECTORY } })
  assert.deepStrictEqual(translateEvent(event, DIRECTORY), {
    type: "session.created",
    properties: { info: { id: "ses_1", directory: DIRECTORY } },
  })
})

test("session.created without an id is dropped instead of reaching the unguarded V1 branch", () => {
  const event = v2Event("session.created", {})
  assert.strictEqual(translateEvent(event, DIRECTORY), null)
})

test("session removal maps to session.deleted", () => {
  const event = v2Event("session.removed", { sessionID: "ses_1" })
  assert.strictEqual(translateEvent(event, DIRECTORY).type, "session.deleted")
})

test("events the V1 handler does not know pass through untouched", () => {
  const event = v2Event("session.tool.called", { sessionID: "ses_1" })
  assert.strictEqual(translateEvent(event, DIRECTORY), event)
})

test("identity prefers the message envelope and falls back to the directory passed in", () => {
  assert.deepStrictEqual(sessionIdentity(v2Event("session.step.ended", { sessionID: "ses_1" }), DIRECTORY), {
    id: "ses_1",
    directory: DIRECTORY,
  })
  assert.deepStrictEqual(sessionIdentity(v2Event("session.step.ended", {}, {}), "C:\\fallback"), {
    id: undefined,
    directory: "C:\\fallback",
  })
})

test("transcript capture reads assistant text parts only", () => {
  const events = transcriptCaptureEvents({
    sessionID: "ses_1",
    messages: [
      { id: "msg_user", role: "user", content: [{ type: "text", text: "asked" }] },
      {
        id: "msg_1",
        role: "assistant",
        content: [
          { type: "reasoning", text: "thinking out loud" },
          { type: "text", text: "an answer" },
          { type: "tool-call", id: "call_1", name: "shell" },
        ],
      },
      { id: "msg_2", role: "assistant", content: [{ type: "text", text: "" }] },
    ],
  })

  assert.deepStrictEqual(events, [
    {
      type: "message.part.updated",
      properties: {
        sessionID: "ses_1",
        part: { id: "msg_1:1", type: "text", text: "an answer" },
      },
    },
  ])
})

test("transcript capture ids stay stable so repeat scans do not re-capture", () => {
  const transcript = {
    sessionID: "ses_1",
    messages: [{ id: "msg_1", role: "assistant", content: [{ type: "text", text: "hello" }] }],
  }
  assert.deepStrictEqual(transcriptCaptureEvents(transcript), transcriptCaptureEvents(transcript))
})

test("transcript capture tolerates a missing transcript", () => {
  assert.deepStrictEqual(transcriptCaptureEvents({}), [])
  assert.deepStrictEqual(transcriptCaptureEvents(undefined), [])
})
