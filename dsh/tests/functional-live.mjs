/**
 * Live functional tests for the DeepSeek Harness bundle.
 *
 * Requires a running MCP Memory Service and a DSH runtime whose fetch trusts
 * its certificate. Not part of CI: the CI runner only picks up *.test.js, and
 * these scenarios need a real service.
 *
 * Run from the repository root:
 *   MCP_API_KEY=... MCP_MEMORY_ENDPOINT=https://localhost:8000 \
 *     ELECTRON_RUN_AS_NODE=1 "$DSH_NODE_EXECUTABLE" dsh/tests/functional-live.mjs
 *
 * Scenarios:
 *   1. project context is injected for a real project (marker walked up),
 *   2. nothing is injected for a directory that is not a project,
 *   3. a decision turn is captured with tags and the classifier's type,
 *   4. #skip suppresses a capture,
 *   5. subagent turns are not captured by default,
 *   6. #remember forces a capture below the length threshold.
 *
 * Every memory the run creates is deleted again at the end.
 */

import assert from 'node:assert/strict'
import { randomUUID } from 'node:crypto'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { apply, findProjectRoot } from '../memory-hooks/index.js'

const ENDPOINT = process.env.MCP_MEMORY_ENDPOINT ?? 'http://127.0.0.1:8000'
const API_KEY = process.env.MCP_API_KEY
const created = new Set()
const cleanupDirs = []

async function request(route, init = {}) {
  const headers = { 'content-type': 'application/json', ...(init.headers ?? {}) }
  if (API_KEY) headers['x-api-key'] = API_KEY
  const response = await fetch(new URL(route, ENDPOINT), {
    ...init,
    headers,
    signal: AbortSignal.timeout(15000),
  })
  const body = await response.json().catch(() => null)
  return { ok: response.ok, status: response.status, body }
}

async function store(content, { tags = ['functional-test'], type = 'decision' } = {}) {
  const result = await request('/api/memories', {
    method: 'POST',
    body: JSON.stringify({ content, tags, memory_type: type }),
  })
  const hash = result.body?.content_hash
  if (hash) created.add(hash)
  return hash
}

async function search(query) {
  const result = await request('/api/search', {
    method: 'POST',
    body: JSON.stringify({ query, n_results: 10 }),
  })
  return (result.body?.results ?? []).map((entry) => entry.memory ?? entry)
}

async function remove(hash) {
  if (!hash) return
  const result = await request(`/api/memories/${hash}`, { method: 'DELETE' })
  // Only a confirmed deletion leaves the cleanup set, so a failed DELETE keeps
  // the hash visible in the final "left behind" report and the exit code.
  if (result.ok && result.body?.success !== false) created.delete(hash)
}

function harness(rawConfig, cwd) {
  const handlers = new Map()
  const contexts = []
  const ctx = {
    systemPrompt: {
      context(entry) {
        contexts.push(entry)
        return () => {}
      },
    },
    workingDirectory: { ensure: async () => cwd },
    on(event, handler) {
      handlers.set(event, handler)
      return () => {}
    },
    effect(fn) {
      return fn()
    },
  }
  apply(ctx, {
    endpoint: ENDPOINT,
    apiKey: API_KEY,
    requireProjectMarker: true,
    ...rawConfig,
  })
  return { handlers, contexts }
}

function emitTurn(handlers, session, { user, assistant }) {
  handlers.get('session/event')(session, { type: 'turn/start', data: { turn: 1 } })
  handlers.get('session/event')(session, {
    type: 'user/message',
    data: { content: [{ type: 'text', text: user }], source: { kind: 'user' } },
  })
  handlers.get('session/event')(session, {
    type: 'assistant/message',
    data: { message: { content: [{ type: 'text', text: assistant }] } },
  })
}

// Record the store response so the test can verify the exact memory without
// depending on semantic ranking right after the write.
const calls = []
const tracked = []
const realFetch = globalThis.fetch
globalThis.fetch = async (url, init) => {
  const response = await realFetch(url, init)
  const entry = { url: String(url), init, bodyPromise: null }
  if (entry.url.endsWith('/api/memories') && init?.method === 'POST') {
    entry.status = response.status
    entry.bodyPromise = response.clone().json().catch(() => null)
    entry.textPromise = response.clone().text().catch(() => '')
    // Track every successful store as soon as its response arrives, before any
    // scenario assertion can fail: a broken test must still clean up after itself.
    tracked.push(entry.bodyPromise.then((body) => {
      if (body?.content_hash) created.add(body.content_hash)
    }).catch(() => {}))
  }
  calls.push(entry)
  return response
}

const results = []
async function scenario(name, fn) {
  calls.length = 0
  try {
    const detail = await fn()
    results.push({ name, status: 'PASS', detail })
  } catch (error) {
    results.push({ name, status: 'FAIL', detail: error.message })
  }
}

const health = await request('/api/health')
assert.ok(health.ok, `service not reachable at ${ENDPOINT} (status ${health.status}); set MCP_MEMORY_ENDPOINT and MCP_API_KEY`)

// Server-side semantic dedup compares embeddings, not just hashes, so every
// stored sentence carries a unique nonce and random words. Two runs of this
// script must stay below the dedup similarity threshold.
const NOISE_WORDS = ['amber', 'basalt', 'cobalt', 'delta', 'ember', 'flint', 'garnet', 'harbor', 'indigo', 'juniper', 'kelp', 'larch', 'marble', 'nickel', 'onyx', 'pewter', 'quartz', 'ridge', 'slate', 'topaz']
function noise(words = 12) {
  return Array.from({ length: words }, () => NOISE_WORDS[Math.floor(Math.random() * NOISE_WORDS.length)]).join(' ')
}

const stamp = Date.now()
const projectRoot = await mkdtemp(path.join(tmpdir(), 'dsh-live-project-'))
cleanupDirs.push(projectRoot)
const nested = path.join(projectRoot, 'packages', 'app')
await mkdir(nested, { recursive: true })
await writeFile(path.join(projectRoot, 'package.json'), '{}')
const project = path.basename(projectRoot)

await scenario('project context: marker above cwd names the project', async () => {
  const sentinel = `LIVE-CTX-${stamp}`
  assert.equal(findProjectRoot(nested), projectRoot, 'marker must be found above the cwd')
  await store(`${project} architecture decisions, recent work and open issues: ${sentinel}. Nonce ${randomUUID()}. ${noise()}`, {
    tags: ['functional-test', project],
  })

  const query = `${project} architecture decisions, recent work and open issues`
  for (let attempt = 0; attempt < 10; attempt += 1) {
    if ((await search(query)).some((m) => (m.content ?? '').includes(sentinel))) break
    await new Promise((resolve) => setTimeout(resolve, 500))
  }

  const { handlers, contexts } = harness({ limit: 10 }, nested)
  const agent = { session: { id: `live-a-${stamp}`, header: {} } }
  await handlers.get('agent/created')({ agent, source: 'startup' })

  const text = contexts[0].text({ agent })
  assert.match(text, new RegExp(`<project-memory project="${project}">`), 'project attribute')
  assert.match(text, new RegExp(sentinel), 'seeded memory must be in the snapshot')
  return `project=${project}, snapshot=${text.length} chars, sentinel found`
})

await scenario('project context: a directory without markers injects nothing', async () => {
  const cwd = await mkdtemp(path.join(tmpdir(), 'dsh-live-noproject-'))
  cleanupDirs.push(cwd)
  assert.equal(findProjectRoot(cwd), null, 'tmp dir must not have a project marker')

  const { handlers, contexts } = harness({}, cwd)
  const agent = { session: { id: `live-b-${stamp}`, header: {} } }
  await handlers.get('agent/created')({ agent, source: 'startup' })

  assert.equal(contexts[0].text({ agent }), '', 'no snapshot expected')
  assert.equal(calls.filter((c) => c.url.endsWith('/api/search')).length, 0, 'no search expected')
  return 'no snapshot, no search request'
})

await scenario('capture: a decision turn is stored with tags and type', async () => {
  const sentinel = `LIVE-CAP-${stamp}`
  const { handlers } = harness({ capture: { minChars: 50 } }, nested)
  const agent = { session: { id: `live-c-${stamp}`, header: {} } }
  emitTurn(handlers, agent.session, {
    user: `Which storage backend should we use? ${noise(6)}`,
    assistant: `We decided to use the sentinel ${sentinel} for the live capture test. Nonce ${randomUUID()}. ${noise()}`,
  })
  await handlers.get('agent/turn-stopping')({ agent, turn: 1 })

  const posts = calls.filter((c) => c.url.endsWith('/api/memories') && c.init?.method === 'POST')
  assert.equal(posts.length, 1, 'exactly one store request')
  const payload = JSON.parse(posts[0].init.body)
  assert.match(payload.content, new RegExp(sentinel))
  assert.equal(payload.memory_type, 'decision')
  assert.ok(payload.tags.includes('dsh-capture'))
  assert.ok(payload.tags.includes(project.toLowerCase()))

  const hash = (await posts[0].bodyPromise)?.content_hash
  assert.ok(hash, `store response must carry a content hash (status ${posts[0].status}: ${(await posts[0].textPromise).slice(0, 200)})`)
  const stored = (await request(`/api/memories/${hash}`)).body
  assert.ok((stored?.content ?? '').includes(sentinel), 'stored content must contain the sentinel')
  return `stored ${hash.slice(0, 12)}, tags=${stored.tags.join(',')}, type=${stored.memory_type}`
})

await scenario('capture: #skip suppresses a matching turn', async () => {
  const sentinel = `LIVE-SKIP-${stamp}`
  const { handlers } = harness({ capture: { minChars: 50 } }, nested)
  const agent = { session: { id: `live-d-${stamp}`, header: {} } }
  emitTurn(handlers, agent.session, {
    user: `#skip we decided to use ${sentinel} ${noise(6)}`,
    assistant: `We decided to use ${sentinel} but the user asked not to store it. ${noise()}`,
  })
  await handlers.get('agent/turn-stopping')({ agent, turn: 1 })
  assert.equal(calls.filter((c) => c.url.endsWith('/api/memories')).length, 0, 'no store request')
  return 'no store request for #skip'
})

await scenario('capture: subagent turns are ignored by default', async () => {
  const sentinel = `LIVE-SUB-${stamp}`
  const { handlers } = harness({ capture: { minChars: 50 } }, nested)
  const agent = { session: { id: `live-e-${stamp}`, header: { parentSession: 'parent' } } }
  emitTurn(handlers, agent.session, {
    user: `Do the subtask. ${noise(6)}`,
    assistant: `We decided to use ${sentinel} for the subtask. ${noise()}`,
  })
  await handlers.get('agent/turn-stopping')({ agent, turn: 1 })
  assert.equal(calls.filter((c) => c.url.endsWith('/api/memories')).length, 0, 'no store request')
  return 'no store request for a subagent session'
})

await scenario('capture: #remember forces a capture below the threshold', async () => {
  const sentinel = `LIVE-REM-${stamp}`
  const { handlers } = harness({ capture: { minChars: 500 } }, nested)
  const agent = { session: { id: `live-f-${stamp}`, header: {} } }
  emitTurn(handlers, agent.session, { user: `#remember keep ${sentinel} ${noise(6)}`, assistant: 'ok' })
  await handlers.get('agent/turn-stopping')({ agent, turn: 1 })
  const posts = calls.filter((c) => c.url.endsWith('/api/memories') && c.init?.method === 'POST')
  assert.equal(posts.length, 1, 'forced store request')
  const payload = JSON.parse(posts[0].init.body)
  assert.equal(payload.memory_type, 'note')
  const hash = (await posts[0].bodyPromise)?.content_hash
  assert.ok(hash, `store response must carry a content hash (status ${posts[0].status}: ${(await posts[0].textPromise).slice(0, 200)})`)
  const stored = (await request(`/api/memories/${hash}`)).body
  assert.ok((stored?.content ?? '').includes(sentinel), 'forced capture must contain the sentinel')
  return 'forced capture stored with type note'
})

await Promise.all(tracked)
for (const dir of cleanupDirs) await rm(dir, { recursive: true, force: true })
for (const hash of [...created]) await remove(hash)

const width = Math.max(...results.map((r) => r.name.length))
console.log('')
for (const result of results) {
  console.log(`${result.status.padEnd(4)}  ${result.name.padEnd(width)}  ${result.detail}`)
}
const failed = results.filter((r) => r.status === 'FAIL')
const leftovers = [...created]
console.log(`\n${results.length - failed.length}/${results.length} scenarios passed; ${leftovers.length} test memories left behind.`)
if (leftovers.length > 0) {
  console.log(`Could not delete: ${leftovers.join(', ')}`)
}
process.exit(failed.length === 0 && leftovers.length === 0 ? 0 : 1)
