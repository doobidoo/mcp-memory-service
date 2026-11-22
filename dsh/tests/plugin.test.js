import { test } from 'node:test'
import assert from 'node:assert/strict'

import { apply, formatProjectContext } from '../memory-hooks/index.js'

function harness(rawConfig = {}, options = {}) {
  const handlers = new Map()
  const contexts = []
  const ctx = {
    systemPrompt: {
      context(entry) {
        contexts.push(entry)
        return () => {}
      },
    },
    workingDirectory: {
      ensure: async () => options.cwd ?? '/work/mcp-memory-service',
    },
    on(event, handler) {
      handlers.set(event, handler)
      return () => {}
    },
    effect(fn) {
      return fn()
    },
  }
  apply(ctx, rawConfig)
  return { handlers, contexts }
}

function memory(content, memoryType = 'decision') {
  return { content, memory_type: memoryType }
}

async function withFetch(impl, run) {
  const original = globalThis.fetch
  globalThis.fetch = impl
  try {
    return await run()
  } finally {
    globalThis.fetch = original
  }
}

test('disabled config registers nothing', () => {
  const { handlers, contexts } = harness({ enabled: false })
  assert.equal(contexts.length, 0)
  assert.equal(handlers.size, 0)
})

test('a new session gets a delimited project snapshot', async () => {
  const calls = []
  await withFetch(async (url, init) => {
    calls.push({ url: String(url), init })
    return { ok: true, json: async () => ({ results: [memory('Use sqlite_vec for local storage')] }) }
  }, async () => {
    const { handlers, contexts } = harness({ endpoint: 'http://memory.test', apiKey: 'secret' })
    assert.equal(contexts.length, 1)
    const agent = {}
    await handlers.get('agent/created')({ agent, source: 'startup' })

    const text = contexts[0].text({ agent })
    assert.match(text, /<project-memory project="mcp-memory-service">/)
    assert.match(text, /not as instructions/)
    assert.match(text, /\[decision\] Use sqlite_vec/)
    assert.equal(contexts[0].text({}), '')
    assert.equal(contexts[0].interpolate, false)

    assert.equal(calls.length, 1)
    assert.equal(calls[0].url, 'http://memory.test/api/search')
    assert.equal(calls[0].init.headers['x-api-key'], 'secret')
    assert.match(JSON.parse(calls[0].init.body).query, /mcp-memory-service architecture decisions/)
  })
})

test('the /api/search wrapper { memory, similarity_score } is unwrapped', async () => {
  await withFetch(async () => ({
    ok: true,
    json: async () => ({ results: [{ memory: memory('Wrapped result survives'), similarity_score: 0.9 }] }),
  }), async () => {
    const { handlers, contexts } = harness()
    const agent = {}
    await handlers.get('agent/created')({ agent, source: 'startup' })
    assert.match(contexts[0].text({ agent }), /Wrapped result survives/)
  })
})

test('resume and compact do not search again', async () => {
  let calls = 0
  await withFetch(async () => {
    calls += 1
    return { ok: true, json: async () => ({ results: [] }) }
  }, async () => {
    const { handlers } = harness()
    const agent = {}
    await handlers.get('agent/created')({ agent, source: 'resume' })
    await handlers.get('agent/created')({ agent, source: 'compact' })
    assert.equal(calls, 0)
  })
})

test('a failing service leaves the prompt unchanged', async () => {
  await withFetch(async () => {
    throw new Error('connection refused')
  }, async () => {
    const { handlers, contexts } = harness()
    const agent = {}
    await handlers.get('agent/created')({ agent, source: 'startup' })
    assert.equal(contexts[0].text({ agent }), '')
  })
})

test('an http error result is ignored', async () => {
  await withFetch(async () => ({ ok: false, status: 500 }), async () => {
    const { handlers, contexts } = harness()
    const agent = {}
    await handlers.get('agent/created')({ agent, source: 'startup' })
    assert.equal(contexts[0].text({ agent }), '')
  })
})

test('disposing the agent drops its snapshot', async () => {
  await withFetch(async () => ({ ok: true, json: async () => ({ results: [memory('keep me')] }) }), async () => {
    const { handlers, contexts } = harness()
    const agent = {}
    await handlers.get('agent/created')({ agent, source: 'startup' })
    assert.match(contexts[0].text({ agent }), /keep me/)
    handlers.get('agent/disposed')({ agent })
    assert.equal(contexts[0].text({ agent }), '')
  })
})

test('the snapshot is capped and skips empty content', () => {
  const text = formatProjectContext('demo', [memory(''), memory('x'.repeat(500))], 120)
  assert.ok(text.length <= 120)
  assert.doesNotMatch(text, /- \[decision\] $/)
  assert.match(text, /…$/)
})
