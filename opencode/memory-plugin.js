import { readFile } from "node:fs/promises"
import { homedir } from "node:os"
import https from "node:https"
import path from "node:path"

process.env.NODE_TLS_REJECT_UNAUTHORIZED = process.env.NODE_TLS_REJECT_UNAUTHORIZED || "0"

const tlsAgent = new https.Agent({ rejectUnauthorized: false })

function httpsFetch(url, options = {}) {
  return new Promise((resolve, reject) => {
    const { method = "GET", headers = {}, body, signal } = options
    const parsed = new URL(url)

    const req = https.request(
      {
        hostname: parsed.hostname,
        port: parsed.port || (parsed.protocol === "https:" ? 443 : 80),
        path: parsed.pathname + parsed.search,
        method,
        headers,
        agent: tlsAgent,
        signal,
      },
      (res) => {
        let data = ""
        res.on("data", (chunk) => { data += chunk })
        res.on("end", () => resolve({ status: res.statusCode, statusText: res.statusMessage, text: () => Promise.resolve(data), ok: res.statusCode >= 200 && res.statusCode < 300 }))
      },
    )
    req.on("error", reject)
    if (body) req.write(body)
    req.end()
  })
}

const DEFAULT_CONFIG = {
  memoryService: {
    endpoint: "http://127.0.0.1:8000",
    apiKey: "",
    timeoutMs: 5000,
    loadTimeoutMs: 2500,
    maxMemoriesPerSession: 8,
    searchTags: [],
    includeProjectTag: false,
    projectQueries: [
      "{project} architecture decisions",
      "{project} recent work",
      "{project} open issues",
    ],
  },
  output: {
    verbose: true,
    includeTimestamps: true,
    maxContentLength: 280,
  },
  autoCapture: {
    enabled: true,
    minMessageLength: 100,
    maxContentLength: 4000,
    patterns: ["decision", "error", "learning", "implementation", "important", "code"],
    tags: ["auto-capture"],
  },
  sessionEnd: {
    enabled: true,
    minSessionLength: 100,
    maxMemoriesPerSession: 3,
    tags: ["opencode-session", "session-summary"],
  },
  harvest: {
    enabled: false,
    dryRunOnFirstUse: true,
    minSessionMessages: 10,
    sessions: 1,
    useLlm: false,
    minConfidence: 0.6,
    types: ["decision", "bug", "convention", "learning", "context"],
  },
}

function parseInteger(value) {
  if (typeof value !== "string" || !value.trim()) return undefined
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) ? parsed : undefined
}

function environmentOverrides() {
  const overrides = {
    memoryService: {},
  }

  const endpoint = process.env.OPENCODE_MEMORY_ENDPOINT || process.env.OPENCODE_MEMORY_URL
  if (endpoint) {
    overrides.memoryService.endpoint = endpoint
  }

  const apiKey = process.env.OPENCODE_MEMORY_API_KEY
  if (apiKey) {
    overrides.memoryService.apiKey = apiKey
  }

  const timeoutMs = parseInteger(process.env.OPENCODE_MEMORY_TIMEOUT_MS)
  if (timeoutMs !== undefined) {
    overrides.memoryService.timeoutMs = timeoutMs
  }

  const loadTimeoutMs = parseInteger(process.env.OPENCODE_MEMORY_LOAD_TIMEOUT_MS)
  if (loadTimeoutMs !== undefined) {
    overrides.memoryService.loadTimeoutMs = loadTimeoutMs
  }

  return overrides
}

function mergeConfig(base, overrides = {}) {
  return {
    ...base,
    ...overrides,
    memoryService: {
      ...base.memoryService,
      ...(overrides.memoryService || {}),
    },
    output: {
      ...base.output,
      ...(overrides.output || {}),
    },
    autoCapture: {
      ...base.autoCapture,
      ...(overrides.autoCapture || {}),
    },
    sessionEnd: {
      ...base.sessionEnd,
      ...(overrides.sessionEnd || {}),
    },
    harvest: {
      ...base.harvest,
      ...(overrides.harvest || {}),
    },
  }
}

function pluginConfigPaths(directory, options = {}) {
  const configDir = path.join(homedir(), ".config", "opencode")
  const paths = [
    typeof options.configPath === "string" ? options.configPath : "",
    process.env.OPENCODE_MEMORY_PLUGIN_CONFIG || "",
    path.join(configDir, "memory-plugin.json"),
    path.join(configDir, "memory-awareness.json"),
  ]
  if (directory) {
    paths.push(
      path.join(directory, ".opencode", "memory-plugin.json"),
      path.join(directory, ".opencode", "memory-awareness.json"),
    )
  }
  return paths.filter(Boolean)
}

async function loadConfig(directory) {
  let config = DEFAULT_CONFIG

  for (const configPath of pluginConfigPaths(directory)) {
    try {
      const raw = await readFile(configPath, "utf8")
      const parsed = JSON.parse(raw)
      config = mergeConfig(config, parsed)
      break
    } catch {
      // Keep searching for a readable config file.
    }
  }

  config = mergeConfig(config, environmentOverrides())

  return config
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function buildUrl(baseUrl, pathname) {
  const normalizedBase = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`
  const normalizedPath = pathname.startsWith("/") ? pathname.slice(1) : pathname
  return new URL(normalizedPath, normalizedBase).toString()
}

function buildHeaders(config, extraHeaders = {}) {
  const headers = {
    Accept: "application/json",
    ...extraHeaders,
  }

  if (config.memoryService.apiKey) {
    headers["X-API-Key"] = config.memoryService.apiKey
  }

  return headers
}

async function requestJson(config, pathname, init = {}) {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), config.memoryService.timeoutMs)

  try {
    const response = await httpsFetch(buildUrl(config.memoryService.endpoint, pathname), {
      ...init,
      headers: buildHeaders(config, init.headers || {}),
      signal: controller.signal,
    })

    const text = await response.text()
    let body = null
    if (text) {
      try {
        body = JSON.parse(text)
      } catch {
        body = { detail: text }
      }
    }

    if (!response.ok) {
      const detail = body?.detail || body?.error || response.statusText
      throw new Error(`${response.status} ${detail}`)
    }

    return body
  } finally {
    clearTimeout(timeout)
  }
}

function projectNameFromDirectory(directory) {
  return path.basename(directory) || "project"
}

function buildQueries(projectName, config) {
  return config.memoryService.projectQueries
    .map((template) => template.replaceAll("{project}", projectName))
    .filter(Boolean)
}

function normalizeMemory(memory) {
  if (!memory || typeof memory !== "object") return null

  const base = memory.memory && typeof memory.memory === "object" ? memory.memory : memory
  const content = base.content || base.preview || ""
  if (!content) return null

  let createdAt = base.created_at_iso || base.created_at || base.created || undefined
  if (typeof createdAt === "number") {
    const timestamp = createdAt < 4102444800 ? createdAt * 1000 : createdAt
    createdAt = new Date(timestamp).toISOString()
  }

  return {
    id: base.content_hash || base.hash || base.id || content,
    content,
    tags: Array.isArray(base.tags) ? base.tags : [],
    createdAt,
    score: memory.similarity_score || base.similarity_score || base.relevanceScore || 0,
  }
}

function dedupeMemories(memories) {
  const seen = new Set()
  const unique = []

  for (const memory of memories) {
    if (!memory) continue
    if (seen.has(memory.id)) continue
    seen.add(memory.id)
    unique.push(memory)
  }

  return unique
}

function sortMemories(memories) {
  return [...memories].sort((left, right) => {
    if ((right.score || 0) !== (left.score || 0)) {
      return (right.score || 0) - (left.score || 0)
    }

    const leftTime = left.createdAt ? Date.parse(left.createdAt) : 0
    const rightTime = right.createdAt ? Date.parse(right.createdAt) : 0
    return rightTime - leftTime
  })
}

function truncateText(content, maxLength) {
  if (content.length <= maxLength) return content
  return `${content.slice(0, maxLength - 3).trimEnd()}...`
}

function formatTimestamp(memory) {
  if (!memory.createdAt) return ""
  const date = new Date(memory.createdAt)
  if (Number.isNaN(date.getTime())) return ""
  return date.toISOString().slice(0, 10)
}

function formatMemories(projectName, memories, config, options = {}) {
  if (!memories.length) return ""

  const includeHeader = options.includeHeader ?? true
  const limit = options.limit || config.memoryService.maxMemoriesPerSession
  const lines = []

  if (includeHeader) {
    lines.push(`# Memory Context - ${projectName}`)
    lines.push("")
    lines.push("Use this as supporting background only. The current repository state and user instructions take precedence.")
    lines.push("")
  }

  lines.push("## Relevant Memories")

  for (const memory of memories.slice(0, limit)) {
    const timestamp = config.output.includeTimestamps ? formatTimestamp(memory) : ""
    const prefix = timestamp ? `- [${timestamp}] ` : "- "
    lines.push(`${prefix}${truncateText(memory.content.replace(/\s+/g, " ").trim(), config.output.maxContentLength)}`)
  }

  return lines.join("\n")
}

function detectOverrides(content) {
  if (!content) return { forceSkip: false, forceRemember: false }
  const text = typeof content === "string" ? content : JSON.stringify(content)
  return {
    forceSkip: /\b#skip\b/i.test(text),
    forceRemember: /\b#remember\b/i.test(text),
  }
}

function splitSentences(text) {
  const blocks = []
  let lastIndex = 0
  const codeBlockRe = /```[\s\S]*?```/g
  let match
  while ((match = codeBlockRe.exec(text)) !== null) {
    if (match.index > lastIndex) {
      blocks.push({ type: "text", content: text.slice(lastIndex, match.index) })
    }
    blocks.push({ type: "code", content: match[0] })
    lastIndex = match.index + match[0].length
  }
  if (lastIndex < text.length) {
    blocks.push({ type: "text", content: text.slice(lastIndex) })
  }
  return blocks
}

function splitTextSentences(text) {
  const result = []
  const re = /[^.!?\n]+[.!?]+\s*/g
  let match
  while ((match = re.exec(text)) !== null) {
    const s = match[0].trim()
    if (s) result.push(s)
  }
  const remainder = text.replace(re, "").trim()
  if (remainder) result.push(remainder)
  return result.length ? result : [text.trim()].filter(Boolean)
}

function detectValuableContent(text, config) {
  const patterns = config.autoCapture.patterns || DEFAULT_CONFIG.autoCapture.patterns
  const minLength = config.autoCapture.minMessageLength || DEFAULT_CONFIG.autoCapture.minMessageLength
  if (!text || text.length < minLength) return { isValuable: false, reason: "too short", memoryType: null, matchedPattern: null }

  const matchers = {
    decision: /\b(decided to|decision|chose to|will use|going with|opting for|better to|should use)\b/i,
    error: /\b(error|bug|crash|failed|broken|exception|stack trace|regression|fixing)\b/i,
    learning: /\b(learned|discovered|realized|turns out|insight|understanding|key finding|important to note)\b/i,
    implementation: /\b(implemented|built|created|added|refactored|extracted|migrated|deployed)\b/i,
    important: /\b(important|critical|notable|significant|worth noting|key takeaway)\b/i,
    code: /```[\s\S]*?```/,
  }

  const blocks = splitSentences(text)
  const matched = []
  const types = new Set()

  for (const block of blocks) {
    if (block.type === "code") {
      if (patterns.includes("code")) {
        matched.push(block.content)
        types.add("code")
      }
      continue
    }
    const sentences = splitTextSentences(block.content)
    for (const sentence of sentences) {
      const lower = sentence.toLowerCase()
      for (const [name, regex] of Object.entries(matchers)) {
        if (name === "code") continue
        if (!patterns.includes(name)) continue
        if (regex.test(lower)) {
          matched.push(sentence)
          types.add(name)
          break
        }
      }
    }
  }

  if (matched.length === 0) {
    return { isValuable: false, reason: "no pattern match", memoryType: null, matchedPattern: null, confidence: 0 }
  }

  const typeOrder = ["decision", "error", "learning", "implementation", "important", "code"]
  const bestType = typeOrder.find((t) => types.has(t)) || [...types][0]
  return { isValuable: true, memoryType: bestType, matchedPattern: "sentence-split", confidence: 0.8, matchedContent: matched.join("\n") }
}

function analyzeSessionMessages(messages) {
  const analysis = {
    topics: [],
    decisions: [],
    insights: [],
    codeChanges: [],
    nextSteps: [],
    sessionLength: 0,
    confidence: 0,
  }

  if (!messages?.length) return analysis

  const text = messages.map((m) => m.content || "").join("\n")
  analysis.sessionLength = text.length

  const topicMatchers = {
    implementation: /implement|building|create|adding/i,
    debugging: /debug|bug|error|fix|issue|problem/i,
    architecture: /architecture|design|structure|pattern|framework/i,
    performance: /performance|optimization|speed|efficient/i,
    testing: /test|testing|coverage|spec/i,
    deployment: /deploy|production|release|ci/i,
    configuration: /config|setup|environment|settings/i,
    database: /database|schema|migration|query/i,
    api: /api|endpoint|rest|service|interface/i,
    ui: /ui|interface|component|styling/i,
  }
  for (const [topic, re] of Object.entries(topicMatchers)) {
    if (re.test(text)) analysis.topics.push(topic)
  }

  const decisionRe = /\b(decided to|decision to|chose to|will use|going with|better to|we should)\b/i
  for (const msg of messages) {
    const c = msg.content || ""
    if (decisionRe.test(c) && c.length > 20) analysis.decisions.push(c.trim().slice(0, 300))
    if (/\b(learned|discovered|realized|turns out|insight)\b/i.test(c) && c.length > 20) analysis.insights.push(c.trim().slice(0, 300))
    if (/\b(implemented|added|created|refactored|fixed|built)\b/i.test(c) && /```/.test(c)) analysis.codeChanges.push(c.trim().slice(0, 300))
    if (/\b(next|todo|need to|should|plan to|continue|follow up)\b/i.test(c) && c.length > 15) analysis.nextSteps.push(c.trim().slice(0, 200))
  }

  analysis.decisions = analysis.decisions.slice(0, 3)
  analysis.insights = analysis.insights.slice(0, 3)
  analysis.codeChanges = analysis.codeChanges.slice(0, 4)
  analysis.nextSteps = analysis.nextSteps.slice(0, 4)

  const total = analysis.topics.length + analysis.decisions.length + analysis.insights.length + analysis.codeChanges.length + analysis.nextSteps.length
  analysis.confidence = Math.min(1, total / 10)

  return analysis
}

function deriveProjectPath(directory) {
  if (!directory) return null
  return directory.split(path.sep).join("-")
}

async function storeMemoryHttp(config, content, tags, memoryType, metadata = {}) {
  const payload = {
    content,
    tags: [...new Set(tags.filter(Boolean).map((t) => String(t).toLowerCase()))],
    memory_type: memoryType || "note",
    metadata: {
      source: "opencode-plugin",
      ...metadata,
      captured_at: new Date().toISOString(),
    },
  }
  return requestJson(config, "/api/memories", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
}

async function postHarvest(config, body) {
  return requestJson(config, "/api/harvest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
}

async function searchMemories(config, query, tags, limit) {
  // /api/search is semantic-only and ignores tag filters server-side
  // (see SemanticSearchRequest in src/mcp_memory_service/web/api/search.py).
  // When the caller passes tags, over-fetch and filter client-side so
  // project-scoped searches actually stay scoped.
  const hasTagFilter = tags.length > 0
  const payload = {
    query,
    n_results: hasTagFilter ? Math.max(limit * 4, 20) : limit,
  }

  const result = await requestJson(config, "/api/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })

  const memories = Array.isArray(result)
    ? result
    : Array.isArray(result?.memories)
      ? result.memories
      : Array.isArray(result?.results)
        ? result.results
        : []

  let normalized = memories.map(normalizeMemory).filter(Boolean)

  if (hasTagFilter) {
    const wanted = new Set(tags)
    normalized = normalized.filter((memory) =>
      memory.tags.some((tag) => wanted.has(tag)),
    )
    normalized = normalized.slice(0, limit)
  }

  return normalized
}

async function getHealth(config) {
  return requestJson(config, "/api/health")
}

function tagsForProject(projectName, config) {
  const tags = [...config.memoryService.searchTags]
  if (config.memoryService.includeProjectTag) {
    tags.push(projectName)
  }
  return tags
}

async function loadSessionMemories({ config, directory, logInfo, logWarn, healthState }) {
  const projectName = projectNameFromDirectory(directory)
  const tags = tagsForProject(projectName, config)
  const queries = buildQueries(projectName, config)
  const perQueryLimit = Math.max(2, Math.ceil(config.memoryService.maxMemoriesPerSession / Math.max(queries.length, 1)))

  if (!healthState.checked) {
    healthState.checked = true
    try {
      const health = await getHealth(config)
      const backend = health?.storage_backend || health?.backend || "unknown"
      await logInfo(`Memory service connected (${backend})`)
    } catch (error) {
      await logWarn(`Memory service unavailable: ${error.message}`)
    }
  }

  const searchResults = await Promise.allSettled(
    queries.map((query) => searchMemories(config, query, tags, perQueryLimit)),
  )

  const found = []
  for (const [index, result] of searchResults.entries()) {
    if (result.status === "fulfilled") {
      found.push(...result.value)
      continue
    }

    await logWarn(`Memory search failed for "${queries[index]}": ${result.reason?.message || result.reason}`)
  }

  const deduped = sortMemories(dedupeMemories(found)).slice(0, config.memoryService.maxMemoriesPerSession)
  if (deduped.length) {
    await logInfo(`Loaded ${deduped.length} memories for ${projectName}`)
  }

  return {
    projectName,
    memories: deduped,
  }
}

const createPlugin = async ({ directory, client }) => {
  const config = await loadConfig(directory)
  const sessionState = new Map()
  const healthState = { checked: false }
  const harvestFirstRun = { done: false }
  const appLog = client?.app?.log?.bind?.(client.app) || (() => {})

  const logInfo = async (message) => {
    if (!config.output.verbose) return
    await appLog({ service: "opencode-memory", level: "info", message }).catch(() => {})
  }

  const logWarn = async (message) => {
    if (!config.output.verbose) return
    await appLog({ service: "opencode-memory", level: "warn", message }).catch(() => {})
  }

  const refreshSession = (sessionID, sessionDirectory) => {
    const existingState = sessionState.get(sessionID)
    if (existingState?.promise) return existingState.promise

    const loadPromise = loadSessionMemories({
      config,
      directory: sessionDirectory,
      logInfo,
      logWarn,
      healthState,
    })
      .then((result) => {
        sessionState.set(sessionID, {
          ...result,
          messages: [],
          promise: null,
        })
      })
      .catch(async (error) => {
        sessionState.set(sessionID, {
          projectName: projectNameFromDirectory(sessionDirectory),
          memories: [],
          messages: [],
          promise: null,
        })
        await logWarn(`Memory load failed: ${error.message}`)
      })

    sessionState.set(sessionID, {
      projectName: projectNameFromDirectory(sessionDirectory),
      memories: [],
      messages: [],
      promise: loadPromise,
    })

    return loadPromise
  }

  const waitForSession = async (sessionID, fallbackDirectory) => {
    let state = sessionState.get(sessionID)

    if (!state) {
      refreshSession(sessionID, fallbackDirectory)
      state = sessionState.get(sessionID)
    }

    if (state?.promise) {
      const loadTimedOut = Symbol("load-timed-out")
      const result = await Promise.race([
        state.promise.then(() => null),
        sleep(config.memoryService.loadTimeoutMs).then(() => loadTimedOut),
      ])

      if (result === loadTimedOut) {
        const latestState = sessionState.get(sessionID)
        if (latestState?.promise) {
          return null
        }
      }
    }

    return sessionState.get(sessionID)
  }

  const handleSessionEnd = async (sessionID, sessionDirectory) => {
    try {
      const state = sessionState.get(sessionID) || { projectName: projectNameFromDirectory(sessionDirectory), messages: [] }

      // --- Session-End Consolidation ---
      if (config.sessionEnd.enabled && state.messages?.length > 0) {
        const text = state.messages.map((m) => m.content || "").join("\n")
        if (text.length >= (config.sessionEnd.minSessionLength || 100)) {
          const analysis = analyzeSessionMessages(state.messages)
          if (analysis.confidence >= 0.1) {
            const tags = [
              ...config.sessionEnd.tags,
              state.projectName,
              ...analysis.topics.slice(0, 3),
              `confidence:${Math.round(analysis.confidence * 100)}`,
            ]
            const consolidation = [
              `## Session Summary — ${state.projectName}`,
              "",
              `**Topics:** ${analysis.topics.join(", ") || "general"}`,
              analysis.decisions.length ? `\n**Decisions:**\n${analysis.decisions.map((d) => `- ${d}`).join("\n")}` : "",
              analysis.insights.length ? `\n**Insights:**\n${analysis.insights.map((d) => `- ${d}`).join("\n")}` : "",
              analysis.codeChanges.length ? `\n**Code Changes:**\n${analysis.codeChanges.map((d) => `- ${d}`).join("\n")}` : "",
              analysis.nextSteps.length ? `\n**Next Steps:**\n${analysis.nextSteps.map((d) => `- ${d}`).join("\n")}` : "",
            ].filter(Boolean).join("\n")

            try {
              await storeMemoryHttp(config, consolidation, tags, "session-summary", {
                session_analysis: {
                  topics: analysis.topics,
                  decisions_count: analysis.decisions.length,
                  insights_count: analysis.insights.length,
                  code_changes_count: analysis.codeChanges.length,
                  next_steps_count: analysis.nextSteps.length,
                  session_length: analysis.sessionLength,
                  confidence: analysis.confidence,
                },
                session_id: sessionID,
              })
              await logInfo(`Session summary stored for ${state.projectName}`)
            } catch (error) {
              await logWarn(`Session summary store failed: ${error.message}`)
            }
          }
        }
      }

      // --- Session-End Harvest ---
      const harvestCfg = config.harvest
      if (harvestCfg.enabled && state.messages?.length >= (harvestCfg.minSessionMessages || 10)) {
        const projectPath = deriveProjectPath(sessionDirectory)
        if (projectPath) {
          const forcedDryRun = harvestCfg.dryRunOnFirstUse !== false && !harvestFirstRun.done
          try {
            const result = await postHarvest(config, {
              sessions: harvestCfg.sessions || 1,
              use_llm: !!harvestCfg.useLlm,
              dry_run: forcedDryRun || !!harvestCfg.dryRun,
              min_confidence: harvestCfg.minConfidence || 0.6,
              types: Array.isArray(harvestCfg.types) ? harvestCfg.types : ["decision", "bug", "convention", "learning"],
              project_path: projectPath,
            })
            const found = result?.results?.reduce((s, r) => s + (r.found || 0), 0) || 0
            const stored = result?.results?.reduce((s, r) => s + (r.stored || 0), 0) || 0
            await logInfo(`Harvest: ${found} candidates, ${stored} stored (dry_run=${forcedDryRun || !!result?.dry_run})`)
            if (forcedDryRun) harvestFirstRun.done = true
          } catch (error) {
            await logWarn(`Harvest failed: ${error.message}`)
          }
        }
      }
    } catch (error) {
      await logWarn(`Session end handler error: ${error.message}`)
    }
  }

  const handleMessagePart = async (sessionID, part) => {
    if (part.type !== "text") return
    const text = part.text
    if (!text || text.length === 0) return

    if (!config.autoCapture.enabled) return
    const overrides = detectOverrides(text)
    if (overrides.forceSkip) return

    let state = sessionState.get(sessionID)
    if (!state) {
      state = { projectName: projectNameFromDirectory(directory), memories: [], messages: [] }
      sessionState.set(sessionID, state)
    }

    if (!state._capturedParts) state._capturedParts = new Set()
    if (state._capturedParts.has(part.id)) return
    state._capturedParts.add(part.id)

    state.messages.push({ role: "unknown", content: text })

    const detection = detectValuableContent(text, config)
    const isValuable = overrides.forceRemember || detection.isValuable

    if (isValuable) {
      const projectName = state.projectName
      const memoryType = overrides.forceRemember ? "note" : detection.memoryType
      const tags = [
        ...config.autoCapture.tags,
        memoryType,
        projectName.toLowerCase(),
      ]
      const maxLen = config.autoCapture.maxContentLength || 4000
      const captureText = detection.matchedContent || text
      const content = captureText.length > maxLen ? captureText.slice(0, maxLen - 3) + "..." : captureText

      try {
        await storeMemoryHttp(config, content, tags, memoryType)
        await logInfo(`Auto-captured ${memoryType}`)
      } catch (error) {
        await logWarn(`Auto-capture failed: ${error.message}`)
      }
    }
  }

  return {
    event: async ({ event }) => {
      if (event.type === "session.created") {
        const sid = event.properties.info.id
        const sdir = event.properties.info.directory || directory
        refreshSession(sid, sdir)
      }

      if (event.type === "session.deleted") {
        const sid = event.properties.info.id
        const sdir = event.properties.info.directory || directory
        await handleSessionEnd(sid, sdir)
        sessionState.delete(sid)
      }

      if (event.type === "message.part.updated") {
        await handleMessagePart(event.properties.sessionID, event.properties.part)
      }
    },



    "experimental.chat.system.transform": async (input, output) => {
      if (!input.sessionID) return

      const state = await waitForSession(input.sessionID, directory)
      if (!state?.memories?.length) return

      const formatted = formatMemories(state.projectName, state.memories, config)
      if (formatted) {
        output.system.push(formatted)
      }
    },

    "experimental.session.compacting": async (input, output) => {
      if (!input.sessionID) return

      const state = await waitForSession(input.sessionID, directory)
      if (!state?.memories?.length) return

      const formatted = formatMemories(state.projectName, state.memories, config, {
        includeHeader: false,
        limit: Math.min(6, config.memoryService.maxMemoriesPerSession),
      })

      if (formatted) {
        output.context.push(formatted)
      }
    },
  }
}

export const OpenCodeMemoryPlugin = createPlugin
export default { id: "opencode-memory", server: createPlugin }
