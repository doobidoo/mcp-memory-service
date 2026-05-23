import { readFile } from "node:fs/promises"
import { homedir } from "node:os"
import path from "node:path"

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
}

function pluginOptionOverrides(options = {}) {
  const { configPath: _configPath, ...rest } = options
  return rest
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
  }
}

function pluginConfigPaths(directory, options = {}) {
  const configDir = path.join(homedir(), ".config", "opencode")
  return [
    typeof options.configPath === "string" ? options.configPath : "",
    process.env.OPENCODE_MEMORY_PLUGIN_CONFIG || "",
    path.join(configDir, "memory-plugin.json"),
    path.join(configDir, "memory-awareness.json"),
    path.join(directory, ".opencode", "memory-plugin.json"),
    path.join(directory, ".opencode", "memory-awareness.json"),
  ].filter(Boolean)
}

async function loadConfig(directory, options) {
  let config = DEFAULT_CONFIG

  for (const configPath of pluginConfigPaths(directory, options)) {
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
  config = mergeConfig(config, pluginOptionOverrides(options))

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
    headers.Authorization = `Bearer ${config.memoryService.apiKey}`
  }

  return headers
}

async function requestJson(config, pathname, init = {}) {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), config.memoryService.timeoutMs)

  try {
    const response = await fetch(buildUrl(config.memoryService.endpoint, pathname), {
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

export const OpenCodeMemoryPlugin = async ({ client, directory }, options = {}) => {
  const config = await loadConfig(directory, options)
  const sessionState = new Map()
  const healthState = { checked: false }
  const appLog = client.app.log.bind(client.app)

  const logInfo = async (message) => {
    if (!config.output.verbose) return
    await appLog({ body: { service: "opencode-memory", level: "info", message } }).catch(() => {})
  }

  const logWarn = async (message) => {
    if (!config.output.verbose) return
    await appLog({ body: { service: "opencode-memory", level: "warn", message } }).catch(() => {})
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
          promise: null,
        })
      })
      .catch(async (error) => {
        sessionState.set(sessionID, {
          projectName: projectNameFromDirectory(sessionDirectory),
          memories: [],
          promise: null,
        })
        await logWarn(`Memory load failed: ${error.message}`)
      })

    sessionState.set(sessionID, {
      projectName: projectNameFromDirectory(sessionDirectory),
      memories: [],
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

  return {
    event: async ({ event }) => {
      if (event.type === "session.created") {
        refreshSession(event.properties.info.id, event.properties.info.directory || directory)
      }

      if (event.type === "session.deleted") {
        sessionState.delete(event.properties.info.id)
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
