/**
 * Auto-capture pattern detection for the DeepSeek Harness adapter.
 *
 * Ported from `claude-hooks/utilities/auto-capture-patterns.js` in this
 * repository so the DSH bundle stays self-contained when installed from its
 * directory. The patterns, priorities and memory types are intentionally the
 * same as the Claude Code and Codex adapters.
 *
 * @module @doobidoo/mcp-memory-service-dsh/patterns
 */

export const PATTERNS = {
  decision: {
    regex: /\b(decided|chose|will use|let's go with|i'll use|we'll use|settled on|going with|picked|selected|opting for|entschieden|gewählt|nehmen wir|verwenden wir|machen wir|nutzen wir|ausgewählt)/i,
    memoryType: 'decision',
    priority: 1,
    confidence: 0.9,
  },
  error: {
    regex: /\b(error|exception|failed|fixed|bug|issue|crash|broken|resolved|solved|debugging|debugged|patched|fehler|behoben|gefixt|problem|kaputt|gelöst|repariert|fehlerbehebung)/i,
    memoryType: 'error',
    priority: 2,
    confidence: 0.85,
  },
  learning: {
    regex: /\b(learned|discovered|realized|found out|turns out|interestingly|til|understanding now|now i see|aha|insight|gelernt|entdeckt|herausgefunden|stellte sich heraus|interessanterweise|jetzt verstehe ich)/i,
    memoryType: 'learning',
    priority: 3,
    confidence: 0.85,
  },
  implementation: {
    regex: /\b(implemented|created|built|added|refactored|set up|configured|deployed|developed|wrote|coding|programmed|implementiert|erstellt|gebaut|hinzugefügt|konfiguriert|eingerichtet|refaktoriert|entwickelt|programmiert)/i,
    memoryType: 'learning',
    priority: 4,
    confidence: 0.8,
  },
  important: {
    regex: /\b(critical|important|remember|note|key|essential|must|never|always|crucial|vital|significant|wichtig|merken|notiz|niemals|immer|kritisch|wesentlich|unbedingt|entscheidend)/i,
    memoryType: 'note',
    priority: 5,
    confidence: 0.75,
  },
  code: {
    regex: /\b(function|class|component|api|endpoint|database|schema|test|config|module|interface|method|funktion|klasse|komponente|datenbank|schnittstelle|konfiguration|modul)/i,
    memoryType: 'reference',
    priority: 6,
    confidence: 0.7,
    minLength: 600,
  },
}

const OVERRIDES = {
  forceRemember: /#remember\b/i,
  forceSkip: /#skip\b/i,
}

/**
 * Classify content by the first matching pattern in priority order.
 * @returns {{ isValuable: boolean, memoryType?: string, matchedPattern?: string, confidence?: number, reason?: string }}
 */
export function detectPatterns(content, { minLength = 300, patterns } = {}) {
  if (!content || typeof content !== 'string') return { isValuable: false, reason: 'Invalid or empty content' }
  if (content.length < minLength) return { isValuable: false, reason: `Content too short (${content.length} < ${minLength} chars)` }
  const enabled = Array.isArray(patterns) && patterns.length > 0 ? new Set(patterns) : null
  const ordered = Object.entries(PATTERNS)
    .filter(([key]) => enabled === null || enabled.has(key))
    .sort((a, b) => a[1].priority - b[1].priority)
  for (const [key, pattern] of ordered) {
    if (pattern.minLength && content.length < pattern.minLength) continue
    if (pattern.regex.test(content)) {
      return { isValuable: true, memoryType: pattern.memoryType, matchedPattern: key, confidence: pattern.confidence }
    }
  }
  return { isValuable: false, reason: 'No pattern matched' }
}

/** Read the `#remember` / `#skip` markers from a user prompt. */
export function hasUserOverride(text) {
  const value = typeof text === 'string' ? text : ''
  return { forceRemember: OVERRIDES.forceRemember.test(value), forceSkip: OVERRIDES.forceSkip.test(value) }
}

/** Cut at a sentence boundary when it keeps most of the text. */
export function truncateContent(content, maxLength = 4000) {
  if (content.length <= maxLength) return content
  const truncated = content.slice(0, maxLength)
  let bestBreak = -1
  for (const delimiter of ['. ', '! ', '? ', '.\n', '!\n', '?\n', '.\t', ';\n', '\n\n']) {
    bestBreak = Math.max(bestBreak, truncated.lastIndexOf(delimiter))
  }
  return bestBreak > maxLength * 0.7
    ? `${truncated.slice(0, bestBreak + 1)}\n[truncated]`
    : `${truncated}\n[truncated]`
}
