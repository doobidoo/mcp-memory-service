/**
 * Auto-Capture Pattern Detection Module
 *
 * Ported from shodh-cloudflare for intelligent automatic memory capture.
 * Detects valuable conversation patterns and classifies memory types.
 *
 * @module auto-capture-patterns
 * @version 1.0.0
 */

'use strict';

/**
 * Pattern definitions for automatic memory capture.
 * Each pattern has:
 * - regex: Pattern to match (case-insensitive)
 * - memoryType: Type to assign when matched
 * - priority: Lower = higher priority (first match wins)
 * - minLength: Optional minimum content length for this pattern
 * - confidence: Base confidence score for this pattern
 */
const PATTERNS = {
    decision: {
        regex: /(decided|chose|will use|let's go with|i'll use|we'll use|settled on|going with|picked|selected|opting for|entschieden|gewählt|nehmen wir|verwenden wir|machen wir|nutzen wir|ausgewählt)/i,
        memoryType: 'Decision',
        priority: 1,
        confidence: 0.9,
        description: 'Decision-making statements'
    },
    error: {
        regex: /(error|exception|failed|fixed|bug|issue|crash|broken|resolved|solved|debugging|debugged|patched|fehler|behoben|gefixt|problem|kaputt|gelöst|repariert|fehlerbehebung)/i,
        memoryType: 'Error',
        priority: 2,
        confidence: 0.85,
        description: 'Error reports and fixes'
    },
    learning: {
        regex: /(learned|discovered|realized|found out|turns out|interestingly|til|understanding now|now i see|aha|insight|gelernt|entdeckt|herausgefunden|stellte sich heraus|interessanterweise|jetzt verstehe ich)/i,
        memoryType: 'Learning',
        priority: 3,
        confidence: 0.85,
        description: 'New knowledge acquisition'
    },
    implementation: {
        regex: /(implemented|created|built|added|refactored|set up|configured|deployed|developed|wrote|coding|programmed|implementiert|erstellt|gebaut|hinzugefügt|konfiguriert|eingerichtet|refaktoriert|entwickelt|programmiert)/i,
        memoryType: 'Learning',
        priority: 4,
        confidence: 0.8,
        description: 'Implementation work'
    },
    important: {
        regex: /(critical|important|remember|note|key|essential|must|never|always|crucial|vital|significant|wichtig|merken|notiz|niemals|immer|kritisch|wesentlich|unbedingt|entscheidend)/i,
        memoryType: 'Context',
        priority: 5,
        confidence: 0.75,
        description: 'Important information markers'
    },
    code: {
        regex: /(function|class|component|api|endpoint|database|schema|test|config|module|interface|method|funktion|klasse|komponente|datenbank|schnittstelle|konfiguration|modul)/i,
        memoryType: 'Context',
        priority: 6,
        confidence: 0.7,
        minLength: 600,
        description: 'Substantial code discussions'
    }
};

/**
 * User override markers for explicit control
 */
const USER_OVERRIDES = {
    forceRemember: /#remember\b/i,
    forceSkip: /#skip\b/i
};

/**
 * Default configuration
 */
const DEFAULT_CONFIG = {
    minLength: 300,
    maxLength: 4000,
    enabledPatterns: ['decision', 'error', 'learning', 'implementation', 'important', 'code'],
    debugMode: false
};

/**
 * Detect patterns in content and determine if it's worth capturing.
 *
 * @param {string} content - The conversation content to analyze
 * @param {Object} options - Configuration options
 * @param {number} options.minLength - Minimum content length (default: 300)
 * @param {string[]} options.enabledPatterns - Which patterns to check
 * @returns {Object} Detection result
 */
function detectPatterns(content, options = {}) {
    const config = { ...DEFAULT_CONFIG, ...options };

    // Validate input
    if (!content || typeof content !== 'string') {
        return {
            isValuable: false,
            reason: 'Invalid or empty content'
        };
    }

    // Length check
    if (content.length < config.minLength) {
        return {
            isValuable: false,
            reason: `Content too short (${content.length} < ${config.minLength} chars)`
        };
    }

    const contentLower = content.toLowerCase();

    // Check patterns in priority order
    const sortedPatterns = Object.entries(PATTERNS)
        .filter(([name]) => config.enabledPatterns.includes(name))
        .sort((a, b) => a[1].priority - b[1].priority);

    for (const [patternName, pattern] of sortedPatterns) {
        // Check minimum length requirement for this pattern
        if (pattern.minLength && content.length < pattern.minLength) {
            continue;
        }

        // Test pattern
        if (pattern.regex.test(contentLower)) {
            const match = contentLower.match(pattern.regex);

            if (config.debugMode) {
                console.log(`[auto-capture] Matched pattern: ${patternName}`);
                console.log(`[auto-capture] Matched text: "${match[0]}"`);
            }

            return {
                isValuable: true,
                memoryType: pattern.memoryType,
                matchedPattern: patternName,
                matchedText: match[0],
                confidence: pattern.confidence,
                description: pattern.description
            };
        }
    }

    return {
        isValuable: false,
        reason: 'No pattern matched'
    };
}

/**
 * Check for user override markers in the message.
 *
 * @param {string} userMessage - The user's message to check
 * @returns {Object} Override flags
 */
function hasUserOverride(userMessage) {
    if (!userMessage || typeof userMessage !== 'string') {
        return {
            forceRemember: false,
            forceSkip: false
        };
    }

    return {
        forceRemember: USER_OVERRIDES.forceRemember.test(userMessage),
        forceSkip: USER_OVERRIDES.forceSkip.test(userMessage)
    };
}

/**
 * Generate automatic tags based on pattern detection result.
 *
 * @param {Object} detectionResult - Result from detectPatterns()
 * @param {string} projectName - Optional project name from cwd
 * @returns {string[]} Array of tags
 */
function generateTags(detectionResult, projectName = null) {
    const tags = ['auto-captured', 'smart-ingest'];

    if (detectionResult.memoryType) {
        tags.push(detectionResult.memoryType.toLowerCase());
    }

    if (detectionResult.matchedPattern) {
        tags.push(detectionResult.matchedPattern.toLowerCase());  // Case-normalize
    }

    if (projectName) {
        tags.push(projectName.toLowerCase());  // Case-normalize
    }

    // Deduplicate case-insensitively
    return [...new Set(tags.map(t => t.toLowerCase()))];
}

/**
 * Truncate content to maximum length while preserving meaning.
 *
 * @param {string} content - Content to truncate
 * @param {number} maxLength - Maximum length (default: 4000)
 * @returns {string} Truncated content
 */
function truncateContent(content, maxLength = 4000) {
    if (content.length <= maxLength) {
        return content;
    }

    const truncated = content.substring(0, maxLength);

    // Search for multiple sentence boundary delimiters
    const delimiters = ['. ', '! ', '? ', '.\n', '!\n', '?\n', '.\t', ';\n', '\n\n'];
    let bestBreak = -1;

    for (const delimiter of delimiters) {
        const pos = truncated.lastIndexOf(delimiter);
        if (pos > bestBreak) {
            bestBreak = pos;
        }
    }

    // Use best break point if it preserves at least 70% of content
    // (Lowered from 80% to allow more flexibility in finding good breaks)
    if (bestBreak > maxLength * 0.7) {
        return truncated.substring(0, bestBreak + 1) + '\n[truncated]';
    } else {
        return truncated + '\n[truncated]';
    }
}

/**
 * Compute SHA-256 hash of content for deduplication.
 *
 * @param {string} content - Content to hash
 * @returns {Promise<string>} Hex-encoded hash
 */
async function computeContentHash(content) {
    const crypto = require('crypto');
    return crypto.createHash('sha256').update(content, 'utf8').digest('hex');
}

/**
 * Extract project name from current working directory.
 *
 * @param {string} cwd - Current working directory path
 * @returns {string|null} Project name or null
 */
function extractProjectName(cwd) {
    if (!cwd) return null;

    // Get the last directory component
    const parts = cwd.replace(/\\/g, '/').split('/').filter(Boolean);
    const lastPart = parts[parts.length - 1];

    // Skip common non-project directories
    const skipDirs = ['home', 'users', 'documents', 'desktop', 'repositories', 'projects', 'src'];
    if (skipDirs.includes(lastPart.toLowerCase())) {
        return parts.length > 1 ? parts[parts.length - 2] : null;
    }

    return lastPart;
}

// Export for Node.js
module.exports = {
    PATTERNS,
    USER_OVERRIDES,
    DEFAULT_CONFIG,
    detectPatterns,
    hasUserOverride,
    generateTags,
    truncateContent,
    computeContentHash,
    extractProjectName
};
