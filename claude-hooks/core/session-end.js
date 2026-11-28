/**
 * Claude Code Session End Hook
 * Automatically consolidates session outcomes and stores them as memories
 */

const fs = require('fs').promises;
const path = require('path');
const { resolveConfigPath } = require('../utilities/config-loader');
const { applySelfSignedCertsOption } = require('../utilities/tls-options');
const https = require('https');
const http = require('http');

// Import utilities
const { detectProjectContext } = require('../utilities/project-detector');
const { formatSessionConsolidation } = require('../utilities/context-formatter');
const { detectUserOverrides, logOverride } = require('../utilities/user-override-detector');
const { MemoryClient } = require('../utilities/memory-client');

/**
 * Load hook configuration
 */
async function loadConfig() {
    try {
        const configPath = resolveConfigPath(__dirname);
        const configData = await fs.readFile(configPath, 'utf8');
        return JSON.parse(configData);
    } catch (error) {
        console.warn('[Memory Hook] Using default configuration:', error.message);
        return {
            memoryService: {
                http: {
                    endpoint: 'http://127.0.0.1:8000',
                    apiKey: 'test-key-123'
                },
                defaultTags: ['claude-code', 'auto-generated'],
                enableSessionConsolidation: true
            },
            sessionAnalysis: {
                extractTopics: true,
                extractDecisions: true,
                extractInsights: true,
                extractCodeChanges: true,
                extractNextSteps: true,
                minSessionLength: 100 // Minimum characters for meaningful session
            }
        };
    }
}

/**
 * Analyze conversation to extract key information
 */
function analyzeConversation(conversationData) {
    try {
        const analysis = {
            topics: [],
            decisions: [],
            insights: [],
            codeChanges: [],
            nextSteps: [],
            sessionLength: 0,
            confidence: 0
        };
        
        if (!conversationData || !conversationData.messages) {
            return analysis;
        }
        
        const messages = conversationData.messages;
        const conversationText = messages.map(msg => msg.content || '').join('\n').toLowerCase();
        analysis.sessionLength = conversationText.length;
        
        // Extract topics (simple keyword matching)
        const topicKeywords = {
            'implementation': /implement|implementing|implementation|build|building|create|creating/g,
            'debugging': /debug|debugging|bug|error|fix|fixing|issue|problem/g,
            'architecture': /architecture|design|structure|pattern|framework|system/g,
            'performance': /performance|optimization|speed|memory|efficient|faster/g,
            'testing': /test|testing|unit test|integration|coverage|spec/g,
            'deployment': /deploy|deployment|production|staging|release/g,
            'configuration': /config|configuration|setup|environment|settings/g,
            'database': /database|db|sql|query|schema|migration/g,
            'api': /api|endpoint|rest|graphql|service|interface/g,
            'ui': /ui|interface|frontend|component|styling|css|html/g
        };
        
        Object.entries(topicKeywords).forEach(([topic, regex]) => {
            if (conversationText.match(regex)) {
                analysis.topics.push(topic);
            }
        });

        // Extract decisions (look for decision language)
        const decisionPatterns = [
            /decided to|decision to|chose to|choosing|will use|going with/g,
            /better to|prefer|recommend|should use|opt for/g,
            /concluded that|determined that|agreed to/g
        ];
        
        messages.forEach(msg => {
            const content = (msg.content || '').toLowerCase();
            decisionPatterns.forEach(pattern => {
                const matches = content.match(pattern);
                if (matches) {
                    // Extract sentences containing decisions
                    const sentences = msg.content.split(/[.!?]+/);
                    sentences.forEach(sentence => {
                        if (pattern.test(sentence.toLowerCase()) && sentence.length > 20) {
                            analysis.decisions.push(sentence.trim());
                        }
                    });
                }
            });
        });
        
        // Extract insights (look for learning language)
        const insightPatterns = [
            /learned that|discovered|realized|found out|turns out/g,
            /insight|understanding|conclusion|takeaway|lesson/g,
            /important to note|key finding|observation/g
        ];
        
        messages.forEach(msg => {
            const content = (msg.content || '').toLowerCase();
            insightPatterns.forEach(pattern => {
                if (pattern.test(content)) {
                    const sentences = msg.content.split(/[.!?]+/);
                    sentences.forEach(sentence => {
                        if (pattern.test(sentence.toLowerCase()) && sentence.length > 20) {
                            analysis.insights.push(sentence.trim());
                        }
                    });
                }
            });
        });
        
        // Extract code changes (look for technical implementations)
        const codePatterns = [
            /added|created|implemented|built|wrote/g,
            /modified|updated|changed|refactored|improved/g,
            /fixed|resolved|corrected|patched/g
        ];
        
        messages.forEach(msg => {
            const content = msg.content || '';
            if (content.includes('```') || /\.(js|py|rs|go|java|cpp|c|ts|jsx|tsx)/.test(content)) {
                // This message contains code
                const lowerContent = content.toLowerCase();
                codePatterns.forEach(pattern => {
                    if (pattern.test(lowerContent)) {
                        const sentences = content.split(/[.!?]+/);
                        sentences.forEach(sentence => {
                            if (pattern.test(sentence.toLowerCase()) && sentence.length > 15) {
                                analysis.codeChanges.push(sentence.trim());
                            }
                        });
                    }
                });
            }
        });
        
        // Extract next steps (look for future language)
        const nextStepsPatterns = [
            /next|todo|need to|should|will|plan to|going to/g,
            /follow up|continue|proceed|implement next|work on/g,
            /remaining|still need|outstanding|future/g
        ];
        
        messages.forEach(msg => {
            const content = (msg.content || '').toLowerCase();
            nextStepsPatterns.forEach(pattern => {
                if (pattern.test(content)) {
                    const sentences = msg.content.split(/[.!?]+/);
                    sentences.forEach(sentence => {
                        if (pattern.test(sentence.toLowerCase()) && sentence.length > 15) {
                            analysis.nextSteps.push(sentence.trim());
                        }
                    });
                }
            });
        });
        
        // Calculate confidence based on extracted information
        const totalExtracted = analysis.topics.length + analysis.decisions.length + 
                              analysis.insights.length + analysis.codeChanges.length + 
                              analysis.nextSteps.length;
        
        analysis.confidence = Math.min(1.0, totalExtracted / 10); // Max confidence at 10+ items
        
        // Limit arrays to prevent overwhelming output
        // Topics: no limit needed (max 10 possible keywords)
        analysis.decisions = analysis.decisions.slice(0, 3);
        analysis.insights = analysis.insights.slice(0, 3);
        analysis.codeChanges = analysis.codeChanges.slice(0, 4);
        analysis.nextSteps = analysis.nextSteps.slice(0, 4);
        
        return analysis;
        
    } catch (error) {
        console.error('[Memory Hook] Error analyzing conversation:', error.message);
        return {
            topics: [],
            decisions: [],
            insights: [],
            codeChanges: [],
            nextSteps: [],
            sessionLength: 0,
            confidence: 0,
            error: error.message
        };
    }
}

/**
 * Decide whether an analyzed session is substantive enough to store.
 *
 * Topics and next-steps are keyword-matched on generic vocabulary
 * ("debugging", "testing", "should", "will") and fire on almost any
 * conversation, so they are NOT counted as evidence. Only decisions,
 * insights, and code changes indicate a session worth remembering.
 * This prevents trivial sessions from producing generic "Session Summary"
 * memories that score 0.0 on quality.
 *
 * The #remember override (forceRemember) bypasses the gate.
 */
function isSessionMeaningful(analysis, { forceRemember = false } = {}) {
    if (forceRemember) return true;
    if (!analysis) return false;
    const substantive = (analysis.decisions?.length || 0)
        + (analysis.insights?.length || 0)
        + (analysis.codeChanges?.length || 0);
    return substantive > 0;
}

/**
 * Trigger quality evaluation for a stored memory (async, non-blocking)
 * This calls the backend's quality scoring system to pre-score the memory
 */
function triggerQualityEvaluation(endpoint, apiKey, contentHash, allowSelfSignedCerts = false) {
    return new Promise((resolve, reject) => {
        const url = new URL(`/api/quality/memories/${contentHash}/evaluate`, endpoint);
        const isHttps = url.protocol === 'https:';
        const requestModule = isHttps ? https : http;

        const postData = JSON.stringify({});

        const options = {
            hostname: url.hostname,
            port: url.port ? Number(url.port) : (isHttps ? 443 : 80),
            path: url.pathname,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(postData),
                'Authorization': `Bearer ${apiKey}`
            },
            timeout: 10000 // 10 second timeout for quality evaluation
        };

        applySelfSignedCertsOption(options, isHttps, allowSelfSignedCerts);

        const req = requestModule.request(options, (res) => {
            let data = '';
            res.on('data', (chunk) => {
                data += chunk;
            });
            res.on('end', () => {
                try {
                    const response = JSON.parse(data);
                    resolve(response);
                } catch (parseError) {
                    resolve({ success: false, error: 'Parse error', data });
                }
            });
        });

        req.on('error', (error) => {
            resolve({ success: false, error: error.message });
        });

        req.on('timeout', () => {
            req.destroy();
            resolve({ success: false, error: 'Quality evaluation timed out' });
        });

        req.write(postData);
        req.end();
    });
}

// Claude Code session ids are UUIDs. Only a plain token is accepted, so the
// value is safe inside a tag; 'unknown' is the placeholder the stdin reader
// below uses when the hook input has no session_id.
const SESSION_ID_PATTERN = /^[a-z0-9][a-z0-9_-]{0,127}$/;
// 'sys:' is the taxonomy namespace for system-generated tags; the
// '<source>:<id>' shape follows the mem0 importer's 'sys:mem0-run:<id>'.
const SESSION_TAG_PREFIX = 'sys:claude-session:';
const SESSION_LOOKUP_LIMIT = 10;
const SESSION_LOOKUP_TIMEOUT_MS = 1000;
// Claude Code gives all SessionEnd hooks 1.5 s together unless
// CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS raises it; a `timeout` on a
// plugin-provided hook does not (hooks reference, "SessionEnd").
const DEFAULT_SESSION_END_BUDGET_MS = 1500;
// Kept back from the lookup and the update, so a slow or failed one still
// leaves time to store the summary the old way; below MIN_LOOKUP_MS the
// lookup is skipped.
const STORE_RESERVE_MS = 500;
const MIN_LOOKUP_MS = 100;
// Floor for the update's timeout, so a nearly spent budget still sends it.
const MIN_REQUEST_MS = 100;
// Process start, which is when Claude Code's budget clock starts too.
const HOOK_STARTED_AT = Date.now() - process.uptime() * 1000;
// Tags the server adds on its own (source:<hostname> with
// MCP_MEMORY_INCLUDE_HOSTNAME, agent:<id> from X-Agent-ID). An update
// replaces the tag list, so these are carried over from the stored summary.
const SERVER_TAG_PREFIXES = ['source:', 'agent:'];

/**
 * When the SessionEnd budget runs out, as an epoch-ms deadline.
 */
function sessionEndDeadline(env = process.env, startedAt = HOOK_STARTED_AT) {
    const configured = Number.parseInt(env.CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS, 10);
    const budget = Number.isFinite(configured) && configured > 0 ? configured : DEFAULT_SESSION_END_BUDGET_MS;
    return startedAt + budget;
}

/**
 * Lookup timeout that leaves STORE_RESERVE_MS of the budget for the fallback
 * store. Without a deadline (library callers, tests) the fixed cap applies.
 */
function lookupTimeoutMs(deadline, now = Date.now()) {
    if (!deadline) return SESSION_LOOKUP_TIMEOUT_MS;
    return Math.min(SESSION_LOOKUP_TIMEOUT_MS, deadline - now - STORE_RESERVE_MS);
}

/**
 * The tags for updating `target`: the freshly generated ones plus the
 * server-added tags `target` carries.
 */
function tagsForUpdate(tags, target) {
    const carried = (target.tags || []).filter(
        (tag) => typeof tag === 'string' && SERVER_TAG_PREFIXES.some((prefix) => tag.startsWith(prefix))
    );
    return [...new Set([...tags, ...carried])];
}

/**
 * Normalize the hook input's session_id, or null when it is missing, the
 * 'unknown' placeholder, or not a plain token.
 */
function normalizeSessionId(sessionId) {
    if (typeof sessionId !== 'string') return null;
    const normalized = sessionId.trim().toLowerCase();
    if (normalized === 'unknown' || !SESSION_ID_PATTERN.test(normalized)) return null;
    return normalized;
}

function sessionTag(sessionId) {
    return `${SESSION_TAG_PREFIX}${sessionId}`;
}

/**
 * Decide how to write a session summary, given the summaries already stored
 * for the same session (newest first, as /api/search/by-tag returns them).
 *
 * Each SessionEnd re-summarizes the whole cumulative transcript, so a
 * resumed-and-exited session would otherwise add a near-identical memory
 * every time (#2150). The server cannot catch these: semantic dedup is off
 * for memory_type 'session' (#898) and the timestamp footer defeats the
 * exact-hash check.
 *
 * When a fallback left two summaries for one session, the longest is the one
 * compared and updated; the other stays as it is.
 *
 * @returns {{action: 'store'|'skip'|'update', target?: object}}
 *   store  - nothing stored for this session yet
 *   skip   - the transcript has not grown since the stored summary
 *   update - it has grown; replace the stored summary's content
 */
function decideSessionWrite(existing, sessionId, sessionLength) {
    const matches = (existing || []).filter(
        (memory) => memory && memory.content_hash && memory.metadata?.session_id === sessionId
    );
    if (matches.length === 0) return { action: 'store' };

    const storedLength = (memory) => Number(memory.metadata?.session_analysis?.session_length) || 0;
    // Strict '>' keeps the newest of equally long summaries.
    const target = matches.reduce((best, memory) => (storedLength(memory) > storedLength(best) ? memory : best));
    if ((Number(sessionLength) || 0) <= storedLength(target)) {
        return { action: 'skip', target };
    }
    return { action: 'update', target };
}

/**
 * Look up the summaries already stored for a session. Never throws: any
 * failure is reported as {success: false} so the caller stores as before.
 */
async function findSessionSummaries(client, sessionId, timeoutMs) {
    try {
        return await client.findMemoriesByTags([sessionTag(sessionId)], {
            matchAll: true,
            limit: SESSION_LOOKUP_LIMIT,
            timeoutMs,
        });
    } catch (error) {
        return { success: false, memories: [], error: error.message };
    }
}

function buildSessionTags(projectContext, analysis, sessionId) {
    const tags = [
        'claude-code-session',
        'session-consolidation',
        projectContext.name,
        projectContext.language ? `language:${projectContext.language}` : null,
        ...analysis.topics.slice(0, 3),
        ...projectContext.frameworks.slice(0, 2),
        `confidence:${Math.round(analysis.confidence * 100)}`,
        sessionId ? sessionTag(sessionId) : null,
    ]
        .filter(Boolean)
        .map((tag) => String(tag).toLowerCase());

    return [...new Set(tags)];
}

function buildSessionMetadata(projectContext, analysis, sessionId) {
    return {
        session_analysis: {
            topics: analysis.topics,
            decisions_count: analysis.decisions.length,
            insights_count: analysis.insights.length,
            code_changes_count: analysis.codeChanges.length,
            next_steps_count: analysis.nextSteps.length,
            session_length: analysis.sessionLength,
            confidence: analysis.confidence,
        },
        project_context: {
            name: projectContext.name,
            language: projectContext.language,
            frameworks: projectContext.frameworks,
        },
        ...(sessionId ? { session_id: sessionId } : {}),
        generated_by: 'claude-code-session-end-hook',
        generated_at: new Date().toISOString(),
    };
}

/**
 * Skip or update the session's stored summary when there is one. Returns the
 * result to report, or null to store a new memory as before (nothing stored
 * yet, too little time left, lookup failed, or the update was rejected).
 *
 * The update replaces the content and the tag list (keeping server-added
 * tags, see SERVER_TAG_PREFIXES); the server merges the new metadata keys
 * into the stored metadata. An update that times out or fails on the network
 * may still have been applied, so it is not followed by a store: at worst the
 * stored summary stays one exit behind.
 */
async function reconcileSessionSummary(client, sessionId, sessionLength, fields, deadline) {
    const timeoutMs = lookupTimeoutMs(deadline);
    if (timeoutMs < MIN_LOOKUP_MS) {
        console.warn('[Memory Hook] Too little of the SessionEnd time budget left to look up the session summary, storing a new summary');
        return null;
    }
    const lookup = await findSessionSummaries(client, sessionId, timeoutMs);
    if (!lookup.success) {
        console.warn(`[Memory Hook] Session summary lookup failed (${lookup.error}), storing a new summary`);
        return null;
    }

    const decision = decideSessionWrite(lookup.memories, sessionId, sessionLength);
    if (decision.action === 'skip') {
        return { success: true, skipped: true, contentHash: decision.target.content_hash };
    }
    if (decision.action === 'update') {
        let update;
        try {
            const updateOpts = deadline
                ? { timeoutMs: Math.max(MIN_REQUEST_MS, deadline - Date.now() - STORE_RESERVE_MS) }
                : {};
            const updateFields = { ...fields, tags: tagsForUpdate(fields.tags, decision.target) };
            update = await client.updateMemory(decision.target.content_hash, updateFields, updateOpts);
        } catch (error) {
            update = { success: false, rejected: false, error: error.message };
        }
        if (update.success) {
            return {
                success: true,
                updated: true,
                contentHash: update.contentHash,
                previousHash: decision.target.content_hash,
            };
        }
        if (!update.rejected) {
            return { success: false, updateUnconfirmed: true, error: update.error };
        }
        console.warn(`[Memory Hook] Session summary update rejected (${update.error}), storing a new summary`);
    }
    return null;
}

/**
 * Store session consolidation to memory service.
 *
 * With a session id and an HTTP connection, at most one summary is kept per
 * session: a repeated SessionEnd on an unchanged transcript is skipped, and a
 * grown one updates the stored summary (PUT, which servers with version
 * history record as a new version). `deadline` (epoch ms) caps the lookup so
 * the fallback store still fits in the SessionEnd budget.
 */
async function storeSessionMemory(endpoint, apiKey, content, projectContext, analysis, allowSelfSignedCerts = false, { sessionId = null, deadline = null } = {}) {
    const normalizedSessionId = normalizeSessionId(sessionId);
    const tags = buildSessionTags(projectContext, analysis, normalizedSessionId);

    const client = new MemoryClient({
        protocol: 'auto',
        preferredProtocol: 'http',
        http: { endpoint, apiKey },
        allowSelfSignedCerts,
    });

    try {
        await client.connect();
    } catch (err) {
        return { success: false, error: `Connect failed: ${err.message}` };
    }

    try {
        const fields = {
            content,
            tags,
            // 'session' is the observation subtype for this; 'session-summary'
            // is not in the ontology and was coerced to 'observation' (#177).
            // The 'claude-code-session' and 'session-consolidation' tags
            // remain the reliable way to find these.
            memoryType: 'session',
            metadata: buildSessionMetadata(projectContext, analysis, normalizedSessionId),
        };
        if (normalizedSessionId && client.activeProtocol === 'http') {
            const reconciled = await reconcileSessionSummary(
                client, normalizedSessionId, analysis.sessionLength, fields, deadline
            );
            if (reconciled) return reconciled;
        } else if (normalizedSessionId) {
            // The MCP client has no tag lookup, so there is nothing to compare with.
            console.log('[Memory Hook] Connected over MCP, which has no tag lookup: storing a new session summary');
        }
        return await client.storeMemory(content, {
            tags: fields.tags,
            memoryType: fields.memoryType,
            metadata: fields.metadata,
        });
    } finally {
        await client.disconnect();
    }
}

/**
 * Main session end hook function
 */
async function onSessionEnd(context) {
    try {
        // Check for user overrides in the last user message (#skip / #remember)
        let lastUserMessage = null;
        if (context.conversation && context.conversation.messages) {
            const userMessages = context.conversation.messages.filter(msg => msg.role === 'user');
            if (userMessages.length > 0) {
                lastUserMessage = userMessages[userMessages.length - 1].content || '';
            }
        }

        const overrides = detectUserOverrides(lastUserMessage);
        if (overrides.forceSkip) {
            logOverride('skip');
            console.log('[Memory Hook] Session consolidation skipped by user override (#skip)');
            return;
        }
        // forceRemember will bypass minSessionLength and confidence checks below

        console.log('[Memory Hook] Session ending - consolidating outcomes...');

        // Load configuration
        const config = await loadConfig();

        if (!config.memoryService.enableSessionConsolidation) {
            console.log('[Memory Hook] Session consolidation disabled in config');
            return;
        }
        
        // Check if session is meaningful enough to store (bypass with #remember)
        if (!overrides.forceRemember && context.conversation && context.conversation.messages) {
            const totalLength = context.conversation.messages
                .map(msg => (msg.content || '').length)
                .reduce((sum, len) => sum + len, 0);

            if (totalLength < config.sessionAnalysis.minSessionLength) {
                console.log('[Memory Hook] Session too short for consolidation');
                return;
            }
        }

        if (overrides.forceRemember) {
            logOverride('remember');
            console.log('[Memory Hook] Force consolidation requested (#remember)');
        }

        // Detect project context
        const projectContext = await detectProjectContext(context.workingDirectory || process.cwd());
        console.log(`[Memory Hook] Consolidating session for project: ${projectContext.name}`);

        // Analyze conversation
        const analysis = analyzeConversation(context.conversation);

        // Only store sessions with real substance (decisions/insights/code
        // changes). Topic- or next-step-only sessions are generic noise.
        // #remember bypasses this gate.
        if (!isSessionMeaningful(analysis, { forceRemember: overrides.forceRemember })) {
            console.log('[Memory Hook] Session not substantive (no decisions/insights/code changes), skipping consolidation');
            return;
        }
        
        console.log(`[Memory Hook] Session analysis: ${analysis.topics.length} topics, ${analysis.decisions.length} decisions, confidence: ${(analysis.confidence * 100).toFixed(1)}%`);
        
        // Format session consolidation
        const consolidation = formatSessionConsolidation(analysis, projectContext);

        // Get endpoint and apiKey from new config structure
        const endpoint = config.memoryService?.http?.endpoint || config.memoryService?.endpoint || 'http://127.0.0.1:8000';
        const apiKey = config.memoryService?.http?.apiKey || config.memoryService?.apiKey || 'test-key-123';

        // Store to memory service
        const result = await storeSessionMemory(
            endpoint,
            apiKey,
            consolidation,
            projectContext,
            analysis,
            config.memoryService?.allowSelfSignedCerts === true,
            { sessionId: context.sessionId, deadline: context.deadline }
        );

        if (result.skipped) {
            console.log('[Memory Hook] Session summary unchanged since it was last stored, skipping');
            return;
        }
        if (result.updateUnconfirmed) {
            // Not followed by a store: the server may have applied it.
            console.warn(`[Memory Hook] Session summary update not confirmed (${result.error}), leaving the stored summary`);
            return;
        }

        const hash = result.content_hash || result.contentHash;
        if (result.success || hash) {
            console.log(result.updated
                ? '[Memory Hook] Session consolidation updated (transcript grew since the last summary)'
                : '[Memory Hook] Session consolidation stored successfully');
            if (hash) {
                console.log(`[Memory Hook] Memory hash: ${hash.substring(0, 8)}...`);

                // Trigger async quality evaluation (non-blocking)
                triggerQualityEvaluation(endpoint, apiKey, hash, config.memoryService?.allowSelfSignedCerts === true)
                    .then(evalResult => {
                        if (evalResult.success) {
                            console.log(`[Memory Hook] Quality evaluated: ${evalResult.quality_score?.toFixed(3)} (${evalResult.quality_provider})`);
                        }
                    })
                    .catch(err => {
                        // Don't fail the hook if quality evaluation fails
                        console.warn('[Memory Hook] Quality evaluation skipped:', err.message);
                    });
            }
        } else {
            console.warn('[Memory Hook] Failed to store session consolidation:', result.error || 'Unknown error');
        }
        
    } catch (error) {
        console.error('[Memory Hook] Error in session end:', error.message);
        // Fail gracefully - don't prevent session from ending
    }
}

/**
 * Hook metadata for Claude Code
 */
module.exports = {
    name: 'memory-awareness-session-end',
    version: '1.0.0',
    description: 'Automatically consolidate and store session outcomes',
    trigger: 'session-end',
    handler: onSessionEnd,
    config: {
        async: true,
        timeout: 15000, // 15 second timeout
        priority: 'normal'
    },
    // Exported for testing
    _internal: {
        parseTranscript: null,  // Will be set after function definition
        analyzeConversation,
        isSessionMeaningful,
        triggerQualityEvaluation,
        storeSessionMemory,
        normalizeSessionId,
        decideSessionWrite,
        sessionTag,
        sessionEndDeadline,
        lookupTimeoutMs,
        tagsForUpdate
    }
};

/**
 * Read JSON context from stdin (provided by Claude Code)
 * Returns: { transcript_path, reason, cwd, session_id, ... }
 */
async function readStdinContext() {
    return new Promise((resolve, reject) => {
        let data = '';

        // Set a timeout in case stdin is empty or never closes
        const timeout = setTimeout(() => {
            resolve(null); // No stdin data - likely manual test run
        }, 100);

        process.stdin.setEncoding('utf8');
        process.stdin.on('readable', () => {
            let chunk;
            while ((chunk = process.stdin.read()) !== null) {
                data += chunk;
            }
        });

        process.stdin.on('end', () => {
            clearTimeout(timeout);
            if (data.trim()) {
                try {
                    resolve(JSON.parse(data));
                } catch (error) {
                    console.error('[Memory Hook] Failed to parse stdin JSON:', error.message);
                    reject(error);
                }
            } else {
                resolve(null);
            }
        });

        process.stdin.on('error', (error) => {
            clearTimeout(timeout);
            console.error('[Memory Hook] Stdin error:', error.message);
            reject(error);
        });
    });
}

/**
 * Parse JSONL transcript file to extract conversation messages
 * @param {string} transcriptPath - Path to the .jsonl transcript file
 * @returns {Object} - { messages: Array<{role, content}> }
 */
async function parseTranscript(transcriptPath) {
    try {
        const content = await fs.readFile(transcriptPath, 'utf8');
        const lines = content.trim().split('\n');
        const messages = [];

        for (const line of lines) {
            if (!line.trim()) continue;

            try {
                const entry = JSON.parse(line);

                // Only process user and assistant messages
                if (entry.type === 'user' || entry.type === 'assistant') {
                    const msg = entry.message;
                    if (msg && msg.role && msg.content) {
                        // Handle content that can be string or array of content blocks
                        let contentText = '';
                        if (typeof msg.content === 'string') {
                            contentText = msg.content;
                        } else if (Array.isArray(msg.content)) {
                            // Extract text from content blocks
                            contentText = msg.content
                                .filter(block => block.type === 'text')
                                .map(block => block.text)
                                .join('\n');
                        }

                        if (contentText) {
                            messages.push({
                                role: msg.role,
                                content: contentText
                            });
                        }
                    }
                }
            } catch (parseError) {
                // Skip malformed lines
                continue;
            }
        }

        return { messages };
    } catch (error) {
        console.error('[Memory Hook] Failed to parse transcript:', error.message);
        return { messages: [] };
    }
}

// Set parseTranscript on exports for testing (after function is defined)
module.exports._internal.parseTranscript = parseTranscript;

/**
 * Mock conversation for manual testing (when no stdin/transcript available)
 */
const mockConversation = {
    messages: [
        {
            role: 'user',
            content: 'I need to implement a memory awareness system for Claude Code'
        },
        {
            role: 'assistant',
            content: 'I\'ll help you create a memory awareness system. We decided to use hooks for session management and implement automatic context injection.'
        },
        {
            role: 'user',
            content: 'Great! I learned that we need project detection and memory scoring algorithms.'
        },
        {
            role: 'assistant',
            content: 'Exactly. I implemented the project detector in project-detector.js and created scoring algorithms. Next we need to test the complete system.'
        }
    ]
};

// Direct execution - reads stdin context from Claude Code
if (require.main === module) {
    (async () => {
        try {
            // Read context from stdin (Claude Code provides this)
            const stdinContext = await readStdinContext();

            let context;

            if (stdinContext && stdinContext.transcript_path) {
                // Real execution: parse transcript file
                console.log(`[Memory Hook] Reading transcript: ${stdinContext.transcript_path}`);
                console.log(`[Memory Hook] Session end reason: ${stdinContext.reason || 'unknown'}`);

                const conversation = await parseTranscript(stdinContext.transcript_path);

                context = {
                    workingDirectory: stdinContext.cwd || process.cwd(),
                    sessionId: stdinContext.session_id || 'unknown',
                    reason: stdinContext.reason,
                    deadline: sessionEndDeadline(),
                    conversation: conversation
                };

                console.log(`[Memory Hook] Parsed ${conversation.messages.length} messages from transcript`);
            } else {
                // Manual test: use mock data
                console.log('[Memory Hook] No stdin context - using mock data for testing');
                context = {
                    workingDirectory: process.cwd(),
                    sessionId: 'test-session',
                    deadline: sessionEndDeadline(),
                    conversation: mockConversation
                };
            }

            await onSessionEnd(context);
            console.log('Session end hook completed');

        } catch (error) {
            console.error('Session end hook failed:', error);
            process.exit(1);
        }
    })();
}