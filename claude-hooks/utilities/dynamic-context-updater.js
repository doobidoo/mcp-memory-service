/**
 * Dynamic Context Updater
 * Orchestrates intelligent context updates during active conversations
 * Phase 2: Intelligent Context Updates
 */

const { analyzeConversation, detectTopicChanges } = require('./conversation-analyzer');
const { scoreMemoryRelevance } = require('./memory-scorer');
const { formatMemoriesForContext } = require('./context-formatter');
const { getSessionTracker } = require('./session-tracker');

/**
 * Dynamic Context Update Manager
 * Coordinates between conversation analysis, memory retrieval, and context injection
 */
class DynamicContextUpdater {
    constructor(options = {}) {
        this.options = {
            updateThreshold: 0.3,           // Minimum significance score to trigger update
            maxMemoriesPerUpdate: 3,        // Maximum memories to inject per update
            updateCooldownMs: 30000,        // Minimum time between updates (30 seconds)
            maxUpdatesPerSession: 10,       // Maximum updates per session
            debounceMs: 5000,               // Debounce rapid conversation changes
            enableCrossSessionContext: true, // Include cross-session intelligence
            ...options
        };

        this.lastUpdateTime = 0;
        this.updateCount = 0;
        this.conversationBuffer = '';
        this.lastAnalysis = null;
        this.loadedMemoryHashes = new Set();
        this.sessionTracker = null;
        this.debounceTimer = null;
    }

    /**
     * Initialize the dynamic context updater
     */
    async initialize(sessionContext = {}) {
        console.log('[Dynamic Context] Initializing dynamic context updater...');
        
        this.sessionContext = sessionContext;
        this.updateCount = 0;
        this.loadedMemoryHashes.clear();
        
        if (this.options.enableCrossSessionContext) {
            this.sessionTracker = getSessionTracker();
            await this.sessionTracker.initialize();
        }

        console.log('[Dynamic Context] Dynamic context updater initialized');
    }

    /**
     * Process conversation update and potentially inject new context
     * @param {string} conversationText - Current conversation content
     * @param {object} memoryServiceConfig - Memory service configuration
     * @param {function} contextInjector - Function to inject context into conversation
     */
    async processConversationUpdate(conversationText, memoryServiceConfig, contextInjector) {
        try {
            // Check rate limiting
            if (!this.shouldProcessUpdate()) {
                return { processed: false, reason: 'rate_limited' };
            }

            // Debounce rapid updates
            if (this.debounceTimer) {
                clearTimeout(this.debounceTimer);
            }

            return new Promise((resolve) => {
                this.debounceTimer = setTimeout(async () => {
                    const result = await this.performContextUpdate(
                        conversationText,
                        memoryServiceConfig,
                        contextInjector
                    );
                    resolve(result);
                }, this.options.debounceMs);
            });

        } catch (error) {
            console.error('[Dynamic Context] Error processing conversation update:', error.message);
            return { processed: false, error: error.message };
        }
    }

    /**
     * Perform the actual context update
     */
    async performContextUpdate(conversationText, memoryServiceConfig, contextInjector) {
        console.log('[Dynamic Context] Processing conversation update...');

        // Analyze current conversation
        const currentAnalysis = analyzeConversation(conversationText, {
            extractTopics: true,
            extractEntities: true,
            detectIntent: true,
            detectCodeContext: true,
            minTopicConfidence: 0.3
        });

        // Detect significant changes
        const changes = detectTopicChanges(this.lastAnalysis, currentAnalysis);

        if (!changes.hasTopicShift || changes.significanceScore < this.options.updateThreshold) {
            console.log(`[Dynamic Context] No significant changes detected (score: ${changes.significanceScore.toFixed(2)})`);
            this.lastAnalysis = currentAnalysis;
            return { processed: false, reason: 'insufficient_change', significanceScore: changes.significanceScore };
        }

        console.log(`[Dynamic Context] Significant conversation change detected (score: ${changes.significanceScore.toFixed(2)})`);
        console.log(`[Dynamic Context] New topics: ${changes.newTopics.map(t => t.name).join(', ')}`);

        // Generate memory queries based on conversation changes
        const queries = this.generateMemoryQueries(currentAnalysis, changes);
        
        if (queries.length === 0) {
            this.lastAnalysis = currentAnalysis;
            return { processed: false, reason: 'no_actionable_queries' };
        }

        // Retrieve memories from memory service
        const memories = await this.retrieveRelevantMemories(queries, memoryServiceConfig);
        
        if (memories.length === 0) {
            this.lastAnalysis = currentAnalysis;
            return { processed: false, reason: 'no_relevant_memories' };
        }

        // Score memories with conversation context
        const scoredMemories = this.scoreMemoriesWithContext(memories, currentAnalysis);
        
        // Select top memories for injection
        const selectedMemories = scoredMemories
            .filter(memory => memory.relevanceScore > 0.3)
            .slice(0, this.options.maxMemoriesPerUpdate);

        if (selectedMemories.length === 0) {
            this.lastAnalysis = currentAnalysis;
            return { processed: false, reason: 'no_high_relevance_memories' };
        }

        // Track loaded memories to avoid duplicates
        selectedMemories.forEach(memory => {
            this.loadedMemoryHashes.add(memory.content_hash);
        });

        // Include cross-session context if enabled
        let crossSessionContext = null;
        if (this.options.enableCrossSessionContext && this.sessionTracker) {
            crossSessionContext = await this.sessionTracker.getConversationContext(
                this.sessionContext.projectContext,
                { maxPreviousSessions: 2, maxDaysBack: 3 }
            );
        }

        // Format context update
        const contextUpdate = this.formatContextUpdate(
            selectedMemories,
            currentAnalysis,
            changes,
            crossSessionContext
        );

        // Inject context into conversation
        if (contextInjector && typeof contextInjector === 'function') {
            await contextInjector(contextUpdate);
        }

        // Update state
        this.lastAnalysis = currentAnalysis;
        this.lastUpdateTime = Date.now();
        this.updateCount++;

        console.log(`[Dynamic Context] Context update completed (update #${this.updateCount})`);
        console.log(`[Dynamic Context] Injected ${selectedMemories.length} memories`);

        return {
            processed: true,
            updateCount: this.updateCount,
            memoriesInjected: selectedMemories.length,
            significanceScore: changes.significanceScore,
            topics: changes.newTopics.map(t => t.name),
            hasConversationContext: true,
            hasCrossSessionContext: !!crossSessionContext
        };
    }

    /**
     * Check if we should process an update based on rate limiting
     */
    shouldProcessUpdate() {
        const now = Date.now();
        
        // Check cooldown period
        if (now - this.lastUpdateTime < this.options.updateCooldownMs) {
            return false;
        }

        // Check maximum updates per session
        if (this.updateCount >= this.options.maxUpdatesPerSession) {
            return false;
        }

        return true;
    }

    /**
     * Generate memory queries from conversation analysis
     */
    generateMemoryQueries(analysis, changes) {
        const queries = [];

        // Query for new topics
        changes.newTopics.forEach(topic => {
            if (topic.confidence > 0.4) {
                queries.push({
                    query: topic.name,
                    type: 'topic',
                    weight: topic.confidence,
                    limit: 2
                });
            }
        });

        // Query for changed intent
        if (changes.changedIntents && analysis.intent && analysis.intent.confidence > 0.5) {
            queries.push({
                query: `${analysis.intent.name} ${this.sessionContext.projectContext?.name || ''}`,
                type: 'intent',
                weight: analysis.intent.confidence,
                limit: 1
            });
        }

        // Query for high-confidence entities
        analysis.entities
            .filter(entity => entity.confidence > 0.7)
            .slice(0, 2)
            .forEach(entity => {
                queries.push({
                    query: `${entity.name} ${entity.type}`,
                    type: 'entity',
                    weight: entity.confidence,
                    limit: 1
                });
            });

        // Sort by weight and limit total queries
        return queries
            .sort((a, b) => b.weight - a.weight)
            .slice(0, 4); // Maximum 4 queries per update
    }

    /**
     * Retrieve memories from memory service for multiple queries
     */
    async retrieveRelevantMemories(queries, memoryServiceConfig) {
        const allMemories = [];
        
        // Import the query function from topic-change hook
        const { queryMemoryService } = require('../core/topic-change');

        for (const queryObj of queries) {
            try {
                const memories = await this.queryMemoryService(
                    memoryServiceConfig.endpoint,
                    memoryServiceConfig.apiKey,
                    queryObj.query,
                    {
                        limit: queryObj.limit,
                        excludeHashes: Array.from(this.loadedMemoryHashes)
                    }
                );

                // Add query context to memories
                memories.forEach(memory => {
                    memory.queryContext = queryObj;
                });

                allMemories.push(...memories);

            } catch (error) {
                console.error(`[Dynamic Context] Failed to query memories for "${queryObj.query}":`, error.message);
            }
        }

        return allMemories;
    }

    /**
     * Simplified memory service query (extracted from topic-change.js pattern)
     */
    async queryMemoryService(endpoint, apiKey, query, options = {}) {
        const https = require('https');
        
        return new Promise((resolve, reject) => {
            const { limit = 3, excludeHashes = [] } = options;

            const postData = JSON.stringify({
                jsonrpc: '2.0',
                id: Date.now(),
                method: 'tools/call',
                params: {
                    name: 'retrieve_memory',
                    arguments: { query: query, limit: limit }
                }
            });

            const url = new URL('/mcp', endpoint);
            const requestOptions = {
                hostname: url.hostname,
                port: url.port,
                path: url.pathname,
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Length': Buffer.byteLength(postData)
                },
                rejectUnauthorized: false,
                timeout: 5000
            };

            const req = https.request(requestOptions, (res) => {
                let data = '';
                res.on('data', (chunk) => { data += chunk; });
                res.on('end', () => {
                    try {
                        const response = JSON.parse(data);
                        if (response.error) {
                            console.error('[Dynamic Context] Memory service error:', response.error);
                            resolve([]);
                            return;
                        }

                        const memories = this.parseMemoryResults(response.result);
                        const filteredMemories = memories.filter(memory => 
                            !excludeHashes.includes(memory.content_hash)
                        );
                        
                        resolve(filteredMemories);
                    } catch (parseError) {
                        console.error('[Dynamic Context] Failed to parse memory response:', parseError.message);
                        resolve([]);
                    }
                });
            });

            req.on('error', (error) => {
                console.error('[Dynamic Context] Memory service request failed:', error.message);
                resolve([]);
            });

            req.on('timeout', () => {
                req.destroy();
                resolve([]);
            });

            req.write(postData);
            req.end();
        });
    }

    /**
     * Parse memory results from MCP response
     */
    parseMemoryResults(result) {
        try {
            if (result && result.content && result.content[0] && result.content[0].text) {
                const text = result.content[0].text;
                const resultsMatch = text.match(/'results':\s*(\[[\s\S]*?\])/);
                if (resultsMatch) {
                    return eval(resultsMatch[1]) || [];
                }
            }
            return [];
        } catch (error) {
            console.error('[Dynamic Context] Error parsing memory results:', error.message);
            return [];
        }
    }

    /**
     * Score memories with enhanced conversation context
     */
    scoreMemoriesWithContext(memories, conversationAnalysis) {
        return scoreMemoryRelevance(memories, this.sessionContext.projectContext || {}, {
            includeConversationContext: true,
            conversationAnalysis: conversationAnalysis,
            weights: {
                timeDecay: 0.2,
                tagRelevance: 0.3,
                contentRelevance: 0.15,
                conversationRelevance: 0.35  // High weight for conversation context
            }
        });
    }

    /**
     * Format the context update message
     */
    formatContextUpdate(memories, analysis, changes, crossSessionContext) {
        let updateMessage = '\n🧠 **Dynamic Context Update**\n\n';

        // Explain the trigger
        if (changes.newTopics.length > 0) {
            updateMessage += `**New topics detected**: ${changes.newTopics.map(t => t.name).join(', ')}\n`;
        }
        if (changes.changedIntents && analysis.intent) {
            updateMessage += `**Focus shifted to**: ${analysis.intent.name}\n`;
        }
        updateMessage += '\n';

        // Add cross-session context if available
        if (crossSessionContext && crossSessionContext.recentSessions.length > 0) {
            updateMessage += '**Recent session context**:\n';
            crossSessionContext.recentSessions.slice(0, 2).forEach(session => {
                const timeAgo = this.formatTimeAgo(session.endTime);
                updateMessage += `• ${session.outcome?.type || 'Session'} completed ${timeAgo}\n`;
            });
            updateMessage += '\n';
        }

        // Add relevant memories
        updateMessage += '**Relevant context**:\n';
        memories.slice(0, 3).forEach((memory, index) => {
            const content = memory.content.length > 100 ? 
                memory.content.substring(0, 100) + '...' : 
                memory.content;
            
            const relevanceIndicator = memory.relevanceScore > 0.7 ? '🔥' : 
                                     memory.relevanceScore > 0.5 ? '⭐' : '💡';
            
            updateMessage += `${relevanceIndicator} ${content}\n`;
            
            if (memory.tags && memory.tags.length > 0) {
                updateMessage += `   *${memory.tags.slice(0, 3).join(', ')}*\n`;
            }
            updateMessage += '\n';
        });

        updateMessage += '---\n';
        return updateMessage;
    }

    /**
     * Format time ago for human readability
     */
    formatTimeAgo(timestamp) {
        const now = new Date();
        const time = new Date(timestamp);
        const diffMs = now - time;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        if (diffMins < 60) return `${diffMins} minutes ago`;
        if (diffHours < 24) return `${diffHours} hours ago`;
        if (diffDays < 7) return `${diffDays} days ago`;
        return time.toLocaleDateString();
    }

    /**
     * Get statistics about dynamic context updates
     */
    getStats() {
        return {
            updateCount: this.updateCount,
            loadedMemoriesCount: this.loadedMemoryHashes.size,
            lastUpdateTime: this.lastUpdateTime,
            hasSessionTracker: !!this.sessionTracker,
            isInitialized: !!this.sessionContext
        };
    }

    /**
     * Reset the updater state for a new conversation
     */
    reset() {
        console.log('[Dynamic Context] Resetting dynamic context updater');
        
        this.lastUpdateTime = 0;
        this.updateCount = 0;
        this.conversationBuffer = '';
        this.lastAnalysis = null;
        this.loadedMemoryHashes.clear();
        
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = null;
        }
    }
}

module.exports = {
    DynamicContextUpdater
};