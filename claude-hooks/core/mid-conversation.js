/**
 * Mid-Conversation Memory Hook
 * Intelligently triggers memory awareness during conversations based on natural language patterns
 */

const { TieredConversationMonitor } = require('../utilities/tiered-conversation-monitor');
const { AdaptivePatternDetector } = require('../utilities/adaptive-pattern-detector');
const { PerformanceManager } = require('../utilities/performance-manager');
const { MemoryClient } = require('../utilities/memory-client');
const { scoreMemoryRelevance } = require('../utilities/memory-scorer');
const { formatMemoriesForContext } = require('../utilities/context-formatter');

class MidConversationHook {
    constructor(config = {}) {
        this.config = config;

        // Decision weighting constants
        this.TRIGGER_WEIGHTS = {
            PATTERN_CONFIDENCE: 0.6,
            CONVERSATION_CONTEXT: 0.4,
            SEMANTIC_SHIFT_BOOST: 0.2,
            QUESTION_PATTERN_BOOST: 0.1,
            PAST_WORK_BOOST: 0.15
        };

        this.THRESHOLD_VALUES = {
            CONVERSATION_PROBABILITY_MIN: 0.3,
            SEMANTIC_SHIFT_MIN: 0.6,
            SPEED_MODE_CONFIDENCE_MIN: 0.8,
            SPEED_MODE_REDUCTION: 0.8
        };

        // Initialize performance management
        this.performanceManager = new PerformanceManager(config.performance);

        // Initialize components with performance awareness
        this.conversationMonitor = new TieredConversationMonitor(
            config.conversationMonitor,
            this.performanceManager
        );

        this.patternDetector = new AdaptivePatternDetector(
            config.patternDetector,
            this.performanceManager
        );

        // Memory client for queries
        this.memoryClient = null;

        // Hook state - read from correct nested config paths
        const midConversationConfig = config.hooks?.midConversation || {};
        const naturalTriggersConfig = config.naturalTriggers || {};

        this.isEnabled = naturalTriggersConfig.enabled !== false;
        this.lastTriggerTime = 0;
        this.cooldownPeriod = naturalTriggersConfig.cooldownPeriod || 30000; // 30 seconds between triggers

        // Analytics
        this.analytics = {
            totalAnalyses: 0,
            triggersExecuted: 0,
            userAcceptanceRate: 0,
            averageLatency: 0,
            totalFeedback: 0
        };
    }

    /**
     * Analyze user message for memory trigger needs
     */
    async analyzeMessage(userMessage, context = {}) {
        if (!this.isEnabled) return null;

        const timing = this.performanceManager.startTiming('mid_conversation_analysis', 'fast');

        try {
            this.analytics.totalAnalyses++;

            // Check cooldown period
            if (Date.now() - this.lastTriggerTime < this.cooldownPeriod) {
                return this.createResult('cooldown', 'Cooldown period active', 0);
            }

            // Phase 1: Conversation monitoring
            const conversationAnalysis = await this.conversationMonitor.analyzeMessage(userMessage, context);

            // Phase 2: Pattern detection
            const patternResults = await this.patternDetector.detectPatterns(userMessage, {
                ...context,
                conversationAnalysis
            });

            // Phase 3: Combined decision making
            const triggerDecision = this.makeTriggerDecision(conversationAnalysis, patternResults, context);

            // Update last trigger time if we're recommending a trigger
            if (triggerDecision.shouldTrigger) {
                this.lastTriggerTime = Date.now();
            }

            // Record performance
            const performanceResult = this.performanceManager.endTiming(timing);
            this.analytics.averageLatency = this.updateAverageLatency(performanceResult.latency);

            return {
                shouldTrigger: triggerDecision.shouldTrigger,
                confidence: triggerDecision.confidence,
                reasoning: triggerDecision.reasoning,
                conversationAnalysis,
                patternResults,
                performance: performanceResult,
                timestamp: Date.now()
            };

        } catch (error) {
            console.error('[Mid-Conversation Hook] Analysis failed:', error.message);
            this.performanceManager.endTiming(timing);
            return this.createResult('error', `Analysis failed: ${error.message}`, 0);
        }
    }

    /**
     * Execute memory retrieval and context injection
     */
    async executeMemoryTrigger(analysisResult, context = {}) {
        if (!analysisResult.shouldTrigger) return null;

        const timing = this.performanceManager.startTiming('memory_trigger_execution', 'intensive');

        try {
            // Initialize memory client if needed
            if (!this.memoryClient) {
                this.memoryClient = new MemoryClient(this.config.memoryService || {});
                await this.memoryClient.connect();
            }

            // Build enhanced query based on analysis
            const memoryQuery = this.buildMemoryQuery(analysisResult, context);

            // Retrieve relevant memories
            const memories = await this.queryMemories(memoryQuery);

            if (memories.length === 0) {
                return this.createResult('no_memories', 'No relevant memories found', analysisResult.confidence);
            }

            // Score and format memories
            const scoredMemories = scoreMemoryRelevance(memories, context.projectContext, {
                verbose: false,
                enhanceRecency: true
            });

            const contextMessage = formatMemoriesForContext(
                scoredMemories.slice(0, this.config.maxMemoriesPerTrigger || 5),
                context.projectContext,
                {
                    includeScore: false,
                    groupByCategory: scoredMemories.length > 3,
                    maxContentLength: 400,
                    includeTimestamp: true
                }
            );

            // Record successful trigger
            this.analytics.triggersExecuted++;

            const performanceResult = this.performanceManager.endTiming(timing);

            return {
                success: true,
                contextMessage,
                memoriesFound: memories.length,
                memoriesUsed: Math.min(scoredMemories.length, this.config.maxMemoriesPerTrigger || 5),
                confidence: analysisResult.confidence,
                performance: performanceResult,
                triggerType: 'mid_conversation'
            };

        } catch (error) {
            console.error('[Mid-Conversation Hook] Memory trigger failed:', error.message);
            this.performanceManager.endTiming(timing);
            return this.createResult('execution_error', `Memory trigger failed: ${error.message}`, analysisResult.confidence);
        }
    }

    /**
     * Make intelligent trigger decision based on all analyses
     */
    makeTriggerDecision(conversationAnalysis, patternResults, context) {
        let confidence = 0;
        const reasons = [];

        // Weight pattern detection heavily for explicit requests
        if (patternResults.triggerRecommendation) {
            confidence += patternResults.confidence * this.TRIGGER_WEIGHTS.PATTERN_CONFIDENCE;
            reasons.push(`Pattern detection: ${patternResults.confidence.toFixed(2)} confidence`);
        }

        // Add conversation context weighting
        if (conversationAnalysis.triggerProbability > this.THRESHOLD_VALUES.CONVERSATION_PROBABILITY_MIN) {
            confidence += conversationAnalysis.triggerProbability * this.TRIGGER_WEIGHTS.CONVERSATION_CONTEXT;
            reasons.push(`Conversation analysis: ${conversationAnalysis.triggerProbability.toFixed(2)} probability`);
        }

        // Boost for semantic shift (topic change)
        if (conversationAnalysis.semanticShift > this.THRESHOLD_VALUES.SEMANTIC_SHIFT_MIN) {
            confidence += this.TRIGGER_WEIGHTS.SEMANTIC_SHIFT_BOOST;
            reasons.push(`Semantic shift detected: ${conversationAnalysis.semanticShift.toFixed(2)}`);
        }

        // Context-specific adjustments
        if (context.isQuestionPattern) {
            confidence += this.TRIGGER_WEIGHTS.QUESTION_PATTERN_BOOST;
            reasons.push('Question pattern detected');
        }

        if (context.mentionsPastWork) {
            confidence += this.TRIGGER_WEIGHTS.PAST_WORK_BOOST;
            reasons.push('References past work');
        }

        // Apply performance profile considerations
        const profile = this.performanceManager.performanceBudget;
        if (profile.maxLatency < 200 && confidence < this.THRESHOLD_VALUES.SPEED_MODE_CONFIDENCE_MIN) {
            // In speed-focused mode, require higher confidence
            confidence *= this.THRESHOLD_VALUES.SPEED_MODE_REDUCTION;
            reasons.push('Speed mode: increased confidence threshold');
        }

        // Final decision threshold
        const threshold = this.config.naturalTriggers?.triggerThreshold || 0.6;
        const shouldTrigger = confidence >= threshold;

        return {
            shouldTrigger,
            confidence: Math.min(confidence, 1.0),
            reasoning: reasons.join('; '),
            threshold,
            details: {
                conversationWeight: conversationAnalysis.triggerProbability * 0.4,
                patternWeight: patternResults.confidence * 0.6,
                contextAdjustments: confidence - (conversationAnalysis.triggerProbability * 0.4 + patternResults.confidence * 0.6)
            }
        };
    }

    /**
     * Build optimized memory query based on analysis
     */
    buildMemoryQuery(analysisResult, context) {
        const query = {
            semanticQuery: '',
            tags: [],
            limit: this.config.maxMemoriesPerTrigger || 5,
            timeFilter: 'last-month'
        };

        // Extract key topics from conversation analysis
        if (analysisResult.conversationAnalysis.topics.length > 0) {
            query.semanticQuery += analysisResult.conversationAnalysis.topics.join(' ');
        }

        // Add project context
        if (context.projectContext) {
            query.semanticQuery += ` ${context.projectContext.name}`;
            query.tags.push(context.projectContext.name);

            if (context.projectContext.language) {
                query.tags.push(`language:${context.projectContext.language}`);
            }
        }

        // Add pattern-based context
        for (const match of analysisResult.patternResults.matches) {
            if (match.category === 'explicitMemoryRequests') {
                query.timeFilter = 'last-week'; // Recent memories for explicit requests
            } else if (match.category === 'technicalDiscussions') {
                query.tags.push('architecture', 'decisions');
            }
        }

        // Ensure we have a meaningful query
        if (!query.semanticQuery.trim()) {
            query.semanticQuery = 'project context decisions';
        }

        return query;
    }

    /**
     * Query memories using unified memory client
     */
    async queryMemories(query) {
        try {
            let memories = [];

            if (query.timeFilter) {
                const timeQuery = `${query.semanticQuery} ${query.timeFilter}`;
                memories = await this.memoryClient.queryMemoriesByTime(timeQuery, query.limit);
            } else {
                memories = await this.memoryClient.queryMemories(query.semanticQuery, query.limit);
            }

            return memories || [];
        } catch (error) {
            console.warn('[Mid-Conversation Hook] Memory query failed:', error.message);
            return [];
        }
    }

    /**
     * Handle user feedback on trigger quality
     */
    recordUserFeedback(analysisResult, wasHelpful, context = {}) {
        // Update analytics
        this.updateAcceptanceRate(wasHelpful);

        // Pass feedback to components for learning
        this.patternDetector.recordUserFeedback(wasHelpful, analysisResult.patternResults, context);
        this.performanceManager.recordUserFeedback(wasHelpful, {
            latency: analysisResult.performance?.latency || 0
        });

        // Log feedback for analysis
        console.log(`[Mid-Conversation Hook] User feedback: ${wasHelpful ? 'helpful' : 'not helpful'} (confidence: ${analysisResult.confidence?.toFixed(2)})`);
    }

    /**
     * Update performance profile
     */
    updatePerformanceProfile(profileName) {
        this.performanceManager.switchProfile(profileName);
        this.conversationMonitor.updatePerformanceProfile(profileName);

        console.log(`[Mid-Conversation Hook] Switched to performance profile: ${profileName}`);
    }

    /**
     * Get hook status and analytics
     */
    getStatus() {
        return {
            enabled: this.isEnabled,
            lastTriggerTime: this.lastTriggerTime,
            cooldownRemaining: Math.max(0, this.cooldownPeriod - (Date.now() - this.lastTriggerTime)),
            analytics: this.analytics,
            performance: this.performanceManager.getPerformanceReport(),
            conversationMonitor: this.conversationMonitor.getPerformanceStatus(),
            patternDetector: this.patternDetector.getStatistics()
        };
    }

    /**
     * Enable or disable the hook
     */
    setEnabled(enabled) {
        this.isEnabled = enabled;
        console.log(`[Mid-Conversation Hook] ${enabled ? 'Enabled' : 'Disabled'}`);
    }

    /**
     * Helper methods
     */

    createResult(type, message, confidence) {
        return {
            shouldTrigger: false,
            confidence,
            reasoning: message,
            type,
            timestamp: Date.now()
        };
    }

    updateAverageLatency(newLatency) {
        const alpha = 0.1; // Exponential moving average factor
        return this.analytics.averageLatency * (1 - alpha) + newLatency * alpha;
    }

    updateAcceptanceRate(wasPositive) {
        // Increment feedback counter
        this.analytics.totalFeedback++;

        const totalFeedback = this.analytics.totalFeedback;
        if (totalFeedback === 1) {
            // First feedback sets the initial rate
            this.analytics.userAcceptanceRate = wasPositive ? 1.0 : 0.0;
        } else {
            // Update running average
            const currentRate = this.analytics.userAcceptanceRate;
            this.analytics.userAcceptanceRate = (currentRate * (totalFeedback - 1) + (wasPositive ? 1 : 0)) / totalFeedback;
        }
    }

    /**
     * Cleanup resources
     */
    async cleanup() {
        if (this.memoryClient) {
            try {
                await this.memoryClient.disconnect();
            } catch (error) {
                // Ignore cleanup errors
            }
            this.memoryClient = null;
        }
    }
}

/**
 * Global hook instance for state management
 */
let globalHookInstance = null;

/**
 * Get or create the hook instance (singleton pattern)
 */
function getHookInstance(config) {
    if (!globalHookInstance) {
        globalHookInstance = new MidConversationHook(config || {});
        console.log('[Mid-Conversation Hook] Created new hook instance');
    }
    return globalHookInstance;
}

/**
 * Reset hook instance (for testing or config changes)
 */
function resetHookInstance() {
    if (globalHookInstance) {
        globalHookInstance.cleanup().catch((error) => {
            // Log cleanup errors during reset but don't throw
            console.debug('[Mid-Conversation Hook] Cleanup error during reset:', error.message);
        });
        globalHookInstance = null;
        console.log('[Mid-Conversation Hook] Reset hook instance');
    }
}

/**
 * Hook function for Claude Code integration
 */
async function onMidConversation(context) {
    // This would be called by Claude Code during conversation flow
    // Implementation depends on how Claude Code exposes mid-conversation hooks

    const hook = getHookInstance(context.config);

    try {
        // Analyze the current message
        const analysis = await hook.analyzeMessage(context.userMessage, context);

        if (analysis && analysis.shouldTrigger) {
            // Execute memory trigger
            const result = await hook.executeMemoryTrigger(analysis, context);

            if (result && result.success && context.injectSystemMessage) {
                await context.injectSystemMessage(result.contextMessage);
                console.log(`[Mid-Conversation Hook] Injected ${result.memoriesUsed} memories (confidence: ${result.confidence.toFixed(2)})`);
            }
        }

    } catch (error) {
        console.error('[Mid-Conversation Hook] Hook execution failed:', error.message);
        // Don't cleanup on error - preserve state for next call
    }
}

module.exports = {
    MidConversationHook,
    onMidConversation,
    getHookInstance,
    resetHookInstance,
    name: 'mid-conversation-memory',
    version: '1.0.0',
    description: 'Intelligent mid-conversation memory awareness with performance optimization',
    trigger: 'mid-conversation',
    handler: onMidConversation,
    config: {
        async: true,
        timeout: 10000,
        priority: 'high'
    }
};