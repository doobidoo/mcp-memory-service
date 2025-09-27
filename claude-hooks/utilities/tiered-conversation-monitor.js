/**
 * Tiered Conversation Monitor
 * Performance-aware semantic analysis with multiple processing levels
 */

const { PerformanceManager } = require('./performance-manager');

class TieredConversationMonitor {
    constructor(config = {}, performanceManager = null) {
        this.config = config;
        this.performanceManager = performanceManager || new PerformanceManager(config.performance);

        // Conversation state
        this.conversationHistory = [];
        this.currentTopics = new Set();
        this.contextWindow = config.contextWindow || 10; // Number of recent messages to analyze

        // Topic tracking
        this.topicWeights = new Map();
        this.semanticCache = new Map();

        // Performance-based configuration
        this.tierConfig = {
            instant: {
                enabled: true,
                methods: ['simplePatternMatch', 'cacheCheck'],
                maxLatency: 50
            },
            fast: {
                enabled: true,
                methods: ['topicExtraction', 'lightweightSemantic'],
                maxLatency: 150
            },
            intensive: {
                enabled: false, // Default off for performance
                methods: ['deepSemanticAnalysis', 'fullContextAnalysis'],
                maxLatency: 500
            }
        };

        this.updateTierConfiguration();
    }

    /**
     * Update tier configuration based on performance manager
     */
    updateTierConfiguration() {
        if (!this.performanceManager) return;

        const profile = this.performanceManager.performanceBudget;

        this.tierConfig.instant.enabled = profile.enabledTiers.includes('instant');
        this.tierConfig.fast.enabled = profile.enabledTiers.includes('fast');
        this.tierConfig.intensive.enabled = profile.enabledTiers.includes('intensive');
    }

    /**
     * Analyze user message with tiered approach
     */
    async analyzeMessage(message, context = {}) {
        const analysis = {
            topics: [],
            semanticShift: 0,
            triggerProbability: 0,
            processingTier: 'none',
            confidence: 0
        };

        // Tier 1: Instant processing (< 50ms)
        if (this.tierConfig.instant.enabled) {
            const timing = this.performanceManager.startTiming('instant_analysis', 'instant');

            try {
                const instantResults = await this.instantAnalysis(message, context);
                analysis.topics.push(...instantResults.topics);
                analysis.triggerProbability = Math.max(analysis.triggerProbability, instantResults.triggerProbability);
                analysis.processingTier = 'instant';

                const result = this.performanceManager.endTiming(timing);

                // If instant analysis is confident enough, return early
                if (instantResults.confidence > 0.8 || !this.tierConfig.fast.enabled) {
                    analysis.confidence = instantResults.confidence;
                    return analysis;
                }
            } catch (error) {
                console.warn('[Monitor] Instant analysis failed:', error.message);
            }
        }

        // Tier 2: Fast processing (< 150ms)
        if (this.tierConfig.fast.enabled && this.performanceManager.shouldRunHook('fast_analysis', 'fast')) {
            const timing = this.performanceManager.startTiming('fast_analysis', 'fast');

            try {
                const fastResults = await this.fastAnalysis(message, context);

                // Merge results with priority to fast analysis
                analysis.topics = this.mergeTopics(analysis.topics, fastResults.topics);
                analysis.semanticShift = fastResults.semanticShift;
                analysis.triggerProbability = Math.max(analysis.triggerProbability, fastResults.triggerProbability);
                analysis.processingTier = 'fast';
                analysis.confidence = fastResults.confidence;

                this.performanceManager.endTiming(timing);

                // If fast analysis is confident, return
                if (fastResults.confidence > 0.7 || !this.tierConfig.intensive.enabled) {
                    return analysis;
                }
            } catch (error) {
                console.warn('[Monitor] Fast analysis failed:', error.message);
            }
        }

        // Tier 3: Intensive processing (< 500ms) - only when needed
        if (this.tierConfig.intensive.enabled &&
            this.performanceManager.shouldRunHook('intensive_analysis', 'intensive') &&
            analysis.triggerProbability > 0.3) {

            const timing = this.performanceManager.startTiming('intensive_analysis', 'intensive');

            try {
                const intensiveResults = await this.intensiveAnalysis(message, context);

                // Use intensive results as authoritative
                analysis.topics = intensiveResults.topics;
                analysis.semanticShift = intensiveResults.semanticShift;
                analysis.triggerProbability = intensiveResults.triggerProbability;
                analysis.confidence = intensiveResults.confidence;
                analysis.processingTier = 'intensive';

                this.performanceManager.endTiming(timing);
            } catch (error) {
                console.warn('[Monitor] Intensive analysis failed:', error.message);
            }
        }

        // Update conversation history
        this.updateConversationHistory(message, analysis);

        return analysis;
    }

    /**
     * Instant analysis: Pattern matching and cache checks
     */
    async instantAnalysis(message, context) {
        const cacheKey = this.generateCacheKey(message);

        // Check cache first
        if (this.semanticCache.has(cacheKey)) {
            const cached = this.semanticCache.get(cacheKey);
            // Update last used timestamp
            cached.lastUsed = Date.now();
            this.semanticCache.set(cacheKey, cached);
            return { ...cached, confidence: 0.9 }; // High confidence for cached results
        }

        // Simple pattern matching for common triggers
        const triggerPatterns = [
            /what (did|do) we (decide|choose|do)/i,
            /remind me (about|how|what)/i,
            /similar to (what|how) we/i,
            /like we (discussed|did|decided)/i,
            /according to (our|previous)/i,
            /remember when we/i,
            /last time we/i
        ];

        let triggerProbability = 0;
        const topics = [];

        // Check for explicit memory trigger patterns
        for (const pattern of triggerPatterns) {
            if (pattern.test(message)) {
                triggerProbability = Math.max(triggerProbability, 0.8);
                topics.push('memory-request');
                break;
            }
        }

        // Extract obvious topics (technology names, frameworks)
        const techPatterns = [
            /\b(react|vue|angular|node|python|java|docker|kubernetes)\b/i,
            /\b(api|database|frontend|backend|ui|ux)\b/i,
            /\b(authentication|oauth|security|performance)\b/i
        ];

        for (const pattern of techPatterns) {
            const matches = message.match(pattern);
            if (matches) {
                topics.push(...matches.map(m => m.toLowerCase()));
                triggerProbability = Math.max(triggerProbability, 0.4);
            }
        }

        const result = {
            topics: [...new Set(topics)], // Remove duplicates
            triggerProbability,
            confidence: triggerProbability > 0.5 ? 0.8 : 0.4,
            lastUsed: Date.now()
        };

        // Cache result
        this.semanticCache.set(cacheKey, result);
        this.cleanCache();

        return result;
    }

    /**
     * Fast analysis: Lightweight semantic processing
     */
    async fastAnalysis(message, context) {
        // Tokenize and extract key phrases
        const tokens = this.tokenizeMessage(message);
        const keyPhrases = this.extractKeyPhrases(tokens);

        // Analyze topic shift from recent history
        const semanticShift = this.calculateSemanticShift(keyPhrases);

        // Calculate trigger probability based on context and content
        let triggerProbability = 0;

        // Check for question patterns that suggest memory need
        if (this.isQuestionPattern(message)) {
            triggerProbability += 0.3;
        }

        // Check for reference to past work
        if (this.referencesPastWork(message)) {
            triggerProbability += 0.4;
        }

        // Check for topic complexity
        if (keyPhrases.length > 3) {
            triggerProbability += 0.2;
        }

        // Semantic shift indicates topic change
        if (semanticShift > 0.5) {
            triggerProbability += 0.3;
        }

        return {
            topics: keyPhrases,
            semanticShift,
            triggerProbability: Math.min(triggerProbability, 1.0),
            confidence: 0.7
        };
    }

    /**
     * Intensive analysis: Deep semantic understanding
     */
    async intensiveAnalysis(message, context) {
        // This would integrate with more sophisticated NLP if available
        // For now, enhance the fast analysis with deeper processing

        const fastResult = await this.fastAnalysis(message, context);

        // Analyze conversation context for better topic understanding
        const contextTopics = this.analyzeConversationContext();
        const mergedTopics = this.mergeTopics(fastResult.topics, contextTopics);

        // More sophisticated semantic shift calculation
        const enhancedSemanticShift = this.calculateEnhancedSemanticShift(message, context);

        // Advanced trigger probability with context weighting
        let enhancedTriggerProbability = fastResult.triggerProbability;

        // Weight based on conversation history
        if (this.conversationHistory.length > 5) {
            const historyWeight = this.calculateHistoryRelevance(message);
            enhancedTriggerProbability += historyWeight * 0.2;
        }

        // Project context relevance
        if (context.projectContext) {
            const projectRelevance = this.calculateProjectRelevance(message, context.projectContext);
            enhancedTriggerProbability += projectRelevance * 0.3;
        }

        return {
            topics: mergedTopics,
            semanticShift: enhancedSemanticShift,
            triggerProbability: Math.min(enhancedTriggerProbability, 1.0),
            confidence: 0.9
        };
    }

    /**
     * Helper methods for analysis
     */

    tokenizeMessage(message) {
        return message.toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(token => token.length > 2);
    }

    extractKeyPhrases(tokens) {
        // Simple key phrase extraction
        const technicalTerms = new Set([
            'react', 'vue', 'angular', 'node', 'python', 'java', 'javascript',
            'api', 'database', 'frontend', 'backend', 'authentication', 'oauth',
            'docker', 'kubernetes', 'security', 'performance', 'architecture',
            'component', 'service', 'endpoint', 'middleware', 'framework'
        ]);

        return tokens.filter(token => technicalTerms.has(token));
    }

    calculateSemanticShift(currentTopics) {
        if (this.currentTopics.size === 0) {
            this.currentTopics = new Set(currentTopics);
            return 0;
        }

        const intersection = new Set([...currentTopics].filter(x => this.currentTopics.has(x)));
        const union = new Set([...currentTopics, ...this.currentTopics]);

        // Prevent division by zero when both sets are empty
        if (union.size === 0) {
            this.currentTopics = new Set(currentTopics);
            return 0;
        }

        const similarity = intersection.size / union.size;
        const shift = 1 - similarity;

        // Update current topics
        this.currentTopics = new Set(currentTopics);

        return shift;
    }

    isQuestionPattern(message) {
        const questionPatterns = [
            /^(what|how|why|when|where|which|who)/i,
            /\?$/,
            /^(can|could|would|should|do|does|did|is|are|was|were)/i
        ];

        return questionPatterns.some(pattern => pattern.test(message.trim()));
    }

    referencesPastWork(message) {
        const pastWorkPatterns = [
            /\b(previous|earlier|before|last time|remember|recall)\b/i,
            /\b(we (did|used|chose|decided|implemented))\b/i,
            /\b(our (approach|solution|decision|choice))\b/i
        ];

        return pastWorkPatterns.some(pattern => pattern.test(message));
    }

    mergeTopics(topics1, topics2) {
        return [...new Set([...topics1, ...topics2])];
    }

    analyzeConversationContext() {
        // Analyze recent conversation for recurring topics
        const recentMessages = this.conversationHistory.slice(-this.contextWindow);
        const allTopics = recentMessages.flatMap(msg => msg.analysis?.topics || []);

        // Count topic frequency
        const topicCounts = {};
        allTopics.forEach(topic => {
            topicCounts[topic] = (topicCounts[topic] || 0) + 1;
        });

        // Return topics mentioned more than once
        return Object.entries(topicCounts)
            .filter(([topic, count]) => count > 1)
            .map(([topic]) => topic);
    }

    calculateEnhancedSemanticShift(message, context) {
        // Enhanced semantic shift with context weighting
        const basicShift = this.calculateSemanticShift(this.extractKeyPhrases(this.tokenizeMessage(message)));

        // Weight by message length and complexity
        const lengthWeight = Math.min(message.length / 500, 1.0);
        const complexityWeight = (message.match(/\b(implement|architecture|design|strategy|approach)\b/gi) || []).length * 0.1;

        return Math.min(basicShift + lengthWeight * 0.2 + complexityWeight, 1.0);
    }

    calculateHistoryRelevance(message) {
        // Calculate how relevant current message is to conversation history
        if (this.conversationHistory.length === 0) return 0;

        const messageTopics = new Set(this.extractKeyPhrases(this.tokenizeMessage(message)));
        const historyTopics = new Set(
            this.conversationHistory
                .flatMap(msg => msg.analysis?.topics || [])
        );

        const intersection = new Set([...messageTopics].filter(x => historyTopics.has(x)));
        return intersection.size / Math.max(messageTopics.size, 1);
    }

    calculateProjectRelevance(message, projectContext) {
        if (!projectContext) return 0;

        const messageTokens = this.tokenizeMessage(message);
        const projectTerms = [
            projectContext.name?.toLowerCase(),
            projectContext.language?.toLowerCase(),
            ...(projectContext.frameworks || []).map(f => f.toLowerCase())
        ].filter(Boolean);

        const relevantTerms = messageTokens.filter(token =>
            projectTerms.some(term => term.includes(token) || token.includes(term))
        );

        return relevantTerms.length / Math.max(messageTokens.length, 1);
    }

    updateConversationHistory(message, analysis) {
        this.conversationHistory.push({
            message,
            analysis,
            timestamp: Date.now()
        });

        // Keep only recent history
        if (this.conversationHistory.length > this.contextWindow * 2) {
            this.conversationHistory.splice(0, this.conversationHistory.length - this.contextWindow);
        }
    }

    generateCacheKey(message) {
        // Generate cache key from message content
        return message.toLowerCase().replace(/[^\w]/g, '').substring(0, 50);
    }

    cleanCache() {
        // Clean cache if it gets too large
        if (this.semanticCache.size > 100) {
            const entries = Array.from(this.semanticCache.entries());
            entries.sort((a, b) => (b[1].lastUsed || 0) - (a[1].lastUsed || 0));

            // Keep only the 50 most recently used entries
            this.semanticCache.clear();
            entries.slice(0, 50).forEach(([key, value]) => {
                this.semanticCache.set(key, value);
            });
        }
    }

    /**
     * Get current performance status
     */
    getPerformanceStatus() {
        return {
            tierConfig: this.tierConfig,
            cacheSize: this.semanticCache.size,
            historyLength: this.conversationHistory.length,
            currentTopics: Array.from(this.currentTopics),
            performanceReport: this.performanceManager.getPerformanceReport()
        };
    }

    /**
     * Update performance profile
     */
    updatePerformanceProfile(profileName) {
        this.performanceManager.switchProfile(profileName);
        this.updateTierConfiguration();
    }
}

module.exports = { TieredConversationMonitor };