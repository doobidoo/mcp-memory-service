/**
 * Adaptive Pattern Detector
 * Detects natural language patterns that suggest memory triggers needed
 */

class AdaptivePatternDetector {
    constructor(config = {}, performanceManager = null) {
        this.config = config;
        this.performanceManager = performanceManager;

        // Pattern sensitivity (0-1, where 1 is most sensitive)
        this.sensitivity = config.sensitivity || 0.7;

        // Pattern categories with performance tiers
        this.patterns = {
            // Tier 1: Instant patterns (regex-based, < 50ms)
            instant: {
                explicitMemoryRequests: [
                    {
                        pattern: /what (did|do) we (decide|choose|do|discuss) (about|regarding|for|with)/i,
                        confidence: 0.9,
                        description: 'Direct memory request'
                    },
                    {
                        pattern: /remind me (about|how|what|of|regarding)/i,
                        confidence: 0.9,
                        description: 'Explicit reminder request'
                    },
                    {
                        pattern: /remember (when|how|what|that) we/i,
                        confidence: 0.8,
                        description: 'Memory recall request'
                    },
                    {
                        pattern: /according to (our|the) (previous|earlier|last)/i,
                        confidence: 0.8,
                        description: 'Reference to past decisions'
                    }
                ],

                pastWorkReferences: [
                    {
                        pattern: /similar to (what|how) we (did|used|implemented)/i,
                        confidence: 0.7,
                        description: 'Comparison to past work'
                    },
                    {
                        pattern: /like (we|the) (discussed|decided|implemented|chose) (before|earlier|previously)/i,
                        confidence: 0.7,
                        description: 'Reference to past implementation'
                    },
                    {
                        pattern: /the (same|approach|solution|pattern) (we|that) (used|implemented|chose)/i,
                        confidence: 0.6,
                        description: 'Reuse of past solutions'
                    }
                ],

                questionPatterns: [
                    {
                        pattern: /^(how do|how did|how should|how can) we/i,
                        confidence: 0.5,
                        description: 'Implementation question'
                    },
                    {
                        pattern: /^(what is|what was|what should be) (our|the) (approach|strategy|pattern)/i,
                        confidence: 0.6,
                        description: 'Strategy question'
                    },
                    {
                        pattern: /^(why did|why do|why should) we (choose|use|implement)/i,
                        confidence: 0.5,
                        description: 'Rationale question'
                    }
                ]
            },

            // Tier 2: Fast patterns (contextual analysis, < 150ms)
            fast: {
                technicalDiscussions: [
                    {
                        pattern: /\b(architecture|design|pattern|approach|strategy|implementation)\b/i,
                        context: ['technical', 'decision'],
                        confidence: 0.4,
                        description: 'Technical architecture discussion'
                    },
                    {
                        pattern: /\b(authentication|authorization|security|oauth|jwt)\b/i,
                        context: ['security', 'implementation'],
                        confidence: 0.5,
                        description: 'Security implementation discussion'
                    },
                    {
                        pattern: /\b(database|storage|persistence|schema|migration)\b/i,
                        context: ['data', 'implementation'],
                        confidence: 0.5,
                        description: 'Data layer discussion'
                    }
                ],

                projectContinuity: [
                    {
                        pattern: /\b(continue|continuing|resume|pick up where)\b/i,
                        context: ['continuation'],
                        confidence: 0.6,
                        description: 'Project continuation'
                    },
                    {
                        pattern: /\b(next step|next phase|moving forward|proceed with)\b/i,
                        context: ['progression'],
                        confidence: 0.4,
                        description: 'Project progression'
                    }
                ],

                problemSolving: [
                    {
                        pattern: /\b(issue|problem|bug|error|failure) (with|in|regarding)/i,
                        context: ['troubleshooting'],
                        confidence: 0.6,
                        description: 'Problem solving discussion'
                    },
                    {
                        pattern: /\b(fix|resolve|solve|debug|troubleshoot)\b/i,
                        context: ['troubleshooting'],
                        confidence: 0.4,
                        description: 'Problem resolution'
                    }
                ]
            },

            // Tier 3: Intensive patterns (semantic analysis, < 500ms)
            intensive: {
                contextualReferences: [
                    {
                        semantic: ['previous discussion', 'earlier conversation', 'past decision'],
                        confidence: 0.7,
                        description: 'Contextual reference to past'
                    },
                    {
                        semantic: ['established pattern', 'agreed approach', 'standard practice'],
                        confidence: 0.6,
                        description: 'Reference to established practices'
                    }
                ],

                complexQuestions: [
                    {
                        semantic: ['best practice', 'recommended approach', 'optimal solution'],
                        confidence: 0.5,
                        description: 'Best practice inquiry'
                    }
                ]
            }
        };

        // Pattern matching statistics
        this.stats = {
            totalMatches: 0,
            patternHits: new Map(),
            falsePositives: 0,
            userFeedback: []
        };

        // Adaptive learning
        this.adaptiveSettings = {
            learningEnabled: config.adaptiveLearning !== false,
            confidenceAdjustments: new Map(),
            userPreferences: new Map()
        };
    }

    /**
     * Detect patterns in user message with tiered approach
     */
    async detectPatterns(message, context = {}) {
        const results = {
            matches: [],
            confidence: 0,
            processingTier: 'none',
            triggerRecommendation: false
        };

        // Tier 1: Instant pattern detection
        if (this.shouldRunTier('instant')) {
            const timing = this.performanceManager?.startTiming('pattern_detection_instant', 'instant');

            const instantMatches = this.detectInstantPatterns(message);
            results.matches.push(...instantMatches);
            results.processingTier = 'instant';

            if (timing) this.performanceManager.endTiming(timing);

            // Early return if high confidence instant match
            const maxConfidence = Math.max(...instantMatches.map(m => m.confidence), 0);
            if (maxConfidence > 0.8) {
                results.confidence = maxConfidence;
                results.triggerRecommendation = true;
                return results;
            }
        }

        // Tier 2: Fast contextual analysis
        if (this.shouldRunTier('fast')) {
            const timing = this.performanceManager?.startTiming('pattern_detection_fast', 'fast');

            const fastMatches = this.detectFastPatterns(message, context);
            results.matches.push(...fastMatches);
            results.processingTier = 'fast';

            if (timing) this.performanceManager.endTiming(timing);
        }

        // Tier 3: Intensive semantic analysis
        if (this.shouldRunTier('intensive') && this.shouldRunIntensiveAnalysis(results.matches)) {
            const timing = this.performanceManager?.startTiming('pattern_detection_intensive', 'intensive');

            const intensiveMatches = await this.detectIntensivePatterns(message, context);
            results.matches.push(...intensiveMatches);
            results.processingTier = 'intensive';

            if (timing) this.performanceManager.endTiming(timing);
        }

        // Calculate overall confidence and recommendation
        results.confidence = this.calculateOverallConfidence(results.matches);
        results.triggerRecommendation = this.shouldRecommendTrigger(results);

        // Record statistics
        this.recordPatternMatch(results);

        return results;
    }

    /**
     * Detect instant patterns using regex
     */
    detectInstantPatterns(message) {
        const matches = [];

        for (const [category, patterns] of Object.entries(this.patterns.instant)) {
            for (const patternDef of patterns) {
                if (patternDef.pattern.test(message)) {
                    const adjustedConfidence = this.adjustConfidence(patternDef.confidence, category);

                    matches.push({
                        type: 'instant',
                        category,
                        pattern: patternDef.description,
                        confidence: adjustedConfidence,
                        message: patternDef.description
                    });

                    // Record pattern hit
                    const key = `${category}:${patternDef.description}`;
                    this.stats.patternHits.set(key, (this.stats.patternHits.get(key) || 0) + 1);
                }
            }
        }

        return matches;
    }

    /**
     * Detect fast patterns with context analysis
     */
    detectFastPatterns(message, context) {
        const matches = [];

        for (const [category, patterns] of Object.entries(this.patterns.fast)) {
            for (const patternDef of patterns) {
                if (patternDef.pattern.test(message)) {
                    // Check if context matches
                    const contextMatch = this.checkContextMatch(patternDef.context, context);
                    const contextBoost = contextMatch ? 0.2 : 0;

                    const adjustedConfidence = this.adjustConfidence(
                        patternDef.confidence + contextBoost,
                        category
                    );

                    matches.push({
                        type: 'fast',
                        category,
                        pattern: patternDef.description,
                        confidence: adjustedConfidence,
                        contextMatch,
                        message: patternDef.description
                    });
                }
            }
        }

        return matches;
    }

    /**
     * Detect intensive patterns with semantic analysis
     */
    async detectIntensivePatterns(message, context) {
        const matches = [];

        // This would integrate with semantic analysis libraries if available
        // For now, we'll do enhanced keyword matching with semantic similarity

        for (const [category, patterns] of Object.entries(this.patterns.intensive)) {
            for (const patternDef of patterns) {
                if (patternDef.semantic) {
                    const semanticMatch = this.checkSemanticMatch(message, patternDef.semantic);
                    if (semanticMatch.isMatch) {
                        const adjustedConfidence = this.adjustConfidence(
                            patternDef.confidence * semanticMatch.similarity,
                            category
                        );

                        matches.push({
                            type: 'intensive',
                            category,
                            pattern: patternDef.description,
                            confidence: adjustedConfidence,
                            similarity: semanticMatch.similarity,
                            message: patternDef.description
                        });
                    }
                }
            }
        }

        return matches;
    }

    /**
     * Check if semantic patterns match (simplified implementation)
     */
    checkSemanticMatch(message, semanticTerms) {
        const messageLower = message.toLowerCase();
        let matchCount = 0;
        let totalTerms = 0;

        for (const term of semanticTerms) {
            totalTerms++;
            // Simple keyword matching - in practice, this would use semantic similarity
            const words = term.toLowerCase().split(' ');
            const termMatches = words.every(word => messageLower.includes(word));

            if (termMatches) {
                matchCount++;
            }
        }

        // Prevent division by zero
        if (totalTerms === 0) {
            return {
                isMatch: false,
                similarity: 0
            };
        }

        const similarity = matchCount / totalTerms;
        return {
            isMatch: similarity > 0.3, // Threshold for semantic match
            similarity
        };
    }

    /**
     * Check if context matches pattern requirements
     */
    checkContextMatch(requiredContext, actualContext) {
        if (!requiredContext || !actualContext) return false;

        return requiredContext.some(reqCtx =>
            Object.keys(actualContext).some(key =>
                key.toLowerCase().includes(reqCtx.toLowerCase()) ||
                (typeof actualContext[key] === 'string' &&
                 actualContext[key].toLowerCase().includes(reqCtx.toLowerCase()))
            )
        );
    }

    /**
     * Adjust confidence based on learning and user preferences
     */
    adjustConfidence(baseConfidence, category) {
        let adjusted = baseConfidence;

        // Apply sensitivity adjustment
        adjusted = adjusted * this.sensitivity;

        // Apply learned adjustments
        if (this.adaptiveSettings.confidenceAdjustments.has(category)) {
            const adjustment = this.adaptiveSettings.confidenceAdjustments.get(category);
            adjusted = Math.max(0, Math.min(1, adjusted + adjustment));
        }

        return adjusted;
    }

    /**
     * Calculate overall confidence from all matches
     */
    calculateOverallConfidence(matches) {
        if (matches.length === 0) return 0;

        // Weight by match type (instant > fast > intensive for reliability)
        const weights = { instant: 1.0, fast: 0.8, intensive: 0.6 };
        let weightedSum = 0;
        let totalWeight = 0;

        for (const match of matches) {
            const weight = weights[match.type] || 0.5;
            weightedSum += match.confidence * weight;
            totalWeight += weight;
        }

        return totalWeight > 0 ? weightedSum / totalWeight : 0;
    }

    /**
     * Determine if we should recommend triggering memory hooks
     */
    shouldRecommendTrigger(results) {
        const confidence = results.confidence;
        const matchCount = results.matches.length;

        // High confidence single match
        if (confidence > 0.8) return true;

        // Medium confidence with multiple matches
        if (confidence > 0.6 && matchCount > 1) return true;

        // Lower threshold for explicit memory requests
        const hasExplicitRequest = results.matches.some(m =>
            m.category === 'explicitMemoryRequests' && m.confidence > 0.5
        );
        if (hasExplicitRequest) return true;

        // Any match with reasonable confidence should trigger
        if (matchCount > 0 && confidence > 0.4) return true;

        return false;
    }

    /**
     * Determine if we should run intensive analysis
     */
    shouldRunIntensiveAnalysis(currentMatches) {
        // Only run intensive if we have some matches but low confidence
        const hasMatches = currentMatches.length > 0;
        const lowConfidence = Math.max(...currentMatches.map(m => m.confidence), 0) < 0.7;

        return hasMatches && lowConfidence;
    }

    /**
     * Check if a processing tier should run
     */
    shouldRunTier(tier) {
        if (!this.performanceManager) return true;

        const tierMap = {
            instant: 'instant',
            fast: 'fast',
            intensive: 'intensive'
        };

        try {
            return this.performanceManager.shouldRunHook(`pattern_detection_${tier}`, tierMap[tier]);
        } catch (error) {
            // If performance manager fails, allow the tier to run
            console.warn(`[Pattern Detector] Performance check failed for ${tier}: ${error.message}`);
            return true;
        }
    }

    /**
     * Record pattern match for statistics and learning
     */
    recordPatternMatch(results) {
        this.stats.totalMatches++;

        // Update pattern hit statistics
        for (const match of results.matches) {
            const key = `${match.category}:${match.pattern}`;
            this.stats.patternHits.set(key, (this.stats.patternHits.get(key) || 0) + 1);
        }
    }

    /**
     * Learn from user feedback
     */
    recordUserFeedback(isPositive, patternResults, context = {}) {
        if (!this.adaptiveSettings.learningEnabled) return;

        const feedback = {
            positive: isPositive,
            patterns: patternResults.matches.map(m => ({
                category: m.category,
                pattern: m.pattern,
                confidence: m.confidence
            })),
            overallConfidence: patternResults.confidence,
            triggerRecommendation: patternResults.triggerRecommendation,
            timestamp: Date.now(),
            context
        };

        this.stats.userFeedback.push(feedback);

        // Adjust confidence for patterns based on feedback
        this.adjustPatternsFromFeedback(feedback);

        // Keep feedback history manageable
        if (this.stats.userFeedback.length > 100) {
            this.stats.userFeedback.splice(0, 20);
        }
    }

    /**
     * Adjust pattern confidence based on user feedback
     */
    adjustPatternsFromFeedback(feedback) {
        const adjustmentFactor = feedback.positive ? 0.05 : -0.05;

        for (const patternInfo of feedback.patterns) {
            const currentAdjustment = this.adaptiveSettings.confidenceAdjustments.get(patternInfo.category) || 0;
            const newAdjustment = Math.max(-0.3, Math.min(0.3, currentAdjustment + adjustmentFactor));
            this.adaptiveSettings.confidenceAdjustments.set(patternInfo.category, newAdjustment);
        }
    }

    /**
     * Get pattern detection statistics
     */
    getStatistics() {
        const recentFeedback = this.stats.userFeedback.slice(-20);
        const positiveRate = recentFeedback.length > 0 ?
            recentFeedback.filter(f => f.positive).length / recentFeedback.length : 0;

        return {
            totalMatches: this.stats.totalMatches,
            patternHitCounts: Object.fromEntries(this.stats.patternHits),
            positiveRate: Math.round(positiveRate * 100),
            confidenceAdjustments: Object.fromEntries(this.adaptiveSettings.confidenceAdjustments),
            sensitivity: this.sensitivity,
            learningEnabled: this.adaptiveSettings.learningEnabled
        };
    }

    /**
     * Update sensitivity setting
     */
    updateSensitivity(newSensitivity) {
        this.sensitivity = Math.max(0, Math.min(1, newSensitivity));
    }

    /**
     * Reset learning data (useful for testing)
     */
    resetLearning() {
        this.adaptiveSettings.confidenceAdjustments.clear();
        this.stats.userFeedback = [];
        this.stats.patternHits.clear();
        this.stats.totalMatches = 0;
    }
}

module.exports = { AdaptivePatternDetector };