/**
 * Performance Manager for Memory Hooks
 * Provides intelligent performance monitoring and adaptive hook management
 */

class PerformanceManager {
    constructor(config = {}) {
        this.config = config;
        this.metrics = {
            totalLatency: [],
            hookLatencies: new Map(),
            userSatisfaction: [],
            degradationEvents: 0
        };

        // Performance tiers
        this.tiers = {
            instant: { maxLatency: 50, priority: 'critical' },
            fast: { maxLatency: 150, priority: 'high' },
            intensive: { maxLatency: 500, priority: 'medium' }
        };

        // Current performance profile
        this.activeProfile = config.defaultProfile || 'balanced';
        this.performanceBudget = this.getProfileBudget(this.activeProfile);

        // Adaptive learning
        this.userPreferences = {
            toleranceLevel: 0.5, // 0 = speed focused, 1 = memory focused
            learningEnabled: true,
            feedbackHistory: []
        };
    }

    /**
     * Get performance budget for a profile
     */
    getProfileBudget(profileName) {
        // Use config profiles first, with hardcoded fallbacks
        const configProfiles = this.config.profiles || {};

        // If profile exists in config, use it (with fallback for missing adaptive calculations)
        if (configProfiles[profileName]) {
            const profile = { ...configProfiles[profileName] };

            // Handle adaptive profile calculations if needed
            if (profileName === 'adaptive') {
                profile.maxLatency = profile.maxLatency || this.calculateAdaptiveLatency();
                profile.enabledTiers = profile.enabledTiers || this.calculateAdaptiveTiers();
            }

            return profile;
        }

        // Fallback to hardcoded profiles if not found in config
        const fallbackProfiles = {
            speed_focused: {
                maxLatency: 100,
                enabledTiers: ['instant'],
                backgroundProcessing: false,
                degradeThreshold: 200
            },
            balanced: {
                maxLatency: 200,
                enabledTiers: ['instant', 'fast'],
                backgroundProcessing: true,
                degradeThreshold: 400
            },
            memory_aware: {
                maxLatency: 500,
                enabledTiers: ['instant', 'fast', 'intensive'],
                backgroundProcessing: true,
                degradeThreshold: 1000
            },
            adaptive: {
                maxLatency: this.calculateAdaptiveLatency(),
                enabledTiers: this.calculateAdaptiveTiers(),
                backgroundProcessing: true,
                degradeThreshold: 800,
                autoAdjust: true
            }
        };

        return fallbackProfiles[profileName] || fallbackProfiles.balanced;
    }

    /**
     * Calculate adaptive latency based on user behavior
     */
    calculateAdaptiveLatency() {
        if (this.metrics.totalLatency.length < 10) {
            return 200; // Default for new users
        }

        const avgLatency = this.metrics.totalLatency.reduce((a, b) => a + b, 0) / this.metrics.totalLatency.length;
        const userTolerance = this.userPreferences?.toleranceLevel || 0.5;

        // Adaptive calculation: balance observed tolerance with user preference
        return Math.min(500, Math.max(100, avgLatency * (1 + userTolerance)));
    }

    /**
     * Calculate which tiers should be enabled adaptively
     */
    calculateAdaptiveTiers() {
        const tolerance = this.userPreferences?.toleranceLevel || 0.5;

        if (tolerance < 0.3) return ['instant'];
        if (tolerance < 0.7) return ['instant', 'fast'];
        return ['instant', 'fast', 'intensive'];
    }

    /**
     * Start timing a hook operation
     */
    startTiming(hookName, tier = 'fast') {
        return {
            hookName,
            tier,
            startTime: Date.now(),
            expectedLatency: this.tiers[tier]?.maxLatency || 150
        };
    }

    /**
     * End timing and record metrics
     */
    endTiming(timingContext) {
        const endTime = Date.now();
        const latency = endTime - timingContext.startTime;

        // Record metrics
        this.recordHookLatency(timingContext.hookName, latency, timingContext.tier);
        this.recordTotalLatency(latency);

        // Check if we exceeded performance budget
        const exceedsThreshold = latency > this.performanceBudget.degradeThreshold;
        if (exceedsThreshold) {
            this.handlePerformanceDegradation(timingContext.hookName, latency);
        }

        return {
            latency,
            tier: timingContext.tier,
            withinBudget: latency <= this.performanceBudget.maxLatency,
            exceedsThreshold
        };
    }

    /**
     * Record hook-specific latency
     */
    recordHookLatency(hookName, latency, tier) {
        if (!this.metrics.hookLatencies.has(hookName)) {
            this.metrics.hookLatencies.set(hookName, []);
        }

        const hookMetrics = this.metrics.hookLatencies.get(hookName);
        hookMetrics.push({ latency, tier, timestamp: Date.now() });

        // Keep only recent measurements (last 50)
        if (hookMetrics.length > 50) {
            hookMetrics.splice(0, hookMetrics.length - 50);
        }
    }

    /**
     * Record total request latency
     */
    recordTotalLatency(latency) {
        this.metrics.totalLatency.push(latency);

        // Keep rolling window of recent measurements
        if (this.metrics.totalLatency.length > 100) {
            this.metrics.totalLatency.splice(0, this.metrics.totalLatency.length - 100);
        }
    }

    /**
     * Handle performance degradation
     */
    handlePerformanceDegradation(hookName, latency) {
        this.metrics.degradationEvents++;

        console.warn(`[Performance] Hook "${hookName}" exceeded threshold: ${latency}ms`);

        // Adaptive response based on profile
        if (this.performanceBudget.autoAdjust) {
            this.adaptToPerformance(hookName, latency);
        }
    }

    /**
     * Adapt hooks based on performance
     */
    adaptToPerformance(hookName, latency) {
        // If a hook consistently performs poorly, suggest tier reduction
        const hookHistory = this.metrics.hookLatencies.get(hookName) || [];
        const recentHistory = hookHistory.slice(-10);

        if (recentHistory.length >= 5) {
            const avgLatency = recentHistory.reduce((a, b) => a + b.latency, 0) / recentHistory.length;

            if (avgLatency > this.performanceBudget.maxLatency * 1.5) {
                // Suggest moving hook to lower tier or disabling
                this.suggestHookOptimization(hookName, avgLatency);
            }
        }
    }

    /**
     * Suggest hook optimization
     */
    suggestHookOptimization(hookName, avgLatency) {
        const suggestion = {
            hookName,
            avgLatency,
            suggestion: avgLatency > 300 ? 'disable' : 'reduce_tier',
            timestamp: Date.now()
        };

        console.log(`[Performance] Suggestion for ${hookName}: ${suggestion.suggestion} (avg: ${avgLatency}ms)`);
        return suggestion;
    }

    /**
     * Check if a hook should run based on current performance profile
     */
    shouldRunHook(hookName, tier = 'fast') {
        const profile = this.performanceBudget;

        // Check if tier is enabled
        if (!profile.enabledTiers.includes(tier)) {
            return false;
        }

        // Check recent performance
        const hookHistory = this.metrics.hookLatencies.get(hookName);
        if (hookHistory && hookHistory.length > 5) {
            const recentLatencies = hookHistory.slice(-5);
            const avgLatency = recentLatencies.reduce((a, b) => a + b.latency, 0) / recentLatencies.length;

            // Don't run if consistently exceeds budget
            if (avgLatency > profile.maxLatency * 1.2) {
                return false;
            }
        }

        return true;
    }

    /**
     * Switch performance profile
     */
    switchProfile(profileName) {
        if (!['speed_focused', 'balanced', 'memory_aware', 'adaptive'].includes(profileName)) {
            throw new Error(`Invalid profile: ${profileName}`);
        }

        this.activeProfile = profileName;
        this.performanceBudget = this.getProfileBudget(profileName);

        console.log(`[Performance] Switched to profile: ${profileName}`);
        return this.performanceBudget;
    }

    /**
     * Learn from user feedback
     */
    recordUserFeedback(isPositive, context = {}) {
        if (!this.userPreferences.learningEnabled) return;

        const feedback = {
            positive: isPositive,
            context,
            latency: context.latency || 0,
            timestamp: Date.now()
        };

        this.userPreferences.feedbackHistory.push(feedback);

        // Update tolerance based on feedback
        this.updateUserTolerance(feedback);

        // Keep feedback history manageable
        if (this.userPreferences.feedbackHistory.length > 50) {
            this.userPreferences.feedbackHistory.splice(0, 10);
        }
    }

    /**
     * Update user tolerance based on feedback patterns
     */
    updateUserTolerance(feedback) {
        const recent = this.userPreferences?.feedbackHistory?.slice(-10) || [];
        const positiveCount = recent.filter(f => f.positive).length;
        const negativeCount = recent.length - positiveCount;

        // Ensure userPreferences is initialized
        if (!this.userPreferences) {
            this.userPreferences = {
                toleranceLevel: 0.5,
                learningEnabled: true,
                feedbackHistory: []
            };
        }

        // Adjust tolerance based on feedback patterns
        if (feedback.positive && feedback.latency > 200) {
            // User satisfied with higher latency, increase tolerance
            this.userPreferences.toleranceLevel = Math.min(1.0, this.userPreferences.toleranceLevel + 0.1);
        } else if (!feedback.positive && feedback.latency > 100) {
            // User dissatisfied with latency, decrease tolerance
            this.userPreferences.toleranceLevel = Math.max(0.0, this.userPreferences.toleranceLevel - 0.1);
        }
    }

    /**
     * Get performance report
     */
    getPerformanceReport() {
        const totalRequests = this.metrics.totalLatency.length;
        const avgLatency = totalRequests > 0 ?
            this.metrics.totalLatency.reduce((a, b) => a + b, 0) / totalRequests : 0;

        const hookSummary = {};
        this.metrics.hookLatencies.forEach((latencies, hookName) => {
            const avgHookLatency = latencies.reduce((a, b) => a + b.latency, 0) / latencies.length;
            hookSummary[hookName] = {
                avgLatency: Math.round(avgHookLatency),
                calls: latencies.length,
                tier: latencies[latencies.length - 1]?.tier || 'unknown'
            };
        });

        return {
            profile: this.activeProfile,
            totalRequests,
            avgLatency: Math.round(avgLatency),
            degradationEvents: this.metrics.degradationEvents,
            userTolerance: this.userPreferences.toleranceLevel,
            hookPerformance: hookSummary,
            budget: this.performanceBudget
        };
    }

    /**
     * Reset metrics (useful for testing)
     */
    resetMetrics() {
        this.metrics = {
            totalLatency: [],
            hookLatencies: new Map(),
            userSatisfaction: [],
            degradationEvents: 0
        };
    }
}

module.exports = { PerformanceManager };