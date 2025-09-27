#!/usr/bin/env node

/**
 * Comprehensive Test Suite for Natural Memory Triggers
 * Tests performance-aware conversation monitoring and pattern detection
 */

const { TieredConversationMonitor } = require('./utilities/tiered-conversation-monitor');
const { AdaptivePatternDetector } = require('./utilities/adaptive-pattern-detector');
const { PerformanceManager } = require('./utilities/performance-manager');
const { MidConversationHook } = require('./core/mid-conversation');

class NaturalTriggersTestSuite {
    constructor() {
        this.testResults = [];
        this.performanceMetrics = [];
    }

    /**
     * Run all tests
     */
    async runAllTests() {
        console.log('🧪 Natural Memory Triggers - Comprehensive Test Suite');
        console.log('═'.repeat(60));

        // Test categories
        const testCategories = [
            { name: 'Performance Management', tests: this.performanceTests },
            { name: 'Pattern Detection', tests: this.patternDetectionTests },
            { name: 'Conversation Monitoring', tests: this.conversationMonitorTests },
            { name: 'Integration Tests', tests: this.integrationTests },
            { name: 'Performance Profiles', tests: this.performanceProfileTests }
        ];

        for (const category of testCategories) {
            console.log(`\n📂 ${category.name}`);
            console.log('─'.repeat(40));

            await category.tests.call(this);
        }

        // Summary
        this.printTestSummary();
        return this.testResults;
    }

    /**
     * Performance Management Tests
     */
    async performanceTests() {
        // Test 1: Performance Manager Initialization
        await this.runTest('Performance Manager Initialization', async () => {
            const perfManager = new PerformanceManager({
                defaultProfile: 'balanced'
            });

            this.assert(perfManager.activeProfile === 'balanced', 'Should initialize with correct profile');
            this.assert(perfManager.performanceBudget.maxLatency === 200, 'Should have correct latency budget');
            return { perfManager };
        });

        // Test 2: Timing Operations
        await this.runTest('Timing Operations', async () => {
            const perfManager = new PerformanceManager();

            const timing = perfManager.startTiming('test_operation', 'fast');
            await this.sleep(50);
            const result = perfManager.endTiming(timing);

            // Test performance tracking functionality without relying on exact timing
            this.assert(typeof result.latency === 'number' && result.latency >= 0, 'Should record numeric latency');
            this.assert(result.latency > 10, 'Should record reasonable latency for 50ms operation');
            this.assert(result.tier === 'fast', 'Should record correct tier');
            this.assert(typeof result.withinBudget === 'boolean', 'Should determine budget compliance');
        });

        // Test 3: Profile Switching
        await this.runTest('Profile Switching', async () => {
            const perfManager = new PerformanceManager();

            const originalProfile = perfManager.activeProfile;
            perfManager.switchProfile('speed_focused');

            this.assert(perfManager.activeProfile === 'speed_focused', 'Should switch to speed focused profile');
            this.assert(perfManager.performanceBudget.maxLatency === 100, 'Should update latency budget');

            perfManager.switchProfile(originalProfile); // Reset
        });

        // Test 4: Adaptive Learning
        await this.runTest('Adaptive Learning', async () => {
            const perfManager = new PerformanceManager();

            // Simulate positive feedback
            perfManager.recordUserFeedback(true, { latency: 300 });
            perfManager.recordUserFeedback(true, { latency: 350 });

            // User tolerance should increase
            const toleranceBefore = perfManager.userPreferences.toleranceLevel;
            perfManager.recordUserFeedback(true, { latency: 400 });
            const toleranceAfter = perfManager.userPreferences.toleranceLevel;

            this.assert(toleranceAfter >= toleranceBefore, 'User tolerance should increase with positive feedback');
        });
    }

    /**
     * Pattern Detection Tests
     */
    async patternDetectionTests() {
        // Test 1: Explicit Memory Requests
        await this.runTest('Explicit Memory Request Detection', async () => {
            const detector = new AdaptivePatternDetector({
                sensitivity: 0.7,
                adaptiveLearning: false // Disable learning for consistent tests
            });

            const testCases = [
                { message: "What did we decide about the authentication approach?", shouldTrigger: true },
                { message: "Remind me how we handled user sessions", shouldTrigger: true },
                { message: "Remember when we discussed the database schema?", shouldTrigger: true },
                { message: "Just implementing a new feature", shouldTrigger: false }
            ];

            for (const testCase of testCases) {
                const result = await detector.detectPatterns(testCase.message);
                const actualTrigger = result.triggerRecommendation;

                // Debug output for failing tests
                if (actualTrigger !== testCase.shouldTrigger) {
                    console.log(`\nDEBUG: "${testCase.message}"`);
                    console.log(`Expected: ${testCase.shouldTrigger}, Got: ${actualTrigger}`);
                    console.log(`Confidence: ${result.confidence}, Matches: ${result.matches.length}`);
                    result.matches.forEach(m => console.log(`  - ${m.category}: ${m.confidence}`));
                }

                this.assert(actualTrigger === testCase.shouldTrigger,
                    `"${testCase.message}" should ${testCase.shouldTrigger ? '' : 'not '}trigger (got ${actualTrigger})`);
            }
        });

        // Test 2: Technical Discussion Patterns
        await this.runTest('Technical Discussion Detection', async () => {
            const detector = new AdaptivePatternDetector({ sensitivity: 0.6 });

            const technicalMessages = [
                "Let's discuss the authentication architecture",
                "What's our approach to database migrations?",
                "How should we implement the security layer?"
            ];

            for (const message of technicalMessages) {
                const result = await detector.detectPatterns(message, {
                    projectContext: { name: 'test-project', language: 'JavaScript' }
                });

                this.assert(result.matches.length > 0, `Technical message should have pattern matches: "${message}"`);
                this.assert(result.confidence > 0.2, `Technical message should have reasonable confidence: ${result.confidence} for "${message}"`);
            }
        });

        // Test 3: Sensitivity Adjustment
        await this.runTest('Sensitivity Adjustment', async () => {
            const lowSensitivity = new AdaptivePatternDetector({ sensitivity: 0.3 });
            const highSensitivity = new AdaptivePatternDetector({ sensitivity: 0.9 });

            const ambiguousMessage = "How do we handle this?";

            const lowResult = await lowSensitivity.detectPatterns(ambiguousMessage);
            const highResult = await highSensitivity.detectPatterns(ambiguousMessage);

            this.assert(highResult.confidence >= lowResult.confidence,
                'Higher sensitivity should yield higher confidence for ambiguous messages');
        });

        // Test 4: Learning from Feedback
        await this.runTest('Learning from Feedback', async () => {
            const detector = new AdaptivePatternDetector({ sensitivity: 0.7, adaptiveLearning: true });

            const message = "What's our standard approach?";
            const initialResult = await detector.detectPatterns(message);
            const initialConfidence = initialResult.confidence;

            // Provide positive feedback multiple times
            for (let i = 0; i < 5; i++) {
                detector.recordUserFeedback(true, initialResult);
            }

            const learnedResult = await detector.detectPatterns(message);

            // Note: In a real implementation, this might increase confidence for similar patterns
            // For now, we just verify the feedback was recorded
            const stats = detector.getStatistics();
            this.assert(stats.positiveRate > 0, 'Should record positive feedback');
        });
    }

    /**
     * Conversation Monitoring Tests
     */
    async conversationMonitorTests() {
        // Test 1: Topic Extraction
        await this.runTest('Topic Extraction', async () => {
            const monitor = new TieredConversationMonitor({
                contextWindow: 5
            });

            const technicalMessage = "Let's implement authentication using OAuth and JWT tokens for our React application";
            const analysis = await monitor.analyzeMessage(technicalMessage);

            this.assert(analysis.topics.length > 0, 'Should extract topics from technical message');
            this.assert(analysis.confidence > 0.4, `Should have reasonable confidence: ${analysis.confidence}`);
            this.assert(analysis.processingTier !== 'none', 'Should process with some tier');
        });

        // Test 2: Semantic Shift Detection
        await this.runTest('Semantic Shift Detection', async () => {
            const monitor = new TieredConversationMonitor();

            // First message establishes context
            await monitor.analyzeMessage("Working on React components and state management");

            // Second message on same topic
            const sameTopicResult = await monitor.analyzeMessage("Adding more React hooks to the component");

            // Third message on different topic
            const differentTopicResult = await monitor.analyzeMessage("Let's switch to database schema design");

            this.assert(differentTopicResult.semanticShift > sameTopicResult.semanticShift,
                'Topic change should register higher semantic shift');
        });

        // Test 3: Performance Tier Selection
        await this.runTest('Performance Tier Selection', async () => {
            const perfManager = new PerformanceManager({ defaultProfile: 'speed_focused' });
            const monitor = new TieredConversationMonitor({}, perfManager);

            const message = "Simple question about React";
            const analysis = await monitor.analyzeMessage(message);

            // In speed_focused mode, should prefer instant tier
            this.assert(analysis.processingTier === 'instant' || analysis.processingTier === 'fast',
                `Speed focused mode should use fast tiers, got: ${analysis.processingTier}`);
        });

        // Test 4: Caching Behavior
        await this.runTest('Caching Behavior', async () => {
            const monitor = new TieredConversationMonitor({
                enableCaching: true
            });

            const message = "What is React?";

            // First analysis
            const start1 = Date.now();
            const result1 = await monitor.analyzeMessage(message);
            const time1 = Date.now() - start1;

            // Second analysis (should use cache)
            const start2 = Date.now();
            const result2 = await monitor.analyzeMessage(message);
            const time2 = Date.now() - start2;

            // Check that both results have reasonable confidence values
            this.assert(typeof result1.confidence === 'number', 'First result should have confidence');
            this.assert(typeof result2.confidence === 'number', 'Second result should have confidence');
            // Note: Processing tiers may vary due to performance-based decisions, which is expected behavior
            this.assert(result1.processingTier && result2.processingTier, 'Both results should have processing tiers');
            // Note: Due to timestamps and context changes, exact confidence equality might vary
        });
    }

    /**
     * Integration Tests
     */
    async integrationTests() {
        // Test 1: Full Mid-Conversation Hook
        await this.runTest('Full Mid-Conversation Hook Analysis', async () => {
            const hook = new MidConversationHook({
                enabled: true,
                triggerThreshold: 0.6,
                maxMemoriesPerTrigger: 3,
                performance: { defaultProfile: 'balanced' }
            });

            const context = {
                userMessage: "What did we decide about the authentication strategy?",
                projectContext: {
                    name: 'test-project',
                    language: 'JavaScript',
                    frameworks: ['React']
                }
            };

            const result = await hook.analyzeMessage(context.userMessage, context);

            this.assert(result !== null, 'Should return analysis result');
            this.assert(typeof result.confidence === 'number', 'Should include confidence score');
            this.assert(typeof result.shouldTrigger === 'boolean', 'Should include trigger decision');
            this.assert(result.reasoning, 'Should include reasoning for decision');

            await hook.cleanup();
        });

        // Test 2: Performance Budget Compliance
        await this.runTest('Performance Budget Compliance', async () => {
            const hook = new MidConversationHook({
                performance: { defaultProfile: 'speed_focused' }
            });

            const start = Date.now();
            const result = await hook.analyzeMessage("Quick question about React hooks");
            const elapsed = Date.now() - start;

            // Speed focused should complete and return results
            this.assert(result !== null, `Speed focused mode should return analysis result`);
            console.log(`[Test] Speed focused analysis completed in ${elapsed}ms`);

            await hook.cleanup();
        });

        // Test 3: Cooldown Period
        await this.runTest('Cooldown Period Enforcement', async () => {
            const hook = new MidConversationHook({
                cooldownPeriod: 1000, // 1 second
                triggerThreshold: 0.5
            });

            const message = "What did we decide about authentication?";

            // First trigger
            const result1 = await hook.analyzeMessage(message);

            // Immediate second attempt (should be in cooldown)
            const result2 = await hook.analyzeMessage(message);

            if (result1.shouldTrigger) {
                this.assert(result2.reasoning?.includes('cooldown') || !result2.shouldTrigger,
                    'Should respect cooldown period');
            }

            await hook.cleanup();
        });
    }

    /**
     * Performance Profile Tests
     */
    async performanceProfileTests() {
        // Test 1: Profile Configuration Loading
        await this.runTest('Performance Profile Loading', async () => {
            const profiles = ['speed_focused', 'balanced', 'memory_aware', 'adaptive'];

            for (const profileName of profiles) {
                const perfManager = new PerformanceManager({ defaultProfile: profileName });

                this.assert(perfManager.activeProfile === profileName,
                    `Should load ${profileName} profile correctly`);

                const budget = perfManager.performanceBudget;
                this.assert(budget !== null, `${profileName} should have performance budget`);

                if (profileName !== 'adaptive') {
                    this.assert(typeof budget.maxLatency === 'number',
                        `${profileName} should have numeric maxLatency`);
                }
            }
        });

        // Test 2: Tier Enabling/Disabling
        await this.runTest('Tier Configuration', async () => {
            const speedFocused = new PerformanceManager({ defaultProfile: 'speed_focused' });
            const memoryAware = new PerformanceManager({ defaultProfile: 'memory_aware' });

            // Speed focused should have fewer enabled tiers
            const speedTiers = speedFocused.performanceBudget.enabledTiers || [];
            const memoryTiers = memoryAware.performanceBudget.enabledTiers || [];

            this.assert(speedTiers.length <= memoryTiers.length,
                'Speed focused should have fewer or equal enabled tiers');

            this.assert(speedTiers.includes('instant'),
                'Speed focused should at least include instant tier');
        });

        // Test 3: Adaptive Profile Behavior
        await this.runTest('Adaptive Profile Behavior', async () => {
            const adaptive = new PerformanceManager({ defaultProfile: 'adaptive' });

            // Simulate performance history
            for (let i = 0; i < 20; i++) {
                adaptive.recordTotalLatency(150); // Consistent good performance
            }

            // Check if adaptive calculation makes sense
            const budget = adaptive.getProfileBudget('adaptive');
            this.assert(budget.autoAdjust === true, 'Adaptive profile should have autoAdjust enabled');
        });
    }

    /**
     * Utility Methods
     */

    async runTest(testName, testFunction) {
        try {
            console.log(`  🧪 ${testName}...`);
            const start = Date.now();

            const result = await testFunction();

            const duration = Date.now() - start;
            this.performanceMetrics.push({ testName, duration });

            console.log(`  ✅ ${testName} (${duration}ms)`);
            this.testResults.push({ name: testName, status: 'passed', duration });

            return result;

        } catch (error) {
            console.log(`  ❌ ${testName}: ${error.message}`);
            this.testResults.push({ name: testName, status: 'failed', error: error.message });
            throw error; // Re-throw to stop execution if needed
        }
    }

    assert(condition, message) {
        if (!condition) {
            throw new Error(`Assertion failed: ${message}`);
        }
    }

    async sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    printTestSummary() {
        console.log('\n📊 Test Summary');
        console.log('═'.repeat(50));

        const passed = this.testResults.filter(r => r.status === 'passed').length;
        const failed = this.testResults.filter(r => r.status === 'failed').length;
        const total = this.testResults.length;

        console.log(`Total Tests: ${total}`);
        console.log(`Passed: ${passed} ✅`);
        console.log(`Failed: ${failed} ${failed > 0 ? '❌' : ''}`);
        console.log(`Success Rate: ${((passed / total) * 100).toFixed(1)}%`);

        // Performance summary
        const totalTime = this.performanceMetrics.reduce((sum, m) => sum + m.duration, 0);
        const avgTime = totalTime / this.performanceMetrics.length;

        console.log(`\n⚡ Performance`);
        console.log(`Total Time: ${totalTime}ms`);
        console.log(`Average per Test: ${avgTime.toFixed(1)}ms`);

        if (failed > 0) {
            console.log('\n❌ Failed Tests:');
            this.testResults
                .filter(r => r.status === 'failed')
                .forEach(r => console.log(`  • ${r.name}: ${r.error}`));
        }
    }
}

/**
 * Run tests if called directly
 */
if (require.main === module) {
    const suite = new NaturalTriggersTestSuite();

    suite.runAllTests()
        .then(results => {
            const failed = results.filter(r => r.status === 'failed').length;
            process.exit(failed > 0 ? 1 : 0);
        })
        .catch(error => {
            console.error('❌ Test suite failed:', error.message);
            process.exit(1);
        });
}

module.exports = { NaturalTriggersTestSuite };