#!/usr/bin/env node

const { AdaptivePatternDetector } = require('./utilities/adaptive-pattern-detector');

async function simpleTest() {
    const detector = new AdaptivePatternDetector({ sensitivity: 0.7 });

    const testCases = [
        { message: "What did we decide about the authentication approach?", shouldTrigger: true },
        { message: "Remind me how we handled user sessions", shouldTrigger: true },
        { message: "Remember when we discussed the database schema?", shouldTrigger: true },
        { message: "Just implementing a new feature", shouldTrigger: false }
    ];

    for (const testCase of testCases) {
        const result = await detector.detectPatterns(testCase.message);
        const actualTrigger = result.triggerRecommendation;

        console.log(`Message: "${testCase.message}"`);
        console.log(`Expected: ${testCase.shouldTrigger}, Actual: ${actualTrigger}`);
        console.log(`Confidence: ${result.confidence}`);
        console.log(`Matches: ${result.matches.length}`);
        console.log('---');
    }
}

simpleTest().catch(console.error);