#!/usr/bin/env node

/**
 * Debug Pattern Detection
 */

const { AdaptivePatternDetector } = require('./utilities/adaptive-pattern-detector');

async function debugPatternDetection() {
    console.log('🔍 Debugging Pattern Detection');
    console.log('═'.repeat(50));

    const detector = new AdaptivePatternDetector({ sensitivity: 0.7 });

    const testMessage = "What did we decide about the authentication approach?";
    console.log(`\nTesting message: "${testMessage}"`);

    const result = await detector.detectPatterns(testMessage);

    console.log('\nResults:');
    console.log('- Matches found:', result.matches.length);
    console.log('- Confidence:', result.confidence);
    console.log('- Processing tier:', result.processingTier);
    console.log('- Trigger recommendation:', result.triggerRecommendation);

    if (result.matches.length > 0) {
        console.log('\nMatches:');
        result.matches.forEach((match, i) => {
            console.log(`  ${i + 1}. Category: ${match.category}`);
            console.log(`     Pattern: ${match.pattern}`);
            console.log(`     Confidence: ${match.confidence}`);
            console.log(`     Type: ${match.type}`);
        });
    }

    // Test the instant patterns directly
    console.log('\n🔍 Testing Instant Patterns Directly');
    const instantMatches = detector.detectInstantPatterns(testMessage);
    console.log('Instant matches:', instantMatches.length);
    instantMatches.forEach((match, i) => {
        console.log(`  ${i + 1}. ${match.category}: ${match.confidence}`);
    });
}

debugPatternDetection().catch(console.error);