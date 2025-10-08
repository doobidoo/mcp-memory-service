#!/usr/bin/env node

/**
 * Test script to validate recency-focused scoring improvements
 */

const { scoreMemoryRelevance, calculateTimeDecay, calculateRecencyBonus } = require('./utilities/memory-scorer');
const config = require('./config.json');

// Test project context
const projectContext = {
    name: 'mcp-memory-service',
    language: 'Python',
    frameworks: ['FastAPI'],
    tools: ['pytest']
};

// Test memories with different ages
const testMemories = [
    {
        content: 'Fixed critical bug in HTTP protocol implementation for memory hooks',
        tags: ['mcp-memory-service', 'bug-fix', 'http-protocol'],
        memory_type: 'bug-fix',
        created_at_iso: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString() // 3 days ago
    },
    {
        content: 'Comprehensive README restructuring and organization completed successfully for MCP Memory Service project',
        tags: ['mcp-memory-service', 'claude-code-reference', 'documentation'],
        memory_type: 'reference',
        created_at_iso: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString() // 60 days ago
    },
    {
        content: 'Implemented dashboard dark mode with improved UX',
        tags: ['mcp-memory-service', 'feature', 'dashboard'],
        memory_type: 'feature',
        created_at_iso: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString() // 5 days ago
    },
    {
        content: 'CONTRIBUTING.md Structure - Created comprehensive contribution guidelines',
        tags: ['mcp-memory-service', 'claude-code-reference', 'documentation'],
        memory_type: 'reference',
        created_at_iso: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString() // 30 days ago
    },
    {
        content: 'Removed ChromaDB backend - major refactoring for v8.0',
        tags: ['mcp-memory-service', 'refactor', 'architecture'],
        memory_type: 'architecture',
        created_at_iso: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString() // 4 days ago
    }
];

// Calculate daysAgo once for each memory (DRY principle)
const memoriesWithAge = testMemories.map(mem => ({
    ...mem,
    daysAgo: Math.floor((Date.now() - new Date(mem.created_at_iso)) / (1000 * 60 * 60 * 24))
}));

console.log('\n=== RECENCY SCORING TEST ===\n');

// Show decay and bonus calculations
console.log('📊 Time Decay and Recency Bonus Analysis:');
console.log('─'.repeat(80));
memoriesWithAge.forEach((mem, idx) => {
    const decayScore = calculateTimeDecay(mem.created_at_iso, config.memoryScoring.timeDecayRate); // Using decay rate from config
    const recencyBonus = calculateRecencyBonus(mem.created_at_iso);

    console.log(`Memory ${idx + 1}: ${mem.daysAgo} days old`);
    console.log(`  Time Decay (${config.memoryScoring.timeDecayRate} rate): ${decayScore.toFixed(3)}`);
    console.log(`  Recency Bonus: ${recencyBonus > 0 ? '+' + recencyBonus.toFixed(3) : '0.000'}`);
    console.log(`  Content: ${mem.content.substring(0, 60)}...`);
    console.log('');
});

// Score memories with new algorithm
console.log('\n📈 Final Scoring Results (New Algorithm):');
console.log('─'.repeat(80));

const scoredMemories = scoreMemoryRelevance(memoriesWithAge, projectContext, {
    verbose: false,
    weights: config.memoryScoring.weights,
    timeDecayRate: config.memoryScoring.timeDecayRate
});

scoredMemories.forEach((memory, index) => {
    console.log(`${index + 1}. Score: ${memory.relevanceScore.toFixed(3)} (${memory.daysAgo} days old)`);
    console.log(`   Content: ${memory.content.substring(0, 70)}...`);
    console.log(`   Breakdown:`);
    console.log(`     - Time Decay: ${memory.scoreBreakdown.timeDecay.toFixed(3)} (weight: ${config.memoryScoring.weights.timeDecay})`);
    console.log(`     - Tag Relevance: ${memory.scoreBreakdown.tagRelevance.toFixed(3)} (weight: ${config.memoryScoring.weights.tagRelevance})`);
    console.log(`     - Content Quality: ${memory.scoreBreakdown.contentQuality.toFixed(3)} (weight: ${config.memoryScoring.weights.contentQuality})`);
    console.log(`     - Recency Bonus: ${memory.scoreBreakdown.recencyBonus.toFixed(3)} (direct boost)`);
    console.log('');
});

console.log('\n✅ Test Summary:');
console.log('─'.repeat(80));
console.log('Expected Behavior:');
console.log('  - Recent memories (3-5 days old) should rank higher');
console.log('  - Recency bonus (+0.15 for <7 days, +0.10 for <14 days, +0.05 for <30 days)');
console.log('  - Gentler time decay (0.05 rate vs old 0.1 rate)');
console.log('  - Higher time weight (0.40 vs old 0.25)');
console.log('  - Old memories with perfect tags should rank lower despite tag advantage\n');

// Check if recent memories are ranked higher
const top3 = scoredMemories.slice(0, 3);
const recentInTop3 = top3.filter(m => m.daysAgo <= 7).length;

if (recentInTop3 >= 2) {
    console.log('✅ SUCCESS: At least 2 of top 3 memories are from the last week');
} else {
    console.log('❌ ISSUE: Recent memories are not prioritized as expected');
}

console.log('');
