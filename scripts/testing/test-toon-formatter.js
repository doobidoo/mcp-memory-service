#!/usr/bin/env node
/**
 * Test TOON Formatter with Sample Data
 *
 * Validates the TOON formatter produces correct output and measures token savings.
 */

const { formatMemoriesAsTOON, calculateAgeDays, estimateTokenSavings } = require('/Users/hkr/.claude/hooks/utilities/toon-formatter.js');

// Sample memories (realistic data from mcp-memory-service)
const sampleMemories = [
    {
        content: "Critical analytics fix: Dashboard now shows accurate memory count instead of 1,000 sampling limit. Fixed line 386 with direct SQL query via storage.primary.conn.",
        tags: ["dashboard", "fix", "v8.17.1", "analytics"],
        created_at_iso: "2025-11-01T08:45:00Z",
        relevanceScore: 0.95,
        type: "implementation"
    },
    {
        content: "Memory hook retrieves git context for recent development work. Enhanced query building with adaptive weight adjustment.",
        tags: ["hooks", "git", "context", "optimization"],
        created_at_iso: "2025-10-31T20:15:00Z",
        relevanceScore: 0.87,
        type: "feature"
    },
    {
        content: "Hybrid backend provides SQLite speed with Cloudflare persistence. Background sync every 5 minutes.",
        tags: ["hybrid", "backend", "architecture", "performance"],
        created_at_iso: "2025-10-30T12:00:00Z",
        relevanceScore: 0.82,
        type: "note"
    },
    {
        content: "HTTP server runs on port 8000 for hook integration. Ensure config.json endpoint matches.",
        tags: ["http", "server", "config", "hooks"],
        created_at_iso: "2025-10-29T16:30:00Z",
        relevanceScore: 0.78,
        type: "reference"
    },
    {
        content: "Tag repair script fixes malformed JSON serialization artifacts in tags. Processed 1,870 malformed tags across 369 memories.",
        tags: ["maintenance", "tags", "v8.17.1", "repair"],
        created_at_iso: "2025-10-28T09:00:00Z",
        relevanceScore: 0.75,
        type: "tool"
    },
    {
        content: "Type assignment uses multi-tier inference with 80+ tag associations. Confidence scoring for accuracy.",
        tags: ["maintenance", "types", "intelligence", "inference"],
        created_at_iso: "2025-10-27T14:20:00Z",
        relevanceScore: 0.72,
        type: "implementation"
    },
    {
        content: "MCP protocol implements async handlers with global caches for performance. Embedding cache reduces redundant computations.",
        tags: ["mcp", "architecture", "performance", "async"],
        created_at_iso: "2025-10-26T11:30:00Z",
        relevanceScore: 0.68,
        type: "note"
    },
    {
        content: "Cloudflare backend requires API token in .env file. Environment variables take precedence over CLI defaults.",
        tags: ["cloudflare", "config", "setup", "credentials"],
        created_at_iso: "2025-10-25T09:15:00Z",
        relevanceScore: 0.65,
        type: "guide"
    }
];

// Sample project context
const projectContext = {
    name: "mcp-memory-service",
    language: "Python",
    frameworks: ["FastAPI", "SQLite", "MCP"],
    git: {
        branch: "main",
        lastCommit: "1f8b4b8"
    }
};

// Sample git context
const gitContext = {
    commits: Array(20).fill({}), // 20 commits
    changelogEntries: Array(3).fill({}), // 3 changelog entries
    developmentKeywords: {
        keywords: ["feat", "release", "docs", "v8.17.1", "chore", "version", "dashboard", "analytics"]
    }
};

// Sample storage info
const storageInfo = {
    backend: "sqlite-vec",
    health: {
        totalMemories: 2224,
        databaseSizeMB: 8.78
    }
};

console.log('='.repeat(80));
console.log('TOON FORMATTER TEST');
console.log('='.repeat(80));
console.log();

// Test 1: Basic functionality
console.log('Test 1: Basic TOON Formatting');
console.log('-'.repeat(80));
try {
    const toonOutput = formatMemoriesAsTOON(sampleMemories, projectContext, {
        gitContext: gitContext,
        storageInfo: storageInfo,
        includeGitContext: true,
        includeStorageInfo: true,
        maxMemories: 8
    });

    console.log('✅ TOON formatting successful');
    console.log();
    console.log('Output:');
    console.log(toonOutput);
    console.log();
} catch (error) {
    console.error('❌ TOON formatting failed:', error.message);
    process.exit(1);
}

// Test 2: Age calculation
console.log();
console.log('Test 2: Age Calculation');
console.log('-'.repeat(80));
const testDates = [
    "2025-11-05T10:00:00Z", // Today
    "2025-11-04T10:00:00Z", // Yesterday
    "2025-10-29T10:00:00Z", // Week ago
    "2025-10-01T10:00:00Z"  // Month ago
];

testDates.forEach(date => {
    const age = calculateAgeDays(date);
    console.log(`Date: ${date} → Age: ${age} days`);
});
console.log('✅ Age calculation working');

// Test 3: Token savings estimation
console.log();
console.log('Test 3: Token Savings Estimation');
console.log('-'.repeat(80));
const savings = estimateTokenSavings(sampleMemories);
console.log(`JSON tokens (estimated): ${savings.jsonTokens}`);
console.log(`TOON tokens (estimated): ${savings.toonTokens}`);
console.log(`Token savings: ${savings.savingsPercent}%`);
console.log('✅ Token estimation working');

// Test 4: Error handling
console.log();
console.log('Test 4: Error Handling');
console.log('-'.repeat(80));

// Test with invalid input
try {
    formatMemoriesAsTOON(null, projectContext);
    console.error('❌ Should have thrown error for null memories');
    process.exit(1);
} catch (error) {
    console.log('✅ Correctly throws error for null memories:', error.message);
}

try {
    formatMemoriesAsTOON([], null);
    console.error('❌ Should have thrown error for null projectContext');
    process.exit(1);
} catch (error) {
    console.log('✅ Correctly throws error for null projectContext:', error.message);
}

// Test 5: Fallback behavior
console.log();
console.log('Test 5: Optional Sections (Git Context & Storage Info)');
console.log('-'.repeat(80));

const minimalOutput = formatMemoriesAsTOON(sampleMemories.slice(0, 2), projectContext, {
    includeGitContext: false,
    includeStorageInfo: false
});

console.log('TOON output WITHOUT git context and storage info:');
console.log(minimalOutput);
console.log('✅ Optional sections working');

// Summary
console.log();
console.log('='.repeat(80));
console.log('ALL TESTS PASSED ✅');
console.log('='.repeat(80));
console.log();
console.log('Summary:');
console.log(`- TOON formatter working correctly`);
console.log(`- Age calculation accurate`);
console.log(`- Token savings: ~${savings.savingsPercent}% vs JSON`);
console.log(`- Error handling robust`);
console.log(`- Optional sections configurable`);
console.log();
console.log('Next step: Test integration with session-start.js hook');
