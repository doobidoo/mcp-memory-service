#!/usr/bin/env node
/**
 * Compare Token Efficiency: TOON vs Markdown
 *
 * Measures actual token count for both formats using the same sample data.
 */

const { formatMemoriesAsTOON } = require('/Users/hkr/.claude/hooks/utilities/toon-formatter.js');
const { formatMemoriesForContext } = require('/Users/hkr/.claude/hooks/utilities/context-formatter.js');

// Sample memories from actual session
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

const projectContext = {
    name: "mcp-memory-service",
    language: "Python",
    frameworks: ["FastAPI", "SQLite", "MCP"],
    git: {
        branch: "main",
        lastCommit: "1f8b4b8"
    }
};

const gitContext = {
    commits: Array(20).fill({}),
    changelogEntries: Array(3).fill({}),
    developmentKeywords: {
        keywords: ["feat", "release", "docs", "v8.17.1", "chore"]
    }
};

const storageInfo = {
    backend: "sqlite-vec",
    health: {
        totalMemories: 2224,
        databaseSizeMB: 8.78
    }
};

// Simple token counter (approximation: ~4 chars per token)
function countTokens(text) {
    return Math.ceil(text.length / 4);
}

console.log('='.repeat(80));
console.log('FORMAT COMPARISON: TOON vs MARKDOWN');
console.log('='.repeat(80));
console.log();

// Generate TOON format
console.log('Generating TOON format...');
const toonOutput = formatMemoriesAsTOON(sampleMemories, projectContext, {
    gitContext: gitContext,
    storageInfo: storageInfo,
    includeGitContext: true,
    includeStorageInfo: true
});

// Generate Markdown format
console.log('Generating Markdown format...');
const markdownOutput = formatMemoriesForContext(sampleMemories, projectContext, {
    includeScore: false,
    groupByCategory: true,
    maxMemories: 8,
    includeTimestamp: true,
    maxContentLength: 500,
    storageInfo: storageInfo,
    adaptiveTruncation: true
});

// Calculate metrics
const toonTokens = countTokens(toonOutput);
const markdownTokens = countTokens(markdownOutput);
const savings = markdownTokens - toonTokens;
const savingsPercent = ((savings / markdownTokens) * 100).toFixed(1);

console.log();
console.log('='.repeat(80));
console.log('RESULTS');
console.log('='.repeat(80));
console.log();
console.log('TOON Format:');
console.log(`  Characters: ${toonOutput.length}`);
console.log(`  Tokens (estimated): ${toonTokens}`);
console.log();
console.log('Markdown Format:');
console.log(`  Characters: ${markdownOutput.length}`);
console.log(`  Tokens (estimated): ${markdownTokens}`);
console.log();
console.log('Savings:');
console.log(`  Token reduction: ${savings} tokens (${savingsPercent}%)`);
console.log();

// Show format preference
if (savingsPercent >= 30) {
    console.log(`✅ TOON format provides significant token savings (${savingsPercent}%)`);
    console.log('   Recommendation: Merge to main branch');
} else if (savingsPercent >= 15) {
    console.log(`⚠️  TOON format provides moderate token savings (${savingsPercent}%)`);
    console.log('   Recommendation: Consider use case before merging');
} else {
    console.log(`❌ TOON format does not provide sufficient savings (${savingsPercent}%)`);
    console.log('   Recommendation: Keep markdown format');
}

console.log();
console.log('='.repeat(80));
console.log();

// Optional: Show sample outputs
if (process.argv.includes('--show-outputs')) {
    console.log('TOON OUTPUT:');
    console.log('-'.repeat(80));
    console.log(toonOutput);
    console.log();
    console.log('MARKDOWN OUTPUT:');
    console.log('-'.repeat(80));
    console.log(markdownOutput);
    console.log();
}
