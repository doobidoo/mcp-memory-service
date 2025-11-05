/**
 * Manual TOON Format Implementation (Spec-Compliant Tabular Format)
 *
 * This is a reference implementation demonstrating the official TOON specification's
 * tabular format for token-efficient data encoding.
 *
 * Performance: 431 tokens (vs 436 markdown, vs 602 library TOON)
 * Format: arrayName[N]{field1,field2}: with CSV-like rows
 *
 * Related PoC: docs/poc/toon-manual-tabular-results.md
 */

/**
 * Escape CSV value (quote if contains comma/newline/quote)
 * RFC 4180 compliant
 */
function escapeCSV(value) {
    if (!value) return '""';

    const needsQuotes = /[",\n\r]/.test(value);
    if (needsQuotes) {
        // Escape internal quotes by doubling them
        const escaped = value.replace(/"/g, '""');
        return `"${escaped}"`;
    }
    return `"${value}"`;
}

/**
 * Format date compactly (e.g., "Nov 1" or "Oct 28")
 * Saves 19 chars per date vs ISO format
 */
function formatCompactDate(isoDate) {
    if (!isoDate) return '';

    try {
        const date = new Date(isoDate);
        const month = date.toLocaleString('en', { month: 'short' });
        const day = date.getDate();
        return `${month} ${day}`;
    } catch {
        return '';
    }
}

/**
 * Truncate content to max length with ellipsis
 */
function truncateContent(content, maxLen) {
    if (!content || content.length <= maxLen) return content || '';
    return content.substring(0, maxLen - 3) + '...';
}

/**
 * Calculate age in days from ISO date string
 */
function calculateAgeDays(isoDate) {
    if (!isoDate || typeof isoDate !== 'string') {
        return 0;
    }

    try {
        const now = new Date();
        const memDate = new Date(isoDate);

        if (isNaN(memDate.getTime())) {
            return 0;
        }

        const daysDiff = (now - memDate) / (1000 * 60 * 60 * 24);
        return Math.floor(daysDiff);
    } catch (error) {
        return 0;
    }
}

/**
 * Format memories using MANUAL tabular TOON format (spec-compliant)
 *
 * Produces true TOON specification tabular format with CSV-like rows.
 * Field names declared once in header, then data rows follow.
 *
 * Example output:
 * ```toon
 * # Session Context - mcp-memory-service
 *
 * memories[8]{content,tags,date,score,type,age}:
 *   "Analytics fix...",["dashboard","fix"],Nov 1,0.95,implementation,4
 *   "Memory hook...",["hooks","git"],Oct 31,0.87,feature,5
 *
 * git[1]{commits,changelog,keywords}:
 *   20,3,"feat,release,docs"
 *
 * storage[1]{backend,memories,size_mb}:
 *   sqlite-vec,2224,8.78
 * ```
 *
 * @param {Array<Object>} memories - Array of memory objects
 * @param {Object} projectContext - Project information with name
 * @param {Object} [options={}] - Formatting options
 * @returns {string} Manual tabular TOON-formatted string
 */
function formatMemoriesAsManualTOON(memories, projectContext, options = {}) {
    // Validate inputs
    if (!Array.isArray(memories)) {
        throw new Error('Memories must be an array');
    }

    if (!projectContext || !projectContext.name) {
        throw new Error('Project context must include a name');
    }

    // Default options
    const {
        gitContext = null,
        storageInfo = null,
        includeGitContext = true,
        includeStorageInfo = true,
        maxMemories = undefined,
        maxContentLength = 250
    } = options;

    try {
        // Filter and limit memories
        let validMemories = memories.filter(m => m && (m.content || m.type || m.tags));

        if (validMemories.length === 0) {
            throw new Error('No valid memories to format');
        }

        if (maxMemories && validMemories.length > maxMemories) {
            validMemories = validMemories.slice(0, maxMemories);
        }

        // Build TOON header
        let toon = `# Session Context - ${projectContext.name}\n\n`;

        // Memory array in tabular format (TOON spec compliant)
        toon += `memories[${validMemories.length}]{content,tags,date,score,type,age}:\n`;

        validMemories.forEach(m => {
            // Escape content (CSV-style: quote if contains comma/newline)
            const content = escapeCSV(truncateContent(m.content || '', maxContentLength));

            // Format tags as JSON array (compact)
            const tags = JSON.stringify(m.tags || []);

            // Compact date (just "Nov 1" or "Oct 28")
            const date = formatCompactDate(m.created_at_iso);

            // Relevance score (2 decimals)
            const score = typeof m.relevanceScore === 'number' ? m.relevanceScore.toFixed(2) : '0.00';

            // Type
            const type = m.type || 'note';

            // Age in days
            const age = calculateAgeDays(m.created_at_iso);

            // Build CSV row (TOON tabular format)
            toon += `  ${content},${tags},${date},${score},${type},${age}\n`;
        });

        // Git context (if enabled and available)
        if (includeGitContext && gitContext) {
            toon += `\ngit[1]{commits,changelog,keywords}:\n`;
            const commits = Array.isArray(gitContext.commits) ? gitContext.commits.length : 0;
            const changelog = Array.isArray(gitContext.changelogEntries) ? gitContext.changelogEntries.length : 0;
            const keywords = gitContext.developmentKeywords?.keywords?.slice(0, 5).join(',') || '';
            toon += `  ${commits},${changelog},"${keywords}"\n`;
        }

        // Storage info (if enabled and available)
        if (includeStorageInfo && storageInfo) {
            toon += `\nstorage[1]{backend,memories,size_mb}:\n`;
            const backend = storageInfo.backend || 'unknown';
            const totalMem = storageInfo.health?.totalMemories || 0;
            const sizeMB = (storageInfo.health?.databaseSizeMB || 0).toFixed(2);
            toon += `  ${backend},${totalMem},${sizeMB}\n`;
        }

        return toon;

    } catch (error) {
        throw new Error(`Manual TOON formatting failed: ${error.message}`);
    }
}

module.exports = {
    formatMemoriesAsManualTOON,
    escapeCSV,
    formatCompactDate,
    truncateContent,
    calculateAgeDays
};
