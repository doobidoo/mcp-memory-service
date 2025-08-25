/**
 * Context Formatting Utility
 * Formats memories for injection into Claude Code sessions
 */

/**
 * Extract meaningful content from session summaries and structured memories
 */
function extractMeaningfulContent(content, maxLength = 300) {
    if (!content || typeof content !== 'string') {
        return 'No content available';
    }
    
    // Check if this is a session summary with structured sections
    if (content.includes('# Session Summary') || content.includes('## 🎯') || content.includes('## 🏛️') || content.includes('## 💡')) {
        const sections = {
            decisions: [],
            insights: [],
            codeChanges: [],
            nextSteps: [],
            topics: []
        };
        
        // Extract structured sections
        const lines = content.split('\n');
        let currentSection = null;
        
        for (const line of lines) {
            const trimmed = line.trim();
            
            if (trimmed.includes('🏛️') && trimmed.includes('Decision')) {
                currentSection = 'decisions';
                continue;
            } else if (trimmed.includes('💡') && (trimmed.includes('Insight') || trimmed.includes('Key'))) {
                currentSection = 'insights';
                continue;
            } else if (trimmed.includes('💻') && trimmed.includes('Code')) {
                currentSection = 'codeChanges';
                continue;
            } else if (trimmed.includes('📋') && trimmed.includes('Next')) {
                currentSection = 'nextSteps';
                continue;
            } else if (trimmed.includes('🎯') && trimmed.includes('Topic')) {
                currentSection = 'topics';
                continue;
            } else if (trimmed.startsWith('##') || trimmed.startsWith('#')) {
                currentSection = null; // Reset on new major section
                continue;
            }
            
            // Collect bullet points under current section
            if (currentSection && trimmed.startsWith('- ') && trimmed.length > 2) {
                const item = trimmed.substring(2).trim();
                if (item.length > 5 && item !== 'implementation' && item !== '...') {
                    sections[currentSection].push(item);
                }
            }
        }
        
        // Build meaningful summary from extracted sections
        const meaningfulParts = [];
        
        if (sections.decisions.length > 0) {
            meaningfulParts.push(`Decisions: ${sections.decisions.slice(0, 2).join('; ')}`);
        }
        if (sections.insights.length > 0) {
            meaningfulParts.push(`Insights: ${sections.insights.slice(0, 2).join('; ')}`);
        }
        if (sections.codeChanges.length > 0) {
            meaningfulParts.push(`Changes: ${sections.codeChanges.slice(0, 2).join('; ')}`);
        }
        if (sections.nextSteps.length > 0) {
            meaningfulParts.push(`Next: ${sections.nextSteps.slice(0, 2).join('; ')}`);
        }
        
        if (meaningfulParts.length > 0) {
            const extracted = meaningfulParts.join(' | ');
            return extracted.length > maxLength ? extracted.substring(0, maxLength - 3) + '...' : extracted;
        }
    }
    
    // For non-structured content, use smart truncation
    if (content.length <= maxLength) {
        return content;
    }
    
    // Try to find a good breaking point (sentence, paragraph, or code block)
    const breakPoints = ['. ', '\n\n', '\n', '; '];
    
    for (const breakPoint of breakPoints) {
        const lastBreak = content.lastIndexOf(breakPoint, maxLength - 3);
        if (lastBreak > maxLength * 0.7) { // Only use if we keep at least 70% of desired length
            return content.substring(0, lastBreak + (breakPoint === '. ' ? 1 : 0)) + '...';
        }
    }
    
    // Fallback to hard truncation
    return content.substring(0, maxLength - 3) + '...';
}

/**
 * Check if memory content appears to be a generic/empty session summary
 */
function isGenericSessionSummary(content) {
    if (!content || typeof content !== 'string') {
        return true;
    }
    
    // Check for generic patterns
    const genericPatterns = [
        /## 🎯 Topics Discussed\s*-\s*implementation\s*-\s*\.\.\.?$/m,
        /Topics Discussed.*implementation.*\.\.\..*$/s,
        /Session Summary.*implementation.*\.\.\..*$/s
    ];
    
    return genericPatterns.some(pattern => pattern.test(content));
}

/**
 * Format a single memory for context display
 */
function formatMemory(memory, index = 0, options = {}) {
    try {
        const {
            includeScore = false,
            includeMetadata = false,
            maxContentLength = 300,
            includeDate = true,
            showOnlyRelevantTags = true
        } = options;
        
        // Extract meaningful content using smart parsing
        const content = extractMeaningfulContent(memory.content || 'No content available', maxContentLength);
        
        // Skip generic/empty session summaries
        if (isGenericSessionSummary(memory.content) && !includeScore) {
            return null; // Signal to skip this memory
        }
        
        // Format date more concisely
        let dateStr = '';
        if (includeDate && memory.created_at_iso) {
            const date = new Date(memory.created_at_iso);
            dateStr = ` (${date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })})`;
        }
        
        // Build formatted memory
        let formatted = `${index + 1}. ${content}${dateStr}`;
        
        // Add only the most relevant tags
        if (showOnlyRelevantTags && memory.tags && memory.tags.length > 0) {
            const relevantTags = memory.tags.filter(tag => {
                const tagLower = tag.toLowerCase();
                return !tagLower.startsWith('source:') && 
                       !tagLower.startsWith('claude-code-session') &&
                       !tagLower.startsWith('session-consolidation') &&
                       tagLower !== 'claude-code' &&
                       tagLower !== 'auto-generated' &&
                       tagLower !== 'implementation' &&
                       tagLower.length > 2;
            });
            
            // Only show tags if they add meaningful context (max 3)
            if (relevantTags.length > 0 && relevantTags.length <= 5) {
                formatted += `\n   Tags: ${relevantTags.slice(0, 3).join(', ')}`;
            }
        }
        
        return formatted;
        
    } catch (error) {
        console.warn('[Context Formatter] Error formatting memory:', error.message);
        return `${index + 1}. [Error formatting memory: ${error.message}]`;
    }
}

/**
 * Deduplicate memories based on content similarity
 */
function deduplicateMemories(memories) {
    if (!Array.isArray(memories) || memories.length <= 1) {
        return memories;
    }
    
    const deduplicated = [];
    const seenContent = new Set();
    
    // Sort by relevance score (highest first) and recency
    const sorted = memories.sort((a, b) => {
        const scoreA = a.relevanceScore || 0;
        const scoreB = b.relevanceScore || 0;
        if (scoreA !== scoreB) return scoreB - scoreA;
        
        // If scores are equal, prefer more recent
        const dateA = new Date(a.created_at_iso || 0);
        const dateB = new Date(b.created_at_iso || 0);
        return dateB - dateA;
    });
    
    for (const memory of sorted) {
        const content = memory.content || '';
        
        // Create a normalized version for comparison
        let normalized = content.toLowerCase()
            .replace(/# session summary.*?\n/gi, '') // Remove session headers
            .replace(/\*\*date\*\*:.*?\n/gi, '')    // Remove date lines
            .replace(/\*\*project\*\*:.*?\n/gi, '') // Remove project lines
            .replace(/\s+/g, ' ')                   // Normalize whitespace
            .trim();
        
        // Skip if content is too generic or already seen
        if (normalized.length < 20 || isGenericSessionSummary(content)) {
            continue;
        }
        
        // Check for substantial similarity
        let isDuplicate = false;
        for (const seenNormalized of seenContent) {
            const similarity = calculateContentSimilarity(normalized, seenNormalized);
            if (similarity > 0.8) { // 80% similarity threshold
                isDuplicate = true;
                break;
            }
        }
        
        if (!isDuplicate) {
            seenContent.add(normalized);
            deduplicated.push(memory);
        }
    }
    
    console.log(`[Context Formatter] Deduplicated ${memories.length} → ${deduplicated.length} memories`);
    return deduplicated;
}

/**
 * Calculate content similarity between two normalized strings
 */
function calculateContentSimilarity(str1, str2) {
    if (!str1 || !str2) return 0;
    if (str1 === str2) return 1;
    
    // Use simple word overlap similarity
    const words1 = new Set(str1.split(/\s+/).filter(w => w.length > 3));
    const words2 = new Set(str2.split(/\s+/).filter(w => w.length > 3));
    
    if (words1.size === 0 && words2.size === 0) return 1;
    if (words1.size === 0 || words2.size === 0) return 0;
    
    const intersection = new Set([...words1].filter(w => words2.has(w)));
    const union = new Set([...words1, ...words2]);
    
    return intersection.size / union.size;
}

/**
 * Group memories by category for better organization
 */
function groupMemoriesByCategory(memories) {
    try {
        // First deduplicate to remove redundant content
        const deduplicated = deduplicateMemories(memories);
        
        const categories = {
            decisions: [],
            architecture: [],
            insights: [],
            bugs: [],
            features: [],
            other: []
        };
        
        deduplicated.forEach(memory => {
            const type = memory.memory_type?.toLowerCase() || 'other';
            const tags = memory.tags || [];
            
            // Categorize based on type and tags
            if (type === 'decision' || tags.some(tag => tag.includes('decision'))) {
                categories.decisions.push(memory);
            } else if (type === 'architecture' || tags.some(tag => tag.includes('architecture'))) {
                categories.architecture.push(memory);
            } else if (type === 'insight' || tags.some(tag => tag.includes('insight'))) {
                categories.insights.push(memory);
            } else if (type === 'bug-fix' || tags.some(tag => tag.includes('bug'))) {
                categories.bugs.push(memory);
            } else if (type === 'feature' || tags.some(tag => tag.includes('feature'))) {
                categories.features.push(memory);
            } else {
                categories.other.push(memory);
            }
        });
        
        return categories;
        
    } catch (error) {
        console.warn('[Context Formatter] Error grouping memories:', error.message);
        return { other: memories };
    }
}

/**
 * Create a context summary from project information
 */
function createProjectSummary(projectContext) {
    try {
        let summary = `**Project**: ${projectContext.name}`;
        
        if (projectContext.language && projectContext.language !== 'Unknown') {
            summary += ` (${projectContext.language})`;
        }
        
        if (projectContext.frameworks && projectContext.frameworks.length > 0) {
            summary += `\n**Frameworks**: ${projectContext.frameworks.join(', ')}`;
        }
        
        if (projectContext.tools && projectContext.tools.length > 0) {
            summary += `\n**Tools**: ${projectContext.tools.join(', ')}`;
        }
        
        if (projectContext.git && projectContext.git.isRepo) {
            summary += `\n**Branch**: ${projectContext.git.branch || 'unknown'}`;
            
            if (projectContext.git.lastCommit) {
                summary += `\n**Last Commit**: ${projectContext.git.lastCommit}`;
            }
        }
        
        return summary;
        
    } catch (error) {
        console.warn('[Context Formatter] Error creating project summary:', error.message);
        return `**Project**: ${projectContext.name || 'Unknown Project'}`;
    }
}

/**
 * Format memories for Claude Code context injection
 */
function formatMemoriesForContext(memories, projectContext, options = {}) {
    try {
        const {
            includeProjectSummary = true,
            includeScore = false,
            groupByCategory = true,
            maxMemories = 8,
            includeTimestamp = true
        } = options;
        
        if (!memories || memories.length === 0) {
            return `## 📋 Memory Context\n\nNo relevant memories found for this session.\n`;
        }
        
        // Filter out null/generic memories and limit number
        const validMemories = [];
        let memoryIndex = 0;
        
        for (const memory of memories) {
            if (validMemories.length >= maxMemories) break;
            
            const formatted = formatMemory(memory, memoryIndex, {
                includeScore,
                maxContentLength: 300,
                includeDate: includeTimestamp,
                showOnlyRelevantTags: true
            });
            
            if (formatted) { // formatMemory returns null for generic summaries
                validMemories.push({ memory, formatted });
                memoryIndex++;
            }
        }
        
        if (validMemories.length === 0) {
            return `## 📋 Memory Context\n\nNo meaningful memories found for this session (filtered out generic content).\n`;
        }
        
        // Start building context message
        let contextMessage = '## 🧠 Memory Context Loaded\n\n';
        
        // Add project summary
        if (includeProjectSummary && projectContext) {
            contextMessage += createProjectSummary(projectContext) + '\n\n';
        }
        
        contextMessage += `**Loaded ${validMemories.length} relevant memories from your project history:**\n\n`;
        
        if (groupByCategory && validMemories.length > 3) {
            // Group and format by category only if we have enough content
            const categories = groupMemoriesByCategory(validMemories.map(v => v.memory));
            
            const categoryTitles = {
                decisions: '### 🎯 Key Decisions',
                architecture: '### 🏗️ Architecture & Design', 
                insights: '### 💡 Insights & Learnings',
                bugs: '### 🐛 Bug Fixes & Issues',
                features: '### ✨ Features & Implementation',
                other: '### 📝 Additional Context'
            };
            
            let hasContent = false;
            Object.entries(categories).forEach(([category, categoryMemories]) => {
                if (categoryMemories.length > 0) {
                    contextMessage += `${categoryTitles[category]}\n`;
                    hasContent = true;
                    
                    categoryMemories.forEach((memory, index) => {
                        const formatted = formatMemory(memory, index, {
                            includeScore,
                            maxContentLength: 300,
                            includeDate: includeTimestamp,
                            showOnlyRelevantTags: true
                        });
                        if (formatted) {
                            contextMessage += `${formatted}\n\n`;
                        }
                    });
                }
            });
            
            if (!hasContent) {
                // Fallback to linear format
                validMemories.forEach(({ formatted }) => {
                    contextMessage += `${formatted}\n\n`;
                });
            }
            
        } else {
            // Simple linear formatting for small lists
            validMemories.forEach(({ formatted }) => {
                contextMessage += `${formatted}\n\n`;
            });
        }
        
        // Add concise footer
        contextMessage += '---\n';
        contextMessage += '*This context was automatically loaded based on your project and recent activities. ';
        contextMessage += 'Use this information to maintain continuity with your previous work and decisions.*';
        
        return contextMessage;
        
    } catch (error) {
        console.error('[Context Formatter] Error formatting memories for context:', error.message);
        return `## 📋 Memory Context\n\n*Error loading context: ${error.message}*\n`;
    }
}

/**
 * Format memory for session-end consolidation
 */
function formatSessionConsolidation(sessionData, projectContext) {
    try {
        const timestamp = new Date().toISOString();
        
        let consolidation = `# Session Summary - ${projectContext.name}\n`;
        consolidation += `**Date**: ${new Date().toLocaleDateString()}\n`;
        consolidation += `**Project**: ${projectContext.name} (${projectContext.language})\n\n`;
        
        if (sessionData.topics && sessionData.topics.length > 0) {
            consolidation += `## 🎯 Topics Discussed\n`;
            sessionData.topics.forEach(topic => {
                consolidation += `- ${topic}\n`;
            });
            consolidation += '\n';
        }
        
        if (sessionData.decisions && sessionData.decisions.length > 0) {
            consolidation += `## 🏛️ Decisions Made\n`;
            sessionData.decisions.forEach(decision => {
                consolidation += `- ${decision}\n`;
            });
            consolidation += '\n';
        }
        
        if (sessionData.insights && sessionData.insights.length > 0) {
            consolidation += `## 💡 Key Insights\n`;
            sessionData.insights.forEach(insight => {
                consolidation += `- ${insight}\n`;
            });
            consolidation += '\n';
        }
        
        if (sessionData.codeChanges && sessionData.codeChanges.length > 0) {
            consolidation += `## 💻 Code Changes\n`;
            sessionData.codeChanges.forEach(change => {
                consolidation += `- ${change}\n`;
            });
            consolidation += '\n';
        }
        
        if (sessionData.nextSteps && sessionData.nextSteps.length > 0) {
            consolidation += `## 📋 Next Steps\n`;
            sessionData.nextSteps.forEach(step => {
                consolidation += `- ${step}\n`;
            });
            consolidation += '\n';
        }
        
        consolidation += `---\n*Session captured by Claude Code Memory Awareness at ${timestamp}*`;
        
        return consolidation;
        
    } catch (error) {
        console.error('[Context Formatter] Error formatting session consolidation:', error.message);
        return `Session Summary Error: ${error.message}`;
    }
}

module.exports = {
    formatMemoriesForContext,
    formatMemory,
    groupMemoriesByCategory,
    createProjectSummary,
    formatSessionConsolidation
};

// Direct execution support for testing
if (require.main === module) {
    // Test with mock data
    const mockMemories = [
        {
            content: 'Decided to use SQLite-vec for better performance, 10x faster than ChromaDB',
            tags: ['mcp-memory-service', 'decision', 'sqlite-vec', 'performance'],
            memory_type: 'decision',
            created_at_iso: '2025-08-19T10:00:00Z',
            relevanceScore: 0.95
        },
        {
            content: 'Implemented Claude Code hooks system for automatic memory awareness. Created session-start, session-end, and topic-change hooks.',
            tags: ['claude-code', 'hooks', 'architecture', 'memory-awareness'],
            memory_type: 'architecture',
            created_at_iso: '2025-08-19T09:30:00Z',
            relevanceScore: 0.87
        },
        {
            content: 'Fixed critical bug in project detector - was not handling pyproject.toml files correctly',
            tags: ['bug-fix', 'project-detector', 'python'],
            memory_type: 'bug-fix',
            created_at_iso: '2025-08-18T15:30:00Z',
            relevanceScore: 0.72
        }
    ];
    
    const mockProjectContext = {
        name: 'mcp-memory-service',
        language: 'JavaScript',
        frameworks: ['Node.js'],
        tools: ['npm'],
        git: {
            isRepo: true,
            branch: 'main',
            lastCommit: 'abc1234 Implement memory awareness hooks'
        }
    };
    
    console.log('\n=== CONTEXT FORMATTING TEST ===');
    const formatted = formatMemoriesForContext(mockMemories, mockProjectContext, {
        includeScore: true,
        groupByCategory: true
    });
    
    console.log(formatted);
    console.log('\n=== END TEST ===');
}