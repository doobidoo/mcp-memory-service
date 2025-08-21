/**
 * Context Formatting Utility
 * Formats memories for injection into Claude Code sessions
 */

/**
 * Format a single memory for context display
 */
function formatMemory(memory, index = 0, options = {}) {
    try {
        const {
            includeScore = false,
            includeMetadata = false,
            maxContentLength = 200,
            includeDate = true
        } = options;
        
        // Truncate content if too long
        let content = memory.content || 'No content available';
        if (content.length > maxContentLength) {
            content = content.substring(0, maxContentLength) + '...';
        }
        
        // Format tags
        const tags = Array.isArray(memory.tags) ? memory.tags.join(', ') : 'No tags';
        
        // Format date
        let dateStr = '';
        if (includeDate && memory.created_at_iso) {
            const date = new Date(memory.created_at_iso);
            dateStr = ` (${date.toLocaleDateString()})`;
        }
        
        // Build formatted memory
        let formatted = `${index + 1}. ${content}${dateStr}`;
        
        // Add tags if they provide useful context
        if (memory.tags && memory.tags.length > 0) {
            const relevantTags = memory.tags.filter(tag => 
                !tag.startsWith('source:') && // Exclude machine source tags
                tag !== 'claude-code' &&      // Exclude generic tags
                tag !== 'auto-generated'
            );
            
            if (relevantTags.length > 0) {
                formatted += `\n   Tags: ${relevantTags.join(', ')}`;
            }
        }
        
        // Add relevance score if requested
        if (includeScore && memory.relevanceScore !== undefined) {
            formatted += `\n   Relevance: ${(memory.relevanceScore * 100).toFixed(1)}%`;
        }
        
        // Add metadata if requested  
        if (includeMetadata && memory.memory_type) {
            formatted += `\n   Type: ${memory.memory_type}`;
        }
        
        return formatted;
        
    } catch (error) {
        console.warn('[Context Formatter] Error formatting memory:', error.message);
        return `${index + 1}. [Error formatting memory: ${error.message}]`;
    }
}

/**
 * Group memories by category for better organization
 */
function groupMemoriesByCategory(memories) {
    try {
        const categories = {
            decisions: [],
            architecture: [],
            insights: [],
            bugs: [],
            features: [],
            other: []
        };
        
        memories.forEach(memory => {
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
        
        // Limit number of memories
        const limitedMemories = memories.slice(0, maxMemories);
        
        // Start building context message
        let contextMessage = '## 🧠 Memory Context Loaded\n\n';
        
        // Add project summary
        if (includeProjectSummary && projectContext) {
            contextMessage += createProjectSummary(projectContext) + '\n\n';
        }
        
        contextMessage += `**Loaded ${limitedMemories.length} relevant memories from your project history:**\n\n`;
        
        if (groupByCategory) {
            // Group and format by category
            const categories = groupMemoriesByCategory(limitedMemories);
            
            const categoryTitles = {
                decisions: '### 🎯 Key Decisions',
                architecture: '### 🏗️ Architecture & Design', 
                insights: '### 💡 Insights & Learnings',
                bugs: '### 🐛 Bug Fixes & Issues',
                features: '### ✨ Features & Implementation',
                other: '### 📝 Additional Context'
            };
            
            Object.entries(categories).forEach(([category, categoryMemories]) => {
                if (categoryMemories.length > 0) {
                    contextMessage += `${categoryTitles[category]}\n`;
                    
                    categoryMemories.forEach((memory, index) => {
                        const formatted = formatMemory(memory, index, {
                            includeScore,
                            maxContentLength: 150,
                            includeDate: includeTimestamp
                        });
                        contextMessage += `${formatted}\n\n`;
                    });
                }
            });
            
        } else {
            // Simple linear formatting
            limitedMemories.forEach((memory, index) => {
                const formatted = formatMemory(memory, index, {
                    includeScore,
                    maxContentLength: 200,
                    includeDate: includeTimestamp
                });
                contextMessage += `${formatted}\n\n`;
            });
        }
        
        // Add footer with usage instructions
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