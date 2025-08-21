/**
 * Memory Relevance Scoring Utility
 * Implements intelligent algorithms to score memories by relevance to current project context
 * Phase 2: Enhanced with conversation context awareness for dynamic memory loading
 */

/**
 * Calculate time decay factor for memory relevance
 * More recent memories get higher scores
 */
function calculateTimeDecay(memoryDate, decayRate = 0.1) {
    try {
        const now = new Date();
        const memoryTime = new Date(memoryDate);
        
        if (isNaN(memoryTime.getTime())) {
            return 0.5; // Default score for invalid dates
        }
        
        // Calculate days since memory creation
        const daysDiff = (now - memoryTime) / (1000 * 60 * 60 * 24);
        
        // Exponential decay: score = e^(-decayRate * days)
        // Recent memories (0-7 days): score 0.8-1.0
        // Older memories (8-30 days): score 0.3-0.8
        // Ancient memories (30+ days): score 0.0-0.3
        const decayScore = Math.exp(-decayRate * daysDiff);
        
        // Ensure score is between 0 and 1
        return Math.max(0.01, Math.min(1.0, decayScore));
        
    } catch (error) {
        console.warn('[Memory Scorer] Time decay calculation error:', error.message);
        return 0.5;
    }
}

/**
 * Calculate tag relevance score
 * Memories with tags matching project context get higher scores
 */
function calculateTagRelevance(memoryTags = [], projectContext) {
    try {
        if (!Array.isArray(memoryTags) || memoryTags.length === 0) {
            return 0.3; // Default score for memories without tags
        }
        
        const contextTags = [
            projectContext.name?.toLowerCase(),
            projectContext.language?.toLowerCase(),
            ...(projectContext.frameworks || []).map(f => f.toLowerCase()),
            ...(projectContext.tools || []).map(t => t.toLowerCase())
        ].filter(Boolean);
        
        if (contextTags.length === 0) {
            return 0.5; // No context to match against
        }
        
        // Calculate tag overlap
        const memoryTagsLower = memoryTags.map(tag => tag.toLowerCase());
        const matchingTags = contextTags.filter(contextTag => 
            memoryTagsLower.some(memoryTag => 
                memoryTag.includes(contextTag) || contextTag.includes(memoryTag)
            )
        );
        
        // Score based on percentage of matching tags
        const overlapScore = matchingTags.length / contextTags.length;
        
        // Bonus for exact project name matches
        const exactProjectMatch = memoryTagsLower.includes(projectContext.name?.toLowerCase());
        const projectBonus = exactProjectMatch ? 0.3 : 0;
        
        // Bonus for exact language matches  
        const exactLanguageMatch = memoryTagsLower.includes(projectContext.language?.toLowerCase());
        const languageBonus = exactLanguageMatch ? 0.2 : 0;
        
        // Bonus for framework matches
        const frameworkMatches = (projectContext.frameworks || []).filter(framework =>
            memoryTagsLower.some(tag => tag.includes(framework.toLowerCase()))
        );
        const frameworkBonus = frameworkMatches.length * 0.1;
        
        const totalScore = Math.min(1.0, overlapScore + projectBonus + languageBonus + frameworkBonus);
        
        return Math.max(0.1, totalScore);
        
    } catch (error) {
        console.warn('[Memory Scorer] Tag relevance calculation error:', error.message);
        return 0.3;
    }
}

/**
 * Calculate content relevance using simple text analysis
 * Memories with content matching project keywords get higher scores
 */
function calculateContentRelevance(memoryContent = '', projectContext) {
    try {
        if (!memoryContent || typeof memoryContent !== 'string') {
            return 0.3;
        }
        
        const content = memoryContent.toLowerCase();
        const keywords = [
            projectContext.name?.toLowerCase(),
            projectContext.language?.toLowerCase(),
            ...(projectContext.frameworks || []).map(f => f.toLowerCase()),
            ...(projectContext.tools || []).map(t => t.toLowerCase()),
            // Add common technical keywords
            'architecture', 'decision', 'implementation', 'bug', 'fix', 
            'feature', 'config', 'setup', 'deployment', 'performance'
        ].filter(Boolean);
        
        if (keywords.length === 0) {
            return 0.5;
        }
        
        // Count keyword occurrences
        let totalMatches = 0;
        let keywordScore = 0;
        
        keywords.forEach(keyword => {
            const occurrences = (content.match(new RegExp(keyword, 'g')) || []).length;
            if (occurrences > 0) {
                totalMatches++;
                keywordScore += Math.log(1 + occurrences) * 0.1; // Logarithmic scoring
            }
        });
        
        // Normalize score
        const matchRatio = totalMatches / keywords.length;
        const contentScore = Math.min(1.0, matchRatio + keywordScore);
        
        return Math.max(0.1, contentScore);
        
    } catch (error) {
        console.warn('[Memory Scorer] Content relevance calculation error:', error.message);
        return 0.3;
    }
}

/**
 * Calculate memory type bonus
 * Certain memory types are more valuable for context injection
 */
function calculateTypeBonus(memoryType) {
    const typeScores = {
        'decision': 0.3,        // Architectural decisions are highly valuable
        'architecture': 0.3,     // Architecture documentation is important
        'reference': 0.2,        // Reference materials are useful
        'session': 0.15,         // Session summaries provide good context
        'insight': 0.2,          // Insights are valuable for learning
        'bug-fix': 0.15,         // Bug fixes provide historical context
        'feature': 0.1,          // Feature descriptions are moderately useful
        'note': 0.05,            // General notes are less critical
        'todo': 0.05,            // TODOs are task-specific
        'temporary': -0.1        // Temporary notes should be deprioritized
    };
    
    return typeScores[memoryType?.toLowerCase()] || 0;
}

/**
 * Calculate conversation context relevance score (Phase 2)
 * Matches memory content with current conversation topics and intent
 */
function calculateConversationRelevance(memory, conversationAnalysis) {
    try {
        if (!conversationAnalysis || !memory.content) {
            return 0.3; // Default score when no conversation context
        }

        const memoryContent = memory.content.toLowerCase();
        let relevanceScore = 0;
        let factorCount = 0;

        // Score based on topic matching
        if (conversationAnalysis.topics && conversationAnalysis.topics.length > 0) {
            conversationAnalysis.topics.forEach(topic => {
                const topicMatches = (memoryContent.match(new RegExp(topic.name, 'gi')) || []).length;
                if (topicMatches > 0) {
                    relevanceScore += topic.confidence * Math.min(topicMatches * 0.2, 0.8);
                    factorCount++;
                }
            });
        }

        // Score based on entity matching
        if (conversationAnalysis.entities && conversationAnalysis.entities.length > 0) {
            conversationAnalysis.entities.forEach(entity => {
                const entityMatches = (memoryContent.match(new RegExp(entity.name, 'gi')) || []).length;
                if (entityMatches > 0) {
                    relevanceScore += entity.confidence * 0.3;
                    factorCount++;
                }
            });
        }

        // Score based on intent alignment
        if (conversationAnalysis.intent) {
            const intentKeywords = {
                'learning': ['learn', 'understand', 'explain', 'how', 'tutorial', 'guide'],
                'problem-solving': ['fix', 'error', 'debug', 'issue', 'problem', 'solve'],
                'development': ['build', 'create', 'implement', 'develop', 'code', 'feature'],
                'optimization': ['optimize', 'improve', 'performance', 'faster', 'better'],
                'review': ['review', 'check', 'analyze', 'audit', 'validate'],
                'planning': ['plan', 'design', 'architecture', 'approach', 'strategy']
            };

            const intentWords = intentKeywords[conversationAnalysis.intent.name] || [];
            let intentMatches = 0;
            intentWords.forEach(word => {
                if (memoryContent.includes(word)) {
                    intentMatches++;
                }
            });

            if (intentMatches > 0) {
                relevanceScore += conversationAnalysis.intent.confidence * (intentMatches / intentWords.length);
                factorCount++;
            }
        }

        // Score based on code context if present
        if (conversationAnalysis.codeContext && conversationAnalysis.codeContext.isCodeRelated) {
            const codeIndicators = ['code', 'function', 'class', 'method', 'variable', 'api', 'library'];
            let codeMatches = 0;
            codeIndicators.forEach(indicator => {
                if (memoryContent.includes(indicator)) {
                    codeMatches++;
                }
            });

            if (codeMatches > 0) {
                relevanceScore += 0.4 * (codeMatches / codeIndicators.length);
                factorCount++;
            }
        }

        // Normalize score
        const normalizedScore = factorCount > 0 ? relevanceScore / factorCount : 0.3;
        return Math.max(0.1, Math.min(1.0, normalizedScore));

    } catch (error) {
        console.warn('[Memory Scorer] Conversation relevance calculation error:', error.message);
        return 0.3;
    }
}

/**
 * Calculate final relevance score for a memory (Enhanced for Phase 2)
 */
function calculateRelevanceScore(memory, projectContext, options = {}) {
    try {
        const {
            weights = {},
            includeConversationContext = false,
            conversationAnalysis = null
        } = options;

        // Default weights for different scoring factors
        const defaultWeights = includeConversationContext ? {
            timeDecay: 0.25,           // Reduced weight for time when conversation context available
            tagRelevance: 0.35,        // Tag matching remains important
            contentRelevance: 0.15,    // Content matching reduced
            conversationRelevance: 0.25, // New conversation context factor
            typeBonus: 0.1             // Memory type provides minor adjustment
        } : {
            timeDecay: 0.3,            // Original Phase 1 weights
            tagRelevance: 0.4,
            contentRelevance: 0.2,
            typeBonus: 0.1
        };
        
        const w = { ...defaultWeights, ...weights };
        
        // Calculate individual scores
        const timeScore = calculateTimeDecay(memory.created_at || memory.created_at_iso);
        const tagScore = calculateTagRelevance(memory.tags, projectContext);
        const contentScore = calculateContentRelevance(memory.content, projectContext);
        const typeBonus = calculateTypeBonus(memory.memory_type);
        
        let finalScore = (
            (timeScore * w.timeDecay) +
            (tagScore * w.tagRelevance) +
            (contentScore * w.contentRelevance) +
            typeBonus // Type bonus is not weighted, acts as adjustment
        );

        const breakdown = {
            timeDecay: timeScore,
            tagRelevance: tagScore,
            contentRelevance: contentScore,
            typeBonus: typeBonus
        };

        // Add conversation context scoring if enabled (Phase 2)
        if (includeConversationContext && conversationAnalysis) {
            const conversationScore = calculateConversationRelevance(memory, conversationAnalysis);
            finalScore += (conversationScore * (w.conversationRelevance || 0));
            breakdown.conversationRelevance = conversationScore;
        }
        
        // Ensure score is between 0 and 1
        const normalizedScore = Math.max(0, Math.min(1, finalScore));
        
        return {
            finalScore: normalizedScore,
            breakdown: breakdown,
            weights: w,
            hasConversationContext: includeConversationContext
        };
        
    } catch (error) {
        console.warn('[Memory Scorer] Score calculation error:', error.message);
        return {
            finalScore: 0.1,
            breakdown: { error: error.message },
            weights: {},
            hasConversationContext: false
        };
    }
}

/**
 * Score and sort memories by relevance
 */
function scoreMemoryRelevance(memories, projectContext, options = {}) {
    try {
        if (!Array.isArray(memories)) {
            console.warn('[Memory Scorer] Invalid memories array');
            return [];
        }
        
        console.log(`[Memory Scorer] Scoring ${memories.length} memories for project: ${projectContext.name}`);
        
        // Score each memory
        const scoredMemories = memories.map(memory => {
            const scoreResult = calculateRelevanceScore(memory, projectContext, options);
            
            return {
                ...memory,
                relevanceScore: scoreResult.finalScore,
                scoreBreakdown: scoreResult.breakdown,
                hasConversationContext: scoreResult.hasConversationContext
            };
        });
        
        // Sort by relevance score (highest first)
        const sortedMemories = scoredMemories.sort((a, b) => b.relevanceScore - a.relevanceScore);
        
        // Log scoring results for debugging
        console.log('[Memory Scorer] Top scored memories:');
        sortedMemories.slice(0, 3).forEach((memory, index) => {
            console.log(`  ${index + 1}. Score: ${memory.relevanceScore.toFixed(3)} - ${memory.content.substring(0, 60)}...`);
        });
        
        return sortedMemories;
        
    } catch (error) {
        console.error('[Memory Scorer] Error scoring memories:', error.message);
        return memories || [];
    }
}

/**
 * Filter memories by minimum relevance threshold
 */
function filterByRelevance(memories, minScore = 0.3) {
    try {
        const filtered = memories.filter(memory => memory.relevanceScore >= minScore);
        console.log(`[Memory Scorer] Filtered ${filtered.length}/${memories.length} memories above threshold ${minScore}`);
        return filtered;
        
    } catch (error) {
        console.warn('[Memory Scorer] Error filtering memories:', error.message);
        return memories;
    }
}

module.exports = {
    scoreMemoryRelevance,
    calculateRelevanceScore,
    calculateTimeDecay,
    calculateTagRelevance,
    calculateContentRelevance,
    calculateTypeBonus,
    filterByRelevance
};

// Direct execution support for testing
if (require.main === module) {
    // Test with mock data
    const mockProjectContext = {
        name: 'mcp-memory-service',
        language: 'JavaScript',
        frameworks: ['Node.js'],
        tools: ['npm']
    };
    
    const mockMemories = [
        {
            content: 'Decided to use SQLite-vec for better performance in MCP Memory Service',
            tags: ['mcp-memory-service', 'decision', 'sqlite-vec'],
            memory_type: 'decision',
            created_at: '2025-08-19T10:00:00Z'
        },
        {
            content: 'Fixed bug in JavaScript hook implementation for Claude Code integration',
            tags: ['javascript', 'bug-fix', 'claude-code'],
            memory_type: 'bug-fix', 
            created_at: '2025-08-18T15:30:00Z'
        },
        {
            content: 'Random note about completely unrelated project',
            tags: ['other-project', 'note'],
            memory_type: 'note',
            created_at: '2025-08-01T08:00:00Z'
        }
    ];
    
    console.log('\n=== MEMORY SCORING TEST ===');
    const scored = scoreMemoryRelevance(mockMemories, mockProjectContext);
    console.log('\n=== SCORED RESULTS ===');
    scored.forEach((memory, index) => {
        console.log(`${index + 1}. Score: ${memory.relevanceScore.toFixed(3)}`);
        console.log(`   Content: ${memory.content.substring(0, 80)}...`);
        console.log(`   Breakdown:`, memory.scoreBreakdown);
        console.log('');
    });
}