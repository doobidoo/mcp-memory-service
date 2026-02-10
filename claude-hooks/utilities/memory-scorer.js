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

        // Handle both Unix timestamps (seconds) and ISO strings
        let memoryTime;
        if (typeof memoryDate === 'string') {
            // ISO string format
            memoryTime = new Date(memoryDate);
        } else if (typeof memoryDate === 'number') {
            // Unix timestamp in seconds, convert to milliseconds
            memoryTime = new Date(memoryDate * 1000);
        } else {
            return 0.5; // Invalid format
        }

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
        // Silently fail with default score to avoid noise
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
        
        // Calculate tag overlap (exact match only to prevent cross-project pollution)
        const memoryTagsLower = memoryTags.map(tag => tag.toLowerCase());
        const matchingTags = contextTags.filter(contextTag =>
            memoryTagsLower.includes(contextTag)
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
        // Silently fail with default score to avoid noise
        return 0.3;
    }
}

/**
 * Calculate content quality score to penalize generic/empty content
 */
function calculateContentQuality(memoryContent = '') {
    try {
        if (!memoryContent || typeof memoryContent !== 'string') {
            return 0.1;
        }
        
        const content = memoryContent.trim();
        
        // Check for generic session summary patterns
        const genericPatterns = [
            /## 🎯 Topics Discussed\s*-\s*implementation\s*-\s*\.\.\.?$/m,
            /Topics Discussed.*implementation.*\.\.\..*$/s,
            /Session Summary.*implementation.*\.\.\..*$/s,
            /^# Session Summary.*Date.*Project.*Topics Discussed.*implementation.*\.\.\..*$/s
        ];
        
        const isGeneric = genericPatterns.some(pattern => pattern.test(content));
        if (isGeneric) {
            return 0.05; // Heavily penalize generic content
        }
        
        // Check content length and substance
        if (content.length < 50) {
            return 0.2; // Short content gets low score
        }
        
        // Check for meaningful content indicators
        const meaningfulIndicators = [
            'decided', 'implemented', 'changed', 'fixed', 'created', 'updated',
            'because', 'reason', 'approach', 'solution', 'result', 'impact',
            'learned', 'discovered', 'found', 'issue', 'problem', 'challenge'
        ];
        
        const meaningfulMatches = meaningfulIndicators.filter(indicator => 
            content.toLowerCase().includes(indicator)
        ).length;
        
        // Calculate information density
        const words = content.split(/\s+/).filter(w => w.length > 2);
        const uniqueWords = new Set(words.map(w => w.toLowerCase()));
        const diversityRatio = uniqueWords.size / Math.max(words.length, 1);
        
        // Combine factors
        const meaningfulnessScore = Math.min(0.4, meaningfulMatches * 0.08);
        const diversityScore = Math.min(0.3, diversityRatio * 0.5);
        const lengthScore = Math.min(0.3, content.length / 1000); // Longer content gets bonus
        
        const qualityScore = meaningfulnessScore + diversityScore + lengthScore;
        return Math.max(0.05, Math.min(1.0, qualityScore));
        
    } catch (error) {
        // Silently fail with default score to avoid noise
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
        // Silently fail with default score to avoid noise
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
 * Calculate recency bonus to prioritize very recent memories
 * Provides explicit boost for memories created within specific time windows
 */
function calculateRecencyBonus(memoryDate) {
    // Recency bonus tiers (days and corresponding bonus values)
    const RECENCY_TIERS = [
        { days: 7, bonus: 0.15 },  // Strong boost for last week
        { days: 14, bonus: 0.10 }, // Moderate boost for last 2 weeks
        { days: 30, bonus: 0.05 }  // Small boost for last month
    ];

    try {
        const now = new Date();

        // Handle both Unix timestamps (seconds) and ISO strings
        let memoryTime;
        if (typeof memoryDate === 'string') {
            // ISO string format
            memoryTime = new Date(memoryDate);
        } else if (typeof memoryDate === 'number') {
            // Unix timestamp in seconds, convert to milliseconds
            memoryTime = new Date(memoryDate * 1000);
        } else {
            return 0; // Invalid format
        }

        if (isNaN(memoryTime.getTime()) || memoryTime > now) {
            return 0; // No bonus for invalid or future dates
        }

        const daysDiff = (now - memoryTime) / (1000 * 60 * 60 * 24);

        // Find the appropriate tier for this memory's age
        for (const tier of RECENCY_TIERS) {
            if (daysDiff <= tier.days) {
                return tier.bonus;
            }
        }

        return 0; // No bonus for older memories

    } catch (error) {
        return 0;
    }
}

/**
 * Extract backend quality score from memory metadata
 * This leverages the AI-based quality scoring from the MCP Memory Service backend
 * (ONNX local SLM, Groq, or implicit signals)
 */
function calculateBackendQuality(memory) {
    try {
        // Check for quality_score in metadata (set by backend quality system)
        if (memory.metadata && typeof memory.metadata.quality_score === 'number') {
            return memory.metadata.quality_score;
        }

        // Also check direct property (some API responses flatten metadata)
        if (typeof memory.quality_score === 'number') {
            return memory.quality_score;
        }

        // Default to neutral score if not available
        // This ensures graceful fallback when backend hasn't scored the memory
        return 0.5;

    } catch (error) {
        return 0.5; // Neutral fallback
    }
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
        // Silently fail with default score to avoid noise
        return 0.3;
    }
}

/**
 * Calculate final relevance score for a memory (Enhanced with quality scoring)
 */
function calculateRelevanceScore(memory, projectContext, options = {}) {
    try {
        const {
            weights = {},
            timeDecayRate = 0.1,       // Default decay rate
            includeConversationContext = false,
            conversationAnalysis = null
        } = options;

        // Default weights including content quality and backend quality factors
        // Backend quality leverages AI-based semantic scoring from MCP Memory Service
        const defaultWeights = includeConversationContext ? {
            timeDecay: 0.15,           // Reduced weight for time
            tagRelevance: 0.25,        // Tag matching remains important
            contentRelevance: 0.10,    // Content matching reduced
            contentQuality: 0.15,      // Heuristic quality factor
            backendQuality: 0.15,      // AI-based backend quality (ONNX/Groq)
            conversationRelevance: 0.20, // Conversation context factor
            typeBonus: 0.05            // Memory type provides minor adjustment
        } : {
            timeDecay: 0.20,           // Reduced time weight
            tagRelevance: 0.30,        // Tag matching important
            contentRelevance: 0.10,    // Content matching reduced
            contentQuality: 0.20,      // Heuristic quality factor
            backendQuality: 0.20,      // AI-based backend quality (ONNX/Groq)
            typeBonus: 0.05            // Type bonus reduced
        };

        const w = { ...defaultWeights, ...weights };

        // Calculate individual scores
        const timeScore = calculateTimeDecay(memory.created_at || memory.created_at_iso, timeDecayRate);
        const tagScore = calculateTagRelevance(memory.tags, projectContext);
        const contentScore = calculateContentRelevance(memory.content, projectContext);
        const qualityScore = calculateContentQuality(memory.content);
        const backendQualityScore = calculateBackendQuality(memory); // AI-based quality from backend
        const typeBonus = calculateTypeBonus(memory.memory_type);
        const recencyBonus = calculateRecencyBonus(memory.created_at || memory.created_at_iso);

        let finalScore = (
            (timeScore * w.timeDecay) +
            (tagScore * w.tagRelevance) +
            (contentScore * w.contentRelevance) +
            (qualityScore * w.contentQuality) +
            (backendQualityScore * (w.backendQuality || 0)) + // Backend AI quality score
            typeBonus + // Type bonus is not weighted, acts as adjustment
            recencyBonus // Recency bonus provides explicit boost for very recent memories
        );

        const breakdown = {
            timeDecay: timeScore,
            tagRelevance: tagScore,
            contentRelevance: contentScore,
            contentQuality: qualityScore,
            backendQuality: backendQualityScore, // AI-based quality from ONNX/Groq
            typeBonus: typeBonus,
            recencyBonus: recencyBonus
        };

        // Add conversation context scoring if enabled (Phase 2)
        if (includeConversationContext && conversationAnalysis) {
            const conversationScore = calculateConversationRelevance(memory, conversationAnalysis);
            finalScore += (conversationScore * (w.conversationRelevance || 0));
            breakdown.conversationRelevance = conversationScore;
        }
        
        // Apply quality penalty for very low quality content (multiplicative)
        if (qualityScore < 0.2) {
            finalScore *= 0.5; // Heavily penalize low quality content
        }

        // Apply project affinity penalty - memories without project tag match get penalized
        // This prevents cross-project memory pollution (e.g., Azure memories in Python project)
        const memoryTags = (memory.tags || []).map(t => t.toLowerCase());
        const memoryContent = (memory.content || '').toLowerCase();
        const projectName = projectContext.name?.toLowerCase();

        // Check for project name in tags OR content
        const hasProjectTag = projectName && (
            memoryTags.some(tag => tag === projectName || tag.includes(projectName)) ||
            memoryContent.includes(projectName)
        );

        if (!hasProjectTag && tagScore < 0.3) {
            // No project reference at all - definitely unrelated memory
            // Hard filter: set score to 0 to exclude from results entirely
            finalScore = 0;
            breakdown.projectAffinity = 'none (filtered)';
        } else if (!hasProjectTag) {
            // Some tag relevance but no project tag - might be related
            finalScore = Math.min(1.0, finalScore) * 0.5; // Cap to prevent bonus inflation, moderate penalty preserves cross-project technology sharing
            breakdown.projectAffinity = 'low';
        } else {
            breakdown.projectAffinity = 'high';
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
        // Silently fail with default score to avoid noise
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
        const { verbose = true } = options;
        
        if (!Array.isArray(memories)) {
            if (verbose) console.warn('[Memory Scorer] Invalid memories array');
            return [];
        }
        
        if (verbose) {
            console.log(`[Memory Scorer] Scoring ${memories.length} memories for project: ${projectContext.name}`);
        }
        
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
        if (verbose) {
            console.log('[Memory Scorer] Top scored memories:');
            sortedMemories.slice(0, 3).forEach((memory, index) => {
                console.log(`  ${index + 1}. Score: ${memory.relevanceScore.toFixed(3)} - ${memory.content.substring(0, 60)}...`);
            });
        }
        
        return sortedMemories;
        
    } catch (error) {
        if (verbose) console.error('[Memory Scorer] Error scoring memories:', error.message);
        return memories || [];
    }
}

/**
 * Filter memories by minimum relevance threshold
 */
function filterByRelevance(memories, minScore = 0.3, options = {}) {
    try {
        const { verbose = true } = options;
        const filtered = memories.filter(memory => memory.relevanceScore >= minScore);
        if (verbose) {
            console.log(`[Memory Scorer] Filtered ${filtered.length}/${memories.length} memories above threshold ${minScore}`);
        }
        return filtered;

    } catch (error) {
        if (verbose) console.warn('[Memory Scorer] Error filtering memories:', error.message);
        return memories;
    }
}

/**
 * Analyze memory age distribution to detect staleness
 * Returns statistics and recommended weight adjustments
 */
function analyzeMemoryAgeDistribution(memories, options = {}) {
    try {
        const { verbose = false } = options;

        if (!Array.isArray(memories) || memories.length === 0) {
            return {
                avgAge: 0,
                medianAge: 0,
                p75Age: 0,
                p90Age: 0,
                recentCount: 0,
                staleCount: 0,
                isStale: false,
                recommendedAdjustments: {}
            };
        }

        const now = new Date();

        // Calculate ages in days
        const ages = memories.map(memory => {
            // Handle both Unix timestamps (seconds) and ISO strings
            let memoryTime;
            if (memory.created_at_iso) {
                memoryTime = new Date(memory.created_at_iso);
            } else if (memory.created_at) {
                // created_at is in seconds, convert to milliseconds
                memoryTime = new Date(memory.created_at * 1000);
            } else {
                return 365; // Default to very old if no timestamp
            }

            if (isNaN(memoryTime.getTime())) return 365; // Default to very old
            return (now - memoryTime) / (1000 * 60 * 60 * 24);
        }).sort((a, b) => a - b);

        // Calculate percentiles
        const avgAge = ages.reduce((sum, age) => sum + age, 0) / ages.length;
        const medianAge = ages[Math.floor(ages.length / 2)];
        const p75Age = ages[Math.floor(ages.length * 0.75)];
        const p90Age = ages[Math.floor(ages.length * 0.90)];

        // Count recent vs stale
        const recentCount = ages.filter(age => age <= 14).length; // Last 2 weeks
        const staleCount = ages.filter(age => age > 30).length;   // Older than 1 month

        // Determine if memory set is stale
        const isStale = medianAge > 30 || (recentCount / ages.length) < 0.2;

        // Recommended adjustments based on staleness
        const recommendedAdjustments = {};

        if (isStale) {
            // Memories are old - boost time decay weight, reduce tag relevance
            recommendedAdjustments.timeDecay = 0.50;      // Increase from default 0.25
            recommendedAdjustments.tagRelevance = 0.20;   // Decrease from default 0.35
            recommendedAdjustments.recencyBonus = 0.25;   // Increase bonus for any recent memories
            recommendedAdjustments.reason = `Stale memory set detected (median: ${Math.round(medianAge)}d old, ${Math.round(recentCount/ages.length*100)}% recent)`;
        } else if (avgAge < 14) {
            // Memories are very recent - balanced approach
            recommendedAdjustments.timeDecay = 0.30;
            recommendedAdjustments.tagRelevance = 0.30;
            recommendedAdjustments.reason = `Recent memory set (avg: ${Math.round(avgAge)}d old)`;
        }

        if (verbose) {
            console.log('[Memory Age Analyzer]', {
                avgAge: Math.round(avgAge),
                medianAge: Math.round(medianAge),
                p75Age: Math.round(p75Age),
                recentPercent: Math.round(recentCount / ages.length * 100),
                isStale,
                adjustments: recommendedAdjustments.reason || 'No adjustments needed'
            });
        }

        return {
            avgAge,
            medianAge,
            p75Age,
            p90Age,
            recentCount,
            staleCount,
            totalCount: ages.length,
            isStale,
            recommendedAdjustments
        };

    } catch (error) {
        if (verbose) console.error('[Memory Age Analyzer] Error:', error.message);
        return {
            avgAge: 0,
            medianAge: 0,
            p75Age: 0,
            p90Age: 0,
            recentCount: 0,
            staleCount: 0,
            isStale: false,
            recommendedAdjustments: {}
        };
    }
}

/**
 * Calculate adaptive git context weight based on memory age and git activity
 * Prevents old git-related memories from dominating when recent development exists
 */
function calculateAdaptiveGitWeight(gitContext, memoryAgeAnalysis, configuredWeight = 1.2, options = {}) {
    try {
        const { verbose = false } = options;

        // No git context or no recent commits - use configured weight
        if (!gitContext || !gitContext.recentCommits || gitContext.recentCommits.length === 0) {
            return { weight: configuredWeight, reason: 'No recent git activity' };
        }

        // Calculate days since most recent commit
        const now = new Date();
        const mostRecentCommit = new Date(gitContext.recentCommits[0].date);
        const daysSinceLastCommit = (now - mostRecentCommit) / (1000 * 60 * 60 * 24);

        // Scenario 1: Recent commits (< 7d) BUT stale memories (median > 30d)
        // Problem: Git boost would amplify old git memories over potential recent work
        if (daysSinceLastCommit <= 7 && memoryAgeAnalysis.medianAge > 30) {
            const reducedWeight = Math.max(1.0, configuredWeight * 0.7); // Reduce by 30%
            const reason = `Recent commits (${Math.round(daysSinceLastCommit)}d ago) but stale memories (median: ${Math.round(memoryAgeAnalysis.medianAge)}d) - reducing git boost`;

            if (verbose) {
                console.log(`[Adaptive Git Weight] ${reason}: ${configuredWeight.toFixed(1)} → ${reducedWeight.toFixed(1)}`);
            }

            return { weight: reducedWeight, reason, adjusted: true };
        }

        // Scenario 2: Both commits and memories are recent (< 14d)
        // Safe to use configured weight, git context is relevant
        if (daysSinceLastCommit <= 14 && memoryAgeAnalysis.avgAge <= 14) {
            return {
                weight: configuredWeight,
                reason: `Recent commits and memories aligned (${Math.round(daysSinceLastCommit)}d commits, ${Math.round(memoryAgeAnalysis.avgAge)}d avg memory age)`,
                adjusted: false
            };
        }

        // Scenario 3: Old commits (> 14d) but recent memories exist
        // Slightly reduce git weight to let recent non-git memories surface
        if (daysSinceLastCommit > 14 && memoryAgeAnalysis.recentCount > 0) {
            const reducedWeight = Math.max(1.0, configuredWeight * 0.85); // Reduce by 15%
            const reason = `Older commits (${Math.round(daysSinceLastCommit)}d ago) with some recent memories - slightly reducing git boost`;

            if (verbose) {
                console.log(`[Adaptive Git Weight] ${reason}: ${configuredWeight.toFixed(1)} → ${reducedWeight.toFixed(1)}`);
            }

            return { weight: reducedWeight, reason, adjusted: true };
        }

        // Default: use configured weight
        return { weight: configuredWeight, reason: 'Using configured weight', adjusted: false };

    } catch (error) {
        if (verbose) console.error('[Adaptive Git Weight] Error:', error.message);
        return { weight: configuredWeight, reason: 'Error - using fallback', adjusted: false };
    }
}

module.exports = {
    scoreMemoryRelevance,
    calculateRelevanceScore,
    calculateTimeDecay,
    calculateTagRelevance,
    calculateContentRelevance,
    calculateContentQuality,
    calculateBackendQuality,  // AI-based quality scoring integration
    calculateConversationRelevance,
    calculateTypeBonus,
    calculateRecencyBonus,
    filterByRelevance,
    analyzeMemoryAgeDistribution,
    calculateAdaptiveGitWeight
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