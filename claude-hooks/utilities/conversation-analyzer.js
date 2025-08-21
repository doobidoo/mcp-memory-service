/**
 * Conversation Analyzer
 * Provides natural language processing and topic detection for dynamic memory loading
 * Phase 2: Intelligent Context Updates
 */

/**
 * Analyze conversation content to extract topics, entities, and context
 * @param {string} conversationText - The conversation text to analyze
 * @param {object} options - Analysis options
 * @returns {object} Analysis results including topics, entities, and intent
 */
function analyzeConversation(conversationText, options = {}) {
    const {
        extractTopics = true,
        extractEntities = true,
        detectIntent = true,
        detectCodeContext = true,
        minTopicConfidence = 0.3
    } = options;

    console.log('[Conversation Analyzer] Analyzing conversation content...');

    const analysis = {
        topics: [],
        entities: [],
        intent: null,
        codeContext: null,
        confidence: 0,
        metadata: {
            length: conversationText.length,
            analysisTime: new Date().toISOString()
        }
    };

    try {
        // Extract topics from conversation
        if (extractTopics) {
            analysis.topics = extractTopicsFromText(conversationText, minTopicConfidence);
        }

        // Extract entities (technologies, frameworks, languages)
        if (extractEntities) {
            analysis.entities = extractEntitiesFromText(conversationText);
        }

        // Detect conversation intent
        if (detectIntent) {
            analysis.intent = detectConversationIntent(conversationText);
        }

        // Detect code-specific context
        if (detectCodeContext) {
            analysis.codeContext = detectCodeContextFromText(conversationText);
        }

        // Calculate overall confidence score
        analysis.confidence = calculateAnalysisConfidence(analysis);

        console.log(`[Conversation Analyzer] Found ${analysis.topics.length} topics, ${analysis.entities.length} entities, confidence: ${(analysis.confidence * 100).toFixed(1)}%`);

        return analysis;

    } catch (error) {
        console.error('[Conversation Analyzer] Error during analysis:', error.message);
        return analysis; // Return partial results
    }
}

/**
 * Extract topics from conversation text using keyword analysis and context
 */
function extractTopicsFromText(text, minConfidence = 0.3) {
    const topics = [];
    
    // Technical topic patterns
    const topicPatterns = [
        // Development activities
        { pattern: /\b(debug|debugging|bug|error|exception|fix|fixing|issue|issues|problem)\b/gi, topic: 'debugging', weight: 0.9 },
        { pattern: /\b(architect|architecture|design|structure|pattern|system|framework)\b/gi, topic: 'architecture', weight: 1.0 },
        { pattern: /\b(implement|implementation|build|develop|code)\b/gi, topic: 'implementation', weight: 0.7 },
        { pattern: /\b(test|testing|unit test|integration|spec)\b/gi, topic: 'testing', weight: 0.7 },
        { pattern: /\b(deploy|deployment|release|production|staging)\b/gi, topic: 'deployment', weight: 0.6 },
        { pattern: /\b(refactor|refactoring|cleanup|optimize|performance)\b/gi, topic: 'refactoring', weight: 0.7 },
        
        // Technologies
        { pattern: /\b(database|db|sql|query|schema|migration|sqlite|postgres|mysql|performance)\b/gi, topic: 'database', weight: 0.9 },
        { pattern: /\b(api|endpoint|rest|graphql|request|response)\b/gi, topic: 'api', weight: 0.7 },
        { pattern: /\b(frontend|ui|ux|interface|component|react|vue)\b/gi, topic: 'frontend', weight: 0.7 },
        { pattern: /\b(backend|server|service|microservice|lambda)\b/gi, topic: 'backend', weight: 0.7 },
        { pattern: /\b(security|auth|authentication|authorization|jwt|oauth)\b/gi, topic: 'security', weight: 0.8 },
        { pattern: /\b(docker|container|kubernetes|deployment|ci\/cd)\b/gi, topic: 'devops', weight: 0.6 },
        
        // Concepts
        { pattern: /\b(memory|storage|cache|persistence|state)\b/gi, topic: 'memory-management', weight: 0.7 },
        { pattern: /\b(hook|plugin|extension|integration)\b/gi, topic: 'integration', weight: 0.6 },
        { pattern: /\b(claude|ai|gpt|llm|automation)\b/gi, topic: 'ai-integration', weight: 0.8 },
    ];

    // Score topics based on pattern matches
    const topicScores = new Map();
    
    topicPatterns.forEach(({ pattern, topic, weight }) => {
        const matches = text.match(pattern) || [];
        if (matches.length > 0) {
            const score = Math.min(matches.length * weight * 0.3, 1.0); // Increased multiplier
            if (score >= minConfidence) {
                topicScores.set(topic, Math.max(topicScores.get(topic) || 0, score));
            }
        }
    });

    // Convert scores to topic objects
    topicScores.forEach((confidence, topicName) => {
        topics.push({
            name: topicName,
            confidence,
            weight: confidence
        });
    });

    // Sort by confidence and return top topics
    return topics
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 10); // Limit to top 10 topics
}

/**
 * Extract entities (technologies, frameworks, languages) from text
 */
function extractEntitiesFromText(text) {
    const entities = [];
    
    const entityPatterns = [
        // Languages
        { pattern: /\b(javascript|js|typescript|ts|python|java|c\+\+|rust|go|php|ruby)\b/gi, type: 'language' },
        
        // Frameworks
        { pattern: /\b(react|vue|angular|next\.js|express|fastapi|django|flask|spring)\b/gi, type: 'framework' },
        
        // Databases
        { pattern: /\b(postgresql|postgres|mysql|mongodb|sqlite|redis|elasticsearch)\b/gi, type: 'database' },
        
        // Tools
        { pattern: /\b(docker|kubernetes|git|github|gitlab|jenkins|webpack|vite)\b/gi, type: 'tool' },
        
        // Cloud/Services
        { pattern: /\b(aws|azure|gcp|vercel|netlify|heroku)\b/gi, type: 'cloud' },
        
        // Specific to our project
        { pattern: /\b(claude|mcp|memory-service|sqlite-vec|chroma)\b/gi, type: 'project' }
    ];

    entityPatterns.forEach(({ pattern, type }) => {
        const matches = text.match(pattern) || [];
        matches.forEach(match => {
            const entity = match.toLowerCase();
            if (!entities.find(e => e.name === entity)) {
                entities.push({
                    name: entity,
                    type,
                    confidence: 0.8
                });
            }
        });
    });

    return entities;
}

/**
 * Detect conversation intent (what the user is trying to accomplish)
 */
function detectConversationIntent(text) {
    const intentPatterns = [
        { pattern: /\b(help|how|explain|understand|learn|guide)\b/gi, intent: 'learning', confidence: 0.7 },
        { pattern: /\b(fix|solve|debug|error|problem|issue)\b/gi, intent: 'problem-solving', confidence: 0.8 },
        { pattern: /\b(build|create|implement|develop|add)\b/gi, intent: 'development', confidence: 0.7 },
        { pattern: /\b(optimize|improve|enhance|refactor|better)\b/gi, intent: 'optimization', confidence: 0.6 },
        { pattern: /\b(review|check|analyze|audit|validate)\b/gi, intent: 'review', confidence: 0.6 },
        { pattern: /\b(plan|design|architect|structure|approach)\b/gi, intent: 'planning', confidence: 0.7 },
    ];

    let bestIntent = null;
    let bestScore = 0;

    intentPatterns.forEach(({ pattern, intent, confidence }) => {
        const matches = text.match(pattern) || [];
        if (matches.length > 0) {
            const score = Math.min(matches.length * confidence * 0.3, 1.0); // Increased multiplier
            if (score > bestScore) {
                bestScore = score;
                bestIntent = {
                    name: intent,
                    confidence: score
                };
            }
        }
    });

    return bestIntent;
}

/**
 * Detect code-specific context from the conversation
 */
function detectCodeContextFromText(text) {
    const context = {
        hasCodeBlocks: /```[\s\S]*?```/g.test(text),
        hasInlineCode: /`[^`]+`/g.test(text),
        hasFilePaths: /\b[\w.-]+\.(js|ts|py|java|cpp|rs|go|php|rb|md|json|yaml|yml)\b/gi.test(text),
        hasErrorMessages: /\b(error|exception|failed|traceback|stack trace)\b/gi.test(text),
        hasCommands: /\$\s+[\w\-\.\/]+/g.test(text),
        hasUrls: /(https?:\/\/[^\s]+)/g.test(text)
    };

    // Extract code languages if present
    const codeLanguages = [];
    const langMatches = text.match(/```(\w+)/g);
    if (langMatches) {
        langMatches.forEach(match => {
            const lang = match.replace('```', '').toLowerCase();
            if (!codeLanguages.includes(lang)) {
                codeLanguages.push(lang);
            }
        });
    }

    context.languages = codeLanguages;
    context.isCodeRelated = Object.values(context).some(v => v === true) || codeLanguages.length > 0;

    return context;
}

/**
 * Calculate overall confidence score for the analysis
 */
function calculateAnalysisConfidence(analysis) {
    let totalConfidence = 0;
    let factors = 0;

    // Factor in topic confidence
    if (analysis.topics.length > 0) {
        const avgTopicConfidence = analysis.topics.reduce((sum, t) => sum + t.confidence, 0) / analysis.topics.length;
        totalConfidence += avgTopicConfidence;
        factors++;
    }

    // Factor in entity confidence
    if (analysis.entities.length > 0) {
        const avgEntityConfidence = analysis.entities.reduce((sum, e) => sum + e.confidence, 0) / analysis.entities.length;
        totalConfidence += avgEntityConfidence;
        factors++;
    }

    // Factor in intent confidence
    if (analysis.intent) {
        totalConfidence += analysis.intent.confidence;
        factors++;
    }

    // Factor in code context
    if (analysis.codeContext && analysis.codeContext.isCodeRelated) {
        totalConfidence += 0.8;
        factors++;
    }

    return factors > 0 ? totalConfidence / factors : 0;
}

/**
 * Compare two conversation analyses to detect topic changes
 * @param {object} previousAnalysis - Previous conversation analysis
 * @param {object} currentAnalysis - Current conversation analysis
 * @returns {object} Topic change detection results
 */
function detectTopicChanges(previousAnalysis, currentAnalysis) {
    const changes = {
        hasTopicShift: false,
        newTopics: [],
        changedIntents: false,
        significanceScore: 0
    };

    if (!currentAnalysis) {
        return changes;
    }

    // If no previous analysis, treat all current topics as new
    if (!previousAnalysis) {
        changes.newTopics = currentAnalysis.topics.filter(topic => topic.confidence > 0.3);
        if (changes.newTopics.length > 0) {
            changes.hasTopicShift = true;
            changes.significanceScore = Math.min(changes.newTopics.length * 0.4, 1.0);
        }
        return changes;
    }

    // Detect new topics
    const previousTopicNames = new Set(previousAnalysis.topics.map(t => t.name));
    changes.newTopics = currentAnalysis.topics.filter(topic => 
        !previousTopicNames.has(topic.name) && topic.confidence > 0.4
    );

    // Check for intent changes
    const previousIntent = previousAnalysis.intent?.name;
    const currentIntent = currentAnalysis.intent?.name;
    changes.changedIntents = previousIntent !== currentIntent && currentIntent;

    // Calculate significance score
    let significance = 0;
    if (changes.newTopics.length > 0) {
        significance += changes.newTopics.length * 0.3;
    }
    if (changes.changedIntents) {
        significance += 0.4;
    }

    changes.significanceScore = Math.min(significance, 1.0);
    changes.hasTopicShift = changes.significanceScore >= 0.3;

    return changes;
}

module.exports = {
    analyzeConversation,
    detectTopicChanges,
    extractTopicsFromText,
    extractEntitiesFromText,
    detectConversationIntent,
    detectCodeContext: detectCodeContextFromText
};