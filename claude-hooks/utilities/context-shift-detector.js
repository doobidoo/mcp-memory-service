/**
 * Context Shift Detection Utility
 * Detects significant context changes that warrant memory refresh
 */

/**
 * Detect if there's been a significant context shift warranting memory refresh
 */
function detectContextShift(currentContext, previousContext, options = {}) {
    try {
        const {
            minTopicShiftScore = 0.4,
            minProjectChangeConfidence = 0.6,
            maxTimeSinceLastRefresh = 30 * 60 * 1000, // 30 minutes
            enableUserRequestDetection = true
        } = options;
        
        if (!previousContext) {
            return {
                shouldRefresh: false,
                reason: 'no-previous-context',
                confidence: 0
            };
        }
        
        const shifts = [];
        let totalScore = 0;
        
        // 1. Check for explicit user requests
        if (enableUserRequestDetection && currentContext.userMessage) {
            const message = currentContext.userMessage.toLowerCase();
            const memoryRequestPatterns = [
                'remember', 'recall', 'what did we', 'previous', 'history',
                'context', 'background', 'refresh', 'load memories',
                'show me what', 'bring up', 'retrieve'
            ];
            
            const hasMemoryRequest = memoryRequestPatterns.some(pattern => 
                message.includes(pattern)
            );
            
            if (hasMemoryRequest) {
                shifts.push({
                    type: 'user-request',
                    confidence: 0.9,
                    description: 'User explicitly requested memory/context'
                });
                totalScore += 0.9;
            }
        }
        
        // 2. Check for project/directory changes
        if (currentContext.workingDirectory !== previousContext.workingDirectory) {
            shifts.push({
                type: 'project-change',
                confidence: 0.8,
                description: `Project changed: ${previousContext.workingDirectory} → ${currentContext.workingDirectory}`
            });
            totalScore += 0.8;
        }
        
        // 3. Check for significant topic/domain shifts
        if (currentContext.topics && previousContext.topics) {
            const topicOverlap = calculateTopicOverlap(currentContext.topics, previousContext.topics);
            if (topicOverlap < (1 - minTopicShiftScore)) {
                const confidence = 1 - topicOverlap;
                shifts.push({
                    type: 'topic-shift',
                    confidence,
                    description: `Significant topic change detected (overlap: ${(topicOverlap * 100).toFixed(1)}%)`
                });
                totalScore += confidence;
            }
        }
        
        // 4. Check for technology/framework changes
        if (currentContext.frameworks && previousContext.frameworks) {
            const frameworkOverlap = calculateArrayOverlap(currentContext.frameworks, previousContext.frameworks);
            if (frameworkOverlap < 0.5) {
                const confidence = 0.6;
                shifts.push({
                    type: 'framework-change',
                    confidence,
                    description: `Framework/technology shift detected`
                });
                totalScore += confidence;
            }
        }
        
        // 5. Check for time-based refresh need
        const timeSinceLastRefresh = currentContext.timestamp - (previousContext.lastMemoryRefresh || 0);
        if (timeSinceLastRefresh > maxTimeSinceLastRefresh) {
            shifts.push({
                type: 'time-based',
                confidence: 0.3,
                description: `Long time since last refresh (${Math.round(timeSinceLastRefresh / 60000)} minutes)`
            });
            totalScore += 0.3;
        }
        
        // 6. Check for conversation complexity increase
        if (currentContext.conversationDepth && previousContext.conversationDepth) {
            const depthIncrease = currentContext.conversationDepth - previousContext.conversationDepth;
            if (depthIncrease > 5) { // More than 5 exchanges since last refresh
                shifts.push({
                    type: 'conversation-depth',
                    confidence: 0.4,
                    description: `Conversation has deepened significantly (${depthIncrease} exchanges)`
                });
                totalScore += 0.4;
            }
        }
        
        // Calculate final decision
        const shouldRefresh = totalScore > 0.5 || shifts.some(s => s.confidence > 0.7);
        const primaryReason = shifts.length > 0 ? shifts.reduce((max, shift) => 
            shift.confidence > max.confidence ? shift : max
        ) : null;
        
        return {
            shouldRefresh,
            reason: primaryReason ? primaryReason.type : 'no-shift',
            confidence: totalScore,
            shifts,
            description: primaryReason ? primaryReason.description : 'No significant context shift detected'
        };
        
    } catch (error) {
        console.warn('[Context Shift Detector] Error detecting context shift:', error.message);
        return {
            shouldRefresh: false,
            reason: 'error',
            confidence: 0,
            error: error.message
        };
    }
}

/**
 * Calculate topic overlap between two topic arrays
 */
function calculateTopicOverlap(topics1, topics2) {
    if (!topics1.length && !topics2.length) return 1;
    if (!topics1.length || !topics2.length) return 0;
    
    const topics1Set = new Set(topics1.map(t => (t.name || t).toLowerCase()));
    const topics2Set = new Set(topics2.map(t => (t.name || t).toLowerCase()));
    
    const intersection = new Set([...topics1Set].filter(t => topics2Set.has(t)));
    const union = new Set([...topics1Set, ...topics2Set]);
    
    return intersection.size / union.size;
}

/**
 * Calculate overlap between two arrays
 */
function calculateArrayOverlap(arr1, arr2) {
    if (!arr1.length && !arr2.length) return 1;
    if (!arr1.length || !arr2.length) return 0;
    
    const set1 = new Set(arr1.map(item => item.toLowerCase()));
    const set2 = new Set(arr2.map(item => item.toLowerCase()));
    
    const intersection = new Set([...set1].filter(item => set2.has(item)));
    const union = new Set([...set1, ...set2]);
    
    return intersection.size / union.size;
}

/**
 * Extract context information from current conversation state
 */
function extractCurrentContext(conversationState, workingDirectory) {
    try {
        return {
            workingDirectory: workingDirectory || process.cwd(),
            timestamp: Date.now(),
            userMessage: conversationState.lastUserMessage || '',
            topics: conversationState.topics || [],
            frameworks: conversationState.frameworks || [],
            conversationDepth: conversationState.exchangeCount || 0,
            lastMemoryRefresh: conversationState.lastMemoryRefresh || 0
        };
    } catch (error) {
        console.warn('[Context Shift Detector] Error extracting context:', error.message);
        return {
            workingDirectory: workingDirectory || process.cwd(),
            timestamp: Date.now(),
            topics: [],
            frameworks: [],
            conversationDepth: 0
        };
    }
}

/**
 * Determine appropriate refresh strategy based on context shift
 */
function determineRefreshStrategy(shiftDetection) {
    const strategies = {
        'user-request': {
            priority: 'high',
            maxMemories: 8,
            includeScore: true,
            message: '🔍 Refreshing memory context as requested...'
        },
        'project-change': {
            priority: 'high',
            maxMemories: 6,
            includeScore: false,
            message: '📁 Loading memories for new project context...'
        },
        'topic-shift': {
            priority: 'medium',
            maxMemories: 5,
            includeScore: false,
            message: '💭 Updating context for topic shift...'
        },
        'framework-change': {
            priority: 'medium',
            maxMemories: 5,
            includeScore: false,
            message: '⚡ Refreshing context for technology change...'
        },
        'time-based': {
            priority: 'low',
            maxMemories: 3,
            includeScore: false,
            message: '⏰ Periodic memory context refresh...'
        },
        'conversation-depth': {
            priority: 'low',
            maxMemories: 4,
            includeScore: false,
            message: '💬 Loading additional context for deep conversation...'
        }
    };
    
    const primaryShift = shiftDetection.shifts.reduce((max, shift) => 
        shift.confidence > max.confidence ? shift : max, 
        { confidence: 0, type: 'none' }
    );
    
    return strategies[primaryShift.type] || {
        priority: 'low',
        maxMemories: 3,
        includeScore: false,
        message: '🧠 Loading relevant memory context...'
    };
}

module.exports = {
    detectContextShift,
    extractCurrentContext,
    determineRefreshStrategy,
    calculateTopicOverlap,
    calculateArrayOverlap
};

// Direct execution support for testing
if (require.main === module) {
    // Test context shift detection
    const mockPreviousContext = {
        workingDirectory: '/old/project',
        timestamp: Date.now() - 40 * 60 * 1000, // 40 minutes ago
        topics: ['javascript', 'react', 'frontend'],
        frameworks: ['React', 'Node.js'],
        conversationDepth: 5,
        lastMemoryRefresh: Date.now() - 35 * 60 * 1000
    };
    
    const mockCurrentContext = {
        workingDirectory: '/new/project',
        timestamp: Date.now(),
        userMessage: 'Can you remind me what we decided about the architecture?',
        topics: ['python', 'fastapi', 'backend'],
        frameworks: ['FastAPI', 'SQLAlchemy'],
        conversationDepth: 12,
        lastMemoryRefresh: Date.now() - 35 * 60 * 1000
    };
    
    console.log('=== CONTEXT SHIFT DETECTION TEST ===');
    const shiftResult = detectContextShift(mockCurrentContext, mockPreviousContext);
    console.log('Shift Detection Result:', JSON.stringify(shiftResult, null, 2));
    
    const strategy = determineRefreshStrategy(shiftResult);
    console.log('Recommended Strategy:', strategy);
    console.log('=== END TEST ===');
}