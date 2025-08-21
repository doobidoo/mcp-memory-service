/**
 * Session Tracker Utility
 * Provides cross-session intelligence and conversation continuity
 * Phase 2: Intelligent Context Updates
 */

const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');

/**
 * Session tracking data structure
 */
class SessionTracker {
    constructor(options = {}) {
        this.options = {
            maxSessionHistory: 50,      // Maximum sessions to track
            maxConversationDepth: 10,   // Maximum conversation thread depth
            sessionExpiryDays: 30,      // Days after which sessions are considered expired
            trackingDataPath: options.trackingDataPath || path.join(__dirname, '../session-tracking.json'),
            ...options
        };

        this.sessions = new Map();
        this.conversationThreads = new Map();
        this.projectSessions = new Map();
        this.loaded = false;
    }

    /**
     * Initialize session tracking system
     */
    async initialize() {
        console.log('[Session Tracker] Initializing session tracking system...');
        
        try {
            await this.loadTrackingData();
            this.cleanupExpiredSessions();
            this.loaded = true;
            
            console.log(`[Session Tracker] Loaded ${this.sessions.size} sessions, ${this.conversationThreads.size} threads`);
        } catch (error) {
            console.error('[Session Tracker] Failed to initialize:', error.message);
            this.loaded = false;
        }
    }

    /**
     * Start tracking a new session
     */
    async startSession(sessionId, context = {}) {
        if (!this.loaded) {
            await this.initialize();
        }

        const session = {
            id: sessionId,
            startTime: new Date().toISOString(),
            endTime: null,
            projectContext: context.projectContext || {},
            workingDirectory: context.workingDirectory,
            initialTopics: [],
            finalTopics: [],
            memoriesLoaded: [],
            memoriesCreated: [],
            conversationSummary: null,
            outcome: null,
            threadId: null,
            parentSessionId: null,
            childSessionIds: [],
            status: 'active'
        };

        // Try to link to existing conversation thread
        await this.linkToConversationThread(session, context);

        this.sessions.set(sessionId, session);
        
        // Track by project
        const projectName = context.projectContext?.name;
        if (projectName) {
            if (!this.projectSessions.has(projectName)) {
                this.projectSessions.set(projectName, []);
            }
            this.projectSessions.get(projectName).push(sessionId);
        }

        console.log(`[Session Tracker] Started session ${sessionId} for project: ${projectName || 'unknown'}`);
        
        await this.saveTrackingData();
        return session;
    }

    /**
     * End session tracking and record outcomes
     */
    async endSession(sessionId, outcome = {}) {
        const session = this.sessions.get(sessionId);
        if (!session) {
            console.warn(`[Session Tracker] Session ${sessionId} not found`);
            return null;
        }

        session.endTime = new Date().toISOString();
        session.status = 'completed';
        session.outcome = outcome;
        session.conversationSummary = outcome.summary;
        session.finalTopics = outcome.topics || [];

        // Update conversation thread with session outcome
        if (session.threadId) {
            await this.updateConversationThread(session.threadId, session);
        }

        console.log(`[Session Tracker] Ended session ${sessionId} with outcome: ${outcome.type || 'unknown'}`);
        
        await this.saveTrackingData();
        return session;
    }

    /**
     * Link session to existing conversation thread or create new one
     */
    async linkToConversationThread(session, context) {
        // Try to find related sessions based on project and recent activity
        const relatedSessions = await this.findRelatedSessions(session, context);
        
        if (relatedSessions.length > 0) {
            // Link to existing thread
            const parentSession = relatedSessions[0];
            session.threadId = parentSession.threadId;
            session.parentSessionId = parentSession.id;
            
            // Update parent session
            if (this.sessions.has(parentSession.id)) {
                this.sessions.get(parentSession.id).childSessionIds.push(session.id);
            }

            console.log(`[Session Tracker] Linked session ${session.id} to thread ${session.threadId}`);
        } else {
            // Create new conversation thread
            const threadId = this.generateThreadId();
            session.threadId = threadId;

            const thread = {
                id: threadId,
                createdAt: new Date().toISOString(),
                projectContext: session.projectContext,
                sessionIds: [session.id],
                topics: new Set(),
                outcomes: [],
                status: 'active'
            };

            this.conversationThreads.set(threadId, thread);
            console.log(`[Session Tracker] Created new conversation thread ${threadId}`);
        }
    }

    /**
     * Find related sessions for conversation threading
     */
    async findRelatedSessions(session, context) {
        const projectName = context.projectContext?.name;
        if (!projectName) {
            return [];
        }

        const projectSessionIds = this.projectSessions.get(projectName) || [];
        const relatedSessions = [];

        // Look for recent sessions in same project
        const cutoffTime = new Date();
        cutoffTime.setHours(cutoffTime.getHours() - 24); // 24 hour window

        for (const sessionId of projectSessionIds.slice(-10)) { // Check last 10 sessions
            const session = this.sessions.get(sessionId);
            if (!session || session.status === 'active') continue;

            const sessionTime = new Date(session.endTime || session.startTime);
            if (sessionTime > cutoffTime) {
                // Calculate relatedness score
                const relatednessScore = this.calculateSessionRelatedness(session, context);
                if (relatednessScore > 0.3) {
                    relatedSessions.push({
                        ...session,
                        relatednessScore
                    });
                }
            }
        }

        // Sort by relatedness score
        return relatedSessions.sort((a, b) => b.relatednessScore - a.relatednessScore);
    }

    /**
     * Calculate how related two sessions are
     */
    calculateSessionRelatedness(existingSession, newContext) {
        let score = 0;

        // Same project bonus
        if (existingSession.projectContext?.name === newContext.projectContext?.name) {
            score += 0.4;
        }

        // Same working directory bonus
        if (existingSession.workingDirectory === newContext.workingDirectory) {
            score += 0.3;
        }

        // Technology stack similarity
        const existingTech = [
            ...(existingSession.projectContext?.languages || []),
            ...(existingSession.projectContext?.frameworks || [])
        ];
        const newTech = [
            ...(newContext.projectContext?.languages || []),
            ...(newContext.projectContext?.frameworks || [])
        ];

        const techOverlap = existingTech.filter(tech => newTech.includes(tech)).length;
        if (existingTech.length > 0) {
            score += (techOverlap / existingTech.length) * 0.3;
        }

        return Math.min(score, 1.0);
    }

    /**
     * Update conversation thread with session information
     */
    async updateConversationThread(threadId, session) {
        const thread = this.conversationThreads.get(threadId);
        if (!thread) {
            console.warn(`[Session Tracker] Thread ${threadId} not found`);
            return;
        }

        // Add session to thread if not already present
        if (!thread.sessionIds.includes(session.id)) {
            thread.sessionIds.push(session.id);
        }

        // Update thread topics
        if (session.finalTopics && session.finalTopics.length > 0) {
            session.finalTopics.forEach(topic => thread.topics.add(topic));
        }

        // Add outcome to thread history
        if (session.outcome) {
            thread.outcomes.push({
                sessionId: session.id,
                outcome: session.outcome,
                timestamp: session.endTime
            });
        }

        thread.lastUpdated = new Date().toISOString();
    }

    /**
     * Get conversation context for a new session
     */
    async getConversationContext(projectContext, options = {}) {
        const {
            maxPreviousSessions = 3,
            maxDaysBack = 7
        } = options;

        const projectName = projectContext?.name;
        if (!projectName) {
            return null;
        }

        const projectSessionIds = this.projectSessions.get(projectName) || [];
        if (projectSessionIds.length === 0) {
            return null;
        }

        // Get recent sessions
        const cutoffTime = new Date();
        cutoffTime.setDate(cutoffTime.getDate() - maxDaysBack);

        const recentSessions = [];
        for (const sessionId of projectSessionIds.slice(-10)) {
            const session = this.sessions.get(sessionId);
            if (!session || session.status === 'active') continue;

            const sessionTime = new Date(session.endTime || session.startTime);
            if (sessionTime > cutoffTime) {
                recentSessions.push(session);
            }
        }

        // Sort by end time and take most recent
        const sortedSessions = recentSessions
            .sort((a, b) => new Date(b.endTime || b.startTime) - new Date(a.endTime || a.startTime))
            .slice(0, maxPreviousSessions);

        if (sortedSessions.length === 0) {
            return null;
        }

        // Build conversation context
        const context = {
            projectName: projectName,
            recentSessions: sortedSessions.map(session => ({
                id: session.id,
                endTime: session.endTime,
                outcome: session.outcome,
                topics: session.finalTopics,
                memoriesCreated: session.memoriesCreated?.length || 0
            })),
            continuityInsights: this.extractContinuityInsights(sortedSessions),
            activeThreads: this.getActiveThreadsForProject(projectName)
        };

        return context;
    }

    /**
     * Extract insights about conversation continuity
     */
    extractContinuityInsights(sessions) {
        const insights = {
            recurringTopics: this.findRecurringTopics(sessions),
            progressionPatterns: this.analyzeProgressionPatterns(sessions),
            uncompletedTasks: this.findUncompletedTasks(sessions)
        };

        return insights;
    }

    /**
     * Find topics that appear across multiple sessions
     */
    findRecurringTopics(sessions) {
        const topicCounts = new Map();
        
        sessions.forEach(session => {
            (session.finalTopics || []).forEach(topic => {
                topicCounts.set(topic, (topicCounts.get(topic) || 0) + 1);
            });
        });

        return Array.from(topicCounts.entries())
            .filter(([topic, count]) => count > 1)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(([topic, count]) => ({ topic, frequency: count }));
    }

    /**
     * Analyze how work progresses across sessions
     */
    analyzeProgressionPatterns(sessions) {
        const patterns = [];
        
        // Look for planning -> implementation -> testing patterns
        const outcomePairs = [];
        for (let i = 0; i < sessions.length - 1; i++) {
            outcomePairs.push([
                sessions[i].outcome?.type,
                sessions[i + 1].outcome?.type
            ]);
        }

        return patterns;
    }

    /**
     * Find tasks or decisions that weren't completed
     */
    findUncompletedTasks(sessions) {
        const tasks = [];
        
        sessions.forEach(session => {
            if (session.outcome?.type === 'planning' || session.outcome?.type === 'partial') {
                tasks.push({
                    sessionId: session.id,
                    description: session.outcome?.summary,
                    timestamp: session.endTime
                });
            }
        });

        return tasks;
    }

    /**
     * Get active conversation threads for a project
     */
    getActiveThreadsForProject(projectName) {
        const threads = [];
        
        this.conversationThreads.forEach((thread, threadId) => {
            if (thread.projectContext?.name === projectName && thread.status === 'active') {
                threads.push({
                    id: threadId,
                    sessionCount: thread.sessionIds.length,
                    topics: Array.from(thread.topics),
                    lastUpdated: thread.lastUpdated
                });
            }
        });

        return threads;
    }

    /**
     * Cleanup expired sessions and threads
     */
    cleanupExpiredSessions() {
        const cutoffTime = new Date();
        cutoffTime.setDate(cutoffTime.getDate() - this.options.sessionExpiryDays);

        let cleanedCount = 0;

        // Cleanup sessions
        for (const [sessionId, session] of this.sessions.entries()) {
            const sessionTime = new Date(session.endTime || session.startTime);
            if (sessionTime < cutoffTime) {
                this.sessions.delete(sessionId);
                cleanedCount++;
            }
        }

        // Cleanup project session references
        this.projectSessions.forEach((sessionIds, projectName) => {
            const validSessions = sessionIds.filter(id => this.sessions.has(id));
            if (validSessions.length !== sessionIds.length) {
                this.projectSessions.set(projectName, validSessions);
            }
        });

        if (cleanedCount > 0) {
            console.log(`[Session Tracker] Cleaned up ${cleanedCount} expired sessions`);
        }
    }

    /**
     * Generate unique thread ID
     */
    generateThreadId() {
        return 'thread-' + crypto.randomBytes(8).toString('hex');
    }

    /**
     * Load tracking data from disk
     */
    async loadTrackingData() {
        try {
            const data = await fs.readFile(this.options.trackingDataPath, 'utf8');
            const parsed = JSON.parse(data);

            // Restore sessions
            if (parsed.sessions) {
                parsed.sessions.forEach(session => {
                    this.sessions.set(session.id, session);
                });
            }

            // Restore conversation threads (convert topics Set back from array)
            if (parsed.conversationThreads) {
                parsed.conversationThreads.forEach(thread => {
                    thread.topics = new Set(thread.topics || []);
                    this.conversationThreads.set(thread.id, thread);
                });
            }

            // Restore project sessions
            if (parsed.projectSessions) {
                Object.entries(parsed.projectSessions).forEach(([project, sessionIds]) => {
                    this.projectSessions.set(project, sessionIds);
                });
            }

        } catch (error) {
            if (error.code !== 'ENOENT') {
                console.warn('[Session Tracker] Failed to load tracking data:', error.message);
            }
            // Initialize empty structures if file doesn't exist
        }
    }

    /**
     * Save tracking data to disk
     */
    async saveTrackingData() {
        try {
            const data = {
                sessions: Array.from(this.sessions.values()),
                conversationThreads: Array.from(this.conversationThreads.values()).map(thread => ({
                    ...thread,
                    topics: Array.from(thread.topics) // Convert Set to Array for JSON
                })),
                projectSessions: Object.fromEntries(this.projectSessions.entries()),
                lastSaved: new Date().toISOString()
            };

            await fs.writeFile(this.options.trackingDataPath, JSON.stringify(data, null, 2));
        } catch (error) {
            console.error('[Session Tracker] Failed to save tracking data:', error.message);
        }
    }

    /**
     * Get statistics about session tracking
     */
    getStats() {
        return {
            totalSessions: this.sessions.size,
            activeSessions: Array.from(this.sessions.values()).filter(s => s.status === 'active').length,
            totalThreads: this.conversationThreads.size,
            trackedProjects: this.projectSessions.size,
            loaded: this.loaded
        };
    }
}

// Create global session tracker instance
let globalSessionTracker = null;

/**
 * Get or create global session tracker instance
 */
function getSessionTracker(options = {}) {
    if (!globalSessionTracker) {
        globalSessionTracker = new SessionTracker(options);
    }
    return globalSessionTracker;
}

module.exports = {
    SessionTracker,
    getSessionTracker
};