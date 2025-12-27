/**
 * Claude Code Session End Hook
 * Automatically consolidates session outcomes and stores them as memories
 */

const fs = require('fs').promises;
const path = require('path');
const https = require('https');
const http = require('http');

// Import utilities
const { detectProjectContext } = require('../utilities/project-detector');
const { formatSessionConsolidation } = require('../utilities/context-formatter');

/**
 * Load hook configuration
 */
async function loadConfig() {
    try {
        const configPath = path.join(__dirname, '../config.json');
        const configData = await fs.readFile(configPath, 'utf8');
        return JSON.parse(configData);
    } catch (error) {
        console.warn('[Memory Hook] Using default configuration:', error.message);
        return {
            memoryService: {
                http: {
                    endpoint: 'http://127.0.0.1:8000',
                    apiKey: 'test-key-123'
                },
                defaultTags: ['claude-code', 'auto-generated'],
                enableSessionConsolidation: true
            },
            sessionAnalysis: {
                extractTopics: true,
                extractDecisions: true,
                extractInsights: true,
                extractCodeChanges: true,
                extractNextSteps: true,
                minSessionLength: 100 // Minimum characters for meaningful session
            }
        };
    }
}

/**
 * Analyze conversation to extract key information
 */
function analyzeConversation(conversationData) {
    try {
        const analysis = {
            topics: [],
            decisions: [],
            insights: [],
            codeChanges: [],
            nextSteps: [],
            sessionLength: 0,
            confidence: 0
        };
        
        if (!conversationData || !conversationData.messages) {
            return analysis;
        }
        
        const messages = conversationData.messages;
        const conversationText = messages.map(msg => msg.content || '').join('\n').toLowerCase();
        analysis.sessionLength = conversationText.length;
        
        // Extract topics (simple keyword matching)
        const topicKeywords = {
            'implementation': /implement|implementing|implementation|build|building|create|creating/g,
            'debugging': /debug|debugging|bug|error|fix|fixing|issue|problem/g,
            'architecture': /architecture|design|structure|pattern|framework|system/g,
            'performance': /performance|optimization|speed|memory|efficient|faster/g,
            'testing': /test|testing|unit test|integration|coverage|spec/g,
            'deployment': /deploy|deployment|production|staging|release/g,
            'configuration': /config|configuration|setup|environment|settings/g,
            'database': /database|db|sql|query|schema|migration/g,
            'api': /api|endpoint|rest|graphql|service|interface/g,
            'ui': /ui|interface|frontend|component|styling|css|html/g
        };
        
        Object.entries(topicKeywords).forEach(([topic, regex]) => {
            if (conversationText.match(regex)) {
                analysis.topics.push(topic);
            }
        });

        // Extract decisions (look for decision language)
        const decisionPatterns = [
            /decided to|decision to|chose to|choosing|will use|going with/g,
            /better to|prefer|recommend|should use|opt for/g,
            /concluded that|determined that|agreed to/g
        ];
        
        messages.forEach(msg => {
            const content = (msg.content || '').toLowerCase();
            decisionPatterns.forEach(pattern => {
                const matches = content.match(pattern);
                if (matches) {
                    // Extract sentences containing decisions
                    const sentences = msg.content.split(/[.!?]+/);
                    sentences.forEach(sentence => {
                        if (pattern.test(sentence.toLowerCase()) && sentence.length > 20) {
                            analysis.decisions.push(sentence.trim());
                        }
                    });
                }
            });
        });
        
        // Extract insights (look for learning language)
        const insightPatterns = [
            /learned that|discovered|realized|found out|turns out/g,
            /insight|understanding|conclusion|takeaway|lesson/g,
            /important to note|key finding|observation/g
        ];
        
        messages.forEach(msg => {
            const content = (msg.content || '').toLowerCase();
            insightPatterns.forEach(pattern => {
                if (pattern.test(content)) {
                    const sentences = msg.content.split(/[.!?]+/);
                    sentences.forEach(sentence => {
                        if (pattern.test(sentence.toLowerCase()) && sentence.length > 20) {
                            analysis.insights.push(sentence.trim());
                        }
                    });
                }
            });
        });
        
        // Extract code changes (look for technical implementations)
        const codePatterns = [
            /added|created|implemented|built|wrote/g,
            /modified|updated|changed|refactored|improved/g,
            /fixed|resolved|corrected|patched/g
        ];
        
        messages.forEach(msg => {
            const content = msg.content || '';
            if (content.includes('```') || /\.(js|py|rs|go|java|cpp|c|ts|jsx|tsx)/.test(content)) {
                // This message contains code
                const lowerContent = content.toLowerCase();
                codePatterns.forEach(pattern => {
                    if (pattern.test(lowerContent)) {
                        const sentences = content.split(/[.!?]+/);
                        sentences.forEach(sentence => {
                            if (pattern.test(sentence.toLowerCase()) && sentence.length > 15) {
                                analysis.codeChanges.push(sentence.trim());
                            }
                        });
                    }
                });
            }
        });
        
        // Extract next steps (look for future language)
        const nextStepsPatterns = [
            /next|todo|need to|should|will|plan to|going to/g,
            /follow up|continue|proceed|implement next|work on/g,
            /remaining|still need|outstanding|future/g
        ];
        
        messages.forEach(msg => {
            const content = (msg.content || '').toLowerCase();
            nextStepsPatterns.forEach(pattern => {
                if (pattern.test(content)) {
                    const sentences = msg.content.split(/[.!?]+/);
                    sentences.forEach(sentence => {
                        if (pattern.test(sentence.toLowerCase()) && sentence.length > 15) {
                            analysis.nextSteps.push(sentence.trim());
                        }
                    });
                }
            });
        });
        
        // Calculate confidence based on extracted information
        const totalExtracted = analysis.topics.length + analysis.decisions.length + 
                              analysis.insights.length + analysis.codeChanges.length + 
                              analysis.nextSteps.length;
        
        analysis.confidence = Math.min(1.0, totalExtracted / 10); // Max confidence at 10+ items
        
        // Limit arrays to prevent overwhelming output
        // Topics: no limit needed (max 10 possible keywords)
        analysis.decisions = analysis.decisions.slice(0, 3);
        analysis.insights = analysis.insights.slice(0, 3);
        analysis.codeChanges = analysis.codeChanges.slice(0, 4);
        analysis.nextSteps = analysis.nextSteps.slice(0, 4);
        
        return analysis;
        
    } catch (error) {
        console.error('[Memory Hook] Error analyzing conversation:', error.message);
        return {
            topics: [],
            decisions: [],
            insights: [],
            codeChanges: [],
            nextSteps: [],
            sessionLength: 0,
            confidence: 0,
            error: error.message
        };
    }
}

/**
 * Trigger quality evaluation for a stored memory (async, non-blocking)
 * This calls the backend's quality scoring system to pre-score the memory
 */
function triggerQualityEvaluation(endpoint, apiKey, contentHash) {
    return new Promise((resolve, reject) => {
        const url = new URL(`/api/quality/memories/${contentHash}/evaluate`, endpoint);
        const isHttps = url.protocol === 'https:';
        const requestModule = isHttps ? https : http;

        const postData = JSON.stringify({});

        const options = {
            hostname: url.hostname,
            port: url.port || (isHttps ? 8443 : 8000),
            path: url.pathname,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(postData),
                'Authorization': `Bearer ${apiKey}`
            },
            timeout: 10000 // 10 second timeout for quality evaluation
        };

        if (isHttps) {
            options.rejectUnauthorized = false;
        }

        const req = requestModule.request(options, (res) => {
            let data = '';
            res.on('data', (chunk) => {
                data += chunk;
            });
            res.on('end', () => {
                try {
                    const response = JSON.parse(data);
                    resolve(response);
                } catch (parseError) {
                    resolve({ success: false, error: 'Parse error', data });
                }
            });
        });

        req.on('error', (error) => {
            resolve({ success: false, error: error.message });
        });

        req.on('timeout', () => {
            req.destroy();
            resolve({ success: false, error: 'Quality evaluation timed out' });
        });

        req.write(postData);
        req.end();
    });
}

/**
 * Store session consolidation to memory service
 */
function storeSessionMemory(endpoint, apiKey, content, projectContext, analysis) {
    return new Promise((resolve, reject) => {
        const url = new URL('/api/memories', endpoint);
        const isHttps = url.protocol === 'https:';
        const requestModule = isHttps ? https : http;

        // Generate tags based on analysis and project context
        const tags = [
            'claude-code-session',
            'session-consolidation',
            projectContext.name,
            `language:${projectContext.language}`,
            ...analysis.topics.slice(0, 3), // Top 3 topics as tags
            ...projectContext.frameworks.slice(0, 2), // Top 2 frameworks
            `confidence:${Math.round(analysis.confidence * 100)}`
        ].filter(Boolean);

        const postData = JSON.stringify({
            content: content,
            tags: tags,
            memory_type: 'session-summary',
            metadata: {
                session_analysis: {
                    topics: analysis.topics,
                    decisions_count: analysis.decisions.length,
                    insights_count: analysis.insights.length,
                    code_changes_count: analysis.codeChanges.length,
                    next_steps_count: analysis.nextSteps.length,
                    session_length: analysis.sessionLength,
                    confidence: analysis.confidence
                },
                project_context: {
                    name: projectContext.name,
                    language: projectContext.language,
                    frameworks: projectContext.frameworks
                },
                generated_by: 'claude-code-session-end-hook',
                generated_at: new Date().toISOString()
            }
        });

        const options = {
            hostname: url.hostname,
            port: url.port || (isHttps ? 8443 : 8000),
            path: url.pathname,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(postData),
                'Authorization': `Bearer ${apiKey}`
            }
        };

        // Only set rejectUnauthorized for HTTPS
        if (isHttps) {
            options.rejectUnauthorized = false; // For self-signed certificates
        }

        const req = requestModule.request(options, (res) => {
            let data = '';
            res.on('data', (chunk) => {
                data += chunk;
            });
            res.on('end', () => {
                try {
                    const response = JSON.parse(data);
                    resolve(response);
                } catch (parseError) {
                    resolve({ success: false, error: 'Parse error', data });
                }
            });
        });

        req.on('error', (error) => {
            resolve({ success: false, error: error.message });
        });

        req.write(postData);
        req.end();
    });
}

/**
 * Main session end hook function
 */
async function onSessionEnd(context) {
    try {
        console.log('[Memory Hook] Session ending - consolidating outcomes...');
        
        // Load configuration
        const config = await loadConfig();
        
        if (!config.memoryService.enableSessionConsolidation) {
            console.log('[Memory Hook] Session consolidation disabled in config');
            return;
        }
        
        // Check if session is meaningful enough to store
        if (context.conversation && context.conversation.messages) {
            const totalLength = context.conversation.messages
                .map(msg => (msg.content || '').length)
                .reduce((sum, len) => sum + len, 0);
                
            if (totalLength < config.sessionAnalysis.minSessionLength) {
                console.log('[Memory Hook] Session too short for consolidation');
                return;
            }
        }
        
        // Detect project context
        const projectContext = await detectProjectContext(context.workingDirectory || process.cwd());
        console.log(`[Memory Hook] Consolidating session for project: ${projectContext.name}`);
        
        // Analyze conversation
        const analysis = analyzeConversation(context.conversation);
        
        if (analysis.confidence < 0.1) {
            console.log('[Memory Hook] Session analysis confidence too low, skipping consolidation');
            return;
        }
        
        console.log(`[Memory Hook] Session analysis: ${analysis.topics.length} topics, ${analysis.decisions.length} decisions, confidence: ${(analysis.confidence * 100).toFixed(1)}%`);
        
        // Format session consolidation
        const consolidation = formatSessionConsolidation(analysis, projectContext);

        // Get endpoint and apiKey from new config structure
        const endpoint = config.memoryService?.http?.endpoint || config.memoryService?.endpoint || 'http://127.0.0.1:8000';
        const apiKey = config.memoryService?.http?.apiKey || config.memoryService?.apiKey || 'test-key-123';

        // Store to memory service
        const result = await storeSessionMemory(
            endpoint,
            apiKey,
            consolidation,
            projectContext,
            analysis
        );
        
        if (result.success || result.content_hash) {
            console.log(`[Memory Hook] Session consolidation stored successfully`);
            if (result.content_hash) {
                console.log(`[Memory Hook] Memory hash: ${result.content_hash.substring(0, 8)}...`);

                // Trigger async quality evaluation (non-blocking)
                triggerQualityEvaluation(endpoint, apiKey, result.content_hash)
                    .then(evalResult => {
                        if (evalResult.success) {
                            console.log(`[Memory Hook] Quality evaluated: ${evalResult.quality_score?.toFixed(3)} (${evalResult.quality_provider})`);
                        }
                    })
                    .catch(err => {
                        // Don't fail the hook if quality evaluation fails
                        console.warn('[Memory Hook] Quality evaluation skipped:', err.message);
                    });
            }
        } else {
            console.warn('[Memory Hook] Failed to store session consolidation:', result.error || 'Unknown error');
        }
        
    } catch (error) {
        console.error('[Memory Hook] Error in session end:', error.message);
        // Fail gracefully - don't prevent session from ending
    }
}

/**
 * Hook metadata for Claude Code
 */
module.exports = {
    name: 'memory-awareness-session-end',
    version: '1.0.0',
    description: 'Automatically consolidate and store session outcomes',
    trigger: 'session-end',
    handler: onSessionEnd,
    config: {
        async: true,
        timeout: 15000, // 15 second timeout
        priority: 'normal'
    },
    // Exported for testing
    _internal: {
        parseTranscript: null,  // Will be set after function definition
        analyzeConversation
    }
};

/**
 * Read JSON context from stdin (provided by Claude Code)
 * Returns: { transcript_path, reason, cwd, session_id, ... }
 */
async function readStdinContext() {
    return new Promise((resolve, reject) => {
        let data = '';

        // Set a timeout in case stdin is empty or never closes
        const timeout = setTimeout(() => {
            resolve(null); // No stdin data - likely manual test run
        }, 100);

        process.stdin.setEncoding('utf8');
        process.stdin.on('readable', () => {
            let chunk;
            while ((chunk = process.stdin.read()) !== null) {
                data += chunk;
            }
        });

        process.stdin.on('end', () => {
            clearTimeout(timeout);
            if (data.trim()) {
                try {
                    resolve(JSON.parse(data));
                } catch (error) {
                    console.error('[Memory Hook] Failed to parse stdin JSON:', error.message);
                    reject(error);
                }
            } else {
                resolve(null);
            }
        });

        process.stdin.on('error', (error) => {
            clearTimeout(timeout);
            console.error('[Memory Hook] Stdin error:', error.message);
            reject(error);
        });
    });
}

/**
 * Parse JSONL transcript file to extract conversation messages
 * @param {string} transcriptPath - Path to the .jsonl transcript file
 * @returns {Object} - { messages: Array<{role, content}> }
 */
async function parseTranscript(transcriptPath) {
    try {
        const content = await fs.readFile(transcriptPath, 'utf8');
        const lines = content.trim().split('\n');
        const messages = [];

        for (const line of lines) {
            if (!line.trim()) continue;

            try {
                const entry = JSON.parse(line);

                // Only process user and assistant messages
                if (entry.type === 'user' || entry.type === 'assistant') {
                    const msg = entry.message;
                    if (msg && msg.role && msg.content) {
                        // Handle content that can be string or array of content blocks
                        let contentText = '';
                        if (typeof msg.content === 'string') {
                            contentText = msg.content;
                        } else if (Array.isArray(msg.content)) {
                            // Extract text from content blocks
                            contentText = msg.content
                                .filter(block => block.type === 'text')
                                .map(block => block.text)
                                .join('\n');
                        }

                        if (contentText) {
                            messages.push({
                                role: msg.role,
                                content: contentText
                            });
                        }
                    }
                }
            } catch (parseError) {
                // Skip malformed lines
                continue;
            }
        }

        return { messages };
    } catch (error) {
        console.error('[Memory Hook] Failed to parse transcript:', error.message);
        return { messages: [] };
    }
}

// Set parseTranscript on exports for testing (after function is defined)
module.exports._internal.parseTranscript = parseTranscript;

/**
 * Mock conversation for manual testing (when no stdin/transcript available)
 */
const mockConversation = {
    messages: [
        {
            role: 'user',
            content: 'I need to implement a memory awareness system for Claude Code'
        },
        {
            role: 'assistant',
            content: 'I\'ll help you create a memory awareness system. We decided to use hooks for session management and implement automatic context injection.'
        },
        {
            role: 'user',
            content: 'Great! I learned that we need project detection and memory scoring algorithms.'
        },
        {
            role: 'assistant',
            content: 'Exactly. I implemented the project detector in project-detector.js and created scoring algorithms. Next we need to test the complete system.'
        }
    ]
};

// Direct execution - reads stdin context from Claude Code
if (require.main === module) {
    (async () => {
        try {
            // Read context from stdin (Claude Code provides this)
            const stdinContext = await readStdinContext();

            let context;

            if (stdinContext && stdinContext.transcript_path) {
                // Real execution: parse transcript file
                console.log(`[Memory Hook] Reading transcript: ${stdinContext.transcript_path}`);
                console.log(`[Memory Hook] Session end reason: ${stdinContext.reason || 'unknown'}`);

                const conversation = await parseTranscript(stdinContext.transcript_path);

                context = {
                    workingDirectory: stdinContext.cwd || process.cwd(),
                    sessionId: stdinContext.session_id || 'unknown',
                    reason: stdinContext.reason,
                    conversation: conversation
                };

                console.log(`[Memory Hook] Parsed ${conversation.messages.length} messages from transcript`);
            } else {
                // Manual test: use mock data
                console.log('[Memory Hook] No stdin context - using mock data for testing');
                context = {
                    workingDirectory: process.cwd(),
                    sessionId: 'test-session',
                    conversation: mockConversation
                };
            }

            await onSessionEnd(context);
            console.log('Session end hook completed');

        } catch (error) {
            console.error('Session end hook failed:', error);
            process.exit(1);
        }
    })();
}