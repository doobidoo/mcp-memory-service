/**
 * Claude Code Session Start Hook
 * Automatically injects relevant memories at the beginning of each session
 */

const fs = require('fs').promises;
const path = require('path');
const https = require('https');

// Import utilities
const { detectProjectContext } = require('../utilities/project-detector');
const { scoreMemoryRelevance } = require('../utilities/memory-scorer');
const { formatMemoriesForContext } = require('../utilities/context-formatter');
const { detectContextShift, extractCurrentContext, determineRefreshStrategy } = require('../utilities/context-shift-detector');

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
                endpoint: 'https://narrowbox.local:8443',
                apiKey: 'test-key-123',
                defaultTags: ['claude-code', 'auto-generated'],
                maxMemoriesPerSession: 8,
                injectAfterCompacting: false
            },
            projectDetection: {
                gitRepository: true,
                packageFiles: ['package.json', 'pyproject.toml', 'Cargo.toml'],
                frameworkDetection: true,
                languageDetection: true
            }
        };
    }
}

/**
 * Query memory service for relevant memories
 */
async function queryMemoryService(endpoint, apiKey, query) {
    return new Promise((resolve, reject) => {
        const url = new URL('/mcp', endpoint);
        const postData = JSON.stringify({
            jsonrpc: '2.0',
            id: 1,
            method: 'tools/call',
            params: {
                name: 'retrieve_memory',
                arguments: {
                    query: query.semanticQuery || '',
                    n_results: query.limit || 10
                }
            }
        });

        const options = {
            hostname: url.hostname,
            port: url.port || 8443,
            path: url.pathname,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(postData),
                'Authorization': `Bearer ${apiKey}`
            },
            rejectUnauthorized: false // For self-signed certificates
        };

        const req = https.request(options, (res) => {
            let data = '';
            res.on('data', (chunk) => {
                data += chunk;
            });
            res.on('end', () => {
                try {
                    const response = JSON.parse(data);
                    if (response.result && response.result.content) {
                        // The response.result.content[0].text contains a Python dict format string
                        let textData = response.result.content[0].text;
                        
                        try {
                            // Convert Python dict format to JSON format safely
                            textData = textData
                                .replace(/'/g, '"')  // Replace single quotes with double quotes
                                .replace(/True/g, 'true')  // Convert Python True to JSON true
                                .replace(/False/g, 'false')  // Convert Python False to JSON false
                                .replace(/None/g, 'null');  // Convert Python None to JSON null
                            
                            const memories = JSON.parse(textData);
                            resolve(memories.results || memories.memories || []);
                        } catch (conversionError) {
                            console.warn('[Memory Hook] Could not parse memory response:', conversionError.message);
                            console.warn('[Memory Hook] Raw text preview:', textData.substring(0, 200) + '...');
                            resolve([]);
                        }
                    } else {
                        resolve([]);
                    }
                } catch (parseError) {
                    console.warn('[Memory Hook] Parse error:', parseError.message);
                    resolve([]);
                }
            });
        });

        req.on('error', (error) => {
            console.warn('[Memory Hook] Network error:', error.message);
            resolve([]);
        });

        req.write(postData);
        req.end();
    });
}

/**
 * Main session start hook function with smart timing
 */
async function onSessionStart(context) {
    try {
        console.log('[Memory Hook] Session starting - initializing memory awareness...');
        
        // Load configuration
        const config = await loadConfig();
        
        // Check if this is triggered by a compacting event and skip if configured to do so
        if (context.trigger === 'compacting' || context.event === 'memory-compacted') {
            if (!config.memoryService.injectAfterCompacting) {
                console.log('[Memory Hook] Skipping memory injection after compacting (disabled in config)');
                return;
            }
            console.log('[Memory Hook] Proceeding with memory injection after compacting (enabled in config)');
        }
        
        // For non-session-start events, use smart timing to decide if refresh is needed
        if (context.trigger !== 'session-start' && context.trigger !== 'start') {
            const currentContext = extractCurrentContext(context.conversationState || {}, context.workingDirectory);
            const previousContext = context.previousContext || context.conversationState?.previousContext;
            
            if (previousContext) {
                const shiftDetection = detectContextShift(currentContext, previousContext);
                
                if (!shiftDetection.shouldRefresh) {
                    console.log(`[Memory Hook] No significant context shift detected (${shiftDetection.reason}), skipping refresh`);
                    return;
                }
                
                console.log(`[Memory Hook] Context shift detected: ${shiftDetection.description}`);
            }
        }
        
        // Detect project context
        const projectContext = await detectProjectContext(context.workingDirectory || process.cwd());
        console.log(`[Memory Hook] Detected project: ${projectContext.name} (${projectContext.language})`);
        
        // Build query for relevant memories
        const memoryQuery = {
            tags: [
                projectContext.name,
                `language:${projectContext.language}`,
                'key-decisions',
                'architecture',
                'recent-insights',
                'claude-code-reference'
            ].filter(Boolean),
            semanticQuery: context.userMessage ? 
                `${projectContext.name} ${context.userMessage}` :
                `${projectContext.name} project context decisions architecture`,
            limit: config.memoryService.maxMemoriesPerSession,
            timeFilter: 'last-2-weeks'
        };
        
        // Query memory service
        const memories = await queryMemoryService(
            config.memoryService.endpoint,
            config.memoryService.apiKey,
            memoryQuery
        );
        
        if (memories.length > 0) {
            console.log(`[Memory Hook] Found ${memories.length} relevant memories`);
            
            // Score memories for relevance
            const scoredMemories = scoreMemoryRelevance(memories, projectContext);
            
            // Determine refresh strategy based on context
            const strategy = context.trigger && context.previousContext ? 
                determineRefreshStrategy(detectContextShift(
                    extractCurrentContext(context.conversationState || {}, context.workingDirectory),
                    context.previousContext
                )) : {
                    maxMemories: config.memoryService.maxMemoriesPerSession,
                    includeScore: false,
                    message: '🧠 Loading relevant memory context...'
                };
            
            // Take top scored memories based on strategy
            const maxMemories = Math.min(strategy.maxMemories || config.memoryService.maxMemoriesPerSession, scoredMemories.length);
            const topMemories = scoredMemories.slice(0, maxMemories);
            
            // Format memories for context injection with strategy-based options
            const contextMessage = formatMemoriesForContext(topMemories, projectContext, {
                includeScore: strategy.includeScore || false,
                groupByCategory: maxMemories > 3,
                maxMemories: maxMemories,
                includeTimestamp: true
            });
            
            // Inject context into session
            if (context.injectSystemMessage) {
                await context.injectSystemMessage(contextMessage);
                console.log(`[Memory Hook] Successfully injected memory context (${maxMemories} memories)`);
            } else {
                // Fallback: log context for manual copying
                console.log('\n=== MEMORY CONTEXT FOR SESSION ===');
                console.log(contextMessage);
                console.log('=== END MEMORY CONTEXT ===\n');
            }
        } else {
            console.log('[Memory Hook] No relevant memories found for this project');
        }
        
    } catch (error) {
        console.error('[Memory Hook] Error in session start:', error.message);
        // Fail gracefully - don't prevent session from starting
    }
}

/**
 * Hook metadata for Claude Code
 */
module.exports = {
    name: 'memory-awareness-session-start',
    version: '2.1.0',
    description: 'Automatically inject relevant memories at session start',
    trigger: 'session-start',
    handler: onSessionStart,
    config: {
        async: true,
        timeout: 10000, // 10 second timeout
        priority: 'high'
    }
};

// Direct execution support for testing
if (require.main === module) {
    // Test the hook with mock context
    const mockContext = {
        workingDirectory: process.cwd(),
        sessionId: 'test-session',
        injectSystemMessage: async (message) => {
            console.log('=== INJECTED MESSAGE ===');
            console.log(message);
            console.log('=== END INJECTION ===');
        }
    };
    
    onSessionStart(mockContext)
        .then(() => console.log('Hook test completed'))
        .catch(error => console.error('Hook test failed:', error));
}