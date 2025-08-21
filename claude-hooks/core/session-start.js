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
                maxMemoriesPerSession: 8
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
                            // Replace Python literals with JS equivalents 
                            const jsCode = textData
                                .replace(/True/g, 'true')
                                .replace(/False/g, 'false')
                                .replace(/None/g, 'null');
                            
                            // Use Function constructor for safe evaluation
                            const memoriesData = new Function('return ' + jsCode)();
                            resolve(memoriesData.results || []);
                        } catch (conversionError) {
                            console.warn('[Memory Hook] Response format conversion error:', conversionError.message);
                            // Fallback: try direct JSON parsing
                            try {
                                const memoriesData = JSON.parse(textData);
                                resolve(memoriesData.results || memoriesData.memories || []);
                            } catch (jsonError) {
                                console.warn('[Memory Hook] JSON fallback failed:', jsonError.message);
                                resolve([]);
                            }
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
 * Main session start hook function
 */
async function onSessionStart(context) {
    try {
        console.log('[Memory Hook] Session starting - initializing memory awareness...');
        
        // Load configuration
        const config = await loadConfig();
        
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
            semanticQuery: `${projectContext.name} project context decisions architecture`,
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
            
            // Take top scored memories
            const topMemories = scoredMemories.slice(0, config.memoryService.maxMemoriesPerSession);
            
            // Format memories for context injection
            const contextMessage = formatMemoriesForContext(topMemories, projectContext);
            
            // Inject context into session
            if (context.injectSystemMessage) {
                await context.injectSystemMessage(contextMessage);
                console.log('[Memory Hook] Successfully injected memory context');
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
    version: '1.0.0',
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