/**
 * On-Demand Memory Retrieval Hook
 * Allows users to manually request context refresh when needed
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
        console.warn('[Memory Retrieval] Using default configuration:', error.message);
        return {
            memoryService: {
                endpoint: 'https://narrowbox.local:8443',
                apiKey: 'test-key-123',
                maxMemoriesPerSession: 5
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
                    n_results: query.limit || 5
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
                        let textData = response.result.content[0].text;
                        
                        try {
                            // Convert Python dict format to JSON format safely
                            textData = textData
                                .replace(/'/g, '"')
                                .replace(/True/g, 'true')
                                .replace(/False/g, 'false')
                                .replace(/None/g, 'null');
                            
                            const memories = JSON.parse(textData);
                            resolve(memories.results || memories.memories || []);
                        } catch (conversionError) {
                            console.warn('[Memory Retrieval] Could not parse memory response:', conversionError.message);
                            resolve([]);
                        }
                    } else {
                        resolve([]);
                    }
                } catch (parseError) {
                    console.warn('[Memory Retrieval] Parse error:', parseError.message);
                    resolve([]);
                }
            });
        });

        req.on('error', (error) => {
            console.warn('[Memory Retrieval] Network error:', error.message);
            resolve([]);
        });

        req.write(postData);
        req.end();
    });
}

/**
 * On-demand memory retrieval function
 */
async function retrieveMemories(context) {
    try {
        console.log('[Memory Retrieval] On-demand memory retrieval requested...');
        
        // Load configuration
        const config = await loadConfig();
        
        // Detect project context
        const projectContext = await detectProjectContext(context.workingDirectory || process.cwd());
        console.log(`[Memory Retrieval] Project context: ${projectContext.name} (${projectContext.language})`);
        
        // Parse user query if provided
        const userQuery = context.query || context.message || '';
        
        // Build memory query
        const memoryQuery = {
            tags: [
                projectContext.name,
                `language:${projectContext.language}`,
                'key-decisions',
                'architecture',
                'recent-insights'
            ].filter(Boolean),
            semanticQuery: userQuery.length > 0 ? 
                `${projectContext.name} ${userQuery}` : 
                `${projectContext.name} project context decisions architecture`,
            limit: config.memoryService.maxMemoriesPerSession || 5,
            timeFilter: 'last-month'
        };
        
        // Query memory service
        const memories = await queryMemoryService(
            config.memoryService.endpoint,
            config.memoryService.apiKey,
            memoryQuery
        );
        
        if (memories.length > 0) {
            console.log(`[Memory Retrieval] Found ${memories.length} relevant memories`);
            
            // Score memories for relevance
            const scoredMemories = scoreMemoryRelevance(memories, projectContext);
            
            // Take top scored memories
            const topMemories = scoredMemories.slice(0, config.memoryService.maxMemoriesPerSession || 5);
            
            // Format memories for display
            const contextMessage = formatMemoriesForContext(topMemories, projectContext, {
                includeScore: true, // Show scores for manual retrieval
                groupByCategory: topMemories.length > 3,
                maxMemories: config.memoryService.maxMemoriesPerSession || 5,
                includeTimestamp: true
            });
            
            // Output formatted context
            if (context.displayResult) {
                await context.displayResult(contextMessage);
                console.log('[Memory Retrieval] Successfully displayed memory context');
            } else {
                // Fallback: log context
                console.log('\n=== RETRIEVED MEMORY CONTEXT ===');
                console.log(contextMessage);
                console.log('=== END CONTEXT ===\n');
            }
            
            return {
                success: true,
                memoriesFound: memories.length,
                memoriesShown: topMemories.length,
                context: contextMessage
            };
            
        } else {
            const message = `## 📋 Memory Retrieval\n\nNo relevant memories found for query: "${userQuery || 'project context'}"\n\nTry a different search term or check if your memory service is running.`;
            
            if (context.displayResult) {
                await context.displayResult(message);
            } else {
                console.log(message);
            }
            
            return {
                success: false,
                memoriesFound: 0,
                memoriesShown: 0,
                context: message
            };
        }
        
    } catch (error) {
        console.error('[Memory Retrieval] Error retrieving memories:', error.message);
        const errorMessage = `## ❌ Memory Retrieval Error\n\n${error.message}\n\nCheck your memory service configuration and connection.`;
        
        if (context.displayResult) {
            await context.displayResult(errorMessage);
        }
        
        return {
            success: false,
            error: error.message
        };
    }
}

/**
 * Hook metadata for Claude Code
 */
module.exports = {
    name: 'on-demand-memory-retrieval',
    version: '1.0.0',
    description: 'Retrieve relevant memories on user request',
    trigger: 'manual', // This hook is triggered manually
    handler: retrieveMemories,
    config: {
        async: true,
        timeout: 10000,
        priority: 'normal'
    }
};

// Direct execution support for testing
if (require.main === module) {
    // Test the retrieval with mock context
    const mockContext = {
        workingDirectory: process.cwd(),
        query: 'architecture decisions',
        displayResult: async (message) => {
            console.log('=== MOCK DISPLAY RESULT ===');
            console.log(message);
            console.log('=== END MOCK DISPLAY ===');
        }
    };
    
    retrieveMemories(mockContext)
        .then(result => console.log('Retrieval test completed:', result))
        .catch(error => console.error('Retrieval test failed:', error));
}