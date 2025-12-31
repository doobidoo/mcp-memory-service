#!/usr/bin/env node
/**
 * HTTP-to-MCP Bridge for MCP Memory Service - FIXED VERSION
 * 
 * This version uses the CORRECT API endpoints from v8.62.9
 * 
 * Fixed endpoints:
 * - POST /api/search (was GET /search)
 * - POST /api/search/by-tag (was GET /memories/search/tags)  
 * - POST /api/search/by-time (NEW)
 * - PUT /api/memories/{hash} (NEW - update operation)
 *
 * Usage in Claude Desktop config:
 * {
 *   "mcpServers": {
 *     "memory": {
 *       "command": "node",
 *       "args": ["/path/to/http-mcp-bridge-FIXED.js"],
 *       "env": {
 *         "MCP_MEMORY_HTTP_ENDPOINT": "https://memory.timkjr.gleeze.com/api",
 *         "MCP_MEMORY_API_KEY": "your-api-key"
 *       }
 *     }
 *   }
 * }
 */

const http = require('http');
const https = require('https');
const { URL } = require('url');

class MCPBridge {
    constructor(endpoint, apiKey) {
        this.endpoint = endpoint;
        this.apiKey = apiKey;
        
        // Parse endpoint URL
        this.endpointUrl = new URL(endpoint);
        this.protocol = this.endpointUrl.protocol === 'https:' ? https : http;
        
        console.error(`MCP Bridge initialized with endpoint: ${endpoint}`);
    }

    /**
     * Make HTTP request to the Memory Service API
     */
    async makeRequest(path, method, body = null) {
        return new Promise((resolve, reject) => {
            // Build full URL - ensure endpoint ends with / and path doesn't start with /
            let baseUrl = this.endpoint;
            if (!baseUrl.endsWith('/')) {
                baseUrl += '/';
            }
            let apiPath = path;
            if (apiPath.startsWith('/')) {
                apiPath = apiPath.substring(1);
            }
            const url = new URL(apiPath, baseUrl);
            
            const options = {
                hostname: url.hostname,
                port: url.port || (this.protocol === https ? 443 : 80),
                path: url.pathname + url.search,
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.apiKey}`
                },
                // Allow self-signed certificates
                rejectUnauthorized: false
            };

            if (body && method !== 'GET') {
                const bodyStr = JSON.stringify(body);
                options.headers['Content-Length'] = Buffer.byteLength(bodyStr);
            }

            console.error(`Making ${method} request to ${url.pathname + url.search}`);

            const req = this.protocol.request(options, (res) => {
                let data = '';

                res.on('data', (chunk) => {
                    data += chunk;
                });

                res.on('end', () => {
                    try {
                        const parsed = JSON.parse(data);
                        resolve({
                            statusCode: res.statusCode,
                            data: parsed
                        });
                    } catch (error) {
                        reject(new Error(`Failed to parse response: ${error.message}`));
                    }
                });
            });

            req.on('error', (error) => {
                reject(error);
            });

            if (body && method !== 'GET') {
                req.write(JSON.stringify(body));
            }

            req.end();
        });
    }

    /**
     * Store memory - POST /api/memories
     */
    async storeMemory(params) {
        try {
            const response = await this.makeRequest('memories', 'POST', {
                content: params.content,
                tags: params.metadata?.tags || params.tags || [],
                memory_type: params.metadata?.type || params.memory_type || 'note',
                metadata: params.metadata || {}
            });

            if (response.statusCode === 200 || response.statusCode === 201) {
                if (response.data.success) {
                    return { 
                        success: true, 
                        message: response.data.message || 'Memory stored successfully',
                        content_hash: response.data.content_hash
                    };
                } else {
                    return { success: false, message: response.data.message || response.data.detail || 'Failed to store memory' };
                }
            } else {
                return { success: false, message: response.data.detail || 'Failed to store memory' };
            }
        } catch (error) {
            return { success: false, message: error.message };
        }
    }

    /**
     * Retrieve memory (semantic search) - POST /api/search
     */
    async retrieveMemory(params) {
        try {
            const response = await this.makeRequest('search', 'POST', {
                query: params.query,
                n_results: params.n_results || 10,
                similarity_threshold: params.similarity_threshold || null
            });

            if (response.statusCode === 200) {
                return {
                    results: response.data.results.map(result => ({
                        content: result.memory.content,
                        content_hash: result.memory.content_hash,
                        similarity_score: result.similarity_score,
                        metadata: {
                            tags: result.memory.tags,
                            type: result.memory.memory_type,
                            created_at: result.memory.created_at_iso
                        }
                    })),
                    total_found: response.data.total_found
                };
            } else {
                return { results: [], error: response.data.detail || 'Search failed' };
            }
        } catch (error) {
            return { results: [], error: error.message };
        }
    }

    /**
     * Search by tag - POST /api/search/by-tag
     */
    async searchByTag(params) {
        try {
            const tags = Array.isArray(params.tags) ? params.tags : [params.tags];
            
            const response = await this.makeRequest('search/by-tag', 'POST', {
                tags: tags,
                match_all: params.match_all || false,
                time_filter: params.time_filter || null
            });

            if (response.statusCode === 200) {
                return {
                    results: response.data.results.map(result => ({
                        content: result.memory.content,
                        content_hash: result.memory.content_hash,
                        metadata: {
                            tags: result.memory.tags,
                            type: result.memory.memory_type,
                            created_at: result.memory.created_at_iso
                        }
                    })),
                    total_found: response.data.total_found
                };
            } else {
                return { results: [], error: response.data.detail || 'Tag search failed' };
            }
        } catch (error) {
            return { results: [], error: error.message };
        }
    }

    /**
     * Recall memory (time-based) - POST /api/search/by-time
     */
    async recallMemory(params) {
        try {
            const response = await this.makeRequest('search/by-time', 'POST', {
                query: params.query || params.time_filter,
                n_results: params.n_results || 10,
                semantic_query: params.semantic_query || null
            });

            if (response.statusCode === 200) {
                return {
                    results: response.data.results.map(result => ({
                        content: result.memory.content,
                        content_hash: result.memory.content_hash,
                        metadata: {
                            tags: result.memory.tags,
                            type: result.memory.memory_type,
                            created_at: result.memory.created_at_iso
                        }
                    })),
                    total_found: response.data.total_found
                };
            } else {
                return { results: [], error: response.data.detail || 'Time search failed' };
            }
        } catch (error) {
            return { results: [], error: error.message };
        }
    }

    /**
     * Update memory - PUT /api/memories/{content_hash}
     */
    async updateMemory(params) {
        try {
            const response = await this.makeRequest(`memories/${params.content_hash}`, 'PUT', {
                tags: params.tags || null,
                memory_type: params.memory_type || null,
                metadata: params.metadata || null
            });

            if (response.statusCode === 200) {
                if (response.data.success) {
                    return {
                        success: true,
                        message: response.data.message || 'Memory updated successfully',
                        content_hash: response.data.content_hash
                    };
                } else {
                    return { success: false, message: response.data.message || 'Update failed' };
                }
            } else {
                return { success: false, message: response.data.detail || 'Update failed' };
            }
        } catch (error) {
            return { success: false, message: error.message };
        }
    }

    /**
     * Delete memory - DELETE /api/memories/{content_hash}
     */
    async deleteMemory(params) {
        try {
            const response = await this.makeRequest(`memories/${params.content_hash}`, 'DELETE');

            if (response.statusCode === 200) {
                return {
                    success: response.data.success,
                    message: response.data.message || 'Memory deleted successfully'
                };
            } else {
                return { success: false, message: response.data.detail || 'Delete failed' };
            }
        } catch (error) {
            return { success: false, message: error.message };
        }
    }

    /**
     * List memories - GET /api/memories
     */
    async listMemories(params) {
        try {
            const queryParams = new URLSearchParams({
                page: params.page || 1,
                page_size: params.page_size || 10
            });

            if (params.tag) queryParams.append('tag', params.tag);
            if (params.memory_type) queryParams.append('memory_type', params.memory_type);

            const response = await this.makeRequest(`memories?${queryParams}`, 'GET');

            if (response.statusCode === 200) {
                return {
                    memories: response.data.memories.map(memory => ({
                        content: memory.content,
                        content_hash: memory.content_hash,
                        metadata: {
                            tags: memory.tags,
                            type: memory.memory_type,
                            created_at: memory.created_at_iso
                        }
                    })),
                    total: response.data.total,
                    page: response.data.page,
                    has_more: response.data.has_more
                };
            } else {
                return { memories: [], total: 0, error: response.data.detail };
            }
        } catch (error) {
            return { memories: [], total: 0, error: error.message };
        }
    }

    /**
     * Check database health - GET /api/health
     */
    async checkHealth(params = {}) {
        try {
            const response = await this.makeRequest('health', 'GET');

            if (response.statusCode === 200) {
                return {
                    status: response.data.status,
                    version: response.data.version,
                    uptime_seconds: response.data.uptime_seconds
                };
            } else {
                return { status: 'unhealthy', error: response.data.detail };
            }
        } catch (error) {
            const errorMessage = error.message || error.code || error.toString() || 'Unknown error';
            return { status: 'error', error: errorMessage };
        }
    }

    /**
     * Handle MCP protocol messages
     */
    async handleMessage(message) {
        const { method, params = {}, id } = message;

        console.error(`Handling MCP method: ${method}`);

        let result;
        switch (method) {
            case 'initialize':
                result = {
                    protocolVersion: "2024-11-05",
                    capabilities: {
                        tools: {}
                    },
                    serverInfo: {
                        name: "memory-service-bridge",
                        version: "1.0.0-fixed"
                    }
                };
                break;

            case 'tools/list':
                result = {
                    tools: [
                        {
                            name: "store_memory",
                            description: "Store a new memory with content, tags, and type",
                            inputSchema: {
                                type: "object",
                                properties: {
                                    content: { type: "string", description: "Memory content" },
                                    tags: { type: "array", items: { type: "string" }, description: "Tags for categorization" },
                                    memory_type: { type: "string", description: "Type of memory" }
                                },
                                required: ["content"]
                            }
                        },
                        {
                            name: "retrieve_memory",
                            description: "Search memories using semantic similarity",
                            inputSchema: {
                                type: "object",
                                properties: {
                                    query: { type: "string", description: "Search query" },
                                    n_results: { type: "integer", description: "Number of results to return" }
                                },
                                required: ["query"]
                            }
                        },
                        {
                            name: "search_by_tag",
                            description: "Search memories by specific tags",
                            inputSchema: {
                                type: "object",
                                properties: {
                                    tags: { 
                                        oneOf: [
                                            { type: "string" },
                                            { type: "array", items: { type: "string" } }
                                        ],
                                        description: "Tag or tags to search for"
                                    },
                                    match_all: { type: "boolean", description: "Require all tags (AND) vs any tag (OR)" }
                                },
                                required: ["tags"]
                            }
                        },
                        {
                            name: "recall_memory",
                            description: "Recall memories from a specific time period",
                            inputSchema: {
                                type: "object",
                                properties: {
                                    query: { type: "string", description: "Time expression like 'yesterday', 'last week'" },
                                    n_results: { type: "integer", description: "Number of results to return" }
                                },
                                required: ["query"]
                            }
                        },
                        {
                            name: "update_memory",
                            description: "Update tags or type of an existing memory",
                            inputSchema: {
                                type: "object",
                                properties: {
                                    content_hash: { type: "string", description: "Hash of memory to update" },
                                    tags: { type: "array", items: { type: "string" }, description: "New tags" },
                                    memory_type: { type: "string", description: "New memory type" }
                                },
                                required: ["content_hash"]
                            }
                        },
                        {
                            name: "delete_memory",
                            description: "Delete a memory by content hash",
                            inputSchema: {
                                type: "object",
                                properties: {
                                    content_hash: { type: "string", description: "Hash of memory to delete" }
                                },
                                required: ["content_hash"]
                            }
                        },
                        {
                            name: "list_memories",
                            description: "List all memories with pagination",
                            inputSchema: {
                                type: "object",
                                properties: {
                                    page: { type: "integer", description: "Page number" },
                                    page_size: { type: "integer", description: "Items per page" }
                                }
                            }
                        },
                        {
                            name: "check_database_health",
                            description: "Check the health of the memory database",
                            inputSchema: {
                                type: "object",
                                properties: {}
                            }
                        }
                    ]
                };
                break;

            case 'tools/call':
                const toolName = params.name;
                const toolParams = params.arguments || {};

                console.error(`Calling tool: ${toolName} with params:`, JSON.stringify(toolParams));

                let toolResult;
                switch (toolName) {
                    case 'store_memory':
                        toolResult = await this.storeMemory(toolParams);
                        break;
                    case 'retrieve_memory':
                        toolResult = await this.retrieveMemory(toolParams);
                        break;
                    case 'search_by_tag':
                        toolResult = await this.searchByTag(toolParams);
                        break;
                    case 'recall_memory':
                        toolResult = await this.recallMemory(toolParams);
                        break;
                    case 'update_memory':
                        toolResult = await this.updateMemory(toolParams);
                        break;
                    case 'delete_memory':
                        toolResult = await this.deleteMemory(toolParams);
                        break;
                    case 'list_memories':
                        toolResult = await this.listMemories(toolParams);
                        break;
                    case 'check_database_health':
                        toolResult = await this.checkHealth(toolParams);
                        break;
                    default:
                        throw new Error(`Unknown tool: ${toolName}`);
                }

                console.error(`Tool result:`, JSON.stringify(toolResult));

                return {
                    jsonrpc: "2.0",
                    id: id,
                    result: {
                        content: [
                            {
                                type: "text",
                                text: JSON.stringify(toolResult, null, 2)
                            }
                        ]
                    }
                };

            case 'notifications/initialized':
                return null; // No response needed

            default:
                throw new Error(`Unknown method: ${method}`);
        }

        return {
            jsonrpc: "2.0",
            id: id,
            result: result
        };
    }

    /**
     * Run the bridge
     */
    async run() {
        console.error('MCP Bridge starting (FIXED version with correct API endpoints)...');

        let inputBuffer = '';

        process.stdin.on('data', async (chunk) => {
            inputBuffer += chunk.toString();

            // Process complete lines
            let newlineIndex;
            while ((newlineIndex = inputBuffer.indexOf('\n')) !== -1) {
                const line = inputBuffer.slice(0, newlineIndex);
                inputBuffer = inputBuffer.slice(newlineIndex + 1);

                if (line.trim()) {
                    try {
                        const message = JSON.parse(line);
                        const response = await this.handleMessage(message);

                        if (response !== null) {
                            process.stdout.write(JSON.stringify(response) + '\n');
                        }
                    } catch (error) {
                        console.error('Error processing message:', error);
                        const errorResponse = {
                            jsonrpc: "2.0",
                            id: null,
                            error: {
                                code: -32603,
                                message: error.message
                            }
                        };
                        process.stdout.write(JSON.stringify(errorResponse) + '\n');
                    }
                }
            }
        });

        process.stdin.on('end', () => {
            console.error('MCP Bridge stopping...');
            process.exit(0);
        });
    }
}

// Main
const endpoint = process.env.MCP_MEMORY_HTTP_ENDPOINT || 'http://localhost:8000/api';
const apiKey = process.env.MCP_MEMORY_API_KEY || '';

if (!apiKey) {
    console.error('WARNING: No API key provided. Set MCP_MEMORY_API_KEY environment variable.');
}

const bridge = new MCPBridge(endpoint, apiKey);
bridge.run().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
});
