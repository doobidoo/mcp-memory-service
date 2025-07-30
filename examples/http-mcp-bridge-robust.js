#!/usr/bin/env node
/**
 * Robust HTTP-to-MCP Bridge for MCP Memory Service
 * 
 * Production-ready bridge that connects MCP clients (like Claude Desktop) to remote
 * HTTP memory services. Features comprehensive error handling, retry logic, and
 * support for various memory operations.
 * 
 * @version 1.0.0
 * @author MCP Memory Service Team
 * @license Apache-2.0
 * 
 * Environment Variables:
 * - MCP_MEMORY_HTTP_ENDPOINT: Remote server API endpoint (required)
 * - MCP_MEMORY_API_KEY: Authentication API key (optional)
 * - NODE_TLS_REJECT_UNAUTHORIZED: Set to "0" to disable cert validation
 */

const http = require('http');
const https = require('https');
const { URL } = require('url');

/**
 * Robust MCP Bridge class that handles HTTP-to-MCP protocol translation
 * with comprehensive error handling and retry logic.
 */
class RobustMCPBridge {
    /**
     * Initialize the bridge with configuration from environment variables
     */
    constructor() {
        this.endpoint = process.env.MCP_MEMORY_HTTP_ENDPOINT;
        
        // Validate required configuration
        if (!this.endpoint) {
            throw new Error('MCP_MEMORY_HTTP_ENDPOINT environment variable is required');
        }
        
        // Ensure endpoint ends with slash for proper URL construction
        if (!this.endpoint.endsWith('/')) {
            this.endpoint += '/';
        }
        
        this.apiKey = process.env.MCP_MEMORY_API_KEY;
        this.requestId = 0;
        
        // Configuration constants
        this.DEFAULT_TIMEOUT = 60000; // 60 seconds
        this.MAX_RETRIES = 3;
        this.BASE_DELAY = 1000; // 1 second
        this.MAX_DELAY = 5000; // 5 seconds
        
        console.error('Robust MCP HTTP Bridge starting...');
        console.error(`Endpoint: ${this.endpoint}`);
        console.error(`API Key: ${this.apiKey ? '[SET]' : '[NOT SET]'}`);
    }

    /**
     * Make HTTP request with retry logic and exponential backoff
     * @param {string} path - API endpoint path
     * @param {string} method - HTTP method (GET, POST, DELETE)
     * @param {Object|null} data - Request body data
     * @param {number} timeout - Request timeout in milliseconds
     * @param {number} maxRetries - Maximum number of retry attempts
     * @returns {Promise<Object>} Response data
     * @throws {Error} If all retry attempts fail
     */
    async makeRequestInternal(path, method = 'GET', data = null, timeout = this.DEFAULT_TIMEOUT, maxRetries = this.MAX_RETRIES) {
        let lastError;
        
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                return await this.singleRequest(path, method, data, timeout);
            } catch (error) {
                lastError = error;
                console.error(`Request attempt ${attempt} failed: ${error.message}`);
                
                if (attempt < maxRetries) {
                    const delay = Math.min(this.BASE_DELAY * Math.pow(2, attempt - 1), this.MAX_DELAY);
                    console.error(`Retrying in ${delay}ms...`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }
        
        throw lastError;
    }

    async singleRequest(path, method = 'GET', data = null, timeout = 60000) {
        return new Promise((resolve, reject) => {
            const url = new URL(path, this.endpoint);
            const protocol = url.protocol === 'https:' ? https : http;
            
            const options = {
                hostname: url.hostname,
                port: url.port,
                path: url.pathname + url.search,
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                    'User-Agent': 'MCP-HTTP-Bridge-Robust/1.0',
                    'Connection': 'keep-alive'
                },
                timeout: timeout,
                // For HTTPS, allow self-signed certificates
                rejectUnauthorized: process.env.NODE_TLS_REJECT_UNAUTHORIZED !== '0'
            };

            if (this.apiKey) {
                options.headers['Authorization'] = `Bearer ${this.apiKey}`;
            }

            if (data) {
                const postData = JSON.stringify(data);
                options.headers['Content-Length'] = Buffer.byteLength(postData);
            }

            const req = protocol.request(options, (res) => {
                let responseData = '';
                
                res.on('data', (chunk) => {
                    responseData += chunk;
                });
                
                res.on('end', () => {
                    try {
                        const parsedData = responseData ? JSON.parse(responseData) : {};
                        resolve({
                            statusCode: res.statusCode,
                            headers: res.headers,
                            data: parsedData
                        });
                    } catch (error) {
                        resolve({
                            statusCode: res.statusCode,
                            headers: res.headers,
                            data: responseData
                        });
                    }
                });
            });

            req.on('error', (error) => {
                reject(new Error(`Network error: ${error.message}`));
            });

            req.on('timeout', () => {
                req.destroy();
                reject(new Error(`Request timeout after ${timeout}ms`));
            });

            if (data) {
                req.write(JSON.stringify(data));
            }
            
            req.end();
        });
    }

    /**
     * Test if an endpoint is healthy
     */
    async testEndpoint(endpoint) {
        try {
            const healthUrl = `${endpoint}/health`;
            const response = await this.makeRequestInternal(healthUrl, 'GET', null, 10000, 2);
            return response.statusCode === 200;
        } catch (error) {
            console.error(`Health check failed: ${error.message}`);
            return false;
        }
    }

    /**
     * Initialize the MCP connection
     */
    async initialize(params) {
        // Use the same protocol version that the client requested
        const protocolVersion = params?.protocolVersion || "2024-11-05";
        
        return {
            protocolVersion: protocolVersion,
            capabilities: {
                tools: {
                    listChanged: true
                }
            },
            serverInfo: {
                name: "mcp-memory-http-bridge-robust", 
                version: "1.0.0"
            }
        };
    }

    /**
     * List available tools
     */
    async listTools(params) {
        return {
            tools: [
                {
                    name: "store_memory",
                    description: "Store new information with optional tags.",
                    inputSchema: {
                        type: "object",
                        properties: {
                            content: {
                                type: "string",
                                description: "The information to store"
                            },
                            tags: {
                                type: "array",
                                items: { type: "string" },
                                description: "Optional tags for organization"
                            }
                        },
                        required: ["content"]
                    }
                },
                {
                    name: "retrieve_memory",
                    description: "Search and retrieve stored information.",
                    inputSchema: {
                        type: "object", 
                        properties: {
                            query: {
                                type: "string",
                                description: "Search query"
                            },
                            n_results: {
                                type: "integer",
                                description: "Number of results to return",
                                default: 5
                            }
                        },
                        required: ["query"]
                    }
                },
                {
                    name: "search_by_tag",
                    description: "Search memories by tag.",
                    inputSchema: {
                        type: "object",
                        properties: {
                            tags: {
                                type: "array",
                                items: { type: "string" },
                                description: "Tags to search for"
                            },
                            n_results: {
                                type: "integer",
                                description: "Number of results to return",
                                default: 10
                            }
                        },
                        required: ["tags"]
                    }
                },
                {
                    name: "delete_memory",
                    description: "Delete specific memories.",
                    inputSchema: {
                        type: "object",
                        properties: {
                            ids: {
                                type: "array",
                                items: { type: "string" },
                                description: "Memory IDs to delete"
                            }
                        },
                        required: ["ids"]
                    }
                },
                {
                    name: "check_database_health", 
                    description: "Check the health status of the memory database.",
                    inputSchema: {
                        type: "object",
                        properties: {}
                    }
                }
            ]
        };
    }

    /**
     * Call a tool
     */
    async callTool(params) {
        const { name, arguments: args } = params;
        console.error(`Calling tool: ${name}`);
        
        let result;
        switch (name) {
            case 'store_memory':
                result = await this.storeMemory(args);
                break;
            case 'retrieve_memory':
                result = await this.retrieveMemory(args);
                break;
            case 'search_by_tag':
                result = await this.searchByTag(args);
                break;
            case 'delete_memory':
                result = await this.deleteMemory(args);
                break;
            case 'check_database_health':
                result = await this.checkDatabaseHealth(args);
                break;
            default:
                throw new Error(`Unknown tool: ${name}`);
        }

        return {
            content: [{
                type: "text",
                text: JSON.stringify(result, null, 2)
            }]
        };
    }

    /**
     * Make a REST API request to the remote server
     */
    async makeRESTRequest(endpoint, method, data) {
        const response = await this.makeRequestInternal(endpoint, method, data);
        if (response.statusCode !== 200) {
            throw new Error(`HTTP ${response.statusCode}: ${JSON.stringify(response.data)}`);
        }
        return response.data;
    }

    /**
     * Store a memory with validation
     * @param {Object} params - Parameters containing content and optional tags
     * @param {string} params.content - The memory content to store
     * @param {string[]} [params.tags] - Optional tags for organization
     * @returns {Promise<Object>} Storage result with content hash
     * @throws {Error} If content is missing or invalid
     */
    async storeMemory(params) {
        if (!params || typeof params.content !== 'string' || !params.content.trim()) {
            throw new Error('Memory content is required and must be a non-empty string');
        }
        
        if (params.tags && !Array.isArray(params.tags)) {
            throw new Error('Tags must be an array of strings');
        }
        
        return await this.makeRESTRequest('memories', 'POST', params);
    }

    /**
     * Retrieve memories using semantic search
     * @param {Object} params - Search parameters
     * @param {string} params.query - Search query string
     * @param {number} [params.n_results=5] - Number of results to return
     * @returns {Promise<Object>} Search results
     * @throws {Error} If query is missing or invalid
     */
    async retrieveMemory(params) {
        if (!params || typeof params.query !== 'string' || !params.query.trim()) {
            throw new Error('Search query is required and must be a non-empty string');
        }
        
        return await this.makeRESTRequest('search', 'POST', params);
    }

    /**
     * Search memories by tag with validation
     * @param {Object} params - Search parameters
     * @param {string[]} params.tags - Array of tags to search for
     * @param {number} [params.n_results=10] - Number of results to return
     * @returns {Promise<Object>} Search results matching tags
     * @throws {Error} If tags are missing or invalid
     */
    async searchByTag(params) {
        if (!params || !Array.isArray(params.tags) || params.tags.length === 0) {
            throw new Error('Tags are required and must be a non-empty array of strings');
        }
        
        // Validate all tags are strings
        for (const tag of params.tags) {
            if (typeof tag !== 'string' || !tag.trim()) {
                throw new Error('All tags must be non-empty strings');
            }
        }
        
        return await this.makeRESTRequest('search/by-tag', 'POST', params);
    }

    /**
     * Delete memory
     */
    async deleteMemory(params) {
        // The API uses content_hash in URL path, but we'll need to handle this differently
        // For now, assume we get content_hash in params
        if (params.content_hash) {
            return await this.makeRESTRequest(`memories/${params.content_hash}`, 'DELETE', null);
        } else if (params.ids && params.ids.length > 0) {
            // If we get IDs, assume they are content hashes
            const results = [];
            for (const id of params.ids) {
                try {
                    const result = await this.makeRESTRequest(`memories/${id}`, 'DELETE', null);
                    results.push(result);
                } catch (error) {
                    results.push({ error: error.message, id: id });
                }
            }
            return { deleted: results };
        }
        throw new Error('Delete requires content_hash or ids parameter');
    }

    /**
     * Check database health
     */
    async checkDatabaseHealth(params) {
        // Try detailed health endpoint first, fallback to basic health
        try {
            return await this.makeRESTRequest('health/detailed', 'GET', null);
        } catch (error) {
            try {
                return await this.makeRESTRequest('health', 'GET', null);
            } catch (fallbackError) {
                throw new Error(`Health check failed: ${fallbackError.message}`);
            }
        }
    }

    /**
     * Process MCP JSON-RPC request
     */
    async processRequest(request) {
        const { method, params, id } = request;
        console.error(`Received request method: ${method}`);
        
        let result;
        try {
            switch (method) {
                case 'initialize':
                    result = await this.initialize(params);
                    console.error('Initialized MCP bridge successfully');
                    break;
                case 'shutdown':
                    console.error('Received shutdown request, will continue running');
                    result = {};
                    break;
                case 'notifications/initialized':
                    // Just acknowledge - no response needed for notifications
                    console.error('Received initialized notification');
                    return null; // Don't send a response for notifications
                case 'tools/list':
                    result = await this.listTools(params);
                    break;
                case 'tools/call':
                    result = await this.callTool(params);
                    break;
                case 'resources/list':
                    // We don't provide resources, return empty list
                    result = { resources: [] };
                    break;
                case 'prompts/list':
                    // We don't provide prompts, return empty list  
                    result = { prompts: [] };
                    break;
                // Legacy direct method calls (for backward compatibility)
                case 'store_memory':
                    result = await this.storeMemory(params);
                    break;
                case 'retrieve_memory':
                    result = await this.retrieveMemory(params);
                    break;
                case 'search_by_tag':
                    result = await this.searchByTag(params);
                    break;
                case 'delete_memory':
                    result = await this.deleteMemory(params);
                    break;
                case 'check_database_health':
                    result = await this.checkDatabaseHealth(params);
                    break;
                default:
                    throw new Error(`Unknown method: ${method}`);
            }

            return {
                jsonrpc: "2.0",
                id: id,
                result: result
            };
        } catch (error) {
            console.error(`Error processing ${method}: ${error.message}`);
            return {
                jsonrpc: "2.0",
                id: id,
                error: {
                    code: -32603,
                    message: error.message
                }
            };
        }
    }

    /**
     * Start the bridge
     */
    async start() {
        if (!this.endpoint) {
            throw new Error('MCP_MEMORY_HTTP_ENDPOINT environment variable is required');
        }

        // Test the endpoint first
        console.error('Testing endpoint connectivity...');
        if (await this.testEndpoint(this.endpoint)) {
            console.error('Endpoint is healthy and reachable');
        } else {
            console.error('Warning: Endpoint health check failed, but continuing anyway');
        }

        // Process stdin for MCP requests
        let buffer = '';
        process.stdin.on('data', async (chunk) => {
            buffer += chunk.toString();
            
            while (buffer.includes('\n')) {
                const newlineIndex = buffer.indexOf('\n');
                const line = buffer.slice(0, newlineIndex).trim();
                buffer = buffer.slice(newlineIndex + 1);
                
                if (line) {
                    try {
                        const request = JSON.parse(line);
                        const response = await this.processRequest(request);
                        // Only send response if it's not null (notifications don't need responses)
                        if (response !== null) {
                            console.log(JSON.stringify(response));
                        }
                    } catch (error) {
                        console.error(`Error processing request: ${error.message}`);
                        console.log(JSON.stringify({
                            jsonrpc: "2.0", 
                            id: null,
                            error: {
                                code: -32700,
                                message: "Parse error"
                            }
                        }));
                    }
                }
            }
        });

        // Handle graceful shutdown - but don't exit on SIGINT/SIGTERM from stdin close
        let shutdownRequested = false;
        const shutdown = () => {
            if (shutdownRequested) return;
            shutdownRequested = true;
            console.error('Shutting down Robust HTTP Bridge...');
            setTimeout(() => {
                console.error('Robust HTTP Bridge shutdown complete');
                process.exit(0);
            }, 1000);
        };

        // Only shutdown on explicit termination signals, not stdin close
        process.on('SIGINT', shutdown);
        process.on('SIGTERM', shutdown);
        
        // Keep the process alive
        process.stdin.on('end', () => {
            console.error('stdin closed, but keeping bridge alive for MCP communication');
        });
    }
}

// Start the bridge
const bridge = new RobustMCPBridge();
bridge.start().catch(error => {
    console.error(`Failed to start bridge: ${error.message}`);
    process.exit(1);
});