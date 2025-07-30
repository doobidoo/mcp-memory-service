#!/usr/bin/env node
/**
 * HTTP-to-MCP Bridge for MCP Memory Service
 * 
 * This bridge allows MCP clients (like Claude Desktop) to connect to a remote
 * MCP Memory Service HTTP server instead of running a local instance.
 * 
 * Features:
 * - Automatic service discovery via mDNS (Bonjour/Zeroconf)
 * - Manual endpoint configuration fallback
 * - HTTPS support with self-signed certificate handling
 * - API key authentication
 * 
 * Usage in Claude Desktop config:
 * 
 * Option 1: Auto-discovery (recommended for local networks)
 * {
 *   "mcpServers": {
 *     "memory": {
 *       "command": "node",
 *       "args": ["/path/to/http-mcp-bridge.js"],
 *       "env": {
 *         "MCP_MEMORY_AUTO_DISCOVER": "true",
 *         "MCP_MEMORY_PREFER_HTTPS": "true",
 *         "MCP_MEMORY_API_KEY": "your-api-key"
 *       }
 *     }
 *   }
 * }
 * 
 * Option 2: Manual configuration
 * {
 *   "mcpServers": {
 *     "memory": {
 *       "command": "node",
 *       "args": ["/path/to/http-mcp-bridge.js"],
 *       "env": {
 *         "MCP_MEMORY_HTTP_ENDPOINT": "https://your-server:8000/api",
 *         "MCP_MEMORY_API_KEY": "your-api-key"
 *       }
 *     }
 *   }
 * }
 */

const http = require('http');
const https = require('https');
const { URL } = require('url');
const dgram = require('dgram');
const dns = require('dns');

/**
 * Simple mDNS service discovery implementation
 */
class MDNSDiscovery {
    constructor() {
        this.services = new Map();
    }
    
    /**
     * Discover MCP Memory Services using mDNS
     */
    async discoverServices(timeout = 5000) {
        return new Promise((resolve) => {
            const socket = dgram.createSocket('udp4');
            const services = [];
            
            // mDNS query for _mcp-memory._tcp.local
            const query = this.createMDNSQuery('_mcp-memory._tcp.local');
            
            socket.on('message', (msg, rinfo) => {
                try {
                    const service = this.parseMDNSResponse(msg, rinfo);
                    if (service) {
                        services.push(service);
                    }
                } catch (error) {
                    // Ignore parsing errors
                }
            });
            
            socket.bind(() => {
                socket.addMembership('224.0.0.251');
                socket.send(query, 5353, '224.0.0.251');
            });
            
            setTimeout(() => {
                socket.close();
                resolve(services);
            }, timeout);
        });
    }
    
    createMDNSQuery(serviceName) {
        // Simplified mDNS query creation
        // This is a basic implementation - in production, use a proper mDNS library
        const header = Buffer.alloc(12);
        header.writeUInt16BE(0, 0); // Transaction ID
        header.writeUInt16BE(0, 2); // Flags
        header.writeUInt16BE(1, 4); // Questions
        header.writeUInt16BE(0, 6); // Answer RRs
        header.writeUInt16BE(0, 8); // Authority RRs
        header.writeUInt16BE(0, 10); // Additional RRs
        
        // Question section (simplified)
        const nameLabels = serviceName.split('.');
        let nameBuffer = Buffer.alloc(0);
        
        for (const label of nameLabels) {
            if (label) {
                const labelBuffer = Buffer.alloc(1 + label.length);
                labelBuffer.writeUInt8(label.length, 0);
                labelBuffer.write(label, 1);
                nameBuffer = Buffer.concat([nameBuffer, labelBuffer]);
            }
        }
        
        const endBuffer = Buffer.alloc(5);
        endBuffer.writeUInt8(0, 0); // End of name
        endBuffer.writeUInt16BE(12, 1); // Type PTR
        endBuffer.writeUInt16BE(1, 3); // Class IN
        
        return Buffer.concat([header, nameBuffer, endBuffer]);
    }
    
    parseMDNSResponse(msg, rinfo) {
        // Simplified mDNS response parsing
        // This is a basic implementation - in production, use a proper mDNS library
        try {
            // Look for MCP Memory Service indicators in the response
            const msgStr = msg.toString('ascii', 0, Math.min(msg.length, 512));
            if (msgStr.includes('mcp-memory') || msgStr.includes('MCP Memory')) {
                // Try common ports for the service
                const possiblePorts = [8000, 8080, 443, 80];
                const host = rinfo.address;
                
                for (const port of possiblePorts) {
                    return {
                        name: 'MCP Memory Service',
                        host: host,
                        port: port,
                        https: port === 443,
                        discovered: true
                    };
                }
            }
        } catch (error) {
            // Ignore parsing errors
        }
        return null;
    }
}

class HTTPMCPBridge {
    constructor() {
        this.endpoint = process.env.MCP_MEMORY_HTTP_ENDPOINT;
        this.apiKey = process.env.MCP_MEMORY_API_KEY;
        this.autoDiscover = process.env.MCP_MEMORY_AUTO_DISCOVER === 'true';
        this.preferHttps = process.env.MCP_MEMORY_PREFER_HTTPS !== 'false';
        this.requestId = 0;
        this.discovery = new MDNSDiscovery();
        this.discoveredEndpoint = null;
    }

    /**
     * Initialize the bridge by discovering or configuring the endpoint
     */
    async initialize() {
        if (this.endpoint) {
            // Manual configuration takes precedence
            console.error(`Using manual endpoint: ${this.endpoint}`);
            return true;
        }
        
        if (this.autoDiscover) {
            console.error('Attempting to discover MCP Memory Service via mDNS...');
            try {
                const services = await this.discovery.discoverServices();
                
                if (services.length > 0) {
                    // Sort services by preference (HTTPS first if preferred)
                    services.sort((a, b) => {
                        if (this.preferHttps) {
                            if (a.https !== b.https) return b.https - a.https;
                        }
                        return a.port - b.port; // Prefer standard ports
                    });
                    
                    const service = services[0];
                    const protocol = service.https ? 'https' : 'http';
                    this.discoveredEndpoint = `${protocol}://${service.host}:${service.port}/api`;
                    this.endpoint = this.discoveredEndpoint;
                    
                    console.error(`Discovered service: ${this.endpoint}`);
                    
                    // Test the discovered endpoint
                    const healthy = await this.testEndpoint(this.endpoint);
                    if (!healthy) {
                        console.error('Discovered endpoint failed health check, trying alternatives...');
                        
                        // Try other discovered services
                        for (let i = 1; i < services.length; i++) {
                            const altService = services[i];
                            const altProtocol = altService.https ? 'https' : 'http';
                            const altEndpoint = `${altProtocol}://${altService.host}:${altService.port}/api`;
                            
                            if (await this.testEndpoint(altEndpoint)) {
                                this.endpoint = altEndpoint;
                                console.error(`Using alternative endpoint: ${this.endpoint}`);
                                return true;
                            }
                        }
                        
                        console.error('No healthy services found');
                        return false;
                    }
                    
                    return true;
                } else {
                    console.error('No MCP Memory Services discovered');
                    return false;
                }
            } catch (error) {
                console.error(`Discovery failed: ${error.message}`);
                return false;
            }
        }
        
        // Default fallback
        this.endpoint = 'http://localhost:8000/api';
        console.error(`Using default endpoint: ${this.endpoint}`);
        return true;
    }
    
    /**
     * Test if an endpoint is healthy
     */
    async testEndpoint(endpoint) {
        try {
            const healthUrl = `${endpoint}/api/health`;
            const response = await this.makeRequestInternal(healthUrl, 'GET', null, 3000); // 3 second timeout
            return response.statusCode === 200;
        } catch (error) {
            return false;
        }
    }

    /**
     * Make HTTP request to the MCP Memory Service
     */
    async makeRequest(path, method = 'GET', data = null) {
        return this.makeRequestInternal(path, method, data);
    }

    /**
     * Internal HTTP request method with timeout support
     */
    async makeRequestInternal(path, method = 'GET', data = null, timeout = 30000) {
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
                    'User-Agent': 'MCP-HTTP-Bridge/2.0'
                },
                timeout: timeout
            };

            // For HTTPS, allow self-signed certificates in development
            if (url.protocol === 'https:') {
                options.rejectUnauthorized = false;
            }

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
                        const result = JSON.parse(responseData);
                        resolve({ statusCode: res.statusCode, data: result });
                    } catch (error) {
                        reject(new Error(`Invalid JSON response: ${responseData}`));
                    }
                });
            });

            req.on('error', (error) => {
                reject(error);
            });

            req.on('timeout', () => {
                req.destroy();
                reject(new Error('Request timeout'));
            });

            if (data) {
                req.write(JSON.stringify(data));
            }
            
            req.end();
        });
    }

    /**
     * Handle MCP store_memory operation
     */
    async storeMemory(params) {
        try {
            const response = await this.makeRequest('/api/memories', 'POST', {
                content: params.content,
                tags: params.metadata?.tags || [],
                memory_type: params.metadata?.type || 'note',
                metadata: params.metadata || {}
            });

            if (response.statusCode === 201) {
                return { success: true, message: 'Memory stored successfully' };
            } else {
                return { success: false, message: response.data.detail || 'Failed to store memory' };
            }
        } catch (error) {
            return { success: false, message: error.message };
        }
    }

    /**
     * Handle MCP retrieve_memory operation
     */
    async retrieveMemory(params) {
        try {
            const queryParams = new URLSearchParams({
                q: params.query,
                n_results: params.n_results || 5
            });

            const response = await this.makeRequest(`/api/search?${queryParams}`, 'GET');

            if (response.statusCode === 200) {
                return {
                    memories: response.data.results.map(result => ({
                        content: result.memory.content,
                        metadata: {
                            tags: result.memory.tags,
                            type: result.memory.memory_type,
                            created_at: result.memory.created_at_iso,
                            relevance_score: result.relevance_score
                        }
                    }))
                };
            } else {
                return { memories: [] };
            }
        } catch (error) {
            return { memories: [] };
        }
    }

    /**
     * Handle MCP search_by_tag operation
     */
    async searchByTag(params) {
        try {
            const queryParams = new URLSearchParams();
            if (Array.isArray(params.tags)) {
                params.tags.forEach(tag => queryParams.append('tags', tag));
            } else if (typeof params.tags === 'string') {
                queryParams.append('tags', params.tags);
            }

            const response = await this.makeRequest(`/api/memories/search/tags?${queryParams}`, 'GET');

            if (response.statusCode === 200) {
                return {
                    memories: response.data.memories.map(memory => ({
                        content: memory.content,
                        metadata: {
                            tags: memory.tags,
                            type: memory.memory_type,
                            created_at: memory.created_at_iso
                        }
                    }))
                };
            } else {
                return { memories: [] };
            }
        } catch (error) {
            return { memories: [] };
        }
    }

    /**
     * Handle MCP delete_memory operation
     */
    async deleteMemory(params) {
        try {
            const response = await this.makeRequest(`/api/memories/${params.content_hash}`, 'DELETE');

            if (response.statusCode === 200) {
                return { success: true, message: 'Memory deleted successfully' };
            } else {
                return { success: false, message: response.data.detail || 'Failed to delete memory' };
            }
        } catch (error) {
            return { success: false, message: error.message };
        }
    }

    /**
     * Handle MCP check_database_health operation
     */
    async checkHealth(params = {}) {
        try {
            const response = await this.makeRequest('/api/health', 'GET');

            if (response.statusCode === 200) {
                return {
                    status: response.data.status || 'healthy',
                    backend: response.data.storage_type || 'remote',
                    statistics: response.data.statistics || {
                        memory_count: 0,
                        uptime_seconds: response.data.uptime_seconds || 0
                    }
                };
            } else {
                return { status: 'unhealthy', backend: 'remote', statistics: {} };
            }
        } catch (error) {
            return { status: 'error', backend: 'unknown', statistics: {}, error: error.message };
        }
    }

    /**
     * Handle MCP initialize operation
     */
    async initialize(params) {
        return {
            capabilities: {
                operations: [
                    "store_memory",
                    "retrieve_memory",
                    "search_by_tag",
                    "delete_memory",
                    "check_database_health"
                ]
            },
            serverInfo: {
                name: "mcp-memory-http-bridge",
                version: "1.0.0"
            }
        };
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
                    result = await this.checkHealth(params);
                    break;
                default:
                    console.error(`Unknown method: ${method}`);
                    throw new Error(`Unknown method: ${method}`);
            }

            return {
                jsonrpc: "2.0",
                id: id,
                result: result
            };
        } catch (error) {
            return {
                jsonrpc: "2.0",
                id: id,
                error: {
                    code: -32000,
                    message: error.message
                }
            };
        }
    }

    /**
     * Start the bridge server
     */
    async start() {
        console.error(`MCP HTTP Bridge starting...`);
        
        // Initialize the bridge (discovery or manual config)
        const initialized = await this.initialize();
        if (!initialized) {
            console.error('Failed to initialize bridge - no endpoint available');
            process.exit(1);
        }
        
        console.error(`Endpoint: ${this.endpoint}`);
        console.error(`API Key: ${this.apiKey ? '[SET]' : '[NOT SET]'}`);
        console.error(`Auto-discovery: ${this.autoDiscover ? 'ENABLED' : 'DISABLED'}`);
        console.error(`Prefer HTTPS: ${this.preferHttps ? 'YES' : 'NO'}`);
        
        if (this.discoveredEndpoint) {
            console.error(`Service discovered automatically via mDNS`);
        }

        let buffer = '';

        process.stdin.on('data', async (chunk) => {
            buffer += chunk.toString();
            
            // Process complete JSON-RPC messages
            let newlineIndex;
            while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
                const line = buffer.slice(0, newlineIndex).trim();
                buffer = buffer.slice(newlineIndex + 1);
                
                if (line) {
                    try {
                        const request = JSON.parse(line);
                        const response = await this.processRequest(request);
                        console.log(JSON.stringify(response));
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

        process.stdin.on('end', () => {
            process.exit(0);
        });

        // Handle graceful shutdown
        process.on('SIGINT', () => {
            console.error('Shutting down HTTP Bridge...');
            // Don't exit immediately to handle pending requests
            setTimeout(() => {
                console.error('HTTP Bridge shutdown complete');
                process.exit(0);
            }, 1000);
        });

        process.on('SIGTERM', () => {
            console.error('Shutting down HTTP Bridge...');
            // Don't exit immediately to handle pending requests
            setTimeout(() => {
                console.error('HTTP Bridge shutdown complete');
                process.exit(0);
            }, 1000);
        });
        
        // Keep the process alive
        setInterval(() => {
            // Heartbeat to prevent the process from exiting
        }, 10000);
    }
}

// Start the bridge if this file is run directly
if (require.main === module) {
    const bridge = new HTTPMCPBridge();
    bridge.start().catch(error => {
        console.error(`Failed to start bridge: ${error.message}`);
        process.exit(1);
    });
}

module.exports = HTTPMCPBridge;