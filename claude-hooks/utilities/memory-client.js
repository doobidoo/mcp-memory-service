/**
 * Unified Memory Client
 * Supports both HTTP and MCP protocols with automatic fallback
 */

const https = require('https');
const http = require('http');
const { MCPClient } = require('./mcp-client');

class MemoryClient {
    constructor(config) {
        this.config = config;
        this.protocol = config.protocol || 'auto';
        this.preferredProtocol = config.preferredProtocol || 'mcp';
        this.fallbackEnabled = config.fallbackEnabled !== false;
        this.httpConfig = config.http || {};
        this.mcpConfig = config.mcp || {};

        // Connection state
        this.activeProtocol = null;
        this.httpAvailable = null;
        this.mcpAvailable = null;
        this.mcpClient = null;

        // Cache successful connections
        this.connectionCache = new Map();
    }

    /**
     * Initialize connection using the configured protocol
     */
    async connect() {
        if (this.protocol === 'http') {
            return this.connectHTTP();
        } else if (this.protocol === 'mcp') {
            return this.connectMCP();
        } else {
            // Auto mode: try preferred first, then fallback
            return this.connectAuto();
        }
    }

    /**
     * Auto-connect: try preferred protocol first, fallback if needed
     */
    async connectAuto() {
        const protocols = this.preferredProtocol === 'mcp' ? ['mcp', 'http'] : ['http', 'mcp'];

        for (const protocol of protocols) {
            try {
                if (protocol === 'mcp') {
                    await this.connectMCP();
                    this.activeProtocol = 'mcp';
                    return { protocol: 'mcp', client: this.mcpClient };
                } else {
                    await this.connectHTTP();
                    this.activeProtocol = 'http';
                    return { protocol: 'http', client: null };
                }
            } catch (error) {
                if (!this.fallbackEnabled || protocols.length === 1) {
                    throw error;
                }
                // Continue to try next protocol
                continue;
            }
        }

        throw new Error('Failed to connect using any available protocol');
    }

    /**
     * Connect using MCP protocol
     */
    async connectMCP() {
        if (this.mcpClient) {
            return this.mcpClient;
        }

        this.mcpClient = new MCPClient(
            this.mcpConfig.serverCommand,
            {
                workingDir: this.mcpConfig.serverWorkingDir,
                connectionTimeout: this.mcpConfig.connectionTimeout || 5000,
                toolCallTimeout: this.mcpConfig.toolCallTimeout || 10000
            }
        );

        // Handle MCP client errors gracefully
        this.mcpClient.on('error', (error) => {
            this.mcpAvailable = false;
        });

        await this.mcpClient.connect();
        this.mcpAvailable = true;
        this.activeProtocol = 'mcp';
        return this.mcpClient;
    }

    /**
     * Connect using HTTP protocol
     */
    async connectHTTP() {
        // Test HTTP connection with a simple health check
        const healthResult = await this.queryHealthHTTP();
        if (!healthResult.success) {
            throw new Error(`HTTP connection failed: ${healthResult.error}`);
        }
        this.httpAvailable = true;
        this.activeProtocol = 'http';
        return true;
    }

    /**
     * Query health status using active protocol
     */
    async getHealthStatus() {
        if (this.activeProtocol === 'mcp' && this.mcpClient) {
            return this.mcpClient.getHealthStatus();
        } else if (this.activeProtocol === 'http') {
            return this.queryHealthHTTP();
        } else {
            throw new Error('No active connection available');
        }
    }

    /**
     * Query health via HTTP with automatic HTTPS → HTTP fallback
     */
    async queryHealthHTTP() {
        const healthPath = this.httpConfig.useDetailedHealthCheck ?
            '/api/health/detailed' : '/api/health';

        // Parse the configured endpoint to extract protocol, host, and port
        let endpointUrl;
        try {
            endpointUrl = new URL(this.httpConfig.endpoint);
        } catch (error) {
            return { success: false, error: `Invalid endpoint URL: ${this.httpConfig.endpoint}` };
        }

        // Try with configured protocol first
        const result = await this._attemptHealthCheck(endpointUrl, healthPath);

        // If HTTPS failed, try HTTP fallback on same host:port
        if (!result.success && endpointUrl.protocol === 'https:') {
            const httpUrl = new URL(endpointUrl);
            httpUrl.protocol = 'http:';
            return this._attemptHealthCheck(httpUrl, healthPath);
        }

        return result;
    }

    /**
     * Attempt health check with specific protocol/host/port
     * @private
     */
    async _attemptHealthCheck(baseUrl, healthPath) {
        return new Promise((resolve) => {
            try {
                const url = new URL(healthPath, baseUrl);

                const requestOptions = {
                    hostname: url.hostname,
                    port: url.port || (url.protocol === 'https:' ? 8443 : 8889),
                    path: url.pathname,
                    method: 'GET',
                    headers: {
                        'X-API-Key': this.httpConfig.apiKey,
                        'Accept': 'application/json'
                    },
                    timeout: this.httpConfig.healthCheckTimeout || 3000,
                    rejectUnauthorized: false  // Allow self-signed certificates
                };

                const protocol = url.protocol === 'https:' ? https : http;
                const req = protocol.request(requestOptions, (res) => {
                    let data = '';
                    res.on('data', (chunk) => data += chunk);
                    res.on('end', () => {
                        try {
                            if (res.statusCode === 200) {
                                const healthData = JSON.parse(data);
                                resolve({ success: true, data: healthData });
                            } else {
                                resolve({ success: false, error: `HTTP ${res.statusCode}`, fallback: true });
                            }
                        } catch (parseError) {
                            resolve({ success: false, error: 'Invalid JSON response', fallback: true });
                        }
                    });
                });

                req.on('error', (error) => {
                    resolve({ success: false, error: error.message, fallback: true });
                });

                req.on('timeout', () => {
                    req.destroy();
                    resolve({ success: false, error: 'Health check timeout', fallback: true });
                });

                req.end();

            } catch (error) {
                resolve({ success: false, error: error.message, fallback: true });
            }
        });
    }

    /**
     * Query memories using active protocol
     */
    async queryMemories(query, limit = 10) {
        if (this.activeProtocol === 'mcp' && this.mcpClient) {
            return this.mcpClient.queryMemories(query, limit);
        } else if (this.activeProtocol === 'http') {
            return this.queryMemoriesHTTP(query, limit);
        } else {
            throw new Error('No active connection available');
        }
    }

    /**
     * Query memories by time using active protocol
     */
    async queryMemoriesByTime(timeQuery, limit = 10, semanticQuery = null) {
        if (this.activeProtocol === 'mcp' && this.mcpClient) {
            // TODO: Update MCP client to support semantic query parameter
            return this.mcpClient.queryMemoriesByTime(timeQuery, limit);
        } else if (this.activeProtocol === 'http') {
            return this.queryMemoriesByTimeHTTP(timeQuery, limit, semanticQuery);
        } else {
            throw new Error('No active connection available');
        }
    }

    /**
     * Private helper: Perform HTTP POST request to API
     * @private
     */
    _performApiPost(path, payload) {
        return new Promise((resolve) => {
            const url = new URL(path, this.httpConfig.endpoint);
            const postData = JSON.stringify(payload);

            const options = {
                hostname: url.hostname,
                port: url.port || (url.protocol === 'https:' ? 8443 : 8889),
                path: url.pathname,
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': Buffer.byteLength(postData),
                    'X-API-Key': this.httpConfig.apiKey
                },
                rejectUnauthorized: false  // Allow self-signed certificates
            };

            const protocol = url.protocol === 'https:' ? https : http;
            const req = protocol.request(options, (res) => {
                let data = '';
                res.on('data', (chunk) => data += chunk);
                res.on('end', () => {
                    try {
                        const response = JSON.parse(data);
                        // REST API returns { results: [{memory: {...}, similarity_score: ...}] }
                        if (response.results && Array.isArray(response.results)) {
                            // Extract memory objects from results and preserve similarity_score
                            const memories = response.results
                                .filter(result => result && result.memory)
                                .map(result => ({
                                    ...result.memory,
                                    similarity_score: result.similarity_score
                                }));
                            resolve(memories);
                        } else {
                            resolve([]);
                        }
                    } catch (parseError) {
                        console.warn('[Memory Client] HTTP parse error:', parseError.message);
                        resolve([]);
                    }
                });
            });

            req.on('error', (error) => {
                console.warn('[Memory Client] HTTP network error:', error.message);
                resolve([]);
            });

            req.write(postData);
            req.end();
        });
    }

    /**
     * Query memories via HTTP REST API
     */
    async queryMemoriesHTTP(query, limit = 10) {
        return this._performApiPost('/api/search', {
            query: query,
            n_results: limit
        });
    }

    /**
     * Query memories by time via HTTP REST API
     */
    async queryMemoriesByTimeHTTP(timeQuery, limit = 10, semanticQuery = null) {
        const payload = {
            query: timeQuery,
            n_results: limit
        };

        // Add semantic query if provided for relevance filtering
        if (semanticQuery) {
            payload.semantic_query = semanticQuery;
        }

        return this._performApiPost('/api/search/by-time', payload);
    }

    /**
     * Get connection status and available protocols
     */
    getConnectionInfo() {
        return {
            activeProtocol: this.activeProtocol,
            httpAvailable: this.httpAvailable,
            mcpAvailable: this.mcpAvailable,
            fallbackEnabled: this.fallbackEnabled,
            preferredProtocol: this.preferredProtocol
        };
    }

    /**
     * Disconnect from active protocol
     */
    async disconnect() {
        if (this.mcpClient) {
            try {
                await this.mcpClient.disconnect();
            } catch (error) {
                // Ignore cleanup errors
            }
            this.mcpClient = null;
        }

        this.activeProtocol = null;
        this.httpAvailable = null;
        this.mcpAvailable = null;
        this.connectionCache.clear();
    }
}

module.exports = { MemoryClient };