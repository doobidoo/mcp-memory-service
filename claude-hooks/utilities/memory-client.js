/**
 * Unified Memory Client
 * Supports both HTTP and MCP protocols with automatic fallback
 */

const https = require('https');
const http = require('http');
const { MCPClient } = require('./mcp-client');

// TODO: Follow-up — extract shared HTTP helper. _attemptHealthCheck, _performApiPost,
// storeMemoryHTTP, and queryMemoriesHTTP duplicate request construction. See Gemini
// review on PR #735.

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
                    port: url.port || (url.protocol === 'https:' ? 8443 : 8000),
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
     * @param {string} query - Search query
     * @param {number} limit - Maximum results to return
     * @param {object} options - Additional options (qualityBoost, qualityWeight)
     */
    async queryMemories(query, limit = 10, options = {}) {
        if (this.activeProtocol === 'mcp' && this.mcpClient) {
            // MCP doesn't support quality boost yet, fall through to semantic search
            return this.mcpClient.queryMemories(query, limit);
        } else if (this.activeProtocol === 'http') {
            return this.queryMemoriesHTTP(query, limit, options);
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
     * Store a memory using the active protocol.
     * @param {string} content - Memory content
     * @param {object} opts - { tags, memoryType, metadata }
     * @returns {Promise<{success: boolean, contentHash?: string, error?: string}>}
     */
    async storeMemory(content, opts = {}) {
        const { tags = [], memoryType = null, metadata = {} } = opts;
        if (this.activeProtocol === 'mcp' && this.mcpClient) {
            return this.storeMemoryMCP(content, { tags, memoryType, metadata });
        } else if (this.activeProtocol === 'http') {
            return this.storeMemoryHTTP(content, { tags, memoryType, metadata });
        } else {
            throw new Error('No active connection available');
        }
    }

    /**
     * Store memory via HTTP REST API.
     * @private
     */
    storeMemoryHTTP(content, { tags, memoryType, metadata }) {
        return new Promise((resolve) => {
            const url = new URL('/api/memories', this.httpConfig.endpoint);
            const isHttps = url.protocol === 'https:';
            const payload = JSON.stringify({
                content,
                tags,
                memory_type: memoryType,
                metadata,
            });

            const options = {
                hostname: url.hostname,
                port: url.port || (isHttps ? 8443 : 8000),
                path: url.pathname,
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': Buffer.byteLength(payload),
                    'X-API-Key': this.httpConfig.apiKey,
                },
                timeout: 5000,
            };

            if (isHttps) {
                options.rejectUnauthorized = false;
            }

            const requestModule = isHttps ? https : http;
            const req = requestModule.request(options, (res) => {
                let data = '';
                res.on('data', (chunk) => (data += chunk));
                res.on('end', () => {
                    if (res.statusCode >= 200 && res.statusCode < 300) {
                        try {
                            const parsed = JSON.parse(data);
                            resolve({
                                success: parsed.success !== false,
                                contentHash: parsed.content_hash || parsed.contentHash,
                            });
                        } catch (err) {
                            resolve({ success: false, error: `Parse error: ${err.message}` });
                        }
                    } else {
                        resolve({ success: false, error: `HTTP ${res.statusCode}: ${data}` });
                    }
                });
            });

            req.on('error', (error) => {
                resolve({ success: false, error: error.message });
            });

            req.on('timeout', () => {
                req.destroy();
                resolve({ success: false, error: 'Request timeout' });
            });

            req.write(payload);
            req.end();
        });
    }

    /**
     * Store memory via MCP protocol.
     * @private
     */
    async storeMemoryMCP(content, { tags, memoryType, metadata }) {
        try {
            const result = await this.mcpClient.storeMemory(content, {
                tags,
                memoryType,
                metadata,
            });
            return {
                success: result?.success !== false,
                contentHash: result?.content_hash || result?.contentHash,
            };
        } catch (err) {
            return { success: false, error: err.message };
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
                port: url.port || (url.protocol === 'https:' ? 8443 : 8000),
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
                                .map(result => {
                                    const memory = { ...result.memory };

                                    // FIX: API returns Unix timestamps in SECONDS, but JavaScript Date expects MILLISECONDS
                                    // Convert created_at and updated_at from seconds to milliseconds
                                    if (memory.created_at && typeof memory.created_at === 'number') {
                                        // Only convert if value looks like seconds (< year 2100 in milliseconds = 4102444800000)
                                        if (memory.created_at < 4102444800) {
                                            memory.created_at = memory.created_at * 1000;
                                        }
                                    }
                                    if (memory.updated_at && typeof memory.updated_at === 'number') {
                                        if (memory.updated_at < 4102444800) {
                                            memory.updated_at = memory.updated_at * 1000;
                                        }
                                    }

                                    return {
                                        ...memory,
                                        similarity_score: result.similarity_score
                                    };
                                });
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
     * @param {string} query - Search query
     * @param {number} limit - Maximum results to return
     * @param {object} options - Additional options
     * @param {boolean} options.qualityBoost - Enable quality-boosted reranking
     * @param {number} options.qualityWeight - Weight for quality in reranking (0.0-1.0)
     */
    async queryMemoriesHTTP(query, limit = 10, options = {}) {
        const payload = {
            query: query,
            n_results: limit
        };

        // Add quality boost parameters if enabled
        if (options.qualityBoost) {
            payload.quality_boost = true;
            if (typeof options.qualityWeight === 'number') {
                payload.quality_weight = options.qualityWeight;
            }
        }

        return this._performApiPost('/api/search', payload);
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
     * Query memories by tags and time (combined filtering)
     * @param {Array<string>} tags - Tags to filter by
     * @param {string} timeQuery - Time-based query (e.g., "last week", "yesterday")
     * @param {number} limit - Maximum results to return
     * @param {boolean} semanticQuery - Optional semantic query for relevance filtering
     */
    async queryMemoriesByTagsAndTime(tags, timeQuery, limit = 10, semanticQuery = false) {
        if (this.activeProtocol === 'mcp' && this.mcpClient) {
            // For MCP, fall back to time-based query (tag filtering not yet supported)
            return this.mcpClient.queryMemoriesByTime(timeQuery, limit);
        } else if (this.activeProtocol === 'http') {
            try {
                // HTTP API: Query by tag FIRST to get project-specific memories, then filter by time client-side
                // This is more efficient than time-first when tags are highly selective (e.g., project names)
                const tagPayload = {
                    tags: Array.isArray(tags) ? tags : [tags],
                    limit: limit * 4  // Over-fetch to ensure we have enough after time filtering
                };

                // Add semantic query if provided
                if (semanticQuery) {
                    tagPayload.query = semanticQuery;
                }

                const tagResults = await this._performApiPost('/api/search/by-tag', tagPayload);
                const memories = tagResults.results ? tagResults.results.map(r => r.memory) : tagResults;

                // Filter by time window client-side
                const now = new Date();
                const filtered = memories.filter(memory => {
                    const createdAt = new Date(memory.created_at_iso || memory.created_at * 1000);
                    const daysDiff = (now - createdAt) / (1000 * 60 * 60 * 24);

                    // Parse time query (simplified - supports common patterns)
                    if (timeQuery.includes('last week') || timeQuery.includes('last-week')) {
                        return daysDiff <= 7;
                    } else if (timeQuery.includes('last 2 weeks') || timeQuery.includes('last-2-weeks')) {
                        return daysDiff <= 14;
                    } else if (timeQuery.includes('last month') || timeQuery.includes('last-month')) {
                        return daysDiff <= 30;
                    } else if (timeQuery.includes('yesterday')) {
                        return daysDiff <= 1;
                    } else {
                        // Default to last month if query not recognized
                        return daysDiff <= 30;
                    }
                });

                console.log(`[Memory Client] Tag-first filter: ${memories.length} tagged → ${filtered.length} within ${timeQuery}`);
                return filtered.slice(0, limit);
            } catch (error) {
                // If tag search fails, fall back to time-only search
                console.warn('[Memory Client] Tag search failed, falling back to time-only search:', error.message);
                return this.queryMemoriesByTimeHTTP(timeQuery, limit, semanticQuery);
            }
        } else {
            throw new Error('No active connection available');
        }
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