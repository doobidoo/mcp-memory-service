/**
 * MCP Client for Memory Hook Integration
 * Provides MCP protocol communication for memory operations
 */

const { spawn } = require('child_process');
const { EventEmitter } = require('events');
const fs = require('fs');
const path = require('path');

class MCPClient extends EventEmitter {
    constructor(serverCommand, options = {}) {
        super();
        this.serverCommand = serverCommand;
        this.serverWorkingDir = options.workingDir || process.cwd();
        this.connectionTimeout = options.connectionTimeout || 5000;
        this.toolCallTimeout = options.toolCallTimeout || 10000;
        this.serverProcess = null;
        this.messageId = 0;
        this.pendingRequests = new Map();
        this.connected = false;
        this.buffer = '';

        // Load environment variables from .env file
        this.loadEnvironment();
    }

    /**
     * Load environment variables from .env file
     */
    loadEnvironment() {
        const envPath = path.join(this.serverWorkingDir, '.env');
        try {
            if (fs.existsSync(envPath)) {
                const envContent = fs.readFileSync(envPath, 'utf8');
                const lines = envContent.split('\n');

                for (const line of lines) {
                    const match = line.match(/^([^#\s][^=]*?)=(.*)$/);
                    if (match) {
                        const [, key, value] = match;
                        if (!process.env[key]) {
                            process.env[key] = value.replace(/^["']|["']$/g, '');
                        }
                    }
                }
            }
        } catch (error) {
            // Ignore .env loading errors
        }
    }

    /**
     * Start MCP server and establish connection
     */
    async connect() {
        return new Promise((resolve, reject) => {
            try {
                // Start MCP server process
                this.serverProcess = spawn(this.serverCommand[0], this.serverCommand.slice(1), {
                    cwd: this.serverWorkingDir,
                    stdio: ['pipe', 'pipe', 'pipe'],
                    env: { ...process.env }
                });

                // Handle server output
                this.serverProcess.stdout.on('data', (data) => {
                    this.buffer += data.toString();
                    this.processMessages();
                });

                this.serverProcess.stderr.on('data', (data) => {
                    const error = data.toString();
                    // Only emit critical errors, ignore warnings and debug info
                    if (error.includes('FATAL') || error.includes('ExceptionGroup')) {
                        this.emit('error', new Error(`Server error: ${error.substring(0, 200)}...`));
                    }
                });

                this.serverProcess.on('error', (error) => {
                    if (!this.connected) {
                        reject(new Error(`Server process error: ${error.message}`));
                    }
                });

                this.serverProcess.on('exit', (code) => {
                    this.connected = false;
                    if (code !== 0 && !this.connected) {
                        reject(new Error(`Server failed to start (exit code ${code})`));
                    }
                });

                // Initialize MCP connection
                this.sendInitialize()
                    .then(() => {
                        this.connected = true;
                        resolve();
                    })
                    .catch(reject);

                // Connection timeout
                setTimeout(() => {
                    if (!this.connected) {
                        reject(new Error('Connection timeout'));
                    }
                }, this.connectionTimeout);

            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Process incoming messages from server
     */
    processMessages() {
        const lines = this.buffer.split('\n');
        this.buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
            if (line.trim()) {
                try {
                    const message = JSON.parse(line);
                    this.handleMessage(message);
                } catch (error) {
                    // Ignore malformed messages
                }
            }
        }
    }

    /**
     * Handle incoming MCP messages
     */
    handleMessage(message) {
        if (message.id && this.pendingRequests.has(message.id)) {
            const { resolve, reject } = this.pendingRequests.get(message.id);
            this.pendingRequests.delete(message.id);

            if (message.error) {
                reject(new Error(message.error.message || 'MCP Error'));
            } else {
                resolve(message.result);
            }
        }
    }

    /**
     * Send MCP initialize request
     */
    async sendInitialize() {
        const message = {
            jsonrpc: '2.0',
            id: ++this.messageId,
            method: 'initialize',
            params: {
                protocolVersion: '2024-11-05',
                capabilities: {
                    tools: {}
                },
                clientInfo: {
                    name: 'claude-hooks-mcp-client',
                    version: '1.0.0'
                }
            }
        };

        return this.sendMessage(message);
    }

    /**
     * Send message to MCP server
     */
    async sendMessage(message) {
        return new Promise((resolve, reject) => {
            if (!this.serverProcess || this.serverProcess.exitCode !== null) {
                reject(new Error('Server not running'));
                return;
            }

            if (message.id) {
                this.pendingRequests.set(message.id, { resolve, reject });

                // Timeout for this specific request
                setTimeout(() => {
                    if (this.pendingRequests.has(message.id)) {
                        this.pendingRequests.delete(message.id);
                        reject(new Error('Request timeout'));
                    }
                }, this.toolCallTimeout);
            }

            try {
                const messageStr = JSON.stringify(message) + '\n';
                this.serverProcess.stdin.write(messageStr);

                if (!message.id) {
                    resolve(); // No response expected
                }
            } catch (error) {
                if (message.id && this.pendingRequests.has(message.id)) {
                    this.pendingRequests.delete(message.id);
                }
                reject(error);
            }
        });
    }

    /**
     * Call a tool via MCP
     */
    async callTool(toolName, args = {}) {
        const message = {
            jsonrpc: '2.0',
            id: ++this.messageId,
            method: 'tools/call',
            params: {
                name: toolName,
                arguments: args
            }
        };

        return this.sendMessage(message);
    }

    /**
     * Get memory service health status
     */
    async getHealthStatus() {
        try {
            const result = await this.callTool('check_database_health');
            return {
                success: true,
                data: result.content ? this.parseToolResponse(result.content) : result
            };
        } catch (error) {
            return {
                success: false,
                error: error.message,
                fallback: true
            };
        }
    }

    /**
     * Query memories using semantic search
     */
    async queryMemories(query, limit = 10) {
        try {
            const result = await this.callTool('retrieve_memory', {
                query: query,
                n_results: limit
            });

            return this.parseToolResponse(result.content);
        } catch (error) {
            console.warn('[MCP Client] Memory query error:', error.message);
            return [];
        }
    }

    /**
     * Query memories by time range
     */
    async queryMemoriesByTime(timeQuery, limit = 10) {
        try {
            const result = await this.callTool('recall_memory', {
                query: timeQuery,
                n_results: limit
            });

            return this.parseToolResponse(result.content);
        } catch (error) {
            console.warn('[MCP Client] Time-based memory query error:', error.message);
            return [];
        }
    }

    /**
     * Parse tool response content
     */
    parseToolResponse(content) {
        if (!content) return [];

        // Handle array of content objects
        if (Array.isArray(content)) {
            const textContent = content.find(c => c.type === 'text')?.text || '';
            return this.parseMemoryResults(textContent);
        }

        // Handle direct text content
        if (typeof content === 'string') {
            return this.parseMemoryResults(content);
        }

        // Handle object with text property
        if (content.text) {
            return this.parseMemoryResults(content.text);
        }

        return [];
    }

    /**
     * Parse memory results from text response
     */
    parseMemoryResults(textData) {
        try {
            // Handle Python dict format conversion to JSON
            let cleanText = textData
                .replace(/'/g, '"')
                .replace(/True/g, 'true')
                .replace(/False/g, 'false')
                .replace(/None/g, 'null');

            const parsed = JSON.parse(cleanText);
            return parsed.results || parsed.memories || parsed || [];
        } catch (error) {
            // Try to extract JSON from text
            const jsonMatch = textData.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                try {
                    const extracted = JSON.parse(jsonMatch[0]);
                    return extracted.results || extracted.memories || [extracted];
                } catch {}
            }

            console.warn('[MCP Client] Could not parse memory results:', error.message);
            return [];
        }
    }

    /**
     * Disconnect and cleanup
     */
    async disconnect() {
        this.connected = false;

        // Clear pending requests
        for (const [id, { reject }] of this.pendingRequests) {
            reject(new Error('Connection closed'));
        }
        this.pendingRequests.clear();

        // Terminate server process
        if (this.serverProcess && this.serverProcess.exitCode === null) {
            this.serverProcess.kill('SIGTERM');

            // Force kill if doesn't exit gracefully
            setTimeout(() => {
                if (this.serverProcess && this.serverProcess.exitCode === null) {
                    this.serverProcess.kill('SIGKILL');
                }
            }, 2000);
        }
    }
}

module.exports = { MCPClient };
