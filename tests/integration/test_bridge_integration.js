/**
 * Integration Tests for HTTP-MCP Bridge
 * 
 * These tests verify the bridge works correctly with a real server
 * or a mock server that accurately simulates real behavior.
 */

const assert = require('assert');
const http = require('http');
const https = require('https');
const path = require('path');
const HTTPMCPBridge = require(path.join(__dirname, '../../examples/http-mcp-bridge.js'));
const { mockResponses, createMockResponse } = require(path.join(__dirname, '../bridge/mock_responses.js'));

describe('Bridge-Server Integration', () => {
    let bridge;
    let testServer;
    let serverPort;
    
    before(async () => {
        // Create a test server that mimics real API behavior
        await startTestServer();
    });
    
    after(async () => {
        if (testServer) {
            await new Promise(resolve => testServer.close(resolve));
        }
    });
    
    beforeEach(() => {
        bridge = new HTTPMCPBridge();
        bridge.endpoint = `http://localhost:${serverPort}/api`;
        bridge.apiKey = 'test-api-key';
    });
    
    async function startTestServer() {
        return new Promise((resolve) => {
            testServer = http.createServer((req, res) => {
                let body = '';
                
                req.on('data', chunk => {
                    body += chunk.toString();
                });
                
                req.on('end', () => {
                    handleRequest(req, res, body);
                });
            });
            
            testServer.listen(0, 'localhost', () => {
                serverPort = testServer.address().port;
                console.log(`Test server started on port ${serverPort}`);
                resolve();
            });
        });
    }
    
    function handleRequest(req, res, body) {
        const url = req.url;
        const method = req.method;
        
        // Verify API key
        if (req.headers.authorization !== 'Bearer test-api-key') {
            res.writeHead(401, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ detail: 'Unauthorized' }));
            return;
        }
        
        // Route requests
        if (url === '/api/health' && method === 'GET') {
            const response = mockResponses.health.healthy;
            res.writeHead(response.status, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(response.body));
        } else if (url === '/api/memories' && method === 'POST') {
            try {
                const data = JSON.parse(body);
                
                // Simulate duplicate detection
                if (data.content === 'duplicate-content') {
                    const response = mockResponses.memories.duplicate;
                    res.writeHead(response.status, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify(response.body));
                } else {
                    const response = mockResponses.memories.createSuccess;
                    res.writeHead(response.status, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify(response.body));
                }
            } catch (e) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ detail: 'Invalid JSON' }));
            }
        } else if (url === '/api/search' && method === 'POST') {
            const response = mockResponses.search.withResults;
            res.writeHead(response.status, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(response.body));
        } else if (url === '/api/search/by-tag' && method === 'POST') {
            const response = mockResponses.search.withResults;
            res.writeHead(response.status, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(response.body));
        } else if (url === '/health' && method === 'GET') {
            // This is the WRONG endpoint - should return 404
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ detail: 'Not Found' }));
        } else {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ detail: 'Not Found' }));
        }
    }
    
    describe('Critical Bug Scenarios', () => {
        it('should use /api/health not /health for health checks', async () => {
            const result = await bridge.checkHealth();
            assert.strictEqual(result.status, 'healthy');
            assert.strictEqual(result.backend, 'sqlite_vec');
        });
        
        it('should handle HTTP 200 with success field for memory storage', async () => {
            const result = await bridge.storeMemory({
                content: 'Test memory content',
                metadata: { tags: ['test'] }
            });
            
            assert.strictEqual(result.success, true);
            assert.strictEqual(result.message, 'Memory stored successfully');
        });
        
        it('should handle duplicate detection with HTTP 200 and success=false', async () => {
            const result = await bridge.storeMemory({
                content: 'duplicate-content',
                metadata: { tags: ['test'] }
            });
            
            assert.strictEqual(result.success, false);
            assert.strictEqual(result.message, 'Duplicate content detected');
        });
        
        it('should construct URLs correctly with /api base path', async () => {
            // This would have failed with the old URL construction bug
            const result = await bridge.retrieveMemory({
                query: 'test',
                n_results: 5
            });
            
            assert(Array.isArray(result.memories));
            assert(result.memories.length > 0);
        });
    });
    
    describe('End-to-End MCP Protocol Flow', () => {
        it('should handle complete MCP session', async () => {
            // 1. Initialize
            let response = await bridge.processRequest({
                method: 'initialize',
                params: {},
                id: 1
            });
            assert.strictEqual(response.result.protocolVersion, '2024-11-05');
            
            // 2. Get tools list
            response = await bridge.processRequest({
                method: 'tools/list',
                params: {},
                id: 2
            });
            assert(response.result.tools.length > 0);
            
            // 3. Store a memory
            response = await bridge.processRequest({
                method: 'tools/call',
                params: {
                    name: 'store_memory',
                    arguments: {
                        content: 'Integration test memory',
                        metadata: { tags: ['test', 'integration'] }
                    }
                },
                id: 3
            });
            const result = JSON.parse(response.result.content[0].text);
            assert.strictEqual(result.success, true);
            
            // 4. Check health
            response = await bridge.processRequest({
                method: 'tools/call',
                params: {
                    name: 'check_database_health',
                    arguments: {}
                },
                id: 4
            });
            const health = JSON.parse(response.result.content[0].text);
            assert.strictEqual(health.status, 'healthy');
        });
    });
    
    describe('Error Recovery', () => {
        it('should handle server unavailability gracefully', async () => {
            // Point to non-existent server (using port 9999 instead of 99999 which is invalid)
            bridge.endpoint = 'http://localhost:9999/api';

            const result = await bridge.checkHealth();
            assert.strictEqual(result.status, 'error');
            // The error message should indicate connection failure or invalid URL
            assert(result.error && (
                result.error.includes('ECONNREFUSED') ||
                result.error.includes('EADDRNOTAVAIL') ||
                result.error.includes('connect') ||
                result.error.includes('ENOTFOUND') ||
                result.error.includes('Invalid URL') || // This can happen with invalid ports
                result.error.includes('ETIMEDOUT')
            ), `Expected connection error but got: ${result.error}`);
        });
        
        it('should handle malformed responses', async () => {
            // Create a server that returns invalid JSON
            const badServer = http.createServer((req, res) => {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end('This is not JSON');
            });
            
            await new Promise(resolve => {
                badServer.listen(0, 'localhost', resolve);
            });
            
            const badPort = badServer.address().port;
            bridge.endpoint = `http://localhost:${badPort}/api`;
            
            const result = await bridge.checkHealth();
            assert.strictEqual(result.status, 'error');
            
            await new Promise(resolve => badServer.close(resolve));
        });
    });
    
    describe('Authentication', () => {
        it('should include API key in requests', async () => {
            bridge.apiKey = 'test-api-key';
            const result = await bridge.checkHealth();
            assert.strictEqual(result.status, 'healthy');
        });
        
        it('should handle authentication failures', async () => {
            bridge.apiKey = 'wrong-api-key';
            const result = await bridge.checkHealth();
            assert.strictEqual(result.status, 'unhealthy');
        });
    });
});

// Run tests if this file is executed directly
if (require.main === module) {
    // Simple test runner for development
    const Mocha = require('mocha');
    const mocha = new Mocha();
    mocha.addFile(__filename);
    mocha.run(failures => {
        process.exit(failures ? 1 : 0);
    });
}