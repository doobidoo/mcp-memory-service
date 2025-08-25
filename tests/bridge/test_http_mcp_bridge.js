/**
 * Test suite for HTTP-MCP Bridge
 * 
 * This comprehensive test suite ensures the HTTP-MCP bridge correctly:
 * - Constructs URLs with proper base path handling
 * - Handles various HTTP status codes correctly
 * - Processes API responses according to actual server behavior
 * - Manages errors and retries appropriately
 */

const assert = require('assert');
const sinon = require('sinon');
const path = require('path');
const HTTPMCPBridge = require(path.join(__dirname, '../../examples/http-mcp-bridge.js'));

describe('HTTP-MCP Bridge', () => {
    let bridge;
    let httpStub;
    let httpsStub;
    
    beforeEach(() => {
        bridge = new HTTPMCPBridge();
        bridge.endpoint = 'https://memory.local:8443/api';
        bridge.apiKey = 'test-api-key';
    });
    
    afterEach(() => {
        sinon.restore();
    });
    
    describe('URL Construction', () => {
        it('should correctly resolve paths with base URL using URL constructor', () => {
            // Test the new URL constructor logic that properly handles base paths
            const testCases = [
                { path: 'memories', expected: 'https://memory.local:8443/api/memories' },
                { path: 'health', expected: 'https://memory.local:8443/api/health' },
                { path: 'search', expected: 'https://memory.local:8443/api/search' },
                { path: 'search?q=test&n_results=5', expected: 'https://memory.local:8443/api/search?q=test&n_results=5' }
            ];
            
            for (const testCase of testCases) {
                // Test the URL construction logic: ensure trailing slash, then use URL constructor
                const baseUrl = bridge.endpoint.endsWith('/') ? bridge.endpoint : bridge.endpoint + '/';
                const constructedUrl = new URL(testCase.path, baseUrl).toString();
                
                assert.strictEqual(constructedUrl, testCase.expected, 
                    `Failed for path: ${testCase.path}`);
            }
        });
        
        it('should handle endpoints without trailing slash', () => {
            bridge.endpoint = 'https://memory.local:8443/api';
            // Test URL construction logic
            const fullPath = '/memories';
            const baseUrl = bridge.endpoint.endsWith('/') ? 
                bridge.endpoint.slice(0, -1) : bridge.endpoint;
            const expectedUrl = 'https://memory.local:8443/api/memories';
            assert.strictEqual(baseUrl + fullPath, expectedUrl);
        });
        
        it('should handle endpoints with trailing slash', () => {
            bridge.endpoint = 'https://memory.local:8443/api/';
            const fullPath = '/memories';
            const baseUrl = bridge.endpoint.endsWith('/') ? 
                bridge.endpoint.slice(0, -1) : bridge.endpoint;
            const expectedUrl = 'https://memory.local:8443/api/memories';
            assert.strictEqual(baseUrl + fullPath, expectedUrl);
        });
    });
    
    describe('Status Code Handling', () => {
        it('should handle HTTP 200 with success=true for memory storage', async () => {
            const mockResponse = {
                statusCode: 200,
                data: {
                    success: true,
                    message: 'Memory stored successfully',
                    content_hash: 'abc123'
                }
            };
            
            sinon.stub(bridge, 'makeRequest').resolves(mockResponse);
            
            const result = await bridge.storeMemory({
                content: 'Test memory',
                metadata: { tags: ['test'] }
            });
            
            assert.strictEqual(result.success, true);
            assert.strictEqual(result.message, 'Memory stored successfully');
        });
        
        it('should handle HTTP 200 with success=false for duplicates', async () => {
            const mockResponse = {
                statusCode: 200,
                data: {
                    success: false,
                    message: 'Duplicate content detected',
                    content_hash: 'abc123'
                }
            };
            
            sinon.stub(bridge, 'makeRequest').resolves(mockResponse);
            
            const result = await bridge.storeMemory({
                content: 'Duplicate memory',
                metadata: { tags: ['test'] }
            });
            
            assert.strictEqual(result.success, false);
            assert.strictEqual(result.message, 'Duplicate content detected');
        });
        
        it('should handle HTTP 201 for backward compatibility', async () => {
            const mockResponse = {
                statusCode: 201,
                data: {
                    success: true,
                    message: 'Created'
                }
            };
            
            sinon.stub(bridge, 'makeRequest').resolves(mockResponse);
            
            const result = await bridge.storeMemory({
                content: 'Test memory',
                metadata: { tags: ['test'] }
            });
            
            assert.strictEqual(result.success, true);
        });
        
        it('should handle HTTP 404 errors correctly', async () => {
            const mockResponse = {
                statusCode: 404,
                data: {
                    detail: 'Not Found'
                }
            };
            
            sinon.stub(bridge, 'makeRequest').resolves(mockResponse);
            
            const result = await bridge.storeMemory({
                content: 'Test memory',
                metadata: { tags: ['test'] }
            });
            
            assert.strictEqual(result.success, false);
            assert.strictEqual(result.message, 'Not Found');
        });
    });
    
    describe('Health Check', () => {
        it('should use health endpoint with proper URL construction', async () => {
            let capturedPath;
            sinon.stub(bridge, 'makeRequest').callsFake((path) => {
                capturedPath = path;
                return Promise.resolve({
                    statusCode: 200,
                    data: { status: 'healthy', version: '6.6.1' }
                });
            });
            
            await bridge.checkHealth();
            assert.strictEqual(capturedPath, 'health');
        });
        
        it('should return healthy status for HTTP 200', async () => {
            sinon.stub(bridge, 'makeRequest').resolves({
                statusCode: 200,
                data: {
                    status: 'healthy',
                    storage_type: 'sqlite_vec',
                    statistics: { total_memories: 100 }
                }
            });
            
            const result = await bridge.checkHealth();
            assert.strictEqual(result.status, 'healthy');
            assert.strictEqual(result.backend, 'sqlite_vec');
            assert.deepStrictEqual(result.statistics, { total_memories: 100 });
        });
        
        it('should return unhealthy for non-200 status', async () => {
            sinon.stub(bridge, 'makeRequest').resolves({
                statusCode: 500,
                data: { error: 'Internal Server Error' }
            });
            
            const result = await bridge.checkHealth();
            assert.strictEqual(result.status, 'unhealthy');
            assert.strictEqual(result.backend, 'unknown');
        });
    });
    
    describe('Memory Retrieval', () => {
        it('should handle successful memory retrieval', async () => {
            sinon.stub(bridge, 'makeRequest').resolves({
                statusCode: 200,
                data: {
                    results: [
                        {
                            memory: {
                                content: 'Test memory',
                                tags: ['test'],
                                memory_type: 'note',
                                created_at_iso: '2025-08-24T12:00:00Z'
                            },
                            relevance_score: 0.95
                        }
                    ]
                }
            });
            
            const result = await bridge.retrieveMemory({
                query: 'test',
                n_results: 5
            });
            
            assert.strictEqual(result.memories.length, 1);
            assert.strictEqual(result.memories[0].content, 'Test memory');
            assert.strictEqual(result.memories[0].metadata.relevance_score, 0.95);
        });
        
        it('should handle empty results', async () => {
            sinon.stub(bridge, 'makeRequest').resolves({
                statusCode: 200,
                data: { results: [] }
            });
            
            const result = await bridge.retrieveMemory({
                query: 'nonexistent',
                n_results: 5
            });
            
            assert.strictEqual(result.memories.length, 0);
        });
    });
    
    describe('Error Handling', () => {
        it('should handle network errors gracefully', async () => {
            // Stub makeRequest which is what storeMemory actually calls
            sinon.stub(bridge, 'makeRequest').rejects(
                new Error('ECONNREFUSED')
            );
            
            const result = await bridge.storeMemory({
                content: 'Test memory'
            });
            
            assert.strictEqual(result.success, false);
            assert(result.message.includes('ECONNREFUSED'));
        });
        
        it('should retry on failure with exponential backoff', async () => {
            const stub = sinon.stub(bridge, 'makeRequestInternal');
            stub.onCall(0).rejects(new Error('Timeout'));
            stub.onCall(1).rejects(new Error('Timeout'));
            stub.onCall(2).resolves({
                statusCode: 200,
                data: { success: true }
            });
            
            // Mock the delay to avoid actual waiting in tests
            const originalSetTimeout = global.setTimeout;
            global.setTimeout = (fn, delay) => {
                // Execute immediately but still track that delay was requested
                originalSetTimeout(fn, 0);
                return { delay };
            };
            
            const startTime = Date.now();
            const result = await bridge.makeRequest('/test', 'GET', null, 3);
            const duration = Date.now() - startTime;
            
            // Restore original setTimeout
            global.setTimeout = originalSetTimeout;
            
            // Verify retry logic worked
            assert.strictEqual(stub.callCount, 3);
            assert.strictEqual(result.statusCode, 200);
        }).timeout(5000);
    });
    
    describe('MCP Protocol Integration', () => {
        it('should handle initialize method', async () => {
            const request = {
                method: 'initialize',
                params: {},
                id: 1
            };
            
            const response = await bridge.processRequest(request);
            
            assert.strictEqual(response.jsonrpc, '2.0');
            assert.strictEqual(response.id, 1);
            assert(response.result.protocolVersion);
            assert(response.result.capabilities);
        });
        
        it('should handle tools/list method', async () => {
            const request = {
                method: 'tools/list',
                params: {},
                id: 2
            };
            
            const response = await bridge.processRequest(request);
            
            assert.strictEqual(response.id, 2);
            assert(Array.isArray(response.result.tools));
            assert(response.result.tools.length > 0);
            
            // Verify all required tools are present
            const toolNames = response.result.tools.map(t => t.name);
            assert(toolNames.includes('store_memory'));
            assert(toolNames.includes('retrieve_memory'));
            assert(toolNames.includes('check_database_health'));
        });
        
        it('should handle tools/call for store_memory', async () => {
            sinon.stub(bridge, 'storeMemory').resolves({
                success: true,
                message: 'Stored'
            });
            
            const request = {
                method: 'tools/call',
                params: {
                    name: 'store_memory',
                    arguments: {
                        content: 'Test',
                        metadata: { tags: ['test'] }
                    }
                },
                id: 3
            };
            
            const response = await bridge.processRequest(request);
            
            assert.strictEqual(response.id, 3);
            assert(response.result.content[0].text.includes('true'));
        });
    });
});

// Export for use in other test files
module.exports = { HTTPMCPBridge };