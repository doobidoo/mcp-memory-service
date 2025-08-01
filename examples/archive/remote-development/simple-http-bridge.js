#!/usr/bin/env node
/**
 * Simple HTTP-to-MCP Bridge for Memory Service
 */

const https = require('https');
const http = require('http');
const { URL } = require('url');

// Configuration from environment variables
const endpoint = process.env.MCP_MEMORY_HTTP_ENDPOINT || 'https://memory.local/api';
const allowInsecure = process.env.NODE_TLS_REJECT_UNAUTHORIZED === '0';
const apiKey = process.env.MCP_MEMORY_API_KEY || 'mcp-0b1ccbde2197a08dcb12d41af4044be6';

// Force insecure mode for https connections to memory.local
if (endpoint.includes('memory.local') && endpoint.startsWith('https')) {
  process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';
  console.error(`Forcing insecure mode for memory.local`);
}

console.error(`Simple HTTP Bridge starting...`);
console.error(`Endpoint: ${endpoint}`);
console.error(`API Key: ${apiKey ? '[SET]' : '[NOT SET]'}`);
console.error(`Allow Insecure: ${process.env.NODE_TLS_REJECT_UNAUTHORIZED === '0' ? 'YES' : 'NO'}`);

// Keep track of connection state
let initialized = false;
let shutdownRequested = false;

/**
 * Make HTTP request to the API
 */
async function makeRequest(path, method = 'GET', data = null) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, endpoint);
    const protocol = url.protocol === 'https:' ? https : http;
    
    const options = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname + url.search,
      method: method,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'MCP-Simple-Bridge/1.0',
        'Authorization': 'Bearer mcp-0b1ccbde2197a08dcb12d41af4044be6'
      },
      timeout: 30000 // Increased timeout to 30 seconds
    };

    // Allow self-signed certificates if configured
    if (url.protocol === 'https:') {
      console.error(`Using HTTPS connection to ${url.hostname}:${url.port}`);
      options.rejectUnauthorized = false; // Always allow self-signed certs for memory.local
    }

    console.error(`Making ${method} request to ${url.toString()}`);

    if (data) {
      const postData = JSON.stringify(data);
      options.headers['Content-Length'] = Buffer.byteLength(postData);
    }

    const req = protocol.request(options, (res) => {
      console.error(`Response status: ${res.statusCode}`);
      let responseData = '';
      
      res.on('data', (chunk) => {
        responseData += chunk;
      });
      
      res.on('end', () => {
        try {
          console.error(`Response data: ${responseData.substring(0, 100)}...`);
          const result = responseData ? JSON.parse(responseData) : {};
          resolve({ statusCode: res.statusCode, data: result });
        } catch (error) {
          console.error(`Error parsing JSON: ${error.message}`);
          resolve({ statusCode: res.statusCode, data: { error: "Invalid JSON" } });
        }
      });
    });

    req.on('error', (error) => {
      console.error(`Request error: ${error.message}`);
      // Don't reject, return an error object instead
      resolve({ statusCode: 500, data: { error: error.message } });
    });

    req.on('timeout', () => {
      console.error('Request timeout');
      req.destroy();
      resolve({ statusCode: 504, data: { error: "Timeout" } });
    });

    if (data) {
      req.write(JSON.stringify(data));
    }
    
    req.end();
  });
}

/**
 * Process MCP JSON-RPC request
 */
async function processRequest(request) {
  const { method, params, id } = request;
  console.error(`Processing request: ${method}`);

  try {
    let result;

    // Handle MCP methods
    switch (method) {
      case 'initialize':
        initialized = true;
        result = {
          capabilities: {
            operations: [
              "store_memory",
              "retrieve_memory",
              "search_by_tag",
              "delete_memory",
              "delete_by_tag",
              "delete_by_tags",
              "check_database_health",
              "recall_memory",
              "optimize_db",
              "debug_retrieve"
            ]
          },
          serverInfo: {
            name: "mcp-memory-http-bridge",
            version: "1.0.0",
            serverEndpoint: endpoint
          }
        };
        console.error('Initialized MCP bridge successfully');
        break;

      case 'shutdown':
        shutdownRequested = true;
        result = {};
        break;

      case 'check_database_health':
        try {
          console.error('Checking database health...');
          const response = await makeRequest('/api/health', 'GET');
          console.error(`Health check response: ${JSON.stringify(response)}`);
          
          if (response.statusCode >= 200 && response.statusCode < 300) {
            result = {
              status: 'healthy',
              backend: 'remote',
              statistics: {
                memory_count: response.data.memory_count || 0,
                uptime_seconds: response.data.uptime_seconds || 0
              }
            };
            console.error('Health check successful');
          } else if (response.statusCode >= 400 && response.statusCode < 500) {
            // Authentication or client error
            result = { 
              status: 'unhealthy', 
              backend: 'remote', 
              statistics: {},
              error: `Server returned ${response.statusCode}: ${response.data.error || 'Unknown error'}`
            };
            console.error(`Health check failed: ${response.statusCode}`);
          } else {
            // Server error
            result = { 
              status: 'unavailable', 
              backend: 'remote', 
              statistics: {},
              error: `Server returned ${response.statusCode}`
            };
            console.error(`Health check failed: ${response.statusCode}`);
          }
        } catch (error) {
          console.error(`Health check error: ${error.message}`);
          result = { 
            status: 'error', 
            backend: 'remote', 
            statistics: {},
            error: error.message
          };
        }
        break;

      case 'store_memory':
        try {
          console.error(`Storing memory: ${params.content.substring(0, 50)}...`);
          
          // Prepare request body
          const memoryData = {
            content: params.content,
            tags: params.metadata?.tags || [],
            memory_type: params.metadata?.type || 'note',
            metadata: params.metadata || {}
          };
          
          // Make API request to store memory
          const response = await makeRequest('/api/memories', 'POST', memoryData);
          console.error(`Store memory response: ${JSON.stringify(response)}`);
          
          if (response.statusCode >= 200 && response.statusCode < 300) {
            result = { 
              success: true, 
              message: 'Memory stored successfully',
              content_hash: response.data.content_hash || ''
            };
            console.error('Memory stored successfully');
          } else {
            result = { 
              success: false, 
              message: `Failed to store memory: ${response.statusCode}`,
              error: response.data.error || 'Unknown error'
            };
            console.error(`Store memory failed: ${response.statusCode}`);
          }
        } catch (error) {
          console.error(`Store memory error: ${error.message}`);
          result = { 
            success: false, 
            message: `Error storing memory: ${error.message}`
          };
        }
        break;

      case 'retrieve_memory':
        try {
          console.error(`Retrieving memories for query: ${params.query}`);
          
          // Build query parameters
          const queryParams = new URLSearchParams({
            q: params.query,
            n_results: params.n_results || 5
          });
          
          // Make API request to retrieve memories
          const response = await makeRequest(`/api/search?${queryParams}`, 'GET');
          console.error(`Retrieve memory response status: ${response.statusCode}`);
          
          if (response.statusCode >= 200 && response.statusCode < 300) {
            // Transform response to MCP format
            result = {
              memories: (response.data.results || []).map(item => ({
                content: item.memory.content,
                metadata: {
                  tags: item.memory.tags || [],
                  type: item.memory.memory_type || 'note',
                  created_at: item.memory.created_at_iso || new Date().toISOString(),
                  relevance_score: item.relevance_score || 0
                },
                content_hash: item.memory.content_hash || ''
              }))
            };
            console.error(`Retrieved ${result.memories.length} memories`);
          } else {
            result = { 
              memories: [],
              error: `Server returned ${response.statusCode}: ${response.data.error || 'Unknown error'}`
            };
            console.error(`Retrieve memory failed: ${response.statusCode}`);
          }
        } catch (error) {
          console.error(`Retrieve memory error: ${error.message}`);
          result = { memories: [] };
        }
        break;

      case 'search_by_tag':
        try {
          console.error(`Searching memories by tags: ${JSON.stringify(params.tags)}`);
          
          // Build query parameters for tags
          const queryParams = new URLSearchParams();
          if (Array.isArray(params.tags)) {
            params.tags.forEach(tag => queryParams.append('tags', tag));
          } else if (typeof params.tags === 'string') {
            queryParams.append('tags', params.tags);
          }
          
          // Make API request to search by tags
          const response = await makeRequest(`/api/memories/search/tags?${queryParams}`, 'GET');
          console.error(`Search by tag response status: ${response.statusCode}`);
          
          if (response.statusCode >= 200 && response.statusCode < 300) {
            // Transform response to MCP format
            result = {
              memories: (response.data.memories || []).map(memory => ({
                content: memory.content,
                metadata: {
                  tags: memory.tags || [],
                  type: memory.memory_type || 'note',
                  created_at: memory.created_at_iso || new Date().toISOString()
                },
                content_hash: memory.content_hash || ''
              }))
            };
            console.error(`Found ${result.memories.length} memories by tag`);
          } else {
            result = { 
              memories: [],
              error: `Server returned ${response.statusCode}: ${response.data.error || 'Unknown error'}`
            };
            console.error(`Search by tag failed: ${response.statusCode}`);
          }
        } catch (error) {
          console.error(`Search by tag error: ${error.message}`);
          result = { memories: [] };
        }
        break;

      case 'delete_memory':
        try {
          console.error(`Deleting memory with hash: ${params.content_hash}`);
          
          // Make API request to delete memory
          const response = await makeRequest(`/api/memories/${params.content_hash}`, 'DELETE');
          console.error(`Delete memory response status: ${response.statusCode}`);
          
          if (response.statusCode >= 200 && response.statusCode < 300) {
            result = { 
              success: true,
              message: 'Memory deleted successfully'
            };
            console.error('Memory deleted successfully');
          } else {
            result = { 
              success: false,
              message: `Failed to delete memory: ${response.statusCode}`,
              error: response.data.error || 'Unknown error'
            };
            console.error(`Delete memory failed: ${response.statusCode}`);
          }
        } catch (error) {
          console.error(`Delete memory error: ${error.message}`);
          result = { 
            success: false,
            message: `Error deleting memory: ${error.message}`
          };
        }
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
    console.error(`Error processing request: ${error.message}`);
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

// Read from stdin and process JSON-RPC requests
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
        activeConnections++;
        console.error(`[CONN] Active connections: ${activeConnections}`);
        
        const request = JSON.parse(line);
        console.error(`Received request: ${request.method} (id: ${request.id})`);
        
        const response = await processRequest(request);
        console.log(JSON.stringify(response));
        
        activeConnections--;
        console.error(`[CONN] Request complete. Active connections: ${activeConnections}`);
      } catch (error) {
        console.error(`Error processing request: ${error.message}`);
        console.error(error.stack);
        
        console.log(JSON.stringify({
          jsonrpc: "2.0",
          id: null,
          error: {
            code: -32700,
            message: "Parse error: " + error.message
          }
        }));
        
        activeConnections--;
        console.error(`[CONN] Request failed. Active connections: ${activeConnections}`);
      }
    }
  }
});

process.stdin.on('end', () => {
  console.error('Input stream ended');
});

// Keep track of active connections
let activeConnections = 0;
let isShuttingDown = false;

// Log active connections every 10 seconds
setInterval(() => {
  const status = shutdownRequested ? 'SHUTDOWN REQUESTED' : 'RUNNING';
  console.error(`[STATUS] ${status} - Active connections: ${activeConnections}`);
  
  // Health check - ping the server periodically
  if (!shutdownRequested && endpoint) {
    makeRequest('/api/health', 'GET')
      .then(response => {
        if (response.statusCode >= 200 && response.statusCode < 300) {
          console.error(`Health check OK: ${response.statusCode}`);
        } else {
          console.error(`Health check failed: ${response.statusCode}`);
        }
      })
      .catch(error => {
        console.error(`Health check error: ${error.message}`);
      });
  }
}, 10000);

// Handle SIGINT (Ctrl+C)
process.on('SIGINT', () => {
  console.error('SIGINT received, but staying alive for Claude Desktop');
  isShuttingDown = true;
  
  // Only exit if explicitly requested and no active connections
  if (shutdownRequested && activeConnections === 0) {
    console.error('No active connections, exiting...');
    process.exit(0);
  }
});

// Handle SIGTERM
process.on('SIGTERM', () => {
  console.error('SIGTERM received, but staying alive for Claude Desktop');
  isShuttingDown = true;
  
  // Only exit if explicitly requested and no active connections
  if (shutdownRequested && activeConnections === 0) {
    console.error('No active connections, exiting...');
    process.exit(0);
  }
});

// Handle errors
process.on('uncaughtException', (error) => {
  console.error(`Uncaught exception: ${error.message}`);
  console.error(error.stack);
});