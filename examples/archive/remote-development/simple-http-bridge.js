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
     