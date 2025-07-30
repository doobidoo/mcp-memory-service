#!/usr/bin/env node
/**
 * Simple test for MCP bridge
 */

const { spawn } = require('child_process');

// Bridge path
const bridgePath = '/Users/hkr/Documents/GitHub/mcp-memory-service/examples/simple-http-bridge.js';

// Environment variables
const env = {
  ...process.env,
  'NODE_TLS_REJECT_UNAUTHORIZED': '0',
  'MCP_MEMORY_HTTP_ENDPOINT': 'https://memory.local/api',
  'MCP_MEMORY_API_KEY': 'mcp-0b1ccbde2197a08dcb12d41af4044be6'
};

// Start the bridge
const bridge = spawn('node', [bridgePath], {
  env,
  stdio: ['pipe', 'pipe', 'inherit']
});

// Initialize request
const initRequest = {
  jsonrpc: '2.0',
  id: 1,
  method: 'initialize',
  params: {
    protocolVersion: '2025-06-18',
    capabilities: {},
    clientInfo: {
      name: 'test-client',
      version: '1.0.0'
    }
  }
};

// Health check request
const healthRequest = {
  jsonrpc: '2.0',
  id: 2,
  method: 'check_database_health',
  params: {}
};

// Handle bridge output
bridge.stdout.on('data', (data) => {
  console.log('Bridge response:', data.toString());
});

// Send requests
setTimeout(() => {
  console.log('Sending initialize request...');
  bridge.stdin.write(JSON.stringify(initRequest) + '\n');
  
  setTimeout(() => {
    console.log('Sending health check request...');
    bridge.stdin.write(JSON.stringify(healthRequest) + '\n');
    
    // Close after receiving responses
    setTimeout(() => {
      console.log('Test complete.');
      bridge.kill();
      process.exit(0);
    }, 2000);
  }, 1000);
}, 1000);

// Handle bridge exit
bridge.on('exit', (code) => {
  console.log(`Bridge exited with code ${code}`);
  process.exit(code);
});