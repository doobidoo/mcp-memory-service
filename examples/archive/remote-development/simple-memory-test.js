#!/usr/bin/env node
/**
 * Simple test for memory.local connection
 */

const fs = require('fs');
const { spawn } = require('child_process');
const path = require('path');

// Set up environment variables
process.env.MCP_MEMORY_HTTP_ENDPOINT = "https://memory.local/api";
process.env.MCP_MEMORY_API_KEY = "mcp-0b1ccbde2197a08dcb12d41af4044be6";
process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

console.log("Starting test for memory.local connection...");

// Spawn the bridge process
const bridgePath = path.join(__dirname, 'http-mcp-bridge.js');
const bridge = spawn('node', [bridgePath], {
  stdio: ['pipe', 'pipe', 'pipe']
});

// Read bridge output for logging
bridge.stdout.on('data', (data) => {
  console.log(`Bridge output: ${data.toString().trim()}`);
});

bridge.stderr.on('data', (data) => {
  console.error(`Bridge log: ${data.toString().trim()}`);
});

// Wait for bridge to start
setTimeout(() => {
  console.log("Sending initialize request...");
  
  // Send initialize request
  bridge.stdin.write(JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {}
  }) + "\n");
  
  // Wait for response and send next request
  setTimeout(() => {
    console.log("Sending health check request...");
    
    bridge.stdin.write(JSON.stringify({
      jsonrpc: "2.0",
      id: 2,
      method: "check_database_health",
      params: {}
    }) + "\n");
    
    // Wait for response and send next request
    setTimeout(() => {
      console.log("Sending store memory request...");
      
      bridge.stdin.write(JSON.stringify({
        jsonrpc: "2.0",
        id: 3,
        method: "store_memory",
        params: {
          content: "Test memory from simple-memory-test.js",
          metadata: {
            tags: ["test", "remote"],
            type: "note"
          }
        }
      }) + "\n");
      
      // Wait for response and send next request
      setTimeout(() => {
        console.log("Sending retrieve memory request...");
        
        bridge.stdin.write(JSON.stringify({
          jsonrpc: "2.0",
          id: 4,
          method: "retrieve_memory",
          params: {
            query: "test memory",
            n_results: 5
          }
        }) + "\n");
        
        // Wait for final response and end test
        setTimeout(() => {
          console.log("Test complete. Shutting down bridge.");
          bridge.kill();
          process.exit(0);
        }, 3000);
      }, 3000);
    }, 3000);
  }, 3000);
}, 3000);