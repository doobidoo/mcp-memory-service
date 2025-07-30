#!/usr/bin/env node
/**
 * Test script for MCP memory bridge
 */

// Create an MCP initialize request
const initRequest = {
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: {}
};

// Stringify and send the request
console.log(JSON.stringify(initRequest));

// Create a health check request
setTimeout(() => {
  const healthRequest = {
    jsonrpc: "2.0",
    id: 2,
    method: "check_database_health",
    params: {}
  };
  console.log(JSON.stringify(healthRequest));
}, 1000);

// Create a store memory request
setTimeout(() => {
  const storeRequest = {
    jsonrpc: "2.0",
    id: 3,
    method: "store_memory",
    params: {
      content: "This is a test memory from the remote bridge",
      metadata: {
        tags: ["test", "remote"],
        type: "note"
      }
    }
  };
  console.log(JSON.stringify(storeRequest));
}, 2000);

// Create a retrieve memory request
setTimeout(() => {
  const retrieveRequest = {
    jsonrpc: "2.0",
    id: 4,
    method: "retrieve_memory",
    params: {
      query: "test memory",
      n_results: 5
    }
  };
  console.log(JSON.stringify(retrieveRequest));
}, 3000);

// Parse and display responses
process.stdin.on('data', (chunk) => {
  const lines = chunk.toString().split('\n');
  for (const line of lines) {
    if (line.trim()) {
      try {
        const response = JSON.parse(line);
        console.error(`\nResponse ID: ${response.id}`);
        console.error(JSON.stringify(response.result || response.error, null, 2));
      } catch (error) {
        console.error(`Error parsing response: ${error.message}`);
      }
    }
  }
});